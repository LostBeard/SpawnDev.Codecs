// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// YUV 4:2:0 (8-bit, BT.601 limited range) -> 16-bit RGB with bilinear chroma, bit-exact with what ComfyUI's LoadVideo
// node produces (its IMAGE tensor is this RGB times 1/65535 in float32). Derived as a black box, without reading
// ComfyUI or FFmpeg: see Docs/comfyui-loadvideo-yuv-to-rgb.md for the measurements and the rule set.
//
// One thread per output pixel. The chroma taps per axis come from a 16.16 position (centred siting, edges clamped,
// nearest when a chroma axis has fewer than 4 samples), 14-bit weights horizontally (the row result truncated to 11
// fraction bits) and 12-bit weights vertically (chroma at 9 fraction bits), then the integer BT.601 matrix. The matrix
// sum needs more than 32 bits; Channel splits it so every term fits an int (equal to the 64-bit form for every input).
// Measured against the plain 64-bit form (2026-11-12, 1080p, identical output): CUDA 0.046 vs 0.086 ms, Velocity and
// the CPU accelerator 2-4% faster, WebGPU equal (1.03-1.15 vs 1.06-1.14 ms, 20 conversions per submit).

using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Images.Yuv;

/// <summary>
/// Converts YUV 4:2:0 planes on the accelerator to RGB the way ComfyUI's LoadVideo does (16-bit RGB, bilinear chroma,
/// BT.601 limited range), as float32 in [0, 1] (<c>value * (1f / 65535f)</c>) or as the 16-bit integers.
/// </summary>
public sealed class Yuv420ToRgb16Gpu
{
    /// <summary>Largest width or height (keeps every position product inside an int).</summary>
    public const int MaxDimension = 16384;

    private Action<Index1D, ArrayView<byte>, ArrayView<float>, int, int, int, int> _toFloat = null!;
    private Action<Index1D, ArrayView<byte>, ArrayView<int>, int, int, int, int> _toInt = null!;
    private readonly Accelerator _accelerator;
    private Task? _kernelsLoad;

    /// <summary>Bind to <paramref name="accelerator"/>. Compiles nothing: await <see cref="EnsureKernelsAsync"/> first.</summary>
    public Yuv420ToRgb16Gpu(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
    }

    /// <summary>Load the kernels once (awaited load: the compile yields to the browser's event loop first).</summary>
    public Task EnsureKernelsAsync()
    {
        if (_kernelsLoad == null || _kernelsLoad.IsFaulted || _kernelsLoad.IsCanceled) _kernelsLoad = LoadKernelsAsync();
        return _kernelsLoad;
    }

    private async Task LoadKernelsAsync()
    {
        var a = _accelerator;
        _toFloat = await a.LoadAutoGroupedStreamKernelAsync<
            Index1D, ArrayView<byte>, ArrayView<float>, int, int, int, int>(FloatKernel);
        _toInt = await a.LoadAutoGroupedStreamKernelAsync<
            Index1D, ArrayView<byte>, ArrayView<int>, int, int, int, int>(IntKernel);
    }

    /// <summary>The 16.16 position step for <paramref name="chroma"/> samples over <paramref name="output"/> pixels.</summary>
    private static int Step(int chroma, int output) => ((chroma << 16) + (output >> 1)) / output;

    /// <summary>Bytes of a tight 4:2:0 picture: Y (width x height), then U, then V ((width + 1) / 2 x (height + 1) / 2).</summary>
    public static long PlanarBytes(int width, int height) => (long)width * height + 2L * ((width + 1) >> 1) * ((height + 1) >> 1);

    private static void Check(ArrayView<byte> planes, int width, int height, long outLength)
    {
        if (width <= 0 || width > MaxDimension) throw new ArgumentOutOfRangeException(nameof(width));
        if (height <= 0 || height > MaxDimension) throw new ArgumentOutOfRangeException(nameof(height));
        if (planes.Length < PlanarBytes(width, height)) throw new ArgumentException("planes must hold PlanarBytes(width, height).", nameof(planes));
        if (outLength < (long)width * height * 3) throw new ArgumentException("output must hold width * height * 3 values.");
    }

    /// <summary>
    /// Queue the conversion of a tight 4:2:0 picture (<see cref="PlanarBytes"/>: Y, U, V; the layout
    /// Vp9DecoderGpu writes) to float32 RGB, interleaved (H x W x 3, ComfyUI's IMAGE layout for one frame); synchronize
    /// before reading <paramref name="rgb"/>. Await <see cref="EnsureKernelsAsync"/> once before the first call. One
    /// view, so WebGPU binds the planes once (three views of one buffer can share a 4-byte word on tiny pictures, which
    /// WebGPU's no-aliasing rule rejects).
    /// </summary>
    public void ConvertToFloat(ArrayView<byte> planes, int width, int height, ArrayView<float> rgb)
    {
        Check(planes, width, height, rgb.Length);
        _toFloat(width * height, planes, rgb, width, height, Step((width + 1) >> 1, width), Step((height + 1) >> 1, height));
    }

    /// <summary>As <see cref="ConvertToFloat"/>, writing the 16-bit values (0..65535) as ints.</summary>
    public void ConvertToUInt16(ArrayView<byte> planes, int width, int height, ArrayView<int> rgb)
    {
        Check(planes, width, height, rgb.Length);
        _toInt(width * height, planes, rgb, width, height, Step((width + 1) >> 1, width), Step((height + 1) >> 1, height));
    }

    /// <summary>
    /// The two chroma taps of output index <paramref name="o"/> on an axis of <paramref name="n"/> chroma samples, weights
    /// out of <paramref name="k"/> (16384 horizontal, 4096 vertical).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Taps(int o, int n, int inc, int k, out int c0, out int w0, out int c1, out int w1)
    {
        int xp = o * inc + ((inc - 1) >> 1) - 32768;
        if (n < 4)
        {
            int t = (xp + 32768) >> 16;
            if (t < 0) t = 0;
            if (t > n - 1) t = n - 1;
            c0 = t; c1 = t; w0 = k; w1 = 0;
            return;
        }
        int i = xp >> 16, r = 0;
        if (xp < 0) i = -1;
        else
        {
            int f = xp & 0xFFFF;
            if (f >= 65405) i++;                        // just below an integer: the next sample, whole
            else r = (f * k + 32768) >> 16;
        }
        if (i < 0) { c0 = 0; c1 = 0; w0 = k; w1 = 0; }
        else if (i >= n - 1) { c0 = n - 1; c1 = n - 1; w0 = k; w1 = 0; }
        else { c0 = i; c1 = i + 1; w0 = k - r; w1 = r; }
    }

    /// <summary>Chroma at 9 fraction bits (value * 512).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int Chroma(ArrayView<byte> p, int at, int cw, int hc0, int hw0, int hc1, int hw1, int vr0, int vw0, int vr1, int vw1)
    {
        int top = (hw0 * p[at + vr0 * cw + hc0] + hw1 * p[at + vr0 * cw + hc1]) >> 3;
        int bottom = (hw0 * p[at + vr1 * cw + hc0] + hw1 * p[at + vr1 * cw + hc1]) >> 3;
        return (vw0 * top + vw1 * bottom) >> 14;
    }

    /// <summary>
    /// One channel: clip(((Y - 16) * 9539 * 512 + du * cu + dv * cv + 8192) >> 14, 0, 65535), with du / dv split at 9
    /// bits so every term fits an int (equal to the 64-bit form for every input).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int Channel(int t, int du, int dv, int cu, int cv)
    {
        int x = t + cu * (du >> 9) + cv * (dv >> 9);
        int z = 8192 + cu * (du & 511) + cv * (dv & 511);
        int v = (x + (z >> 9)) >> 5;
        return v < 0 ? 0 : v > 65535 ? 65535 : v;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Pixel(int index, ArrayView<byte> planes, int width, int height,
        int incX, int incY, out int r, out int g, out int b)
    {
        int y = index / width, x = index - y * width;
        int cw = (width + 1) >> 1, ch = (height + 1) >> 1;
        Taps(x, cw, incX, 16384, out int hc0, out int hw0, out int hc1, out int hw1);
        Taps(y, ch, incY, 4096, out int vr0, out int vw0, out int vr1, out int vw1);
        int uAt = width * height, vAt = uAt + cw * ch;
        int du = Chroma(planes, uAt, cw, hc0, hw0, hc1, hw1, vr0, vw0, vr1, vw1) - 65536;
        int dv = Chroma(planes, vAt, cw, hc0, hw0, hc1, hw1, vr0, vw0, vr1, vw1) - 65536;
        int t = (planes[index] - 16) * 9539;
        r = Channel(t, du, dv, 0, 13075);
        g = Channel(t, du, dv, -3209, -6660);
        b = Channel(t, du, dv, 16525, 0);
    }

    private static void FloatKernel(Index1D index, ArrayView<byte> planes,
        ArrayView<float> rgb, int width, int height, int incX, int incY)
    {
        Pixel(index, planes, width, height, incX, incY, out int r, out int g, out int b);
        const float Scale = 1f / 65535f;   // ComfyUI: value * float32(1 / 65535)
        int o = index * 3;
        rgb[o] = r * Scale;
        rgb[o + 1] = g * Scale;
        rgb[o + 2] = b * Scale;
    }

    private static void IntKernel(Index1D index, ArrayView<byte> planes,
        ArrayView<int> rgb, int width, int height, int incX, int incY)
    {
        Pixel(index, planes, width, height, incX, incY, out int r, out int g, out int b);
        int o = index * 3;
        rgb[o] = r;
        rgb[o + 1] = g;
        rgb[o + 2] = b;
    }
}
