// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// BMP encoder, accelerator-resident: one GPU thread per OUTPUT byte writes the headers, the palette and the
// padded bottom-up pixel rows straight from the RGBA8 source. The host computes the file layout (a handful of
// ints), dispatches, and (for the managed / TypedArray overloads) moves the finished file to the sink.
//
// Formats (BmpEncodeFormat):
//   Bgra32   (default) 32 bit, BITMAPV4HEADER with BI_BITFIELDS masks R=00FF0000 G=0000FF00 B=000000FF A=FF000000.
//            Keeps the alpha channel. Browsers and Pillow read it as RGBA.
//   Bgr24    24 bit BI_RGB, BITMAPINFOHEADER. Alpha is dropped (not composited): the RGB values are written as they are.
//   Indexed8 8 bit BI_RGB with a palette, BITMAPINFOHEADER. The palette is found on the accelerator
//            (exact-colour hash set, sorted ascending by 0xRRGGBB); the image must have at most 256 distinct
//            RGB colours (alpha is ignored), otherwise InvalidOperationException.
// All files are bottom-up with rows padded to a multiple of 4 bytes; resolution is written as 2835 px/m (72 dpi).

using System.Runtime.Versioning;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnDev.Codecs.Images.Bmp;

/// <summary>Pixel layout written by <see cref="BmpEncoderGpu"/>.</summary>
public enum BmpEncodeFormat
{
    /// <summary>32 bit BGRA with alpha (V4 header, BI_BITFIELDS).</summary>
    Bgra32 = 0,
    /// <summary>24 bit BGR; alpha is dropped.</summary>
    Bgr24 = 1,
    /// <summary>8 bit palette; needs at most 256 distinct RGB colours.</summary>
    Indexed8 = 2,
}

/// <summary>Options for <see cref="BmpEncoderGpu"/>.</summary>
public sealed record BmpEncodeOptions
{
    /// <summary>Pixel layout to write. Default <see cref="BmpEncodeFormat.Bgra32"/>.</summary>
    public BmpEncodeFormat Format { get; init; } = BmpEncodeFormat.Bgra32;
}

/// <summary>BMP encoder running on an ILGPU accelerator.</summary>
/// <remarks>Not thread-safe: one in-flight call per instance.</remarks>
public sealed class BmpEncoderGpu : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly BmpKernels _kernels;
    private readonly ExactPaletteBuilder _palette;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _params;
    private readonly StageBuffer<byte> _stageIn;
    private readonly StageBuffer<byte> _stageOut;

    /// <summary>Compile the kernels.</summary>
    public BmpEncoderGpu(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernels = new BmpKernels(accelerator);
        _palette = new ExactPaletteBuilder(accelerator);
        _params = accelerator.Allocate1D<int>(8);
        _stageIn = new StageBuffer<byte>(accelerator);
        _stageOut = new StageBuffer<byte>(accelerator);
    }

    private static int BitsPerPixel(BmpEncodeFormat f) => f switch
    {
        BmpEncodeFormat.Bgra32 => 32,
        BmpEncodeFormat.Bgr24 => 24,
        BmpEncodeFormat.Indexed8 => 8,
        _ => throw new ArgumentOutOfRangeException(nameof(f)),
    };

    private static long RowStride(int width, BmpEncodeFormat f) => (((long)width * BitsPerPixel(f) + 31) / 32) * 4;

    private static void ValidateSize(int width, int height)
    {
        if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width));
        if (height <= 0) throw new ArgumentOutOfRangeException(nameof(height));
        if ((long)width * height * 4 > int.MaxValue - 2048)
            throw new ArgumentOutOfRangeException(nameof(width), $"{width}x{height} is too large for one BMP.");
    }

    /// <summary>
    /// Upper bound on the encoded size: exact for <see cref="BmpEncodeFormat.Bgra32"/> and
    /// <see cref="BmpEncodeFormat.Bgr24"/>, the 256-entry-palette size for <see cref="BmpEncodeFormat.Indexed8"/>.
    /// </summary>
    public static int GetMaxEncodedSize(int width, int height, BmpEncodeOptions? options = null)
    {
        ValidateSize(width, height);
        var f = options?.Format ?? BmpEncodeFormat.Bgra32;
        long header = f == BmpEncodeFormat.Bgra32 ? 14 + 108 : 14 + 40;
        long palette = f == BmpEncodeFormat.Indexed8 ? 1024 : 0;
        return checked((int)(header + palette + RowStride(width, f) * height));
    }

    /// <summary>
    /// Encode RGBA8 pixels that are already on the accelerator into a caller-owned output view. Only metadata
    /// scalars cross to the host.
    /// </summary>
    /// <param name="rgba">Tightly packed RGBA8, rows top to bottom (<c>width * height * 4</c> bytes).</param>
    /// <param name="width">Image width.</param>
    /// <param name="height">Image height.</param>
    /// <param name="output">Receives the file; at least <see cref="GetMaxEncodedSize"/> bytes.</param>
    /// <param name="options">Pixel layout; default 32 bit BGRA.</param>
    /// <returns>The file length in bytes.</returns>
    /// <exception cref="InvalidOperationException"><see cref="BmpEncodeFormat.Indexed8"/> with more than 256 colours.</exception>
    public async Task<int> EncodeAsync(
        ArrayView1D<byte, Stride1D.Dense> rgba, int width, int height,
        ArrayView1D<byte, Stride1D.Dense> output, BmpEncodeOptions? options = null)
    {
        ValidateSize(width, height);
        var format = options?.Format ?? BmpEncodeFormat.Bgra32;
        int pixels = width * height;
        if (rgba.Length < (long)pixels * 4)
            throw new ArgumentException($"rgba must hold {(long)pixels * 4} bytes for {width}x{height}.", nameof(rgba));
        int max = GetMaxEncodedSize(width, height, options);
        if (output.Length < max)
            throw new ArgumentException($"output must hold {max} bytes (GetMaxEncodedSize).", nameof(output));

        int stride = (int)RowStride(width, format);
        int paletteCount = 0;
        int headerLen = format == BmpEncodeFormat.Bgra32 ? 14 + 108 : 14 + 40;
        if (format == BmpEncodeFormat.Indexed8)
        {
            _palette.Reset();
            _palette.SetLimit(ExactPaletteBuilder.MaxPalette);
            _palette.Insert(rgba, pixels, 0);
            _palette.Finish();
            await _accelerator.SynchronizeAsync();
            var state = await _palette.State.CopyToHostAsync();
            if (state[ExactPaletteState.Exact] == 0)
                throw new InvalidOperationException(
                    "Indexed8 BMP needs at most 256 distinct colours; this image has more. Use Bgr24 or Bgra32.");
            paletteCount = state[ExactPaletteState.PaletteCount];
        }
        int dataStart = headerLen + paletteCount * 4;
        long total = dataStart + (long)stride * height;
        if (total > int.MaxValue) throw new ArgumentOutOfRangeException(nameof(width), $"{width}x{height} is too large for one BMP.");

        var p = new[] { (int)format, width, height, (int)total, dataStart, stride, stride * height, paletteCount };
        _params.View.CopyFromCPU(ref p[0], p.Length);
        _kernels.Encode(rgba, output, _palette.Palette, _params.View, (int)total);
        await _accelerator.SynchronizeAsync();
        return (int)total;
    }

    // ------------------------------------------------------------------
    // Managed overload
    // ------------------------------------------------------------------

    /// <summary>
    /// Encode managed RGBA8 pixels. Returns the file as a managed array (a terminal CPU sink; in a browser
    /// prefer the <see cref="TypedArray"/> overload).
    /// </summary>
    public async Task<byte[]> EncodeAsync(ReadOnlyMemory<byte> rgba, int width, int height, BmpEncodeOptions? options = null)
    {
        ValidateSize(width, height);
        if (rgba.Length < (long)width * height * 4)
            throw new ArgumentException($"rgba must hold {(long)width * height * 4} bytes for {width}x{height}.", nameof(rgba));
        _stageIn.Ensure(rgba.Length);
        DeviceTransfer.Upload(_stageIn.View, rgba.Span);
        _stageOut.Ensure(GetMaxEncodedSize(width, height, options));
        int length = await EncodeAsync(_stageIn.View, width, height, _stageOut.View, options);
        return await DeviceTransfer.DownloadAsync(_stageOut.View, length);
    }

    // ------------------------------------------------------------------
    // TypedArray overload (browser: bulk bytes stay in JS)
    // ------------------------------------------------------------------

    /// <summary>
    /// Encode RGBA8 pixels held in a JS typed array (browser accelerators only). The pixels go to the GPU
    /// with <c>CopyFromJS</c> and the file comes back as a JS <see cref="Uint8Array"/>; none of it enters the
    /// .NET heap. The caller disposes the result.
    /// </summary>
    [SupportedOSPlatform("browser")]
    public async Task<Uint8Array> EncodeAsync(TypedArray rgba, int width, int height, BmpEncodeOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(rgba);
        ValidateSize(width, height);
        long length = rgba.ByteLength;
        if (length < (long)width * height * 4)
            throw new ArgumentException($"rgba must hold {(long)width * height * 4} bytes for {width}x{height}.", nameof(rgba));
        _stageIn.Ensure(length);
        DeviceTransfer.Upload(_stageIn.View, rgba);
        _stageOut.Ensure(GetMaxEncodedSize(width, height, options));
        int encoded = await EncodeAsync(_stageIn.View, width, height, _stageOut.View, options);
        return await DeviceTransfer.DownloadUint8ArrayAsync(_stageOut.View, encoded);
    }

    /// <summary>Release kernel resources and buffers.</summary>
    public void Dispose()
    {
        _kernels.Dispose();
        _palette.Dispose();
        _params.Dispose();
        _stageIn.Dispose();
        _stageOut.Dispose();
    }
}
