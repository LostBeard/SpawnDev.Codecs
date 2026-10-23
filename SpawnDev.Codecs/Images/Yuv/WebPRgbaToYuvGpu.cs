// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// RGBA8 -> YUV 4:2:0 (+ alpha plane) exactly as libwebp's lossy encoder imports a picture by default
// (libwebp 1.6.0 src/enc/picture_csp_enc.c ImportYUVAFromRGBA, no dithering, no sharp YUV), followed by
// WebPCleanupTransparentArea (src/enc/picture_tools_enc.c, YUV path), which libwebp runs unless config->exact.
//
//   Y = (16839 r + 33059 g + 6420 b + 2^15 + (16 << 16)) >> 16                         (VP8RGBToY, no clip)
//   U, V from the 2x2 block averaged in a gamma-linear space: GammaToLinear table (gamma 0.80, 12-bit) summed over the
//   four samples, back through the interpolated LinearToGamma table, then VP8RGBToU/V with YUV_HALF << 2 rounding and the
//   0..255 clip. A block whose alpha sum is neither 0 nor 4 * 255 is ALPHA-WEIGHTED (sum a_i * lin_i, times the
//   kInvAlpha[sum a] reciprocal, >> 17). An odd last column / row repeats the sample (libwebp's SUM2 / rgb_stride = 0
//   cases are exactly the 2x2 formula with the duplicated sample).
//   Cleanup: every 8x8 luma block that is partly transparent gets its alpha-0 luma replaced by the integer mean of its
//   visible luma (SmoothenBlock, also for the partial right / bottom blocks); a run of fully transparent whole 8x8 blocks
//   along a block row is flattened (Y 8x8, U/V 4x4) to the Y/U/V of the first block of the run.
//
// Kernels: one thread per chroma sample (writes its <= 4 luma, 1 U, 1 V and <= 4 alpha bytes); one thread per 8x8 block for
// the smoothing; one thread per 8-row strip for the flattening (a run carries its value along the strip, so a strip is
// serial). The host uploads the constant tables once and dispatches; nothing returns to the host.

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs.Images.Yuv;

/// <summary>Converts RGBA8 on the accelerator to YUV 4:2:0 + alpha exactly like libwebp's lossy encoder input.</summary>
public sealed class WebPRgbaToYuvGpu : IDisposable
{
    // Table layout in one int buffer.
    private const int GammaToLinearBase = 0;       // 256 entries
    private const int LinearToGammaBase = 256;     // 33 entries (GAMMA_TAB_SIZE + 1)
    private const int InvAlphaBase = 289;          // 1021 entries
    private const int TableSize = 289 + 1021;

    private readonly MemoryBuffer1D<int, Stride1D.Dense> _tables;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, int, int, int> _convert;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, int, int> _smoothen;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, int, int> _flatten;

    /// <summary>Compile the kernels and upload the libwebp tables.</summary>
    public WebPRgbaToYuvGpu(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _tables = accelerator.Allocate1D<int>(TableSize);
        _tables.View.CopyFromCPU(BuildTables());
        _convert = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, int, int, int>(ConvertKernel);
        _smoothen = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, int, int>(SmoothenKernel);
        _flatten = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, int, int>(FlattenKernel);
    }

    /// <summary>libwebp's InitGammaTables and kInvAlpha, computed exactly as the C source does (double pow, + .5 truncation).</summary>
    internal static int[] BuildTables()
    {
        var t = new int[TableSize];
        const double kGamma = 0.80;
        const int kGammaScale = (1 << 12) - 1;
        double scale = (double)(1 << 7) / kGammaScale;
        double norm = 1.0 / 255.0;
        for (int v = 0; v <= 255; v++)
            t[GammaToLinearBase + v] = (ushort)(Math.Pow(norm * v, kGamma) * kGammaScale + .5);
        for (int v = 0; v <= 32; v++)
            t[LinearToGammaBase + v] = (int)(255.0 * Math.Pow(scale * v, 1.0 / kGamma) + .5);
        t[InvAlphaBase] = 0;
        for (int a = 1; a <= 4 * 255; a++)
            t[InvAlphaBase + a] = (1 << 19) / a;
        return t;
    }

    /// <summary>Size of the U (or V) plane: ((width + 1) / 2) x ((height + 1) / 2).</summary>
    public static long ChromaPlaneSize(int width, int height) => (long)((width + 1) >> 1) * ((height + 1) >> 1);

    /// <summary>
    /// Queue the conversion of tightly packed RGBA8 (<paramref name="width"/> x <paramref name="height"/>) into tightly packed
    /// planes; synchronize before reading them. <paramref name="alpha"/> receives the alpha plane (width x height).
    /// </summary>
    /// <param name="cleanupTransparent">libwebp's default (<c>exact = false</c>): smooth / flatten fully transparent areas.</param>
    public void Convert(
        ArrayView<byte> rgba, int width, int height,
        ArrayView<byte> yPlane, ArrayView<byte> uPlane, ArrayView<byte> vPlane, ArrayView<byte> alpha,
        bool cleanupTransparent = true)
    {
        if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width));
        if (height <= 0) throw new ArgumentOutOfRangeException(nameof(height));
        long pixels = (long)width * height;
        if (pixels > int.MaxValue / 4) throw new ArgumentOutOfRangeException(nameof(width), "picture too large");
        long chroma = ChromaPlaneSize(width, height);
        if (rgba.Length < pixels * 4) throw new ArgumentException("rgba too short.", nameof(rgba));
        if (yPlane.Length < pixels) throw new ArgumentException("yPlane too short.", nameof(yPlane));
        if (uPlane.Length < chroma) throw new ArgumentException("uPlane too short.", nameof(uPlane));
        if (vPlane.Length < chroma) throw new ArgumentException("vPlane too short.", nameof(vPlane));
        if (alpha.Length < pixels) throw new ArgumentException("alpha too short.", nameof(alpha));

        int cw = (width + 1) >> 1;
        _convert((int)chroma, rgba, yPlane, uPlane, vPlane, alpha, _tables.View, width, height, cw);
        if (!cleanupTransparent) return;
        int bw = (width + 7) >> 3, bh = (height + 7) >> 3;
        _smoothen(bw * bh, alpha, yPlane, uPlane, width, height);   // uPlane unused there; keeps one signature shape
        int strips = height >> 3;
        if (strips > 0 && width >= 8) _flatten(strips, alpha, yPlane, uPlane, vPlane, width, height);
    }

    // ------------------------------------------------------------------
    // Kernels
    // ------------------------------------------------------------------

    /// <summary>libwebp Interpolate + LinearToGamma(base, 0): a 4-sample linear sum (premultiplied by 4) back to gamma, U/V scale.</summary>
    private static int LinearToGamma(ArrayView<int> t, int v)
    {
        int tabPos = v >> 9;            // GAMMA_TAB_FIX + 2
        int x = v & 511;                // (kGammaTabScale << 2) - 1
        int v0 = t[LinearToGammaBase + tabPos];
        int v1 = t[LinearToGammaBase + tabPos + 1];
        int y = v1 * x + v0 * (512 - x);
        return (y + 64) >> 7;           // kGammaTabRounder, GAMMA_TAB_FIX
    }

    private static int ClipUV(int uv)
    {
        int s = (uv + (1 << 17) + (128 << 18)) >> 18;   // rounding YUV_HALF << 2, YUV_FIX + 2
        bool inRange = (s & ~0xff) == 0;
        int clamped = s < 0 ? 0 : 255;
        return inRange ? s : clamped;
    }

    private static int Channel(ArrayView<int> t, int c00, int c01, int c10, int c11,
        int a00, int a01, int a10, int a11, int aSum, bool weighted)
    {
        int l00 = t[GammaToLinearBase + c00], l01 = t[GammaToLinearBase + c01];
        int l10 = t[GammaToLinearBase + c10], l11 = t[GammaToLinearBase + c11];
        int plain = l00 + l01 + l10 + l11;
        // DIVIDE_BY_ALPHA(sum, a) = (sum * kInvAlpha[a]) >> (kAlphaFix - 2); the product fits 32 unsigned bits (libwebp asserts it)
        long sum = (long)a00 * l00 + (long)a01 * l01 + (long)a10 * l10 + (long)a11 * l11;
        int inv = t[InvAlphaBase + aSum];
        int divided = (int)((ulong)(sum * inv) >> 17);
        return LinearToGamma(t, weighted ? divided : plain);
    }

    private static void ConvertKernel(
        Index1D index, ArrayView<byte> rgba, ArrayView<byte> yPlane, ArrayView<byte> uPlane, ArrayView<byte> vPlane,
        ArrayView<byte> alpha, ArrayView<int> t, int width, int height, int cw)
    {
        int cx = index % cw;
        int cy = index / cw;
        int x0 = cx * 2, y0 = cy * 2;
        bool hasX1 = x0 + 1 < width;
        bool hasY1 = y0 + 1 < height;
        int x1 = hasX1 ? x0 + 1 : x0;
        int y1 = hasY1 ? y0 + 1 : y0;

        int p00 = (y0 * width + x0) * 4, p01 = (y0 * width + x1) * 4;
        int p10 = (y1 * width + x0) * 4, p11 = (y1 * width + x1) * 4;

        // Luma and alpha of the (up to) four pixels this block owns.
        WriteLumaAlpha(rgba, yPlane, alpha, p00, y0 * width + x0);
        if (hasX1) WriteLumaAlpha(rgba, yPlane, alpha, p01, y0 * width + x1);
        if (hasY1) WriteLumaAlpha(rgba, yPlane, alpha, p10, y1 * width + x0);
        bool both = hasX1 & hasY1;
        if (both) WriteLumaAlpha(rgba, yPlane, alpha, p11, y1 * width + x1);

        int a00 = rgba[p00 + 3], a01 = rgba[p01 + 3], a10 = rgba[p10 + 3], a11 = rgba[p11 + 3];
        int aSum = a00 + a01 + a10 + a11;
        bool weighted = (aSum != 0) & (aSum != 4 * 255);

        int r = Channel(t, rgba[p00], rgba[p01], rgba[p10], rgba[p11], a00, a01, a10, a11, aSum, weighted);
        int g = Channel(t, rgba[p00 + 1], rgba[p01 + 1], rgba[p10 + 1], rgba[p11 + 1], a00, a01, a10, a11, aSum, weighted);
        int b = Channel(t, rgba[p00 + 2], rgba[p01 + 2], rgba[p10 + 2], rgba[p11 + 2], a00, a01, a10, a11, aSum, weighted);

        uPlane[index] = (byte)ClipUV(-9719 * r - 19081 * g + 28800 * b);
        vPlane[index] = (byte)ClipUV(28800 * r - 24116 * g - 4684 * b);
    }

    private static void WriteLumaAlpha(ArrayView<byte> rgba, ArrayView<byte> yPlane, ArrayView<byte> alpha, int p, int i)
    {
        int luma = 16839 * rgba[p] + 33059 * rgba[p + 1] + 6420 * rgba[p + 2];
        yPlane[i] = (byte)((luma + (1 << 15) + (16 << 16)) >> 16);
        alpha[i] = rgba[p + 3];
    }

    /// <summary>
    /// SmoothenBlock on block (bx, by): alpha-0 luma of a PARTLY transparent block := mean of its visible luma. Blocks are 8x8,
    /// the right / bottom ones clipped to the picture (libwebp smooths those too). Fully transparent blocks are left alone here.
    /// </summary>
    private static void SmoothenKernel(Index1D index, ArrayView<byte> alpha, ArrayView<byte> yPlane, ArrayView<byte> unused, int width, int height)
    {
        int bw = (width + 7) >> 3;
        int bx = index % bw, by = index / bw;
        int x0 = bx * 8, y0 = by * 8;
        int w = width - x0 < 8 ? width - x0 : 8;
        int h = height - y0 < 8 ? height - y0 : 8;
        int sum = 0, count = 0;
        for (int y = 0; y < h; y++)
        {
            int row = (y0 + y) * width + x0;
            for (int x = 0; x < w; x++)
            {
                bool visible = alpha[row + x] != 0;
                if (visible) { count = count + 1; sum = sum + yPlane[row + x]; }
            }
        }
        bool partly = (count > 0) & (count < w * h);
        if (!partly) return;
        byte avg = (byte)(sum / count);
        for (int y = 0; y < h; y++)
        {
            int row = (y0 + y) * width + x0;
            for (int x = 0; x < w; x++)
            {
                bool hidden = alpha[row + x] == 0;
                if (hidden) yPlane[row + x] = avg;
            }
        }
    }

    /// <summary>
    /// The flattening pass of WebPCleanupTransparentArea along one whole 8-row strip: each run of fully transparent whole 8x8
    /// blocks takes the Y / U / V of its first block (U / V at x / 2 of that block's top-left).
    /// </summary>
    private static void FlattenKernel(Index1D strip, ArrayView<byte> alpha, ArrayView<byte> yPlane, ArrayView<byte> uPlane, ArrayView<byte> vPlane, int width, int height)
    {
        int cw = (width + 1) >> 1;
        int y0 = strip * 8;
        int blocks = width >> 3;
        bool needReset = true;
        int vy = 0, vu = 0, vv = 0;
        for (int b = 0; b < blocks; b++)
        {
            int x0 = b * 8;
            bool transparent = true;
            for (int y = 0; y < 8; y++)
                for (int x = 0; x < 8; x++)
                {
                    bool visible = alpha[(y0 + y) * width + x0 + x] != 0;
                    if (visible) transparent = false;
                }
            if (transparent)
            {
                if (needReset)
                {
                    vy = yPlane[y0 * width + x0];
                    vu = uPlane[(y0 >> 1) * cw + (x0 >> 1)];
                    vv = vPlane[(y0 >> 1) * cw + (x0 >> 1)];
                    needReset = false;
                }
                for (int y = 0; y < 8; y++)
                    for (int x = 0; x < 8; x++)
                        yPlane[(y0 + y) * width + x0 + x] = (byte)vy;
                for (int y = 0; y < 4; y++)
                    for (int x = 0; x < 4; x++)
                    {
                        int c = ((y0 >> 1) + y) * cw + (x0 >> 1) + x;
                        uPlane[c] = (byte)vu;
                        vPlane[c] = (byte)vv;
                    }
            }
            else
            {
                needReset = true;
            }
        }
    }

    /// <summary>Release the table buffer.</summary>
    public void Dispose() => _tables.Dispose();
}
