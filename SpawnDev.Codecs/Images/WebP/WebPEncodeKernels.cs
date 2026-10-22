// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// The WebP lossless encode kernels. Every stage runs on the accelerator; the host dispatches them in a fixed order
// and reads back a few scalars (the output length, the cost of a colour-cache variant).
//
//   1. ToArgbKernel          (1 thread per pixel)   RGBA8 -> ARGB, optionally clearing RGB under alpha 0.
//   2. AnalyzeKernel         (1 thread)             alpha, distinct colours (palette when <= 256, sorted), delta-coded palette.
//   3. PaletteIndexKernel    (1 thread per packed pixel)   colour -> palette index, 2/4/8 pixels bundled per word.
//   4. SubtractGreenKernel   (1 thread per pixel)   forward subtract green.
//   5. PredictorModeKernel   (1 thread per tile)    picks the predictor mode per tile by minimum residual magnitude.
//   6. PredictorResidualKernel (1 thread per pixel) residuals against the ORIGINAL neighbours (fully parallel).
//   7. CrossColorModeKernel / CrossColorKernel      per-tile multipliers and the forward cross-colour transform.
//   8. TokenizeKernel        (1 thread)             LZ77 hash chain + colour cache -> tokens + histograms.
//   9. BuildCodesKernel      (1 thread per code)    length-limited Huffman codes of the five prefix codes.
//  10. EmitKernel            (1 thread)             header, transforms (sub-images), prefix codes, tokens -> bits.
//  11. FinalizeKernel + CopyBlobKernel              RIFF / VP8X / chunk headers, ICCP / EXIF / XMP payloads.
//
// The transforms read only ORIGINAL pixels, so they are parallel; the bitstream (variable-length codes in one
// stream) is a sequential dependency chain, hence the single-thread kernels - still accelerator-resident.
// Stages that do not apply to the image (palette decided on the accelerator) return at once, so the host
// needs no mid-pipeline readback.

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs.Images.WebP;

/// <summary>Loads and launches the WebP lossless encode kernels.</summary>
public sealed class WebPEncodeKernels : IDisposable
{
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<uint>, int> _toArgb;
    private readonly Action<Index1D, ArrayView<uint>, ArrayView<int>, ArrayView<int>> _analyze;
    private readonly Action<Index1D, ArrayView<uint>, ArrayView<uint>, ArrayView<int>, ArrayView<int>> _paletteIndex;
    private readonly Action<Index1D, ArrayView<uint>, ArrayView<int>> _subtractGreen;
    private readonly Action<Index1D, ArrayView<uint>, ArrayView<int>, ArrayView<int>> _predictorMode;
    private readonly Action<Index1D, ArrayView<uint>, ArrayView<uint>, ArrayView<int>, ArrayView<int>> _predictorResidual;
    private readonly Action<Index1D, ArrayView<uint>, ArrayView<int>, ArrayView<int>> _crossMode;
    private readonly Action<Index1D, ArrayView<uint>, ArrayView<int>, ArrayView<int>> _crossForward;
    private readonly Action<Index1D, ArrayView<uint>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int> _tokenize;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>> _buildCodes;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>> _emit;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>> _finalize;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, int, int> _copyBlob;

    /// <summary>Compile the kernels.</summary>
    public WebPEncodeKernels(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _toArgb = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<uint>, int>(ToArgbKernel);
        _analyze = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<uint>, ArrayView<int>, ArrayView<int>>(AnalyzeKernel);
        _paletteIndex = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<uint>, ArrayView<uint>, ArrayView<int>, ArrayView<int>>(PaletteIndexKernel);
        _subtractGreen = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<uint>, ArrayView<int>>(SubtractGreenKernel);
        _predictorMode = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<uint>, ArrayView<int>, ArrayView<int>>(PredictorModeKernel);
        _predictorResidual = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<uint>, ArrayView<uint>, ArrayView<int>, ArrayView<int>>(PredictorResidualKernel);
        _crossMode = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<uint>, ArrayView<int>, ArrayView<int>>(CrossColorModeKernel);
        _crossForward = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<uint>, ArrayView<int>, ArrayView<int>>(CrossColorKernel);
        _tokenize = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<uint>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int>(TokenizeKernel);
        _buildCodes = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>>(BuildCodesKernel);
        _emit = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>>(EmitKernel);
        _finalize = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<int>>(FinalizeKernel);
        _copyBlob = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, int, int>(CopyBlobKernel);
    }

    // ------------------------------------------------------------------
    // Launchers
    // ------------------------------------------------------------------

    /// <summary>RGBA8 to ARGB over <paramref name="pixels"/> pixels; <paramref name="exact"/> 0 clears RGB where alpha is 0.</summary>
    public void ToArgb(ArrayView<byte> rgba, ArrayView<uint> argb, int pixels, int exact) => _toArgb(pixels, rgba, argb, exact);

    /// <summary>Colour analysis (alpha, palette).</summary>
    public void Analyze(ArrayView<uint> argb, ArrayView<int> work, ArrayView<int> meta) => _analyze(1, argb, work, meta);

    /// <summary>Palette indices (bundled) from <paramref name="src"/> into <paramref name="dst"/>; extent = the pixel count (surplus threads return).</summary>
    public void PaletteIndex(ArrayView<uint> src, ArrayView<uint> dst, ArrayView<int> work, ArrayView<int> meta, int pixels)
        => _paletteIndex(pixels, src, dst, work, meta);

    /// <summary>Forward subtract green, in place.</summary>
    public void SubtractGreen(ArrayView<uint> argb, ArrayView<int> meta, int pixels) => _subtractGreen(pixels, argb, meta);

    /// <summary>Predictor mode per tile.</summary>
    public void PredictorMode(ArrayView<uint> argb, ArrayView<int> work, ArrayView<int> meta, int tiles)
        => _predictorMode(tiles, argb, work, meta);

    /// <summary>Predictor residuals of <paramref name="src"/> into <paramref name="dst"/>.</summary>
    public void PredictorResidual(ArrayView<uint> src, ArrayView<uint> dst, ArrayView<int> work, ArrayView<int> meta, int pixels)
        => _predictorResidual(pixels, src, dst, work, meta);

    /// <summary>Cross-colour multipliers per tile of the (residual) image.</summary>
    public void CrossColorMode(ArrayView<uint> argb, ArrayView<int> work, ArrayView<int> meta, int tiles)
        => _crossMode(tiles, argb, work, meta);

    /// <summary>Forward cross-colour transform, in place.</summary>
    public void CrossColor(ArrayView<uint> argb, ArrayView<int> work, ArrayView<int> meta, int pixels)
        => _crossForward(pixels, argb, work, meta);

    /// <summary>LZ77 + colour cache tokenisation of the main image.</summary>
    public void Tokenize(ArrayView<uint> main, ArrayView<int> work, ArrayView<int> tok, ArrayView<int> meta, ArrayView<int> tables, int cacheBits)
        => _tokenize(1, main, work, tok, meta, tables, cacheBits);

    /// <summary>Huffman codes of the five main prefix codes (one thread each).</summary>
    public void BuildCodes(ArrayView<int> work, ArrayView<int> meta) => _buildCodes(5, work, meta);

    /// <summary>Bitstream: header, transforms, codes, tokens.</summary>
    public void Emit(ArrayView<byte> output, ArrayView<int> work, ArrayView<int> tok, ArrayView<int> meta, ArrayView<int> tables)
        => _emit(1, output, work, tok, meta, tables);

    /// <summary>RIFF / VP8X / chunk headers and the total length.</summary>
    public void FinalizeFile(ArrayView<byte> output, ArrayView<int> meta) => _finalize(1, output, meta);

    /// <summary>Copy <paramref name="count"/> blob bytes from <c>blobs[srcOffset ..]</c> to <c>output[meta[dstSlot] ..]</c>.</summary>
    public void CopyBlob(ArrayView<byte> blobs, ArrayView<byte> output, ArrayView<int> meta, int srcOffset, int count, int dstSlot)
    {
        if (count > 0) _copyBlob(count, blobs, output, meta, srcOffset, dstSlot);
    }

    // ------------------------------------------------------------------
    // 1. RGBA -> ARGB
    // ------------------------------------------------------------------

    private static void ToArgbKernel(Index1D index, ArrayView<byte> rgba, ArrayView<uint> argb, int exact)
    {
        int o = index.X * 4;
        uint r = rgba[o + 0];
        uint g = rgba[o + 1];
        uint b = rgba[o + 2];
        uint a = rgba[o + 3];
        bool clear = (exact == 0) & (a == 0);
        if (clear)
        {
            r = 0;
            g = 0;
            b = 0;
        }
        argb[index.X] = (a << 24) | (r << 16) | (g << 8) | b;
    }

    // ------------------------------------------------------------------
    // 2. Analysis
    // ------------------------------------------------------------------

    private static uint SubPixels(uint a, uint b)
    {
        uint ag = 0x00ff00ffu + (a & 0xff00ff00u) - (b & 0xff00ff00u);
        uint rb = 0xff00ff00u + (a & 0x00ff00ffu) - (b & 0x00ff00ffu);
        return (ag & 0xff00ff00u) | (rb & 0x00ff00ffu);
    }

    private static void AnalyzeKernel(Index1D _, ArrayView<uint> argb, ArrayView<int> w, ArrayView<int> meta)
    {
        int width = meta[WebPEncMeta.Width];
        int height = meta[WebPEncMeta.Height];
        int n = width * height;
        int keys = Vp8lEncLayout.Analysis;
        int flags = Vp8lEncLayout.Analysis + 1024;
        for (int i = 0; i < 1024; i++) w[flags + i] = 0;

        int count = 0;
        int hasAlpha = 0;
        bool collecting = meta[WebPEncMeta.AllowPalette] == 1;
        for (int i = 0; i < n; i++)
        {
            uint px = argb[i];
            if ((px >> 24) != 255u) hasAlpha = 1;
            if (collecting)
            {
                int h = (int)((px * 0x9E3779B1u) >> 22);
                bool probing = true;
                while (probing)
                {
                    if (w[flags + h] == 0)
                    {
                        if (count == 256)
                        {
                            collecting = false;
                        }
                        else
                        {
                            w[flags + h] = 1;
                            w[keys + h] = (int)px;
                            count++;
                        }
                        probing = false;
                    }
                    else if ((uint)w[keys + h] == px)
                    {
                        probing = false;
                    }
                    else
                    {
                        h = (h + 1) & 1023;
                    }
                }
            }
        }
        meta[WebPEncMeta.HasAlpha] = hasAlpha;

        int palette = 0;
        if (collecting)
        {
            palette = count;
            int pal = Vp8lEncLayout.Palette;
            int k = 0;
            for (int i = 0; i < 1024; i++)
            {
                if (w[flags + i] == 1)
                {
                    // insertion sort by unsigned value
                    uint v = (uint)w[keys + i];
                    int j = k - 1;
                    bool shifting = j >= 0;
                    while (shifting)
                    {
                        bool greater = (uint)w[pal + j] > v;
                        if (greater)
                        {
                            w[pal + j + 1] = w[pal + j];
                            j--;
                        }
                        shifting = greater & (j >= 0);
                    }
                    w[pal + j + 1] = (int)v;
                    k++;
                }
            }
            int delta = Vp8lEncLayout.PaletteDelta;
            w[delta] = w[pal];
            for (int i = 1; i < count; i++)
                w[delta + i] = (int)SubPixels((uint)w[pal + i], (uint)w[pal + i - 1]);
        }
        meta[WebPEncMeta.PaletteAvailable] = palette;
    }

    // ------------------------------------------------------------------
    // 3. Palette indices
    // ------------------------------------------------------------------

    private static void PaletteIndexKernel(
        Index1D index, ArrayView<uint> src, ArrayView<uint> dst, ArrayView<int> w, ArrayView<int> meta)
    {
        int count = meta[WebPEncMeta.PaletteCount];
        int width = meta[WebPEncMeta.Width];
        int height = meta[WebPEncMeta.Height];
        int bits = meta[WebPEncMeta.PaletteBits];
        int packedWidth = meta[WebPEncMeta.PackedWidth];
        int i = index.X;
        bool active = (count > 0) & (i < packedWidth * height);
        if (active)
        {
            int y = i / packedWidth;
            int xp = i - y * packedWidth;
            int per = 1 << bits;
            int bpp = 8 >> bits;
            uint green = 0;
            for (int k = 0; k < per; k++)
            {
                int x = xp * per + k;
                if (x < width)
                {
                    uint px = src[y * width + x];
                    // binary search in the sorted palette
                    int lo = 0;
                    int hi = count - 1;
                    while (lo < hi)
                    {
                        int mid = (lo + hi) >> 1;
                        bool below = (uint)w[Vp8lEncLayout.Palette + mid] < px;
                        if (below) lo = mid + 1;
                        else hi = mid;
                    }
                    green |= (uint)lo << (k * bpp);
                }
            }
            dst[i] = 0xff000000u | (green << 8);
        }
    }

    // ------------------------------------------------------------------
    // 4-7. Transforms
    // ------------------------------------------------------------------

    private static void SubtractGreenKernel(Index1D index, ArrayView<uint> argb, ArrayView<int> meta)
    {
        bool active = meta[WebPEncMeta.UseSubtractGreen] == 1;
        if (active)
        {
            uint p = argb[index.X];
            uint g = (p >> 8) & 0xFFu;
            uint r = (((p >> 16) & 0xFFu) - g) & 0xFFu;
            uint b = ((p & 0xFFu) - g) & 0xFFu;
            argb[index.X] = (p & 0xff00ff00u) | (r << 16) | b;
        }
    }

    private static int AbsResidual(uint residual)
    {
        int cost = 0;
        for (int i = 0; i < 4; i++)
        {
            int v = (int)((residual >> (i * 8)) & 0xFFu);
            cost += v < 128 ? v : 256 - v;
        }
        return cost;
    }

    private static int ModeCandidate(int set, int i)
    {
        // set 0: a handful of cheap, usually good modes; set 1: all of them
        int m = i;
        if (set == 0)
        {
            if (i == 0) m = 1;
            else if (i == 1) m = 2;
            else if (i == 2) m = 11;
            else if (i == 3) m = 12;
            else m = 13;
        }
        return m;
    }

    private static void PredictorModeKernel(Index1D index, ArrayView<uint> argb, ArrayView<int> w, ArrayView<int> meta)
    {
        bool active = meta[WebPEncMeta.UsePredictor] == 1;
        if (active)
        {
            int width = meta[WebPEncMeta.PackedWidth];
            int height = meta[WebPEncMeta.Height];
            int bits = meta[WebPEncMeta.PredictorBits];
            int set = meta[WebPEncMeta.ModeSet];
            int tilesPerRow = (width + (1 << bits) - 1) >> bits;
            int t = index.X;
            int ty = t / tilesPerRow;
            int tx = t - ty * tilesPerRow;
            int x0 = tx << bits;
            int y0 = ty << bits;
            int x1 = x0 + (1 << bits);
            if (x1 > width) x1 = width;
            int y1 = y0 + (1 << bits);
            if (y1 > height) y1 = height;
            int xStart = x0 > 0 ? x0 : 1;
            int yStart = y0 > 0 ? y0 : 1;

            int candidates = set == 0 ? 5 : 14;
            int best = ModeCandidate(set, 0);
            int bestCost = int.MaxValue;
            for (int c = 0; c < candidates; c++)
            {
                int mode = ModeCandidate(set, c);
                int cost = 0;
                for (int y = yStart; y < y1; y++)
                {
                    for (int x = xStart; x < x1; x++)
                    {
                        int i = y * width + x;
                        uint pred = WebPKernels.Predict(mode, argb[i - 1], argb[i - width], argb[i - width + 1], argb[i - width - 1]);
                        cost += AbsResidual(SubPixels(argb[i], pred));
                    }
                }
                if (cost < bestCost)
                {
                    bestCost = cost;
                    best = mode;
                }
            }
            w[meta[WebPEncMeta.RegionPredictor] + t] = (int)(0xff000000u | ((uint)best << 8));
        }
    }

    private static void PredictorResidualKernel(
        Index1D index, ArrayView<uint> src, ArrayView<uint> dst, ArrayView<int> w, ArrayView<int> meta)
    {
        bool active = meta[WebPEncMeta.UsePredictor] == 1;
        if (active)
        {
            int width = meta[WebPEncMeta.PackedWidth];
            int bits = meta[WebPEncMeta.PredictorBits];
            int tilesPerRow = (width + (1 << bits) - 1) >> bits;
            int i = index.X;
            int y = i / width;
            int x = i - y * width;
            uint pred;
            if (i == 0)
            {
                pred = 0xff000000u;
            }
            else if (y == 0)
            {
                pred = src[i - 1];
            }
            else if (x == 0)
            {
                pred = src[i - width];
            }
            else
            {
                int mode = (w[meta[WebPEncMeta.RegionPredictor] + (y >> bits) * tilesPerRow + (x >> bits)] >> 8) & 0xF;
                pred = WebPKernels.Predict(mode, src[i - 1], src[i - width], src[i - width + 1], src[i - width - 1]);
            }
            dst[i] = SubPixels(src[i], pred);
        }
    }

    private static int Signed8(int v) => v >= 128 ? v - 256 : v;

    private static int ColorDelta(int multiplier, int color) => (multiplier * color) >> 5;

    /// <summary>Residual magnitude of the red channel of <paramref name="p"/> after removing greenToRed * green.</summary>
    private static int RedCost(uint p, int greenToRed)
    {
        int green = Signed8((int)((p >> 8) & 0xFFu));
        int red = (int)((p >> 16) & 0xFFu);
        int r = (red - ColorDelta(greenToRed, green)) & 0xFF;
        return r < 128 ? r : 256 - r;
    }

    private static int BlueCost(uint p, int greenToBlue, int redToBlue)
    {
        int green = Signed8((int)((p >> 8) & 0xFFu));
        int red = Signed8((int)((p >> 16) & 0xFFu));
        int blue = (int)(p & 0xFFu);
        int b = (blue - ColorDelta(greenToBlue, green) - ColorDelta(redToBlue, red)) & 0xFF;
        return b < 128 ? b : 256 - b;
    }

    private static void CrossColorModeKernel(Index1D index, ArrayView<uint> argb, ArrayView<int> w, ArrayView<int> meta)
    {
        bool active = meta[WebPEncMeta.UseCrossColor] == 1;
        if (active)
        {
            int width = meta[WebPEncMeta.PackedWidth];
            int height = meta[WebPEncMeta.Height];
            int bits = meta[WebPEncMeta.CrossColorBits];
            int tilesPerRow = (width + (1 << bits) - 1) >> bits;
            int t = index.X;
            int ty = t / tilesPerRow;
            int tx = t - ty * tilesPerRow;
            int x0 = tx << bits;
            int y0 = ty << bits;
            int x1 = x0 + (1 << bits);
            if (x1 > width) x1 = width;
            int y1 = y0 + (1 << bits);
            if (y1 > height) y1 = height;

            // greedy search over a coarse grid of multipliers (-64..+63 in steps of 8), channel by channel
            int bestG2R = 0;
            int bestCost = int.MaxValue;
            for (int c = -8; c < 8; c++)
            {
                int m = c * 8;
                int cost = 0;
                for (int y = y0; y < y1; y++)
                    for (int x = x0; x < x1; x++)
                        cost += RedCost(argb[y * width + x], m);
                if (cost < bestCost)
                {
                    bestCost = cost;
                    bestG2R = m;
                }
            }
            int bestG2B = 0;
            bestCost = int.MaxValue;
            for (int c = -8; c < 8; c++)
            {
                int m = c * 8;
                int cost = 0;
                for (int y = y0; y < y1; y++)
                    for (int x = x0; x < x1; x++)
                        cost += BlueCost(argb[y * width + x], m, 0);
                if (cost < bestCost)
                {
                    bestCost = cost;
                    bestG2B = m;
                }
            }
            int bestR2B = 0;
            bestCost = int.MaxValue;
            for (int c = -8; c < 8; c++)
            {
                int m = c * 8;
                int cost = 0;
                for (int y = y0; y < y1; y++)
                    for (int x = x0; x < x1; x++)
                        cost += BlueCost(argb[y * width + x], bestG2B, m);
                if (cost < bestCost)
                {
                    bestCost = cost;
                    bestR2B = m;
                }
            }
            // element: red_to_blue in the red byte, green_to_blue in green, green_to_red in blue
            uint e = 0xff000000u | ((uint)(bestR2B & 0xFF) << 16) | ((uint)(bestG2B & 0xFF) << 8) | (uint)(bestG2R & 0xFF);
            w[meta[WebPEncMeta.RegionCross] + t] = (int)e;
        }
    }

    private static void CrossColorKernel(Index1D index, ArrayView<uint> argb, ArrayView<int> w, ArrayView<int> meta)
    {
        bool active = meta[WebPEncMeta.UseCrossColor] == 1;
        if (active)
        {
            int width = meta[WebPEncMeta.PackedWidth];
            int bits = meta[WebPEncMeta.CrossColorBits];
            int tilesPerRow = (width + (1 << bits) - 1) >> bits;
            int i = index.X;
            int y = i / width;
            int x = i - y * width;
            uint m = (uint)w[meta[WebPEncMeta.RegionCross] + (y >> bits) * tilesPerRow + (x >> bits)];
            int greenToRed = Signed8((int)(m & 0xFFu));
            int greenToBlue = Signed8((int)((m >> 8) & 0xFFu));
            int redToBlue = Signed8((int)((m >> 16) & 0xFFu));
            uint p = argb[i];
            int green = Signed8((int)((p >> 8) & 0xFFu));
            int red = (int)((p >> 16) & 0xFFu);
            int blue = (int)(p & 0xFFu);
            int newRed = (red - ColorDelta(greenToRed, green)) & 0xFF;
            // the decoder adds red_to_blue * (reconstructed = original) red, so the encoder subtracts it with the ORIGINAL red
            int newBlue = (blue - ColorDelta(greenToBlue, green) - ColorDelta(redToBlue, Signed8(red))) & 0xFF;
            argb[i] = (p & 0xff00ff00u) | ((uint)newRed << 16) | (uint)newBlue;
        }
    }

    // ------------------------------------------------------------------
    // 8-10. Entropy coding
    // ------------------------------------------------------------------

    private static void TokenizeKernel(
        Index1D _, ArrayView<uint> main, ArrayView<int> w, ArrayView<int> tok, ArrayView<int> meta, ArrayView<int> tables, int cacheBits)
    {
        int height = meta[WebPEncMeta.Height];
        int xsize = meta[WebPEncMeta.PackedWidth];
        Vp8lEncodeGpu.Tokenize(main, xsize, height, w, tok, meta, tables, cacheBits);
    }

    private static void BuildCodesKernel(Index1D index, ArrayView<int> w, ArrayView<int> meta)
    {
        int k = index.X;
        int cacheBits = meta[WebPEncMeta.CacheBits];
        int cacheSize = cacheBits > 0 ? 1 << cacheBits : 0;
        int set = meta[WebPEncMeta.RegionMain];
        int scratch = meta[WebPEncMeta.RegionBuild] + k * Vp8lEncLayout.BuildScratch;
        Vp8lEncodeGpu.BuildOneCode(w, set, cacheSize, k, scratch);
        meta[WebPEncMeta.CodeCost + k] = Vp8lEncodeGpu.CodeCostBits(w, set, cacheSize, k);
    }

    private static void EmitKernel(
        Index1D _, ArrayView<byte> output, ArrayView<int> w, ArrayView<int> tok, ArrayView<int> meta, ArrayView<int> tables)
    {
        int start = meta[WebPEncMeta.PayloadStart];
        var bw = new Vp8lBitWriter { Pos = start, Acc = 0, Nbits = 0, Cap = meta[WebPEncMeta.OutputCapacity], Overflow = 0 };
        int width = meta[WebPEncMeta.Width];
        int height = meta[WebPEncMeta.Height];
        int palette = meta[WebPEncMeta.PaletteCount];
        int sub = meta[WebPEncMeta.RegionSub];
        int build = meta[WebPEncMeta.RegionBuild];
        int ws = meta[WebPEncMeta.RegionWriter];

        Vp8lEncodeGpu.PutBits(ref bw, output, 0x2Fu, 8);
        Vp8lEncodeGpu.PutBits(ref bw, output, (uint)(width - 1), 14);
        Vp8lEncodeGpu.PutBits(ref bw, output, (uint)(height - 1), 14);
        Vp8lEncodeGpu.PutBits(ref bw, output, (uint)meta[WebPEncMeta.HasAlpha], 1);
        Vp8lEncodeGpu.PutBits(ref bw, output, 0, 3);

        int packedWidth = meta[WebPEncMeta.PackedWidth];
        if (palette > 0)
        {
            Vp8lEncodeGpu.PutBits(ref bw, output, 1, 1);
            Vp8lEncodeGpu.PutBits(ref bw, output, 3, 2);
            Vp8lEncodeGpu.PutBits(ref bw, output, (uint)(palette - 1), 8);
            Vp8lEncodeGpu.WriteSubImage(ref bw, output, w, Vp8lEncLayout.PaletteDelta, palette, sub, build, ws);
        }
        else if (meta[WebPEncMeta.UseSubtractGreen] == 1)
        {
            Vp8lEncodeGpu.PutBits(ref bw, output, 1, 1);
            Vp8lEncodeGpu.PutBits(ref bw, output, 2, 2);
        }
        if (meta[WebPEncMeta.UsePredictor] == 1)
        {
            int bits = meta[WebPEncMeta.PredictorBits];
            Vp8lEncodeGpu.PutBits(ref bw, output, 1, 1);
            Vp8lEncodeGpu.PutBits(ref bw, output, 0, 2);
            Vp8lEncodeGpu.PutBits(ref bw, output, (uint)(bits - 2), 3);
            Vp8lEncodeGpu.WriteSubImage(ref bw, output, w, meta[WebPEncMeta.RegionPredictor],
                Vp8lEncLayout.Tiles(packedWidth, height, bits), sub, build, ws);
        }
        if (meta[WebPEncMeta.UseCrossColor] == 1)
        {
            int bits = meta[WebPEncMeta.CrossColorBits];
            Vp8lEncodeGpu.PutBits(ref bw, output, 1, 1);
            Vp8lEncodeGpu.PutBits(ref bw, output, 1, 2);
            Vp8lEncodeGpu.PutBits(ref bw, output, (uint)(bits - 2), 3);
            Vp8lEncodeGpu.WriteSubImage(ref bw, output, w, meta[WebPEncMeta.RegionCross],
                Vp8lEncLayout.Tiles(packedWidth, height, bits), sub, build, ws);
        }
        Vp8lEncodeGpu.PutBits(ref bw, output, 0, 1); // no more transforms

        int cacheBits = meta[WebPEncMeta.CacheBits];
        if (cacheBits > 0)
        {
            Vp8lEncodeGpu.PutBits(ref bw, output, 1, 1);
            Vp8lEncodeGpu.PutBits(ref bw, output, (uint)cacheBits, 4);
        }
        else
        {
            Vp8lEncodeGpu.PutBits(ref bw, output, 0, 1);
        }
        Vp8lEncodeGpu.PutBits(ref bw, output, 0, 1); // no meta prefix codes: one group
        int cacheSize = cacheBits > 0 ? 1 << cacheBits : 0;
        Vp8lEncodeGpu.WriteCodeSet(ref bw, output, w, meta[WebPEncMeta.RegionMain], cacheSize, ws);
        Vp8lEncodeGpu.WriteTokens(ref bw, output, w, tok, meta, tables, meta[WebPEncMeta.PackedWidth], cacheBits);
        Vp8lEncodeGpu.Flush(ref bw, output);
        meta[WebPEncMeta.PayloadLength] = bw.Pos - start;
        meta[WebPEncMeta.Error] = bw.Overflow;
    }

    // ------------------------------------------------------------------
    // 11. Container
    // ------------------------------------------------------------------

    private static void PutLe32(ArrayView<byte> o, int p, int v)
    {
        o[p] = (byte)(v & 0xFF);
        o[p + 1] = (byte)((v >> 8) & 0xFF);
        o[p + 2] = (byte)((v >> 16) & 0xFF);
        o[p + 3] = (byte)((v >> 24) & 0xFF);
    }

    private static void PutFcc(ArrayView<byte> o, int p, int a, int b, int c, int d)
    {
        o[p] = (byte)a;
        o[p + 1] = (byte)b;
        o[p + 2] = (byte)c;
        o[p + 3] = (byte)d;
    }

    private static void FinalizeKernel(Index1D _, ArrayView<byte> o, ArrayView<int> meta)
    {
        int cap = meta[WebPEncMeta.OutputCapacity];
        int start = meta[WebPEncMeta.PayloadStart];
        int payload = meta[WebPEncMeta.PayloadLength];
        int icc = meta[WebPEncMeta.IccLength];
        int exif = meta[WebPEncMeta.ExifLength];
        int xmp = meta[WebPEncMeta.XmpLength];
        int extended = meta[WebPEncMeta.ExtendedFile];
        int end = start + payload + (payload & 1);
        int p = end;
        int exifHeader = 0;
        int xmpHeader = 0;
        if (exif > 0)
        {
            exifHeader = p;
            meta[WebPEncMeta.ExifDst] = p + 8;
            p += 8 + exif + (exif & 1);
        }
        if (xmp > 0)
        {
            xmpHeader = p;
            meta[WebPEncMeta.XmpDst] = p + 8;
            p += 8 + xmp + (xmp & 1);
        }
        meta[WebPEncMeta.Total] = p;
        if ((meta[WebPEncMeta.Error] == 0) & (p <= cap))
        {
            PutFcc(o, 0, 'R', 'I', 'F', 'F');
            PutLe32(o, 4, p - 8);
            PutFcc(o, 8, 'W', 'E', 'B', 'P');
            int q = 12;
            if (extended == 1)
            {
                int flags = 0;
                if (meta[WebPEncMeta.HasAlpha] == 1) flags |= 0x10;
                if (icc > 0) flags |= 0x20;
                if (exif > 0) flags |= 0x08;
                if (xmp > 0) flags |= 0x04;
                int w1 = meta[WebPEncMeta.Width] - 1;
                int h1 = meta[WebPEncMeta.Height] - 1;
                PutFcc(o, q, 'V', 'P', '8', 'X');
                PutLe32(o, q + 4, 10);
                o[q + 8] = (byte)flags;
                o[q + 9] = 0;
                o[q + 10] = 0;
                o[q + 11] = 0;
                o[q + 12] = (byte)(w1 & 0xFF);
                o[q + 13] = (byte)((w1 >> 8) & 0xFF);
                o[q + 14] = (byte)((w1 >> 16) & 0xFF);
                o[q + 15] = (byte)(h1 & 0xFF);
                o[q + 16] = (byte)((h1 >> 8) & 0xFF);
                o[q + 17] = (byte)((h1 >> 16) & 0xFF);
                q += 18;
                if (icc > 0)
                {
                    PutFcc(o, q, 'I', 'C', 'C', 'P');
                    PutLe32(o, q + 4, icc);
                    meta[WebPEncMeta.IccDst] = q + 8;
                    if ((icc & 1) == 1) o[q + 8 + icc] = 0;
                }
            }
            PutFcc(o, start - 8, 'V', 'P', '8', 'L');
            PutLe32(o, start - 4, payload);
            if ((payload & 1) == 1) o[start + payload] = 0;
            if (exif > 0)
            {
                PutFcc(o, exifHeader, 'E', 'X', 'I', 'F');
                PutLe32(o, exifHeader + 4, exif);
                if ((exif & 1) == 1) o[exifHeader + 8 + exif] = 0;
            }
            if (xmp > 0)
            {
                PutFcc(o, xmpHeader, 'X', 'M', 'P', ' ');
                PutLe32(o, xmpHeader + 4, xmp);
                if ((xmp & 1) == 1) o[xmpHeader + 8 + xmp] = 0;
            }
        }
    }

    private static void CopyBlobKernel(
        Index1D index, ArrayView<byte> blobs, ArrayView<byte> output, ArrayView<int> meta, int srcOffset, int dstSlot)
    {
        bool ok = meta[WebPEncMeta.Error] == 0;
        if (ok) output[meta[dstSlot] + index.X] = blobs[srcOffset + index.X];
    }

    /// <summary>No-op (kernels are owned by the accelerator).</summary>
    public void Dispose() { }
}
