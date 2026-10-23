// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Lossy WebP encoder, kernel 1: macroblock analysis (libwebp 1.6.0 analysis_enc.c MBAnalyze, methods >= 2), ONE THREAD PER
// MACROBLOCK. libwebp "pretends a perfect lossless reconstruction": the analysis imports the top / left boundary from the
// SOURCE picture, so macroblocks are independent. Per macroblock: the DCT-coefficient histogram "alpha" of the I16 DC and
// TM predictions (best = higher alpha) and of the chroma DC and TM predictions (best mode = smaller alpha), mixed into the
// macroblock susceptibility; the I16 mode is written to the prediction map (VP8SetIntra16Mode), the chroma mode to the
// macroblock record. Segment assignment (k-means over the alphas) is kernel 2.
//
// Boundary rules (VP8IteratorImport with a scratch): left absent on the first column (x == 0: predictions use no left;
// InitLeft's 129s are never read by the DC / TM predictors because MakeLuma16Preds passes NULL), top absent on the first row;
// a partial macroblock replicates its last column / row, and so do the imported top / left lines; the top-left corner is
// 127 on the first row, else the source pixel.

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs.Images.WebP.Lossy;

internal static class WebPLossyAnalysisKernel
{
    private const int MaxCoeffThresh = 31;
    private const int MaxAlpha = 255;
    private const int AlphaScale = 2 * MaxAlpha;

    private static int Min(int a, int b) => a < b ? a : b;
    private static int Abs(int v) => v < 0 ? -v : v;
    private static int Clip8(int v) => v < 0 ? 0 : (v > 255 ? 255 : v);

    /// <summary>Source sample of plane `p` at (px, py) relative to a block origin, replicated inside the w x h valid area.</summary>
    private static int Src(ArrayView<byte> p, int stride, int ox, int oy, int px, int py, int w, int h)
        => p[(oy + Min(py, h - 1)) * stride + ox + Min(px, w - 1)];

    /// <summary>
    /// libwebp FTransform of a 4x4 residual (src - pred) held in `d` (16 ints, row-major), into `o` (16 ints).
    /// </summary>
    private static void FTransform(ArrayView<int> d, ArrayView<int> tmp, ArrayView<int> o)
    {
        for (int i = 0; i < 4; ++i)
        {
            int d0 = d[i * 4 + 0], d1 = d[i * 4 + 1], d2 = d[i * 4 + 2], d3 = d[i * 4 + 3];
            int a0 = d0 + d3, a1 = d1 + d2, a2 = d1 - d2, a3 = d0 - d3;
            tmp[0 + i * 4] = (a0 + a1) * 8;
            tmp[1 + i * 4] = (a2 * 2217 + a3 * 5352 + 1812) >> 9;
            tmp[2 + i * 4] = (a0 - a1) * 8;
            tmp[3 + i * 4] = (a3 * 2217 - a2 * 5352 + 937) >> 9;
        }
        for (int i = 0; i < 4; ++i)
        {
            int a0 = tmp[0 + i] + tmp[12 + i];
            int a1 = tmp[4 + i] + tmp[8 + i];
            int a2 = tmp[4 + i] - tmp[8 + i];
            int a3 = tmp[0 + i] - tmp[12 + i];
            o[0 + i] = (short)((a0 + a1 + 7) >> 4);
            int nz = a3 != 0 ? 1 : 0;
            o[4 + i] = (short)(((a2 * 2217 + a3 * 5352 + 12000) >> 16) + nz);
            o[8 + i] = (short)((a0 - a1 + 7) >> 4);
            o[12 + i] = (short)((a3 * 2217 - a2 * 5352 + 51000) >> 16);
        }
    }

    /// <summary>Histogram -> libwebp GetAlpha: ALPHA_SCALE * last_non_zero / max_value (0 when max_value <= 1).</summary>
    private static int AlphaOf(ArrayView<int> dist)
    {
        int maxValue = 0, lastNonZero = 1;
        for (int k = 0; k <= MaxCoeffThresh; ++k)
        {
            int value = dist[k];
            bool pos = value > 0;
            if (pos)
            {
                if (value > maxValue) maxValue = value;
                lastNonZero = k;
            }
        }
        return maxValue > 1 ? AlphaScale * lastNonZero / maxValue : 0;
    }

    private static void AddBlock(ArrayView<int> d, ArrayView<int> tmp, ArrayView<int> o, ArrayView<int> dist)
    {
        FTransform(d, tmp, o);
        for (int k = 0; k < 16; ++k)
        {
            int v = Abs(o[k]) >> 3;
            int bin = v > MaxCoeffThresh ? MaxCoeffThresh : v;
            dist[bin] = dist[bin] + 1;
        }
    }

    /// <summary>
    /// Analyse macroblock `index` (raster order). Writes mbRecs[mb * MbStride + MbAlpha / MbUVAlphaRaw / MbUVMode / MbType]
    /// and the 4x4 prediction map entries (I16 mode) at preds[predsBase + 4 * (y * predsW + x) ...].
    /// </summary>
    public static void Kernel(
        Index1D index, ArrayView<byte> yP, ArrayView<byte> uP, ArrayView<byte> vP, ArrayView<byte> aP,
        ArrayView<int> mbRecs, ArrayView<byte> preds, ArrayView<int> meta)
    {
        var d = LocalMemory.Allocate<int>(16);
        var tmp = LocalMemory.Allocate<int>(16);
        var o = LocalMemory.Allocate<int>(16);
        var dist = LocalMemory.Allocate<int>(MaxCoeffThresh + 1);
        var top = LocalMemory.Allocate<int>(16);
        var left = LocalMemory.Allocate<int>(16);

        int width = meta[WebPLossyEncMeta.Width], height = meta[WebPLossyEncMeta.Height];
        int mbW = meta[WebPLossyEncMeta.MbW];
        int mb = index.X;
        int mx = mb % mbW, my = mb / mbW;
        int w = Min(width - mx * 16, 16), h = Min(height - my * 16, 16);
        int uvW = (w + 1) >> 1, uvH = (h + 1) >> 1;
        int cw = (width + 1) >> 1;
        bool hasTop = my != 0, hasLeft = mx != 0;
        int ox = mx * 16, oy = my * 16;

        // ---------------- I16: modes DC (0) and TM (1) ----------------
        int topLeft = 0, sumTop = 0, sumLeft = 0;
        if (hasTop)
            for (int i = 0; i < 16; ++i) { top[i] = yP[(oy - 1) * width + ox + Min(i, w - 1)]; sumTop += top[i]; }
        if (hasLeft)
        {
            for (int i = 0; i < 16; ++i) { left[i] = yP[(oy + Min(i, h - 1)) * width + ox - 1]; sumLeft += left[i]; }
            topLeft = hasTop ? yP[(oy - 1) * width + ox - 1] : 127;
        }
        int dc16;
        if (hasTop) dc16 = hasLeft ? (sumTop + sumLeft + 16) >> 5 : (sumTop + sumTop + 16) >> 5;
        else if (hasLeft) dc16 = (sumLeft + sumLeft + 16) >> 5;
        else dc16 = 0x80;

        int bestAlpha = -1, bestMode = 0;
        for (int mode = 0; mode < 2; ++mode)
        {
            for (int k = 0; k <= MaxCoeffThresh; ++k) dist[k] = 0;
            for (int blk = 0; blk < 16; ++blk)
            {
                int bx = (blk & 3) * 4, by = (blk >> 2) * 4;
                for (int j = 0; j < 4; ++j)
                    for (int i = 0; i < 4; ++i)
                    {
                        int px = bx + i, py = by + j;
                        int pred;
                        if (mode == 0) pred = dc16;
                        else
                        {
                            // TrueMotion: both -> clip(left + top - tl); left only -> H; top only -> V; none -> 129
                            if (hasLeft) pred = hasTop ? Clip8(left[py] + top[px] - topLeft) : left[py];
                            else pred = hasTop ? top[px] : 129;
                        }
                        d[j * 4 + i] = Src(yP, width, ox, oy, px, py, w, h) - pred;
                    }
                AddBlock(d, tmp, o, dist);
            }
            int alpha = AlphaOf(dist);
            if (alpha > bestAlpha) { bestAlpha = alpha; bestMode = mode; }
        }

        // ---------------- UV: modes DC (0) and TM (1) over the U then V blocks ----------------
        int uox = mx * 8, uoy = my * 8;
        int bestUvAlpha = -1, smallestAlpha = 0, bestUvMode = 0;
        for (int mode = 0; mode < 2; ++mode)
        {
            for (int k = 0; k <= MaxCoeffThresh; ++k) dist[k] = 0;
            for (int ch = 0; ch < 2; ++ch)
            {
                // boundary of this chroma plane
                int cTopLeft = 0, cSumTop = 0, cSumLeft = 0;
                if (hasTop)
                    for (int i = 0; i < 8; ++i)
                    {
                        int t = ch == 0 ? uP[(uoy - 1) * cw + uox + Min(i, uvW - 1)] : vP[(uoy - 1) * cw + uox + Min(i, uvW - 1)];
                        top[i] = t; cSumTop += t;
                    }
                if (hasLeft)
                {
                    for (int i = 0; i < 8; ++i)
                    {
                        int l = ch == 0 ? uP[(uoy + Min(i, uvH - 1)) * cw + uox - 1] : vP[(uoy + Min(i, uvH - 1)) * cw + uox - 1];
                        left[i] = l; cSumLeft += l;
                    }
                    if (hasTop) cTopLeft = ch == 0 ? uP[(uoy - 1) * cw + uox - 1] : vP[(uoy - 1) * cw + uox - 1];
                    else cTopLeft = 127;
                }
                int dc8;
                if (hasTop) dc8 = hasLeft ? (cSumTop + cSumLeft + 8) >> 4 : (cSumTop + cSumTop + 8) >> 4;
                else if (hasLeft) dc8 = (cSumLeft + cSumLeft + 8) >> 4;
                else dc8 = 0x80;

                for (int blk = 0; blk < 4; ++blk)
                {
                    int bx = (blk & 1) * 4, by = (blk >> 1) * 4;
                    for (int j = 0; j < 4; ++j)
                        for (int i = 0; i < 4; ++i)
                        {
                            int px = bx + i, py = by + j;
                            int pred;
                            if (mode == 0) pred = dc8;
                            else
                            {
                                if (hasLeft) pred = hasTop ? Clip8(left[py] + top[px] - cTopLeft) : left[py];
                                else pred = hasTop ? top[px] : 129;
                            }
                            int s = ch == 0 ? Src(uP, cw, uox, uoy, px, py, uvW, uvH) : Src(vP, cw, uox, uoy, px, py, uvW, uvH);
                            d[j * 4 + i] = s - pred;
                        }
                    AddBlock(d, tmp, o, dist);
                }
            }
            int alpha = AlphaOf(dist);
            if (alpha > bestUvAlpha) bestUvAlpha = alpha;
            bool smaller = (mode == 0) | (alpha < smallestAlpha);
            if (smaller) { smallestAlpha = alpha; bestUvMode = mode; }
        }

        // ---------------- final susceptibility (MBAnalyze) ----------------
        int mixed = (3 * bestAlpha + bestUvAlpha + 2) >> 2;
        mixed = MaxAlpha - mixed;
        mixed = mixed < 0 ? 0 : (mixed > MaxAlpha ? MaxAlpha : mixed);

        int rec = mb * WebPLossyEncMeta.MbStride;
        mbRecs[rec + WebPLossyEncMeta.MbType] = 1;
        mbRecs[rec + WebPLossyEncMeta.MbUVMode] = bestUvMode;
        mbRecs[rec + WebPLossyEncMeta.MbSkip] = 0;
        mbRecs[rec + WebPLossyEncMeta.MbSegment] = 0;
        mbRecs[rec + WebPLossyEncMeta.MbAlpha] = mixed;
        mbRecs[rec + WebPLossyEncMeta.MbUVAlphaRaw] = bestUvAlpha;
        // transparency of this macroblock's pixels (libwebp WebPPictureHasTransparency, reduced in kernel 2)
        int transparent = 0;
        int rows = meta[WebPLossyEncMeta.AlphaPlanePresent] != 0 ? h : 0;
        for (int py = 0; py < rows; ++py)
            for (int px = 0; px < w; ++px)
                if (aP[(oy + py) * width + ox + px] != 255) transparent = 1;
        mbRecs[rec + WebPLossyEncMeta.MbTransparent] = transparent;

        int predsW = 4 * mbW + 1;
        int pBase = 1 + predsW + my * 4 * predsW + mx * 4;
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c) preds[pBase + r * predsW + c] = (byte)bestMode;
    }
}
