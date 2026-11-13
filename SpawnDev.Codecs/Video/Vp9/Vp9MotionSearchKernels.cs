// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt). See NOTICE.md for upstream attributions.
//
// VP9 inter frame encoder, the motion search (Plans/PLAN-VP9-Encoder-Inter-Frames.md, stage 2): one MV per leaf
// (16x16 position), every leaf independent of the others, so each launch is one thread per leaf (or per pixel).
//
//   DownsampleKernel  2x2 box filter: the source and the reference luma at 1/2 and 1/4 size (a pyramid)
//   FullPelKernel     coarse to fine: every offset within +-8 at 1/4 size (+-32 pixels), +-2 around it at 1/2 size,
//                     +-2 at full size: sum of absolute differences, a small cost on the MV's length (zero motion and
//                     short MVs win ties, so flat areas do not get noise for MVs)
//   RefineKernel      a few candidates per leaf, each scored on the prediction the decoder will really make
//                     (Vp9InterPred.PredictBlockPlane: the EIGHTTAP filter, the edge handling): the half-pel
//                     neighbours, then the quarter-pel neighbours, then the neighbouring leaves' MVs and zero (a leaf
//                     that takes a neighbour's MV is coded NEARESTMV / NEARMV instead of NEWMV + the MV)
//
// The search only chooses MVs. The prediction, and so the reference the encoder keeps, is made by the decoder's
// InterPredictKernel from the MVs in the mode info, whatever the search did.

using ILGPU;
using SpawnDev.Codecs.Video.Vp9.Decoder;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>The motion search's frame arguments, after the encoder's (<see cref="Vp9IArg"/>).</summary>
internal static class Vp9MArg
{
    /// <summary>The picture's size (crop), and the pyramid levels' sizes.</summary>
    public const int Width = Vp9IArg.Count, Height = Width + 1, HalfW = Width + 2, HalfH = Width + 3, QuarterW = Width + 4, QuarterH = Width + 5;
    /// <summary>The source's and the reference's luma: stride (both), absolute offsets in the source buffer and the pool.</summary>
    public const int Stride = Width + 6, SrcAt = Width + 7, RefAt = Width + 8;
    /// <summary>Cost (in SAD units) of a pixel of MV length in the full-pel search.</summary>
    public const int LengthCost = Width + 9;
    /// <summary>Cost (in SAD units) of an MV that is none of the leaf's neighbours' and not zero: a NEWMV.</summary>
    public const int NewMvCost = Width + 10;
    public const int Count = Width + 11;
}

internal static class Vp9MotionSearchKernels
{
    /// <summary>The search range at 1/4 size (+-8 = +-32 pixels) and the refinement range at 1/2 and full size.</summary>
    public const int CoarseRange = 8, FineRange = 2;
    /// <summary>Bytes of prediction scratch per leaf (one 16x16 block).</summary>
    public const int LeafBytes = 256;

    /// <summary>
    /// One thread per pixel of the smaller plane (launch <paramref name="dstW"/> x its height): the rounded mean of the
    /// 2x2 pixels of the larger one (positions past its edge repeat the edge).
    /// </summary>
    public static void DownsampleKernel(Index1D index, ArrayView<byte> src, ArrayView<byte> dst, int srcAt, int srcStride, int srcW, int srcH,
        int dstAt, int dstW)
    {
        int y = index / dstW, x = index - y * dstW;
        int x0 = 2 * x, x1 = x0 + 1, y0 = 2 * y, y1 = y0 + 1;
        if (x1 > srcW - 1) x1 = srcW - 1;
        if (y1 > srcH - 1) y1 = srcH - 1;
        int r0 = srcAt + y0 * srcStride, r1 = srcAt + y1 * srcStride;
        dst[dstAt + index] = (byte)((src[r0 + x0] + src[r0 + x1] + src[r1 + x0] + src[r1 + x1] + 2) >> 2);
    }

    /// <summary>
    /// Sum of absolute differences between the <paramref name="w"/> x <paramref name="h"/> block of the current plane at
    /// (x, y) and the reference plane's block at (x + dx, y + dy); positions outside a plane repeat its edge.
    /// </summary>
    private static int Sad(ArrayView<byte> cur, int curAt, ArrayView<byte> reference, int refAt, int stride, int planeW, int planeH,
        int x, int y, int w, int h, int dx, int dy)
    {
        int sad = 0;
        for (int r = 0; r < h; r++)
        {
            int cy = y + r, ry = cy + dy;
            if (cy < 0) cy = 0;
            if (cy > planeH - 1) cy = planeH - 1;
            if (ry < 0) ry = 0;
            if (ry > planeH - 1) ry = planeH - 1;
            for (int c = 0; c < w; c++)
            {
                int cx = x + c, rx = cx + dx;
                if (cx < 0) cx = 0;
                if (cx > planeW - 1) cx = planeW - 1;
                if (rx < 0) rx = 0;
                if (rx > planeW - 1) rx = planeW - 1;
                int d = cur[curAt + cy * stride + cx] - reference[refAt + ry * stride + rx];
                sad += d < 0 ? -d : d;
            }
        }
        return sad;
    }

    /// <summary>
    /// One thread per leaf: its full-pel MV, coarse to fine over the pyramid (<paramref name="quarter"/> and
    /// <paramref name="half"/> hold the source's plane then the reference's, tightly packed; the full size planes are
    /// in <paramref name="src"/> and <paramref name="pool"/>). Writes the MV (1/8 pixel) to <paramref name="mvs"/>.
    /// </summary>
    public static void FullPelKernel(Index1D index, ArrayView<byte> quarter, ArrayView<byte> half, ArrayView<byte> src, ArrayView<byte> pool,
        ArrayView<int> a, ArrayView<int> mvs)
    {
        int miCols = a[Vp9RArg.MiCols], miRows = a[Vp9RArg.MiRows];
        int cols16 = (miCols + 1) >> 1;
        int r16 = index / cols16, c16 = index - r16 * cols16;
        int shape = Vp9InterEncodeKernels.LeafShape(r16 * 2, c16 * 2, miRows, miCols);
        int bw = shape == 0 | shape == 1 ? 16 : 8, bh = shape == 0 | shape == 2 ? 16 : 8;
        int x = c16 * 16, y = r16 * 16;
        int lengthCost = a[Vp9MArg.LengthCost];

        // 1/4 size: an 8x8 window around the leaf's 4x4 (32x32 pixels of picture), every offset in the range
        int qw = a[Vp9MArg.QuarterW], qh = a[Vp9MArg.QuarterH];
        int bestX = 0, bestY = 0, best = int.MaxValue;
        for (int dy = -CoarseRange; dy <= CoarseRange; dy++)
            for (int dx = -CoarseRange; dx <= CoarseRange; dx++)
            {
                int cost = Sad(quarter, 0, quarter, qw * qh, qw, qw, qh, (x >> 2) - 2, (y >> 2) - 2, 8, 8, dx, dy) * 16
                    + ((dx < 0 ? -dx : dx) + (dy < 0 ? -dy : dy)) * lengthCost;
                if (cost < best)
                {
                    best = cost;
                    bestX = dx;
                    bestY = dy;
                }
            }
        // 1/2 size: the leaf's own 8x8, around twice the 1/4 size offset
        int hw = a[Vp9MArg.HalfW], hh = a[Vp9MArg.HalfH];
        int cx = bestX * 2, cy = bestY * 2;
        best = int.MaxValue;
        for (int dy = cy - FineRange; dy <= cy + FineRange; dy++)
            for (int dx = cx - FineRange; dx <= cx + FineRange; dx++)
            {
                int cost = Sad(half, 0, half, hw * hh, hw, hw, hh, x >> 1, y >> 1, bw >> 1, bh >> 1, dx, dy) * 4
                    + ((dx < 0 ? -dx : dx) + (dy < 0 ? -dy : dy)) * lengthCost * 2;
                if (cost < best)
                {
                    best = cost;
                    bestX = dx;
                    bestY = dy;
                }
            }
        // full size: the leaf, around twice the 1/2 size offset
        int w = a[Vp9MArg.Width], h = a[Vp9MArg.Height], stride = a[Vp9MArg.Stride];
        cx = bestX * 2;
        cy = bestY * 2;
        best = int.MaxValue;
        for (int dy = cy - FineRange; dy <= cy + FineRange; dy++)
            for (int dx = cx - FineRange; dx <= cx + FineRange; dx++)
            {
                int cost = Sad(src, a[Vp9MArg.SrcAt], pool, a[Vp9MArg.RefAt], stride, w, h, x, y, bw, bh, dx, dy)
                    + ((dx < 0 ? -dx : dx) + (dy < 0 ? -dy : dy)) * lengthCost;
                if (cost < best)
                {
                    best = cost;
                    bestX = dx;
                    bestY = dy;
                }
            }
        mvs[index] = Vp9ParseKernels.PackMv(bestY * 8, bestX * 8);
    }

    /// <summary>
    /// One thread per leaf, after <see cref="Vp9InterEncodeKernels.SetupKernel"/>: of a few candidate MVs, the one whose
    /// real prediction (<see cref="Vp9InterPred.PredictBlockPlane"/>, luma) is nearest to the source, into the leaf's
    /// mode info record and <paramref name="mvsOut"/>. The candidates start from the leaf's MV in <paramref name="mvsIn"/>:
    /// <paramref name="mode"/> 0 its 8 half-pel neighbours, 1 its 8 quarter-pel neighbours, 2 zero and the MVs of the
    /// leaves to the left, above, right and below (from <paramref name="mvsIn"/>: every leaf reads the same snapshot).
    /// One prediction call site, in the candidate loop.
    /// </summary>
    public static void RefineKernel(Index1D index, ArrayView<byte> pool, ArrayView<byte> src, ArrayView<byte> tmp, ArrayView<int> a,
        ArrayView<int> mi, ArrayView<int> mvsIn, ArrayView<int> mvsOut, ArrayView<int> scratch, int mode)
    {
        int miCols = a[Vp9RArg.MiCols], miRows = a[Vp9RArg.MiRows];
        int cols16 = (miCols + 1) >> 1, rows16 = (miRows + 1) >> 1;
        int r16 = index / cols16, c16 = index - r16 * cols16;
        int miRow = r16 * 2, miCol = c16 * 2;
        int shape = Vp9InterEncodeKernels.LeafShape(miRow, miCol, miRows, miCols);
        int bw = shape == 0 | shape == 1 ? 16 : 8, bh = shape == 0 | shape == 2 ? 16 : 8;
        int b = (miRow * miCols + miCol) * Vp9Mi.Stride;
        int stride = a[Vp9MArg.Stride];
        int srcAt = a[Vp9MArg.SrcAt] + r16 * 16 * stride + c16 * 16;
        int tmpAt = index * LeafBytes;
        var plane = Vp9ReconKernels.RefPlane(a, Vp9Mi.LastFrame, 0);
        int center = mvsIn[index];
        int step = mode == 0 ? 4 : 2;   // half-pel, quarter-pel (1/8 pixel units)
        int newMvCost = a[Vp9MArg.NewMvCost];

        // merge pass: the leaf's own MV is free too when it is zero or a neighbouring leaf already has it
        bool centerFree = center == 0;
        if (mode == 2)
            for (int k = 0; k < 4; k++)
            {
                int nr = r16 + (k == 1 ? -1 : k == 3 ? 1 : 0), nc = c16 + (k == 0 ? -1 : k == 2 ? 1 : 0);
                if (nr >= 0 & nc >= 0 & nr < rows16 & nc < cols16)
                    if (mvsIn[nr * cols16 + nc] == center) centerFree = true;
            }

        // score only the pixels inside the picture (an edge leaf reaches into the padding)
        int insideW = a[Vp9MArg.Width] - c16 * 16, insideH = a[Vp9MArg.Height] - r16 * 16;
        if (bw > insideW) bw = insideW;
        if (bh > insideH) bh = insideH;

        int bestMv = center, best = int.MaxValue;
        for (int k = 0; k < 9; k++)
        {
            // candidate k: 0 = the leaf's MV; then the 8 sub-pel neighbours, or zero and the 4 neighbouring leaves' MVs
            int cand = center;
            bool valid = k == 0;
            bool free = centerFree;   // costs no MV in the stream: zero, or a neighbouring leaf's MV
            if (mode == 2)
            {
                if (k == 1)
                {
                    cand = 0;
                    valid = true;
                }
                else if (k >= 2 & k <= 5)
                {
                    int nr = r16, nc = c16;
                    if (k == 2) nc = c16 - 1;
                    if (k == 3) nr = r16 - 1;
                    if (k == 4) nc = c16 + 1;
                    if (k == 5) nr = r16 + 1;
                    valid = nr >= 0 & nc >= 0 & nr < rows16 & nc < cols16;
                    if (valid) cand = mvsIn[nr * cols16 + nc];
                }
                if (k != 0) free = true;
                if (cand == center & k != 0) valid = false;   // already scored
            }
            else if (k != 0)
            {
                int j = k - 1 + (k > 4 ? 1 : 0);   // 0..8 without the centre (4): a 3x3 neighbourhood
                cand = Vp9ParseKernels.PackMv(Vp9ParseKernels.MvRow(center) + (j / 3 - 1) * step, Vp9ParseKernels.MvCol(center) + (j % 3 - 1) * step);
                valid = true;
            }
            if (valid)
            {
                mi[b + Vp9Mi.Mv0] = cand;
                Vp9InterPred.PredictBlockPlane(0, 0, miRow, miCol, miRows, miCols, mi, b, pool, plane, plane, tmp, tmpAt, 16,
                    scratch, index * Vp9InterPred.ScratchInts);
                int sad = 0;
                for (int r = 0; r < bh; r++)
                    for (int c = 0; c < bw; c++)
                    {
                        int d = src[srcAt + r * stride + c] - tmp[tmpAt + r * 16 + c];
                        sad += d < 0 ? -d : d;
                    }
                // in the merge pass an MV of the leaf's own costs a NEWMV unless it is zero
                int cost = sad + (mode == 2 & !free ? newMvCost : 0);
                if (cost < best)
                {
                    best = cost;
                    bestMv = cand;
                }
            }
        }
        mi[b + Vp9Mi.Mv0] = bestMv;
        mvsOut[index] = bestMv;
    }
}
