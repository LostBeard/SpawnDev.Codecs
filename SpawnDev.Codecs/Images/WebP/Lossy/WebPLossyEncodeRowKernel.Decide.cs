// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Kernel 3, part 3: the RD mode decisions (PickBestIntra16 / PickBestIntra4 / PickBestUV), RecordTokens, the token
// probability refresh, SaveBoundary and the row driver. Follows Vp8LossyEncoderReference statement by statement.

using ILGPU;
using ILGPU.Runtime;
using M = SpawnDev.Codecs.Images.WebP.Lossy.WebPLossyEncMeta;
using T = SpawnDev.Codecs.Images.WebP.Lossy.WebPLossyEncTables;

namespace SpawnDev.Codecs.Images.WebP.Lossy;

internal static partial class WebPLossyEncodeRowKernel
{
    /// <summary>VP8TopLeftI4: position of each 4x4 block's top sample inside the 37-sample boundary.</summary>
    private static int TopLeftI4(int i4)
    {
        int x = i4 & 3, y = i4 >> 2;
        return 17 + 4 * x - 4 * y;
    }

    /// <summary>Row-level context shared by the helpers (passed by ref).</summary>
    private struct RowCtx
    {
        public int X, Y, MbW, MbH, Width, Height, Seg, PredsW, PredsIdx, OutBuf, Out2Buf, YTopOff, UVTopOff;
        public bool ErrDiff;
    }

    private static void SwapOut(ref RowCtx c) { int tmp = c.OutBuf; c.OutBuf = c.Out2Buf; c.Out2Buf = tmp; }

    private static void SetIntra16Mode(ArrayView<byte> preds, ref RowCtx c, int mode)
    {
        for (int y = 0; y < 4; ++y)
            for (int k = 0; k < 4; ++k) preds[c.PredsIdx + y * c.PredsW + k] = (byte)mode;
    }

    // ------------------------------------------------------------------ PickBestIntra16

    /// <summary>Lanes of the Intra16 stage's thread group: 16 per mode, one per 4x4 block.</summary>
    public const int I16Lanes = 64;

    private static int I16LaneRecon(int mode) => BI16Lanes + (mode >> 1) * 16 * BPS + (mode & 1) * 16;

    /// <summary>
    /// PickBestIntra16 by a thread group of <see cref="I16Lanes"/> lanes (every lane calls it; barriers inside): lanes
    /// 16m .. 16m + 15 reconstruct and score mode m into the mode's scratch (a level slot, tmps, nz contexts, the 16x16
    /// reconstruction at <see cref="I16LaneRecon"/>), a lane per 4x4 block: the forward transforms; the DC transform and
    /// its quantization (the mode's first lane); the AC quantizations; the inverse DC transform (first lane); then each
    /// block's reconstruction, SSE, spectral distortion and token cost (a block's nz context is its top / left neighbour's
    /// level nz, all known once the blocks are quantized) - the same integers as ReconstructIntra16 / GetCostLuma16 /
    /// Disto16x16 in one thread, summed by the mode's first lane. Then lane 0 replays libwebp's walk over the 4 results in
    /// order - the flatness that doubles D / SD holds while every mode so far was flat (a prefix AND), the first minimum
    /// wins - copies the winner's levels into slot A and its reconstruction into the output buffer (libwebp swaps the two
    /// output buffers instead: the contents later stages read are the same). Only lane 0's <paramref name="rd"/> / mode count.
    /// </summary>
    private static void PickBestIntra16(int lane, ArrayView<byte> sb, ArrayView<int> si, ArrayView<int> meta, ArrayView<int> t,
        ArrayView<int> coeffs, ArrayView<int> lc, ArrayView<int> nzRow, ArrayView<byte> preds, ArrayView<int> mbRecs,
        ref RowCtx c, ref WebPModeScore rd, ref int rdModeI16)
    {
        int segBase = M.Seg + c.Seg * M.SegStride;
        int lambda = meta[segBase + M.SLambdaI16];
        int tlambda = meta[segBase + M.STLambda];
        {
            int mode = lane >> 4, n = lane & 15;
            int lo = II16Lanes + mode * I16LaneStride;
            int dst = I16LaneRecon(mode);
            int refO = BPred + I16Off(mode);
            int slot = lo + I16LSlot, tmp = lo + I16LTmp, dcTmp = lo + I16LDc;
            int lv = slot + LAc + n * 16;
            var blkTmp = si.SubView(lo + I16LBlk + n * 16, 16);
            int part = lo + I16LPart + n * I16PartStride;
            // the forward transform of block n; the mode's first lane also takes the macroblock's nz contexts
            FTransform(sb, BIn + ScanY(n), sb, refO + ScanY(n), si, tmp + n * 16, blkTmp);
            if (n == 0)
            {
                NzToBytes(nzRow, c.X, si, lo + I16LTopNz, lo + I16LLeftNz);
                // left[8] (the left DC block's nz) is NOT rebuilt by NzToBytes: it is state across macroblocks (zeroed at
                // a row start, set by RecordTokens) - give the mode's copy the shared value (it read garbage until 2026-11-15)
                si[lo + I16LLeftNz + 8] = si[ILeftNz + 8];
            }
            Group.Barrier();   // every block's coefficients
            if (n == 0)
            {
                FTransformWHT(si, tmp, si, dcTmp, si.SubView(lo + I16LT16, 16));
                si[lo + I16LNzDc] = QuantizeBlock(si, dcTmp, si, slot + LDc, meta, c.Seg, 1, t);
            }
            Group.Barrier();   // the DCs are taken (the AC quantization clears them)
            si[tmp + n * 16] = 0;
            si[part + PQ] = QuantizeBlock(si, tmp + n * 16, si, lv, meta, c.Seg, 0, t);
            si[part + PNzb] = LastNonZero(si, lv) >= 0 ? 1 : 0;
            int nonZeroAc = 0;
            for (int i = 1; i < 16; ++i) nonZeroAc += si[lv + i] != 0 ? 1 : 0;
            si[part + PFlat] = nonZeroAc;
            Group.Barrier();   // every block quantized (its DC slot is 0 again)
            if (n == 0) TransformWHT(si, dcTmp, si, tmp, si.SubView(lo + I16LT16, 16));
            Group.Barrier();   // every block's DC is back
            ITransformOne(sb, refO + ScanY(n), si, tmp + n * 16, sb, dst + ScanY(n), blkTmp);
            si[part + PSse] = SSE(sb, BIn + ScanY(n), sb, dst + ScanY(n), 4, 4);
            si[part + PDisto] = tlambda != 0 ? Disto4x4(sb, BIn + ScanY(n), sb, dst + ScanY(n), t, blkTmp) : 0;
            int xx = n & 3, yy = n >> 2;
            int topCtx = yy == 0 ? si[lo + I16LTopNz + xx] : si[part - 4 * I16PartStride + PNzb];
            int leftCtx = xx == 0 ? si[lo + I16LLeftNz + yy] : si[part - I16PartStride + PNzb];
            si[part + PCost] = GetResidualCost(t, coeffs, lc, topCtx + leftCtx, 0, 1, si, lv);
            Group.Barrier();   // every block scored
            if (n == 0)
            {
                var cur = new WebPModeScore();
                uint nz = (uint)si[lo + I16LNzDc] << 24;
                long d = 0;
                int disto = 0, r = 0, flatCount = 0;
                int p0 = lo + I16LPart;
                for (int b = 0; b < 16; ++b)
                {
                    int pb = p0 + b * I16PartStride;
                    nz |= (uint)si[pb + PQ] << b;
                    d += si[pb + PSse];
                    disto += si[pb + PDisto];
                    r += si[pb + PCost];
                    flatCount += si[pb + PFlat];
                }
                cur.Nz = nz;
                cur.D = d;
                cur.SD = tlambda != 0 ? (tlambda * disto + 128) >> 8 : 0;
                cur.H = t[T.FixedCostsI16 + mode];
                cur.R = GetResidualCost(t, coeffs, lc, si[lo + I16LTopNz + 8] + si[lo + I16LLeftNz + 8], 1, 0, si, slot + LDc) + r;
                StoreLaneScore(si, lo + I16LScore, ref cur);
                si[lo + I16LFlat] = flatCount <= 0 ? 1 : 0;   // IsFlat(levels, 16 blocks, thresh 0)
            }
        }
        Group.Barrier();   // the 4 modes are scored
        if (lane != 0) return;
        bool isFlat = IsFlatSource16(sb);
        int best = 0;
        var rdBest = new WebPModeScore();
        for (int mode = 0; mode < 4; ++mode)
        {
            int lo = II16Lanes + mode * I16LaneStride;
            var cur = LoadLaneScore(si, lo + I16LScore);
            if (isFlat)
            {
                isFlat = si[lo + I16LFlat] != 0;
                if (isFlat) { cur.D *= 2; cur.SD *= 2; }
            }
            cur.SetRD(lambda);
            if ((mode == 0) | (cur.Score < rdBest.Score))
            {
                rdBest = cur;
                best = mode;
            }
        }
        int blo = II16Lanes + best * I16LaneStride + I16LSlot;
        for (int i = 0; i < 400; ++i) si[IRdA + i] = si[blo + i];
        CopyBlock(sb, I16LaneRecon(best), c.OutBuf, 16, 16);
        rd = rdBest;
        rdModeI16 = best;
        rd.SetRD(meta[segBase + M.SLambdaMode]);
        SetIntra16Mode(preds, ref c, rdModeI16);
        mbRecs[(c.Y * c.MbW + c.X) * M.MbStride + M.MbType] = 1;
        bool blocky = ((rd.Nz & 0x100ffffu) == 0x1000000u) & (rd.D > meta[segBase + M.SMinDisto]);
        if (blocky)
        {
            int v0 = Abs(si[IRdA + LDc + 1]), v1 = Abs(si[IRdA + LDc + 2]), v2 = Abs(si[IRdA + LDc + 4]);
            int maxV = v1 > v0 ? v1 : v0;
            maxV = v2 > maxV ? v2 : maxV;
            // a maximum over the frame (order-free): atomic, as a wavefront diagonal runs macroblocks of one segment at once
            Atomic.Max(ref meta[segBase + M.SMaxEdge], maxV);
        }
    }

    // ------------------------------------------------------------------ PickBestIntra4

    private static void StartI4(ArrayView<byte> sb, ArrayView<byte> yTop, ref RowCtx c, ArrayView<int> nzRow, ArrayView<int> si)
    {
        for (int i = 0; i < 17; ++i) sb[BI4 + i] = sb[BLeft + YLeft + 15 - i];
        for (int i = 0; i < 16; ++i) sb[BI4 + 17 + i] = yTop[c.YTopOff + i];
        bool notLast = c.X < c.MbW - 1;
        for (int i = 16; i < 20; ++i)
            sb[BI4 + 17 + i] = notLast ? yTop[c.YTopOff + i] : sb[BI4 + 17 + 15];
        NzToBytes(nzRow, c.X, si);
    }

    /// <summary>VP8IteratorRotateI4 for block i4 (the reconstruction is at blocks base `outBase`); returns false after block 15.</summary>
    private static void RotateI4(ArrayView<byte> sb, int i4, int outBase)
    {
        int blk = outBase + ScanY(i4);
        int top = BI4 + TopLeftI4(i4);
        for (int i = 0; i <= 3; ++i) sb[top - 4 + i] = sb[blk + i + 3 * BPS];
        if ((i4 & 3) != 3)
        {
            for (int i = 0; i <= 2; ++i) sb[top + i] = sb[blk + 3 + (2 - i) * BPS];
        }
        else
        {
            for (int i = 0; i <= 3; ++i) sb[top + i] = sb[top + i + 4];
        }
    }

    /// <summary>Lanes of the Intra4 stage's thread group: one per mode (10), padded to a power of two.</summary>
    public const int I4Lanes = 16;

    private static int I4LaneBlock(int mode) => BI4Lanes + (mode >> 3) * 4 * BPS + (mode & 7) * 4;

    private static void StoreLaneScore(ArrayView<int> si, int o, ref WebPModeScore r)
    {
        si[o + 0] = (int)r.D; si[o + 1] = (int)(r.D >> 32);
        si[o + 2] = (int)r.SD; si[o + 3] = (int)(r.SD >> 32);
        si[o + 4] = (int)r.H; si[o + 5] = (int)(r.H >> 32);
        si[o + 6] = (int)r.R; si[o + 7] = (int)(r.R >> 32);
        si[o + 8] = (int)r.Score; si[o + 9] = (int)(r.Score >> 32);
        si[o + 10] = (int)r.Nz;
    }

    private static WebPModeScore LoadLaneScore(ArrayView<int> si, int o)
    {
        var r = new WebPModeScore();
        r.D = Long(si, o + 0); r.SD = Long(si, o + 2); r.H = Long(si, o + 4); r.R = Long(si, o + 6); r.Score = Long(si, o + 8);
        r.Nz = (uint)si[o + 10];
        return r;
    }

    /// <summary>
    /// PickBestIntra4 by a thread group of <see cref="I4Lanes"/> lanes (every lane calls it; barriers inside): per 4x4
    /// block, lane m (m &lt; 10) reconstructs and scores mode m into its own scratch (levels, tmps, the reconstruction at
    /// <see cref="I4LaneBlock"/>), and lane 0 keeps libwebp's choice and bookkeeping. libwebp walks the modes in order,
    /// skips the residual cost of a mode whose score without it already reaches the best (adding the cost only raises
    /// the score) and keeps a mode only when strictly better: the first minimum of the FULL scores - so the lanes score
    /// every mode in full and lane 0 takes the first minimum. Same choices, same bits. Only lane 0's
    /// <paramref name="rd"/> and <paramref name="c"/> count. Barriers stay outside every branch (WGSL's uniformity rule).
    /// </summary>
    private static bool PickBestIntra4(int lane, ArrayView<byte> sb, ArrayView<int> si, ArrayView<int> meta, ArrayView<int> t,
        ArrayView<int> coeffs, ArrayView<int> lc, ArrayView<int> nzRow, ArrayView<byte> preds, ArrayView<byte> yTop,
        ArrayView<int> mbRecs, ref RowCtx c, ref WebPModeScore rd)
    {
        int segBase = M.Seg + c.Seg * M.SegStride;
        int lambda = meta[segBase + M.SLambdaI4];
        int tlambda = meta[segBase + M.STLambda];
        int lambdaMode = meta[segBase + M.SLambdaMode];
        int maxI4HeaderBits = meta[M.PassMaxI4HeaderBits];

        int bestBlocks = c.Out2Buf;
        int totalHeaderBits = 0;   // lane 0's
        var rdBest = new WebPModeScore();   // lane 0's
        rdBest.Init();
        rdBest.H = 211;
        rdBest.SetRD(lambdaMode);
        if (lane == 0)
        {
            if (maxI4HeaderBits != 0) StartI4(sb, yTop, ref c, nzRow, si);
            si[II4Ok] = maxI4HeaderBits != 0 ? 1 : 0;
        }
        for (int i4 = 0; i4 < 16; ++i4)
        {
            Group.Barrier();   // the previous block's bookkeeping (modes, nz, the rotated boundary) is visible
            bool ok = si[II4Ok] != 0;
            int bx = i4 & 3, by = i4 >> 2;
            if (ok & (lane == 0)) Intra4Preds(sb, BI4 + TopLeftI4(i4));
            Group.Barrier();   // the 10 predictions are visible
            if (ok & (lane < 10))
            {
                int mode = lane;
                int src = BIn + ScanY(i4);
                int left = bx == 0 ? preds[c.PredsIdx + by * c.PredsW - 1] : si[IModesI4 + i4 - 1];
                int top = by == 0 ? preds[c.PredsIdx - c.PredsW + bx] : si[IModesI4 + i4 - 4];
                int lo = II4Lanes + lane * I4LaneStride;
                int dst = I4LaneBlock(mode);
                var r = new WebPModeScore();
                r.Nz = (uint)ReconstructIntra4(sb, si, meta, t, c.Seg, lo + I4LLevels, src, dst, mode, lo + I4LTmpA, lo + I4LTmpB) << i4;
                r.D = SSE(sb, src, sb, dst, 4, 4);
                r.SD = tlambda != 0 ? (tlambda * Disto4x4(sb, src, sb, dst, t, si.SubView(lo + I4LTmpA, 16)) + 128) >> 8 : 0;
                r.H = t[T.FixedCostsI4 + (top * 10 + left) * 10 + mode];
                r.R = (mode > 0) & IsFlat(si, lo + I4LLevels, 1, 3) ? 140 : 0;
                r.R += GetResidualCost(t, coeffs, lc, si[ITopNz + bx] + si[ILeftNz + by], 3, 0, si, lo + I4LLevels);
                r.SetRD(lambda);
                StoreLaneScore(si, lo + I4LScore, ref r);
            }
            Group.Barrier();   // every mode is scored
            if (ok & (lane == 0))
            {
                int bestMode = 0;
                var rdI4 = LoadLaneScore(si, II4Lanes + I4LScore);
                for (int mode = 1; mode < 10; ++mode)
                {
                    var r = LoadLaneScore(si, II4Lanes + mode * I4LaneStride + I4LScore);
                    if (r.Score < rdI4.Score) { rdI4 = r; bestMode = mode; }
                }
                int blo = II4Lanes + bestMode * I4LaneStride + I4LLevels;
                for (int k = 0; k < 16; ++k) si[II4Best + i4 * 16 + k] = si[blo + k];
                rdI4.SetRD(lambdaMode);
                rdBest.D += rdI4.D; rdBest.SD += rdI4.SD; rdBest.R += rdI4.R; rdBest.H += rdI4.H;
                rdBest.Nz |= rdI4.Nz; rdBest.Score += rdI4.Score;
                bool still = rdBest.Score < rd.Score;
                if (still)
                {
                    totalHeaderBits += (int)rdI4.H;
                    if (totalHeaderBits > maxI4HeaderBits) still = false;
                }
                if (still)
                {
                    CopyBlock(sb, I4LaneBlock(bestMode), bestBlocks + ScanY(i4), 4, 4);
                    si[IModesI4 + i4] = bestMode;
                    int nzb = rdI4.Nz != 0 ? 1 : 0;
                    si[ITopNz + bx] = nzb;
                    si[ILeftNz + by] = nzb;
                    if (i4 < 15) RotateI4(sb, i4, bestBlocks);
                }
                si[II4Ok] = still ? 1 : 0;
            }
        }
        Group.Barrier();   // lane 0's last decision is visible
        if (si[II4Ok] == 0) return false;
        if (lane == 0)
        {
            rd.D = rdBest.D; rd.SD = rdBest.SD; rd.R = rdBest.R; rd.H = rdBest.H; rd.Nz = rdBest.Nz; rd.Score = rdBest.Score;
            for (int y = 0; y < 4; ++y)
                for (int k = 0; k < 4; ++k) preds[c.PredsIdx + y * c.PredsW + k] = (byte)si[IModesI4 + y * 4 + k];
            mbRecs[(c.Y * c.MbW + c.X) * M.MbStride + M.MbType] = 0;
            for (int i = 0; i < 256; ++i) si[IRdA + LAc + i] = si[II4Best + i];
        }
        SwapOut(ref c);
        return true;
    }

    // ------------------------------------------------------------------ PickBestUV

    /// <summary>Lanes of the UV stage's thread group: one per mode.</summary>
    public const int UVLanes = 32;   // 8 per mode, one per 4x4 chroma block

    // a lane's chroma reconstruction buffer: 8 rows of BPS, chroma at columns 16..31 (the BOutA layout)
    private static int UVLaneRecon(int mode) => BUVLanes + mode * 8 * BPS - 16;

    /// <summary>
    /// PickBestUV by a thread group of <see cref="UVLanes"/> lanes (every lane calls it; barriers inside): lanes 8m ..
    /// 8m + 7 reconstruct and score mode m into the mode's scratch (uv levels, tmps, error-diffusion output, nz contexts,
    /// the chroma reconstruction at <see cref="UVLaneRecon"/>), a lane per 4x4 block: the forward transforms; the
    /// error-diffusion DC correction (the mode's first lane: sequential over the 4 DCs of a plane); the quantizations;
    /// each block's reconstruction, SSE and token cost (a block's nz context is its top / left neighbour's level nz) -
    /// the same integers as ReconstructUV / GetCostUV in one thread, summed by the mode's first lane. Lane 0 keeps the
    /// first minimum (libwebp's walk: a mode is kept only when strictly better), copies its levels, error-diffusion values
    /// and reconstruction where libwebp's walk leaves them, and updates the error-diffusion state. Only lane 0's
    /// <paramref name="rd"/> counts.
    /// </summary>
    private static void PickBestUV(int lane, ArrayView<byte> sb, ArrayView<int> si, ArrayView<int> meta, ArrayView<int> t,
        ArrayView<int> coeffs, ArrayView<int> lc, ArrayView<int> nzRow, ArrayView<int> topDerr, ArrayView<int> mbRecs,
        ref RowCtx c, ref WebPModeScore rd)
    {
        int segBase = M.Seg + c.Seg * M.SegStride;
        int lambda = meta[segBase + M.SLambdaUV];
        {
            int mode = lane >> 3, n = lane & 7;
            int lo = IUVLanes + mode * UVLaneStride;
            int dst = UVLaneRecon(mode);
            int refO = BPred + UVOff(mode);
            int uvTmp = lo + UVLTmp;
            int lv = lo + UVLLevels + n * 16;
            var blkTmp = si.SubView(lo + UVLBlk + n * 16, 16);
            int part = lo + UVLPart + n * I16PartStride;
            FTransform(sb, BIn + 16 + ScanUV(n), sb, refO + ScanUV(n), si, uvTmp + n * 16, blkTmp);
            if (n == 0) NzToBytes(nzRow, c.X, si, lo + UVLTopNz, lo + UVLLeftNz);
            Group.Barrier();   // every block's coefficients
            if ((n == 0) & c.ErrDiff) CorrectDCValues(si, topDerr, c.X, meta, c.Seg, lo + UVLDerr, uvTmp);
            Group.Barrier();   // the corrected DCs
            si[part + PQ] = QuantizeBlock(si, uvTmp + n * 16, si, lv, meta, c.Seg, 2, t);
            si[part + PNzb] = LastNonZero(si, lv) >= 0 ? 1 : 0;
            int nonZeroAc = 0;
            for (int i = 1; i < 16; ++i) nonZeroAc += si[lv + i] != 0 ? 1 : 0;
            si[part + PFlat] = nonZeroAc;
            Group.Barrier();   // every block quantized
            ITransformOne(sb, refO + ScanUV(n), si, uvTmp + n * 16, sb, dst + 16 + ScanUV(n), blkTmp);
            si[part + PSse] = SSE(sb, BIn + 16 + ScanUV(n), sb, dst + 16 + ScanUV(n), 4, 4);
            // GetCostUV's order: plane k (U, V), then 2x2 blocks; block n = 4k + xx + 2y
            int k = n >> 2, xx = n & 1, yy = (n >> 1) & 1;
            int topCtx = yy == 0 ? si[lo + UVLTopNz + 4 + 2 * k + xx] : si[part - 2 * I16PartStride + PNzb];
            int leftCtx = xx == 0 ? si[lo + UVLLeftNz + 4 + 2 * k + yy] : si[part - I16PartStride + PNzb];
            si[part + PCost] = GetResidualCost(t, coeffs, lc, topCtx + leftCtx, 2, 0, si, lv);
            Group.Barrier();   // every block scored
            if (n == 0)
            {
                var rdUv = new WebPModeScore();
                uint nz = 0;
                long d = 0;
                int r = 0, nonZero = 0;
                int p0 = lo + UVLPart;
                for (int b = 0; b < 8; ++b)
                {
                    int pb = p0 + b * I16PartStride;
                    nz |= (uint)si[pb + PQ] << b;
                    d += si[pb + PSse];
                    r += si[pb + PCost];
                    nonZero += si[pb + PFlat];
                }
                rdUv.Nz = nz << 16;
                rdUv.D = d;
                rdUv.SD = 0;
                rdUv.H = t[T.FixedCostsUV + mode];
                bool flatPenalty = (mode > 0) & (nonZero <= 2);   // IsFlat(levels, 8 blocks, thresh 2)
                rdUv.R = r + (flatPenalty ? 140 * 8 : 0);
                rdUv.SetRD(lambda);
                StoreLaneScore(si, lo + UVLScore, ref rdUv);
            }
        }
        Group.Barrier();   // the 4 modes are scored (the error-diffusion state they read is updated only below)
        if (lane != 0) return;
        int modeUv = 0;
        var rdBest = LoadLaneScore(si, IUVLanes + UVLScore);
        for (int mode = 1; mode < 4; ++mode)
        {
            var r = LoadLaneScore(si, IUVLanes + mode * UVLaneStride + UVLScore);
            if (r.Score < rdBest.Score) { rdBest = r; modeUv = mode; }
        }
        int blo = IUVLanes + modeUv * UVLaneStride;
        for (int i = 0; i < 128; ++i) si[IRdA + LUv + i] = si[blo + UVLLevels + i];
        if (c.ErrDiff) for (int i = 0; i < 6; ++i) si[IDerrA + i] = si[blo + UVLDerr + i];
        mbRecs[(c.Y * c.MbW + c.X) * M.MbStride + M.MbUVMode] = modeUv;
        rd.D += rdBest.D; rd.SD += rdBest.SD; rd.R += rdBest.R; rd.H += rdBest.H; rd.Nz |= rdBest.Nz; rd.Score += rdBest.Score;
        CopyBlock(sb, UVLaneRecon(modeUv) + 16, c.OutBuf + 16, 16, 8);
        if (c.ErrDiff)
        {
            for (int ch = 0; ch <= 1; ++ch)
            {
                int topO = c.X * 4 + ch * 2;
                int leftO = ILeftDerr + ch * 2;
                si[leftO + 0] = si[IDerrA + ch * 3 + 0];
                si[leftO + 1] = (sbyte)((3 * si[IDerrA + ch * 3 + 2]) >> 2);
                topDerr[topO + 0] = si[IDerrA + ch * 3 + 1];
                topDerr[topO + 1] = (sbyte)(si[IDerrA + ch * 3 + 2] - si[leftO + 1]);
            }
        }
    }
}
