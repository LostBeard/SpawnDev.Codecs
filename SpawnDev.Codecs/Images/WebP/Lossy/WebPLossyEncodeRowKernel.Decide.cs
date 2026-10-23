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

    private static void PickBestIntra16(ArrayView<byte> sb, ArrayView<int> si, ArrayView<int> meta, ArrayView<int> t,
        ArrayView<int> coeffs, ArrayView<int> lc, ArrayView<int> nzRow, ArrayView<byte> preds, ArrayView<int> mbRecs,
        ref RowCtx c, ref WebPModeScore rd, ref int rdModeI16)
    {
        int segBase = M.Seg + c.Seg * M.SegStride;
        int lambda = meta[segBase + M.SLambdaI16];
        int tlambda = meta[segBase + M.STLambda];
        var tmpV = si.SubView(IT16b, 16);
        bool isFlat = IsFlatSource16(sb);
        // rd (slot A) and rd_tmp (slot B); `bestIsA` tracks libwebp's rd_best pointer, the other slot is rd_cur.
        var sA = rd;   // rd as initialised by VP8Decimate (score = MAX_COST)
        var sB = new WebPModeScore();
        sB.Init();
        int modeA = -1, modeB = -1;
        bool bestIsA = true;
        for (int mode = 0; mode < 4; ++mode)
        {
            int curSlot = bestIsA ? IRdB : IRdA;
            var cur = new WebPModeScore();
            int tmpDst = c.Out2Buf;
            cur.Nz = ReconstructIntra16(sb, si, meta, t, c.Seg, curSlot, tmpDst, mode);
            cur.D = SSE(sb, BIn, sb, tmpDst, 16, 16);
            cur.SD = tlambda != 0 ? (tlambda * Disto16x16(sb, BIn, sb, tmpDst, t, tmpV) + 128) >> 8 : 0;
            cur.H = t[T.FixedCostsI16 + mode];
            cur.R = GetCostLuma16(t, coeffs, lc, nzRow, c.X, si, curSlot);
            if (isFlat)
            {
                isFlat = IsFlat(si, curSlot + LAc, 16, 0);
                if (isFlat) { cur.D *= 2; cur.SD *= 2; }
            }
            cur.SetRD(lambda);
            long bestScore = bestIsA ? sA.Score : sB.Score;
            bool better = (mode == 0) | (cur.Score < bestScore);
            // store cur into its slot's scalars
            if (bestIsA) { sB = cur; modeB = mode; } else { sA = cur; modeA = mode; }
            if (better)
            {
                bestIsA = !bestIsA;
                SwapOut(ref c);
            }
        }
        if (!bestIsA)
        {
            // memcpy(rd, rd_best): the levels and the scalars of slot B become rd's
            for (int i = 0; i < 400; ++i) si[IRdA + i] = si[IRdB + i];
            sA = sB;
            modeA = modeB;
        }
        rd = sA;
        rdModeI16 = modeA;
        rd.SetRD(meta[segBase + M.SLambdaMode]);
        SetIntra16Mode(preds, ref c, rdModeI16);
        mbRecs[(c.Y * c.MbW + c.X) * M.MbStride + M.MbType] = 1;
        bool blocky = ((rd.Nz & 0x100ffffu) == 0x1000000u) & (rd.D > meta[segBase + M.SMinDisto]);
        if (blocky)
        {
            int v0 = Abs(si[IRdA + LDc + 1]), v1 = Abs(si[IRdA + LDc + 2]), v2 = Abs(si[IRdA + LDc + 4]);
            int maxV = v1 > v0 ? v1 : v0;
            maxV = v2 > maxV ? v2 : maxV;
            if (maxV > meta[segBase + M.SMaxEdge]) meta[segBase + M.SMaxEdge] = maxV;
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

    private static bool PickBestIntra4(ArrayView<byte> sb, ArrayView<int> si, ArrayView<int> meta, ArrayView<int> t,
        ArrayView<int> coeffs, ArrayView<int> lc, ArrayView<int> nzRow, ArrayView<byte> preds, ArrayView<byte> yTop,
        ArrayView<int> mbRecs, ref RowCtx c, ref WebPModeScore rd)
    {
        int segBase = M.Seg + c.Seg * M.SegStride;
        int lambda = meta[segBase + M.SLambdaI4];
        int tlambda = meta[segBase + M.STLambda];
        int lambdaMode = meta[segBase + M.SLambdaMode];
        int maxI4HeaderBits = meta[M.PassMaxI4HeaderBits];
        var tmpV = si.SubView(IT16, 16);
        if (maxI4HeaderBits == 0) return false;

        int bestBlocks = c.Out2Buf;
        int totalHeaderBits = 0;
        var rdBest = new WebPModeScore();
        rdBest.Init();
        rdBest.H = 211;
        rdBest.SetRD(lambdaMode);
        StartI4(sb, yTop, ref c, nzRow, si);
        bool ok = true;
        for (int i4 = 0; i4 < 16; ++i4)
        {
            if (ok)
            {
                int bx = i4 & 3, by = i4 >> 2;
                int src = BIn + ScanY(i4);
                int left = bx == 0 ? preds[c.PredsIdx + by * c.PredsW - 1] : si[IModesI4 + i4 - 1];
                int top = by == 0 ? preds[c.PredsIdx - c.PredsW + bx] : si[IModesI4 + i4 - 4];
                int modeCosts = T.FixedCostsI4 + (top * 10 + left) * 10;
                int bestBlock = bestBlocks + ScanY(i4);
                int tmpDst = BPred + I4TMP;
                int bestMode = -1;
                var rdI4 = new WebPModeScore();
                rdI4.Init();
                Intra4Preds(sb, BI4 + TopLeftI4(i4));
                for (int mode = 0; mode < 10; ++mode)
                {
                    var rdTmp = new WebPModeScore();
                    rdTmp.Nz = (uint)ReconstructIntra4(sb, si, meta, t, c.Seg, ITmpLv, src, tmpDst, mode) << i4;
                    rdTmp.D = SSE(sb, src, sb, tmpDst, 4, 4);
                    rdTmp.SD = tlambda != 0 ? (tlambda * Disto4x4(sb, src, sb, tmpDst, t, tmpV) + 128) >> 8 : 0;
                    rdTmp.H = t[modeCosts + mode];
                    bool flatPenalty = (mode > 0) & IsFlat(si, ITmpLv, 1, 3);
                    rdTmp.R = flatPenalty ? 140 : 0;
                    rdTmp.SetRD(lambda);
                    bool skipMode = (bestMode >= 0) & (rdTmp.Score >= rdI4.Score);
                    if (!skipMode)
                    {
                        rdTmp.R += GetResidualCost(t, coeffs, lc, si[ITopNz + bx] + si[ILeftNz + by], 3, 0, si, ITmpLv);
                        rdTmp.SetRD(lambda);
                        bool better = (bestMode < 0) | (rdTmp.Score < rdI4.Score);
                        if (better)
                        {
                            rdI4 = rdTmp;
                            bestMode = mode;
                            int sw = tmpDst; tmpDst = bestBlock; bestBlock = sw;
                            for (int k = 0; k < 16; ++k) si[II4Best + i4 * 16 + k] = si[ITmpLv + k];
                        }
                    }
                }
                rdI4.SetRD(lambdaMode);
                rdBest.D += rdI4.D; rdBest.SD += rdI4.SD; rdBest.R += rdI4.R; rdBest.H += rdI4.H;
                rdBest.Nz |= rdI4.Nz; rdBest.Score += rdI4.Score;
                if (rdBest.Score >= rd.Score) ok = false;
                if (ok)
                {
                    totalHeaderBits += (int)rdI4.H;
                    if (totalHeaderBits > maxI4HeaderBits) ok = false;
                }
                if (ok)
                {
                    int home = bestBlocks + ScanY(i4);
                    if (bestBlock != home) CopyBlock(sb, bestBlock, home, 4, 4);
                    si[IModesI4 + i4] = bestMode;
                    int nzb = rdI4.Nz != 0 ? 1 : 0;
                    si[ITopNz + bx] = nzb;
                    si[ILeftNz + by] = nzb;
                    if (i4 < 15) RotateI4(sb, i4, bestBlocks);
                }
            }
        }
        if (!ok) return false;
        rd.D = rdBest.D; rd.SD = rdBest.SD; rd.R = rdBest.R; rd.H = rdBest.H; rd.Nz = rdBest.Nz; rd.Score = rdBest.Score;
        for (int y = 0; y < 4; ++y)
            for (int k = 0; k < 4; ++k) preds[c.PredsIdx + y * c.PredsW + k] = (byte)si[IModesI4 + y * 4 + k];
        mbRecs[(c.Y * c.MbW + c.X) * M.MbStride + M.MbType] = 0;
        SwapOut(ref c);
        for (int i = 0; i < 256; ++i) si[IRdA + LAc + i] = si[II4Best + i];
        return true;
    }

    // ------------------------------------------------------------------ PickBestUV

    private static void PickBestUV(ArrayView<byte> sb, ArrayView<int> si, ArrayView<int> meta, ArrayView<int> t,
        ArrayView<int> coeffs, ArrayView<int> lc, ArrayView<int> nzRow, ArrayView<int> topDerr, ArrayView<int> mbRecs,
        ref RowCtx c, ref WebPModeScore rd)
    {
        int segBase = M.Seg + c.Seg * M.SegStride;
        int lambda = meta[segBase + M.SLambdaUV];
        int tmpDst = c.Out2Buf;
        int dst0 = c.OutBuf;
        int dst = dst0;
        var rdBest = new WebPModeScore();
        rdBest.Init();
        int modeUv = -1;
        for (int mode = 0; mode < 4; ++mode)
        {
            var rdUv = new WebPModeScore();
            rdUv.Nz = ReconstructUV(sb, si, meta, t, topDerr, c.X, c.ErrDiff, c.Seg, IUvLv, IDerrUv, tmpDst, mode);
            rdUv.D = SSE(sb, BIn + 16, sb, tmpDst + 16, 16, 8);
            rdUv.SD = 0;
            rdUv.H = t[T.FixedCostsUV + mode];
            rdUv.R = GetCostUV(t, coeffs, lc, nzRow, c.X, si, IUvLv);
            bool flatPenalty = (mode > 0) & IsFlat(si, IUvLv, 8, 2);
            if (flatPenalty) rdUv.R += 140 * 8;
            rdUv.SetRD(lambda);
            bool better = (mode == 0) | (rdUv.Score < rdBest.Score);
            if (better)
            {
                rdBest = rdUv;
                modeUv = mode;
                for (int i = 0; i < 128; ++i) si[IRdA + LUv + i] = si[IUvLv + i];
                if (c.ErrDiff) for (int i = 0; i < 6; ++i) si[IDerrA + i] = si[IDerrUv + i];
                int sw = dst; dst = tmpDst; tmpDst = sw;
            }
        }
        mbRecs[(c.Y * c.MbW + c.X) * M.MbStride + M.MbUVMode] = modeUv;
        rd.D += rdBest.D; rd.SD += rdBest.SD; rd.R += rdBest.R; rd.H += rdBest.H; rd.Nz |= rdBest.Nz; rd.Score += rdBest.Score;
        if (dst != dst0) CopyBlock(sb, dst + 16, dst0 + 16, 16, 8);
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
