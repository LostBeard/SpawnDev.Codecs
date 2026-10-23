// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Kernel 3, part 4: the row driver (VP8EncTokenLoop's do-while over one macroblock row), import, RecordTokens, the
// probability refresh (FinalizeTokenProbas, shared with the final kernel) and SaveBoundary.

using ILGPU;
using ILGPU.Runtime;
using M = SpawnDev.Codecs.Images.WebP.Lossy.WebPLossyEncMeta;
using T = SpawnDev.Codecs.Images.WebP.Lossy.WebPLossyEncTables;

namespace SpawnDev.Codecs.Images.WebP.Lossy;

internal static partial class WebPLossyEncodeRowKernel
{
    /// <summary>
    /// One macroblock row. Buffers: yuv (Y | U | V, tightly packed), mbRecs, work (preds | yTop | byte scratch), state
    /// (nz | topDerr | coeffs | stats | levelCost | int scratch), meta, constant tables, token buffer.
    /// </summary>
    public static void Kernel(
        Index1D _, ArrayView<byte> yuv, ArrayView<int> mbRecs, ArrayView<byte> work, ArrayView<int> state,
        ArrayView<int> meta, ArrayView<int> t, ArrayView<int> tokens, int row)
    {
        if (meta[M.Error] != 0) return;
        int mbW = meta[M.MbW], mbH = meta[M.MbH];
        int width = meta[M.Width], height = meta[M.Height];
        int cw = (width + 1) >> 1;
        int predsW = 4 * mbW + 1;
        var preds = work.SubView(0, predsW * (4 * mbH + 1));
        var yTop = work.SubView(meta[M.WorkYTopOff], 2 * mbW * 16);
        var sb = work.SubView(meta[M.WorkScratchOff], ByteScratchSize);
        var nzRow = state.SubView(meta[M.StNzOff], mbW + 1);
        var topDerr = state.SubView(meta[M.StTopDerrOff], mbW * 4);
        var coeffs = state.SubView(meta[M.StCoeffsOff], WebPLossySetupKernel.ProbaCount);
        var stats = state.SubView(meta[M.StStatsOff], WebPLossySetupKernel.ProbaCount);
        var lc = state.SubView(meta[M.StLevelCostOff], WebPLossySetupKernel.LevelCostCount);
        var si = state.SubView(meta[M.StScratchOff], IntScratchSize);
        int uOff = meta[M.YuvUOff], vOff = meta[M.YuvVOff];
        bool errDiff = meta[M.ErrorDiffusion] != 0;

        if (row == 0)
        {
            // InitTop
            for (int i = 0; i < 2 * mbW * 16; ++i) yTop[i] = 127;
            for (int i = 0; i < mbW + 1; ++i) nzRow[i] = 0;
            for (int i = 0; i < mbW * 4; ++i) topDerr[i] = 0;
        }
        // VP8IteratorSetRow -> InitLeft
        byte corner = (byte)(row > 0 ? 129 : 127);
        sb[BLeft + YLeft - 1] = corner; sb[BLeft + ULeft - 1] = corner; sb[BLeft + VLeft - 1] = corner;
        for (int i = 0; i < 16; ++i) sb[BLeft + YLeft + i] = 129;
        for (int i = 0; i < 8; ++i) { sb[BLeft + ULeft + i] = 129; sb[BLeft + VLeft + i] = 129; }
        si[ILeftNz + 8] = 0;
        for (int i = 0; i < 4; ++i) si[ILeftDerr + i] = 0;

        var c = new RowCtx
        {
            Y = row, MbW = mbW, MbH = mbH, Width = width, Height = height, PredsW = predsW,
            OutBuf = BOutA, Out2Buf = BOutB, ErrDiff = errDiff,
        };
        int maxCount = meta[M.MaxCount];
        long sizeP0 = ((long)meta[M.SizeP0Hi] << 32) | (uint)meta[M.SizeP0Lo];

        for (int x = 0; x < mbW; ++x)
        {
            c.X = x;
            c.PredsIdx = 1 + predsW + row * 4 * predsW + x * 4;
            c.YTopOff = x * 16;
            c.UVTopOff = mbW * 16 + x * 16;
            int rec = (row * mbW + x) * M.MbStride;
            c.Seg = mbRecs[rec + M.MbSegment];

            // ---- VP8IteratorImport (no boundary)
            int w = Min(width - x * 16, 16), h = Min(height - row * 16, 16);
            int uvW = (w + 1) >> 1, uvH = (h + 1) >> 1;
            for (int py = 0; py < 16; ++py)
                for (int px = 0; px < 16; ++px)
                    sb[BIn + py * BPS + px] = yuv[(row * 16 + Min(py, h - 1)) * width + x * 16 + Min(px, w - 1)];
            for (int py = 0; py < 8; ++py)
                for (int px = 0; px < 8; ++px)
                {
                    int s = (row * 8 + Min(py, uvH - 1)) * cw + x * 8 + Min(px, uvW - 1);
                    sb[BIn + 16 + py * BPS + px] = yuv[uOff + s];
                    sb[BIn + 24 + py * BPS + px] = yuv[vOff + s];
                }

            // ---- probability refresh (`if (--cnt < 0)`)
            int cnt = meta[M.MbCountDown] - 1;
            if (cnt < 0)
            {
                FinalizeTokenProbas(coeffs, stats, t, meta);
                if (meta[M.ProbaDirty] != 0)
                {
                    WebPLossySetupKernel.CalculateLevelCosts(coeffs, lc, t);
                    meta[M.ProbaDirty] = 0;
                }
                cnt = maxCount;
            }
            meta[M.MbCountDown] = cnt;

            // ---- VP8Decimate (RD_OPT_BASIC)
            var rd = new WebPModeScore();
            rd.Init();
            int leftOff = x != 0 ? BLeft + YLeft : -1;
            int topOff = row != 0 ? c.YTopOff : -1;
            for (int mode = 0; mode < 4; ++mode)
                PredictBlock(sb, BPred + I16Off(mode), 16, 16, 5, leftOff, yTop, topOff, mode);
            int uLeftOff = x != 0 ? BLeft + ULeft : -1;
            int uvTopOff = row != 0 ? c.UVTopOff : -1;
            int vTopOff = row != 0 ? c.UVTopOff + 8 : -1;
            int vLeftOff = x != 0 ? BLeft + VLeft : -1;
            for (int mode = 0; mode < 4; ++mode)
            {
                PredictBlock(sb, BPred + UVOff(mode), 8, 8, 4, uLeftOff, yTop, uvTopOff, mode);
                PredictBlock(sb, BPred + UVOff(mode) + 8, 8, 8, 4, vLeftOff, yTop, vTopOff, mode);
            }
            int modeI16 = -1;
            PickBestIntra16(sb, si, meta, t, coeffs, lc, nzRow, preds, mbRecs, ref c, ref rd, ref modeI16);
            PickBestIntra4(sb, si, meta, t, coeffs, lc, nzRow, preds, yTop, mbRecs, ref c, ref rd);
            PickBestUV(sb, si, meta, t, coeffs, lc, nzRow, topDerr, mbRecs, ref c, ref rd);
            mbRecs[rec + M.MbSkip] = rd.Nz == 0 ? 1 : 0;

            // ---- RecordTokens
            RecordTokens(si, nzRow, x, coeffs, stats, t, meta, tokens, mbRecs[rec + M.MbType] == 1);
            sizeP0 += rd.H;

            // ---- VP8IteratorSaveBoundary
            int o = c.OutBuf;
            if (x < mbW - 1)
            {
                for (int i = 0; i < 16; ++i) sb[BLeft + YLeft + i] = sb[o + 15 + i * BPS];
                for (int i = 0; i < 8; ++i)
                {
                    sb[BLeft + ULeft + i] = sb[o + 16 + 7 + i * BPS];
                    sb[BLeft + VLeft + i] = sb[o + 16 + 15 + i * BPS];
                }
                sb[BLeft + YLeft - 1] = yTop[c.YTopOff + 15];
                sb[BLeft + ULeft - 1] = yTop[c.UVTopOff + 0 + 7];
                sb[BLeft + VLeft - 1] = yTop[c.UVTopOff + 8 + 7];
            }
            if (row < mbH - 1)
            {
                for (int i = 0; i < 16; ++i) yTop[c.YTopOff + i] = sb[o + 15 * BPS + i];
                for (int i = 0; i < 16; ++i) yTop[c.UVTopOff + i] = sb[o + 16 + 7 * BPS + i];
            }
        }
        meta[M.SizeP0Lo] = (int)sizeP0;
        meta[M.SizeP0Hi] = (int)(sizeP0 >> 32);
    }

    // ================================================================== tokens (token_enc.c) + statistics

    private static int RecordStats(int bit, ArrayView<int> stats, int i)
    {
        uint p = (uint)stats[i];
        if (p >= 0xfffe0000u) p = ((p + 1u) >> 1) & 0x7fff7fffu;
        p += 0x00010000u + (uint)bit;
        stats[i] = (int)p;
        return bit;
    }

    private static void PushToken(ArrayView<int> tokens, ArrayView<int> meta, int value)
    {
        int n = meta[M.TokenCount];
        if (n < meta[M.TokenCapacity]) tokens[n] = value;
        else meta[M.Error] = (int)WebPLossyEncError.TokenBufferTooSmall;
        meta[M.TokenCount] = n + 1;
    }

    private static int AddToken(ArrayView<int> tokens, ArrayView<int> meta, ArrayView<int> stats, int bit, int probaIdx, int statIdx)
    {
        PushToken(tokens, meta, (bit << 15) | probaIdx);
        RecordStats(bit, stats, statIdx);
        return bit;
    }

    private static void AddConstantToken(ArrayView<int> tokens, ArrayView<int> meta, int bit, int proba)
        => PushToken(tokens, meta, (bit << 15) | FixedProbaBit | proba);

    /// <summary>VP8RecordCoeffTokens; returns 1 when the block has a non-zero coefficient.</summary>
    private static int RecordCoeffTokens(ArrayView<int> si, int lvOff, int ctx, int type, int first,
        ArrayView<int> tokens, ArrayView<int> meta, ArrayView<int> stats, ArrayView<int> t)
    {
        int last = LastNonZero(si, lvOff);
        int n = first;
        int baseId = PIdx(type, n, ctx);
        if (AddToken(tokens, meta, stats, last >= 0 ? 1 : 0, baseId + 0, baseId + 0) == 0) return 0;
        bool go = n < 16;
        while (go)
        {
            int cv = si[lvOff + n];
            n++;
            bool sign = cv < 0;
            int v = sign ? -cv : cv;
            bool eob = false;
            if (AddToken(tokens, meta, stats, v != 0 ? 1 : 0, baseId + 1, baseId + 1) == 0)
            {
                baseId = PIdx(type, t[T.EncBands + n], 0);
            }
            else
            {
                if (AddToken(tokens, meta, stats, v > 1 ? 1 : 0, baseId + 2, baseId + 2) == 0)
                {
                    baseId = PIdx(type, t[T.EncBands + n], 1);
                }
                else
                {
                    if (AddToken(tokens, meta, stats, v > 4 ? 1 : 0, baseId + 3, baseId + 3) == 0)
                    {
                        if (AddToken(tokens, meta, stats, v != 2 ? 1 : 0, baseId + 4, baseId + 4) != 0)
                            AddToken(tokens, meta, stats, v == 4 ? 1 : 0, baseId + 5, baseId + 5);
                    }
                    else if (AddToken(tokens, meta, stats, v > 10 ? 1 : 0, baseId + 6, baseId + 6) == 0)
                    {
                        if (AddToken(tokens, meta, stats, v > 6 ? 1 : 0, baseId + 7, baseId + 7) == 0)
                            AddConstantToken(tokens, meta, v == 6 ? 1 : 0, 159);
                        else
                        {
                            AddConstantToken(tokens, meta, v >= 9 ? 1 : 0, 165);
                            AddConstantToken(tokens, meta, (v & 1) == 0 ? 1 : 0, 145);
                        }
                    }
                    else
                    {
                        int mask;
                        int tab;
                        int residue = v - 3;
                        if (residue < (8 << 1))
                        {
                            AddToken(tokens, meta, stats, 0, baseId + 8, baseId + 8);
                            AddToken(tokens, meta, stats, 0, baseId + 9, baseId + 9);
                            residue -= 8 << 0; mask = 1 << 2; tab = T.Cat3;
                        }
                        else if (residue < (8 << 2))
                        {
                            AddToken(tokens, meta, stats, 0, baseId + 8, baseId + 8);
                            AddToken(tokens, meta, stats, 1, baseId + 9, baseId + 9);
                            residue -= 8 << 1; mask = 1 << 3; tab = T.Cat4;
                        }
                        else if (residue < (8 << 3))
                        {
                            // libwebp records the statistic of p[10] in slot 9 for Cat5 / Cat6
                            AddToken(tokens, meta, stats, 1, baseId + 8, baseId + 8);
                            AddToken(tokens, meta, stats, 0, baseId + 10, baseId + 9);
                            residue -= 8 << 2; mask = 1 << 4; tab = T.Cat5;
                        }
                        else
                        {
                            AddToken(tokens, meta, stats, 1, baseId + 8, baseId + 8);
                            AddToken(tokens, meta, stats, 1, baseId + 10, baseId + 9);
                            residue -= 8 << 3; mask = 1 << 10; tab = T.Cat6;
                        }
                        bool more = mask != 0;
                        while (more)
                        {
                            AddConstantToken(tokens, meta, (residue & mask) != 0 ? 1 : 0, t[tab]);
                            tab++;
                            mask >>= 1;
                            more = mask != 0;
                        }
                    }
                    baseId = PIdx(type, t[T.EncBands + n], 2);
                }
                AddConstantToken(tokens, meta, sign ? 1 : 0, 128);
                bool atEnd = n == 16;
                if (atEnd) eob = true;
                else if (AddToken(tokens, meta, stats, n <= last ? 1 : 0, baseId + 0, baseId + 0) == 0) eob = true;
            }
            go = (n < 16) & !eob;
        }
        return 1;
    }

    private static void RecordTokens(ArrayView<int> si, ArrayView<int> nzRow, int x, ArrayView<int> coeffs,
        ArrayView<int> stats, ArrayView<int> t, ArrayView<int> meta, ArrayView<int> tokens, bool i16)
    {
        NzToBytes(nzRow, x, si);
        int acType = 3, acFirst = 0;
        if (i16)
        {
            int ctx = si[ITopNz + 8] + si[ILeftNz + 8];
            int nzDc = RecordCoeffTokens(si, IRdA + LDc, ctx, 1, 0, tokens, meta, stats, t);
            si[ITopNz + 8] = nzDc;
            si[ILeftNz + 8] = nzDc;
            acType = 0; acFirst = 1;
        }
        for (int y = 0; y < 4; ++y)
            for (int xx = 0; xx < 4; ++xx)
            {
                int ctx = si[ITopNz + xx] + si[ILeftNz + y];
                int nzb = RecordCoeffTokens(si, IRdA + LAc + (xx + y * 4) * 16, ctx, acType, acFirst, tokens, meta, stats, t);
                si[ITopNz + xx] = nzb;
                si[ILeftNz + y] = nzb;
            }
        for (int ch = 0; ch <= 2; ch += 2)
            for (int y = 0; y < 2; ++y)
                for (int xx = 0; xx < 2; ++xx)
                {
                    int ctx = si[ITopNz + 4 + ch + xx] + si[ILeftNz + 4 + ch + y];
                    int nzb = RecordCoeffTokens(si, IRdA + LUv + (ch * 2 + xx + y * 2) * 16, ctx, 2, 0, tokens, meta, stats, t);
                    si[ITopNz + 4 + ch + xx] = nzb;
                    si[ILeftNz + 4 + ch + y] = nzb;
                }
        BytesToNz(nzRow, x, si);
    }

    /// <summary>frame_enc.c FinalizeTokenProbas: new probabilities where the saving pays for the update; sets ProbaDirty.</summary>
    public static void FinalizeTokenProbas(ArrayView<int> coeffs, ArrayView<int> stats, ArrayView<int> t, ArrayView<int> meta)
    {
        bool hasChanged = false;
        for (int i = 0; i < WebPLossySetupKernel.ProbaCount; ++i)
        {
            uint s = (uint)stats[i];
            int nb = (int)(s & 0xffff);
            int total = (int)((s >> 16) & 0xffff);
            int updateProba = t[T.CoeffsUpdateProba + i];
            int oldP = t[T.CoeffsProba0 + i];
            int newP = nb != 0 ? 255 - nb * 255 / total : 255;
            int oldCost = nb * BitCost(t, 1, oldP) + (total - nb) * BitCost(t, 0, oldP) + BitCost(t, 0, updateProba);
            int newCost = nb * BitCost(t, 1, newP) + (total - nb) * BitCost(t, 0, newP) + BitCost(t, 1, updateProba) + 8 * 256;
            bool useNew = oldCost > newCost;
            if (useNew)
            {
                coeffs[i] = newP;
                if (newP != oldP) hasChanged = true;
            }
            else coeffs[i] = oldP;
        }
        meta[M.ProbaDirty] = hasChanged ? 1 : 0;
    }
}
