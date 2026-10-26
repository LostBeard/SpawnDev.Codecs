// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Kernel 3, part 4: RecordTokens and the probability refresh (FinalizeTokenProbas, shared with the final kernel). The
// per-macroblock driver is the stage kernels (WebPLossyEncodeRowKernel.Stages.cs).

using ILGPU;
using ILGPU.Runtime;
using M = SpawnDev.Codecs.Images.WebP.Lossy.WebPLossyEncMeta;
using T = SpawnDev.Codecs.Images.WebP.Lossy.WebPLossyEncTables;

namespace SpawnDev.Codecs.Images.WebP.Lossy;

internal static partial class WebPLossyEncodeRowKernel
{
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
