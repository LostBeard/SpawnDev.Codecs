// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Lossy WebP encoder, kernel 2 (ONE thread): what libwebp does between the analysis and the first macroblock of the token
// loop - analysis_enc.c AssignSegments (k-means over the 256-bin susceptibility histogram, SetSegmentAlphas) or
// ResetAllMBInfo for a single segment, then SetLoopParams: quant_enc.c VP8SetSegmentParams (quantizers from the host's
// config-only power-law table, SetupFilterStrength with the RESET sharpness 0 as on libwebp's first pass, SimplifySegments,
// SetupMatrices), frame_enc.c SetSegmentProbas, the default token probabilities (VP8DefaultProbas) and the level cost table
// (VP8CalculateLevelCosts). Statement-level port of SpawnDev.Codecs.References' Vp8LossyEncoderReference.

using ILGPU;
using ILGPU.Runtime;
using M = SpawnDev.Codecs.Images.WebP.Lossy.WebPLossyEncMeta;
using T = SpawnDev.Codecs.Images.WebP.Lossy.WebPLossyEncTables;

namespace SpawnDev.Codecs.Images.WebP.Lossy;

internal static class WebPLossySetupKernel
{
    public const int NumTypes = 4, NumBands = 8, NumCtx = 3, NumProbas = 11;
    public const int ProbaCount = NumTypes * NumBands * NumCtx * NumProbas;   // 1056
    public const int MaxVariableLevel = 67;
    public const int LevelCostStride = MaxVariableLevel + 1;                   // 68
    public const int LevelCostCount = NumTypes * NumBands * NumCtx * LevelCostStride;

    private static int Clip(int v, int m, int mx) => v < m ? m : (v > mx ? mx : v);
    private static int Abs(int v) => v < 0 ? -v : v;

    public static int BitCost(ArrayView<int> t, int bit, int proba)
        => bit == 0 ? t[T.EntropyCost + proba] : t[T.EntropyCost + 255 - proba];

    private static int SegOff(int s, int field) => M.Seg + s * M.SegStride + field;

    /// <summary>ExpandMatrix for one VP8Matrix stored as q / iq / bias / zthresh pairs at the given meta slots; returns the average q.</summary>
    private static int ExpandMatrix(ArrayView<int> meta, ArrayView<int> t, int s, int type, int q0, int q1, int qSlot, int iqSlot, int biasSlot, int ztSlot)
    {
        int sb = M.Seg + s * M.SegStride;
        meta[sb + qSlot] = q0;
        meta[sb + qSlot + 1] = q1;
        for (int i = 0; i < 2; ++i)
        {
            int q = i == 0 ? q0 : q1;
            int bias = t[T.BiasMatrices + type * 2 + i];
            int iq = (1 << 17) / q;
            int b = bias << (17 - 8);
            meta[sb + iqSlot + i] = iq;
            meta[sb + biasSlot + i] = b;
            meta[sb + ztSlot + i] = (int)(((1u << 17) - 1u - (uint)b) / (uint)iq);
        }
        int sum = q0 + 15 * q1;
        if (type == 0)
            for (int i = 0; i < 16; ++i)
            {
                int qi = i == 0 ? q0 : q1;
                meta[sb + M.SSharpY1 + i] = (t[T.FreqSharpening + i] * qi) >> 11;
            }
        return (sum + 8) >> 4;
    }

    private static int CheckLambda(int v) => v < 1 ? 1 : v;

    private static void CopySegment(ArrayView<int> meta, int dst, int src)
    {
        for (int f = 0; f < M.SegStride; ++f) meta[M.Seg + dst * M.SegStride + f] = meta[M.Seg + src * M.SegStride + f];
    }

    public static void SetupKernel(
        Index1D _, ArrayView<int> mbRecs, ArrayView<int> meta, ArrayView<int> t, ArrayView<int> qTable,
        ArrayView<int> coeffs, ArrayView<int> stats, ArrayView<int> levelCost)
    {
        var alphas = LocalMemory.Allocate<int>(256);
        var map = LocalMemory.Allocate<int>(256);
        var centers = LocalMemory.Allocate<int>(4);
        var accum = LocalMemory.Allocate<int>(4);
        var distAccum = LocalMemory.Allocate<int>(4);
        var cnt = LocalMemory.Allocate<int>(4);

        int mbW = meta[M.MbW], mbH = meta[M.MbH];
        int totalMb = mbW * mbH;
        int segmentsCfg = meta[M.SegmentsCfg];
        int sns = meta[M.SnsStrength];
        int method = meta[M.Method];

        for (int f = 0; f < 4 * M.SegStride; ++f) meta[M.Seg + f] = 0;
        int anyTransparent = 0;
        for (int n = 0; n < totalMb; ++n) anyTransparent |= mbRecs[n * M.MbStride + M.MbTransparent];
        meta[M.HasAlpha] = anyTransparent;

        // ------------------------------------------------ VP8EncAnalyze tail
        int nb = segmentsCfg < 4 ? segmentsCfg : 4;
        bool doSegments = (segmentsCfg > 1) | (method <= 1);
        if (doSegments)
        {
            for (int a = 0; a < 256; ++a) { alphas[a] = 0; map[a] = 0; }
            int sumAlpha = 0, sumUvAlpha = 0;
            for (int n = 0; n < totalMb; ++n)
            {
                int ra = mbRecs[n * M.MbStride + M.MbAlpha];
                alphas[ra] = alphas[ra] + 1;
                sumAlpha += ra;
                sumUvAlpha += mbRecs[n * M.MbStride + M.MbUVAlphaRaw];
            }
            meta[M.Alpha] = sumAlpha / totalMb;
            meta[M.UVAlpha] = sumUvAlpha / totalMb;

            // AssignSegments
            int minA = 0;
            bool scan = alphas[0] == 0;
            while (scan)
            {
                minA++;
                scan = IsEmptyBin(alphas, minA);
            }
            int maxA = 255;
            bool scan2 = (maxA > minA) & (alphas[maxA] == 0);
            while (scan2) { maxA--; scan2 = (maxA > minA) & (alphas[maxA] == 0); }
            int rangeA = maxA - minA;
            for (int k = 0; k < nb; ++k) centers[k] = minA + ((2 * k + 1) * rangeA) / (2 * nb);

            int weightedAverage = 0;
            bool iterate = true;
            for (int k = 0; k < 6; ++k)
            {
                if (iterate)
                {
                    for (int n = 0; n < nb; ++n) { accum[n] = 0; distAccum[n] = 0; }
                    int c = 0;
                    for (int a = minA; a <= maxA; ++a)
                    {
                        if (alphas[a] != 0)
                        {
                            bool move = NextCenterIsCloser(a, c, nb, centers);
                            while (move)
                            {
                                c++;
                                move = NextCenterIsCloser(a, c, nb, centers);
                            }
                            map[a] = c;
                            distAccum[c] = distAccum[c] + a * alphas[a];
                            accum[c] = accum[c] + alphas[a];
                        }
                    }
                    int displaced = 0;
                    weightedAverage = 0;
                    int totalWeight = 0;
                    for (int n = 0; n < nb; ++n)
                    {
                        if (accum[n] != 0)
                        {
                            int newCenter = (distAccum[n] + accum[n] / 2) / accum[n];
                            displaced += Abs(centers[n] - newCenter);
                            centers[n] = newCenter;
                            weightedAverage += newCenter * accum[n];
                            totalWeight += accum[n];
                        }
                    }
                    weightedAverage = (weightedAverage + totalWeight / 2) / totalWeight;
                    if (displaced < 5) iterate = false;
                }
            }
            for (int n = 0; n < totalMb; ++n)
            {
                int r = n * M.MbStride;
                int a = mbRecs[r + M.MbAlpha];
                mbRecs[r + M.MbSegment] = map[a];
                mbRecs[r + M.MbAlpha] = centers[map[a]];
            }
            // SetSegmentAlphas (preprocessing & 1 smoothing is rejected on the host)
            int mn = centers[0], mxv = centers[0];
            if (nb > 1)
                for (int n = 0; n < nb; ++n) { if (mn > centers[n]) mn = centers[n]; if (mxv < centers[n]) mxv = centers[n]; }
            if (mxv == mn) mxv = mn + 1;
            for (int n = 0; n < nb; ++n)
            {
                int alpha = 255 * (centers[n] - weightedAverage) / (mxv - mn);
                int beta = 255 * (centers[n] - mn) / (mxv - mn);
                meta[SegOff(n, M.SAlpha)] = Clip(alpha, -127, 127);
                meta[SegOff(n, M.SBeta)] = Clip(beta, 0, 255);
            }
        }
        else
        {
            // ResetAllMBInfo
            for (int n = 0; n < totalMb; ++n)
            {
                int r = n * M.MbStride;
                mbRecs[r + M.MbType] = 1;
                mbRecs[r + M.MbUVMode] = 0;
                mbRecs[r + M.MbSkip] = 0;
                mbRecs[r + M.MbSegment] = 0;
                mbRecs[r + M.MbAlpha] = 0;
            }
            meta[M.Alpha] = 0;
            meta[M.UVAlpha] = 0;
        }

        // ------------------------------------------------ VP8SetSegmentParams
        int numSegments = segmentsCfg;
        for (int i = 0; i < numSegments; ++i)
            meta[SegOff(i, M.SQuant)] = qTable[meta[SegOff(i, M.SAlpha)] + 127];
        int baseQuant = meta[SegOff(0, M.SQuant)];
        meta[M.BaseQuant] = baseQuant;
        for (int i = numSegments; i < 4; ++i) meta[SegOff(i, M.SQuant)] = baseQuant;

        int dqUvAc = (meta[M.UVAlpha] - 64) * (6 - (-4)) / (100 - 30);
        dqUvAc = dqUvAc * sns / 100;
        dqUvAc = Clip(dqUvAc, -4, 6);
        int dqUvDc = Clip(-4 * sns / 100, -15, 15);
        meta[M.DqY1Dc] = 0; meta[M.DqY2Dc] = 0; meta[M.DqY2Ac] = 0;
        meta[M.DqUVDc] = dqUvDc; meta[M.DqUVAc] = dqUvAc;

        // SetupFilterStrength (first pass: filter_hdr.sharpness is still the reset value 0)
        int level0 = 5 * meta[M.FilterStrength];
        for (int i = 0; i < 4; ++i)
        {
            int qstep = t[T.AcTable + Clip(meta[SegOff(i, M.SQuant)], 0, 127)] >> 2;
            int pos = qstep < 64 ? qstep : 63;
            int baseStrength = t[T.LevelsFromDelta + 0 * 64 + pos];
            int f = baseStrength * level0 / (256 + meta[SegOff(i, M.SBeta)]);
            meta[SegOff(i, M.SFStrength)] = f < 2 ? 0 : (f > 63 ? 63 : f);
        }
        meta[M.FilterLevel] = meta[SegOff(0, M.SFStrength)];
        meta[M.FilterSimple] = meta[M.FilterType] == 0 ? 1 : 0;
        meta[M.FilterSharpness] = meta[M.FilterSharpnessCfg];

        // SimplifySegments
        if (numSegments > 1)
        {
            var segMap = cnt;   // reuse: map[s] for s < 4
            segMap[0] = 0; segMap[1] = 1; segMap[2] = 2; segMap[3] = 3;
            int numFinal = 1;
            for (int s1 = 1; s1 < numSegments; ++s1)
            {
                bool found = false;
                int s2found = numFinal;
                for (int s2 = 0; s2 < numFinal; ++s2)
                {
                    bool same = (meta[SegOff(s1, M.SQuant)] == meta[SegOff(s2, M.SQuant)])
                              & (meta[SegOff(s1, M.SFStrength)] == meta[SegOff(s2, M.SFStrength)]);
                    bool take = same & !found;
                    if (take) { found = true; s2found = s2; }
                }
                segMap[s1] = s2found;
                if (!found)
                {
                    if (numFinal != s1) CopySegment(meta, numFinal, s1);
                    ++numFinal;
                }
            }
            if (numFinal < numSegments)
            {
                for (int n = 0; n < totalMb; ++n)
                {
                    int r = n * M.MbStride + M.MbSegment;
                    mbRecs[r] = segMap[mbRecs[r]];
                }
                for (int i = numFinal; i < numSegments; ++i) CopySegment(meta, i, numFinal - 1);
                numSegments = numFinal;
            }
        }
        meta[M.NumSegments] = numSegments;

        // SetupMatrices
        int tlambdaScale = method >= 4 ? sns : 0;
        for (int i = 0; i < numSegments; ++i)
        {
            int q = meta[SegOff(i, M.SQuant)];
            int qI4 = ExpandMatrix(meta, t, i, 0, t[T.DcTable + Clip(q, 0, 127)], t[T.AcTable + Clip(q, 0, 127)],
                M.SQY1, M.SIQY1, M.SBiasY1, M.SZtY1);
            int qI16 = ExpandMatrix(meta, t, i, 1, t[T.DcTable + Clip(q, 0, 127)] * 2, t[T.AcTable2 + Clip(q, 0, 127)],
                M.SQY2, M.SIQY2, M.SBiasY2, M.SZtY2);
            int qUV = ExpandMatrix(meta, t, i, 2, t[T.DcTable + Clip(q + dqUvDc, 0, 117)], t[T.AcTable + Clip(q + dqUvAc, 0, 127)],
                M.SQUV, M.SIQUV, M.SBiasUV, M.SZtUV);
            meta[SegOff(i, M.SLambdaI4)] = CheckLambda((3 * qI4 * qI4) >> 7);
            meta[SegOff(i, M.SLambdaI16)] = CheckLambda(3 * qI16 * qI16);
            meta[SegOff(i, M.SLambdaUV)] = CheckLambda((3 * qUV * qUV) >> 6);
            meta[SegOff(i, M.SLambdaMode)] = CheckLambda((1 * qI4 * qI4) >> 7);
            meta[SegOff(i, M.STLambda)] = CheckLambda((tlambdaScale * qI4) >> 5);
            meta[SegOff(i, M.SMinDisto)] = 20 * meta[SegOff(i, M.SQY1)];
            meta[SegOff(i, M.SMaxEdge)] = 0;
            meta[SegOff(i, M.SI4PenaltyLo)] = 1000 * qI4 * qI4;
        }

        // ------------------------------------------------ SetSegmentProbas
        for (int s = 0; s < 4; ++s) cnt[s] = 0;
        for (int n = 0; n < totalMb; ++n)
        {
            int sgm = mbRecs[n * M.MbStride + M.MbSegment];
            cnt[sgm] = cnt[sgm] + 1;
        }
        if (numSegments > 1)
        {
            int p0 = GetProba(cnt[0] + cnt[1], cnt[2] + cnt[3]);
            int p1 = GetProba(cnt[0], cnt[1]);
            int p2 = GetProba(cnt[2], cnt[3]);
            meta[M.SegProba0] = p0; meta[M.SegProba1] = p1; meta[M.SegProba2] = p2;
            bool update = (p0 != 255) | (p1 != 255) | (p2 != 255);
            meta[M.UpdateMap] = update ? 1 : 0;
            if (!update)
                for (int n = 0; n < totalMb; ++n) mbRecs[n * M.MbStride + M.MbSegment] = 0;
            meta[M.SegmentHdrSize] =
                cnt[0] * (BitCost(t, 0, p0) + BitCost(t, 0, p1)) +
                cnt[1] * (BitCost(t, 0, p0) + BitCost(t, 1, p1)) +
                cnt[2] * (BitCost(t, 1, p0) + BitCost(t, 0, p2)) +
                cnt[3] * (BitCost(t, 1, p0) + BitCost(t, 1, p2));
        }
        else
        {
            meta[M.SegProba0] = 255; meta[M.SegProba1] = 255; meta[M.SegProba2] = 255;
            meta[M.UpdateMap] = 0;
            meta[M.SegmentHdrSize] = 0;
        }

        // ------------------------------------------------ VP8DefaultProbas + ResetStats (VP8CalculateLevelCosts)
        for (int i = 0; i < ProbaCount; ++i) { coeffs[i] = t[T.CoeffsProba0 + i]; stats[i] = 0; }
        meta[M.UseSkipProba] = 0;
        meta[M.NbSkip] = 0;
        CalculateLevelCosts(coeffs, levelCost, t);
        meta[M.ProbaDirty] = 0;
    }

    /// <summary>libwebp's `n <= MAX_ALPHA && alphas[n] == 0` without reading past the histogram.</summary>
    private static bool IsEmptyBin(ArrayView<int> alphas, int n)
    {
        if (n > 255) return false;
        return alphas[n] == 0;
    }

    /// <summary>libwebp's `n + 1 < nb && abs(a - centers[n + 1]) < abs(a - centers[n])` without reading centers[nb].</summary>
    private static bool NextCenterIsCloser(int a, int c, int nb, ArrayView<int> centers)
    {
        if (c + 1 >= nb) return false;
        return Abs(a - centers[c + 1]) < Abs(a - centers[c]);
    }

    private static int GetProba(int a, int b)
    {
        int total = a + b;
        return total == 0 ? 255 : (255 * a + total / 2) / total;
    }

    /// <summary>VP8CalculateLevelCosts (always recomputes; the caller tracks `dirty`).</summary>
    public static void CalculateLevelCosts(ArrayView<int> coeffs, ArrayView<int> levelCost, ArrayView<int> t)
    {
        for (int ctype = 0; ctype < NumTypes; ++ctype)
            for (int band = 0; band < NumBands; ++band)
                for (int ctx = 0; ctx < NumCtx; ++ctx)
                {
                    int p = NumProbas * (ctx + NumCtx * (band + NumBands * ctype));
                    int table = LevelCostStride * (ctx + NumCtx * (band + NumBands * ctype));
                    int cost0 = ctx > 0 ? BitCost(t, 1, coeffs[p + 0]) : 0;
                    int costBase = BitCost(t, 1, coeffs[p + 1]) + cost0;
                    levelCost[table + 0] = BitCost(t, 0, coeffs[p + 1]) + cost0;
                    for (int v = 1; v <= MaxVariableLevel; ++v)
                    {
                        int pattern = t[T.LevelCodes + (v - 1) * 2 + 0];
                        int bits = t[T.LevelCodes + (v - 1) * 2 + 1];
                        int cost = 0;
                        int i = 2;
                        bool go = pattern != 0;
                        while (go)
                        {
                            if ((pattern & 1) != 0) cost += BitCost(t, bits & 1, coeffs[p + i]);
                            bits >>= 1;
                            pattern >>= 1;
                            ++i;
                            go = pattern != 0;
                        }
                        levelCost[table + v] = costBase + cost;
                    }
                }
    }
}
