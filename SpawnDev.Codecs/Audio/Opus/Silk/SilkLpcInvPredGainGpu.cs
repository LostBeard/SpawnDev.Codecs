// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable SILK LPC inverse prediction gain. Mirror of
// SilkLpcInvPredGain.Compute (libopus silk/LPC_inv_pred_gain.c).
// Verifies LPC filter stability + computes gain in Q30; returns 0
// when unstable / gain > MAX_PREDICTION_POWER_GAIN.
//
// Single-thread dispatch: the reverse-Levinson recursion is
// sequential. Caller provides an int[order] scratch buffer.
//
// All silk macros (LSHIFT, RSHIFT, SMMUL, SMULL, SMULWB, SUB32,
// SUB_SAT32, MUL32_FRAC_Q, RSHIFT_ROUND64, abs, CLZ32) inlined here.
// silk_INVERSE32_varQ delegates to SilkInverseQ32Gpu.Compute.

using ILGPU;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// GPU-callable SILK LPC inverse prediction gain. Mirror of
/// <see cref="SilkLpcInvPredGain"/>.Compute.
/// </summary>
public static class SilkLpcInvPredGainGpu
{
    private const int QA = 24;
    private const int A_LIMIT = 16773022;
    private const int INV_GAIN_Q30_MIN = 107374;

    /// <summary>
    /// Compute the inverse LPC prediction gain in Q30 from Q12
    /// coefficients. Returns 0 if the filter is unstable or gain too
    /// large.
    /// </summary>
    /// <param name="aQ12">Input LPC coefficients in Q12.</param>
    /// <param name="aQ12Base">Base offset.</param>
    /// <param name="order">LPC prediction order.</param>
    /// <param name="aQAScratch">Per-call int scratch (length order).
    /// Contents replaced.</param>
    /// <param name="scratchBase">Base offset.</param>
    /// <returns>Inverse prediction gain in Q30, or 0 if unstable.</returns>
    public static int Compute(
        ArrayView<short> aQ12, long aQ12Base, int order,
        ArrayView<int> aQAScratch, long scratchBase)
    {
        int dcResp = 0;
        for (int k = 0; k < order; k++)
        {
            int v = aQ12[aQ12Base + k];
            dcResp += v;
            aQAScratch[scratchBase + k] = v << (QA - 12);
        }
        if (dcResp >= 4096) return 0;
        return LpcInversePredGainQA(aQAScratch, scratchBase, order);
    }

    private static int LpcInversePredGainQA(
        ArrayView<int> aQA, long aBase, int order)
    {
        int invGainQ30 = 1 << 30;

        int k;
        for (k = order - 1; k > 0; k--)
        {
            int aK = aQA[aBase + k];
            if (aK > A_LIMIT || aK < -A_LIMIT) return 0;

            int rcQ31 = -(aK << (31 - QA));

            // SMMUL(rcQ31, rcQ31) = ((long)rcQ31 * rcQ31) >> 32
            int rcMult1Q30 = (1 << 30) - (int)(((long)rcQ31 * rcQ31) >> 32);

            invGainQ30 = ((int)(((long)invGainQ30 * rcMult1Q30) >> 32)) << 2;
            if (invGainQ30 < INV_GAIN_Q30_MIN) return 0;

            int absRc = rcMult1Q30 < 0 ? -rcMult1Q30 : rcMult1Q30;
            int mult2Q = 32 - Clz32(absRc);
            int rcMult2 = SilkInverseQ32Gpu.Compute(rcMult1Q30, mult2Q + 30);

            int halfK = (k + 1) >> 1;
            for (int n = 0; n < halfK; n++)
            {
                int tmp1 = aQA[aBase + n];
                int tmp2 = aQA[aBase + k - n - 1];

                // silk_MUL32_FRAC_Q(a, b, Q) = (int)RSHIFT_ROUND64(SMULL(a, b), Q)
                int frac1 = MulFracQ(tmp2, rcQ31, 31);
                int sub1 = SubSat32(tmp1, frac1);
                long mul1 = (long)sub1 * rcMult2;
                long round1 = RShiftRound64(mul1, mult2Q);
                if (round1 > int.MaxValue || round1 < int.MinValue) return 0;
                aQA[aBase + n] = (int)round1;

                int frac2 = MulFracQ(tmp1, rcQ31, 31);
                int sub2 = SubSat32(tmp2, frac2);
                long mul2 = (long)sub2 * rcMult2;
                long round2 = RShiftRound64(mul2, mult2Q);
                if (round2 > int.MaxValue || round2 < int.MinValue) return 0;
                aQA[aBase + k - n - 1] = (int)round2;
            }
        }

        int a0 = aQA[aBase + 0];
        if (a0 > A_LIMIT || a0 < -A_LIMIT) return 0;

        int rcQ31Last = -(a0 << (31 - QA));
        int rcMult1Q30Last = (1 << 30) - (int)(((long)rcQ31Last * rcQ31Last) >> 32);
        invGainQ30 = ((int)(((long)invGainQ30 * rcMult1Q30Last) >> 32)) << 2;
        if (invGainQ30 < INV_GAIN_Q30_MIN) return 0;

        return invGainQ30;
    }

    /// <summary>silk_MUL32_FRAC_Q.</summary>
    private static int MulFracQ(int a32, int b32, int Q)
    {
        long product = (long)a32 * b32;
        return (int)RShiftRound64(product, Q);
    }

    /// <summary>silk_RSHIFT_ROUND64.</summary>
    private static long RShiftRound64(long a, int shift)
    {
        if (shift <= 0) return a;
        // ((a >> (shift-1)) + 1) >> 1 - rounded right shift.
        return ((a >> (shift - 1)) + 1) >> 1;
    }

    /// <summary>silk_SUB_SAT32.</summary>
    private static int SubSat32(int a, int b)
    {
        long diff = (long)a - b;
        if (diff > int.MaxValue) return int.MaxValue;
        if (diff < int.MinValue) return int.MinValue;
        return (int)diff;
    }

    /// <summary>silk_CLZ32.</summary>
    private static int Clz32(int x)
    {
        if (x == 0) return 32;
        uint u = (uint)x;
        int n = 0;
        if ((u & 0xFFFF0000u) == 0) { n += 16; u <<= 16; }
        if ((u & 0xFF000000u) == 0) { n += 8; u <<= 8; }
        if ((u & 0xF0000000u) == 0) { n += 4; u <<= 4; }
        if ((u & 0xC0000000u) == 0) { n += 2; u <<= 2; }
        if ((u & 0x80000000u) == 0) { n += 1; }
        return n;
    }
}
