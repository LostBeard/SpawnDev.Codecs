// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable SILK LPC synthesis filter inner loop. Mirror of
// SilkLpcSynthesisFilter.Apply (libopus silk/decode_core.c synthesis path).
// Inverse of the LPC analysis filter; reconstructs PCM by adding the LPC
// prediction to the residual + gain-scaling to int16.
//
// Sequential per-stream: stateQ14[maxOrder + i] depends on prior outputs
// stateQ14[maxOrder + i - 1..i - order]. One-thread-per-stream on the GPU
// (multiple independent streams parallelize across threads).
//
// All silk macros (RSHIFT, SMLAWB, ADD_SAT32, LSHIFT_SAT32, SMULWW,
// SMULWB, RSHIFT_ROUND, SAT16) inlined here.

using ILGPU;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// GPU-callable SILK LPC synthesis filter inner loop. Mirror of
/// <see cref="SilkLpcSynthesisFilter"/>.Apply.
/// </summary>
public static class SilkLpcSynthesisFilterGpu
{
    private const int MAX_LPC_ORDER = 16;

    /// <summary>
    /// Apply the LPC synthesis filter for <paramref name="subfrLen"/> samples,
    /// updating the in-place state buffer and writing scaled int16 PCM to
    /// <paramref name="pcmOut"/>. Bit-exact vs the CPU SilkLpcSynthesisFilter.Apply.
    /// </summary>
    /// <param name="stateQ14">State buffer in Q14. Length &gt;= MAX_LPC_ORDER + subfrLen.
    /// History at [0..MAX_LPC_ORDER); new samples written at [MAX_LPC_ORDER..MAX_LPC_ORDER+subfrLen).</param>
    /// <param name="presQ14">Residual signal in Q14. Length &gt;= subfrLen.</param>
    /// <param name="aQ12">LPC coefficients in Q12. Length &gt;= order.</param>
    /// <param name="gainQ10">Gain in Q10.</param>
    /// <param name="order">LPC order (10 or 16).</param>
    /// <param name="subfrLen">Subframe length in samples.</param>
    /// <param name="pcmOut">Output PCM (int16). Length &gt;= subfrLen.</param>
    public static void ApplyAt(
        ArrayView<int> stateQ14, long stateBase,
        ArrayView<int> presQ14, long presBase,
        ArrayView<short> aQ12, long aBase,
        int gainQ10, int order, int subfrLen,
        ArrayView<short> pcmOut, long pcmBase)
    {
        int maxOrder = MAX_LPC_ORDER;
        int orderHalf = order >> 1;

        for (int i = 0; i < subfrLen; i++)
        {
            long sIdx = stateBase + maxOrder + i;

            // LPC_pred_Q10 = order/2 (rounding bias) + sum SMLAWB.
            int lpcPredQ10 = orderHalf;
            lpcPredQ10 = SmlaWB(lpcPredQ10, stateQ14[sIdx - 1], aQ12[aBase + 0]);
            lpcPredQ10 = SmlaWB(lpcPredQ10, stateQ14[sIdx - 2], aQ12[aBase + 1]);
            lpcPredQ10 = SmlaWB(lpcPredQ10, stateQ14[sIdx - 3], aQ12[aBase + 2]);
            lpcPredQ10 = SmlaWB(lpcPredQ10, stateQ14[sIdx - 4], aQ12[aBase + 3]);
            lpcPredQ10 = SmlaWB(lpcPredQ10, stateQ14[sIdx - 5], aQ12[aBase + 4]);
            lpcPredQ10 = SmlaWB(lpcPredQ10, stateQ14[sIdx - 6], aQ12[aBase + 5]);
            lpcPredQ10 = SmlaWB(lpcPredQ10, stateQ14[sIdx - 7], aQ12[aBase + 6]);
            lpcPredQ10 = SmlaWB(lpcPredQ10, stateQ14[sIdx - 8], aQ12[aBase + 7]);
            lpcPredQ10 = SmlaWB(lpcPredQ10, stateQ14[sIdx - 9], aQ12[aBase + 8]);
            lpcPredQ10 = SmlaWB(lpcPredQ10, stateQ14[sIdx - 10], aQ12[aBase + 9]);
            if (order == 16)
            {
                lpcPredQ10 = SmlaWB(lpcPredQ10, stateQ14[sIdx - 11], aQ12[aBase + 10]);
                lpcPredQ10 = SmlaWB(lpcPredQ10, stateQ14[sIdx - 12], aQ12[aBase + 11]);
                lpcPredQ10 = SmlaWB(lpcPredQ10, stateQ14[sIdx - 13], aQ12[aBase + 12]);
                lpcPredQ10 = SmlaWB(lpcPredQ10, stateQ14[sIdx - 14], aQ12[aBase + 13]);
                lpcPredQ10 = SmlaWB(lpcPredQ10, stateQ14[sIdx - 15], aQ12[aBase + 14]);
                lpcPredQ10 = SmlaWB(lpcPredQ10, stateQ14[sIdx - 16], aQ12[aBase + 15]);
            }

            // state[i] = ADD_SAT32(presQ14[i], LSHIFT_SAT32(lpcPredQ10, 4)).
            int newSample = AddSat32(presQ14[presBase + i], LShiftSat32(lpcPredQ10, 4));
            stateQ14[sIdx] = newSample;

            // pcmOut[i] = SAT16(RSHIFT_ROUND(SMULWW(newSample, gainQ10), 8)).
            int scaled = SmulWW(newSample, gainQ10);
            int rounded = (scaled + (1 << 7)) >> 8;
            if (rounded > short.MaxValue) rounded = short.MaxValue;
            else if (rounded < short.MinValue) rounded = short.MinValue;
            pcmOut[pcmBase + i] = (short)rounded;
        }
    }

    /// <summary>silk_SMLAWB(c, a, b) = c + (int)((long)a * (short)b >> 16).</summary>
    private static int SmlaWB(int c32, int a32, short b16) =>
        c32 + (int)((long)a32 * b16 >> 16);

    /// <summary>silk_SMULWW(a, b) = SMULWB(a, b) + a * RSHIFT_ROUND(b, 16).</summary>
    private static int SmulWW(int a32, int b32)
    {
        int smulwb = (int)((long)a32 * (short)b32 >> 16);
        int rshiftRound = (b32 + (1 << 15)) >> 16;
        return smulwb + a32 * rshiftRound;
    }

    /// <summary>silk_ADD_SAT32: saturating int32 add.</summary>
    private static int AddSat32(int a, int b)
    {
        long sum = (long)a + b;
        if (sum > int.MaxValue) return int.MaxValue;
        if (sum < int.MinValue) return int.MinValue;
        return (int)sum;
    }

    /// <summary>silk_LSHIFT_SAT32: saturating left shift.</summary>
    private static int LShiftSat32(int a, int shift)
    {
        if (shift <= 0) return a;
        if (shift >= 32) return a == 0 ? 0 : (a > 0 ? int.MaxValue : int.MinValue);
        int max = int.MaxValue >> shift;
        int min = int.MinValue >> shift;
        if (a > max) return int.MaxValue;
        if (a < min) return int.MinValue;
        return a << shift;
    }
}
