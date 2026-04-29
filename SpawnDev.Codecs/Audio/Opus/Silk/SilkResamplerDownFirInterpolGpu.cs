// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable SILK polyphase interpolated FIR downsampler. Mirror of
// silk_resampler_private_down_FIR_INTERPOL inside libopus
// silk/resampler_private_down_FIR.c. Three variants by FIR order:
//   - Fir0 (order 18): polyphase 3/4 + 2/3 downsample (firFracs > 1).
//   - Fir1 (order 24): symmetric 1/2 downsample (12 taps mirrored).
//   - Fir2 (order 36): symmetric 1/3, 1/4, 1/6 downsample (18 taps mirrored).
//
// Per-output-sample independent: each output reads buf[bufStart..bufStart+order-1]
// (read-only) and writes one short. One thread per output sample is the canonical
// parallel pattern across all 6 ILGPU backends.
//
// All silk macros (SMULWB, SMLAWB, RSHIFT_ROUND, SAT16, ADD32) inlined here.

using ILGPU;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// GPU-callable SILK polyphase interpolated FIR downsampler. Mirror of libopus
/// <c>silk_resampler_private_down_FIR_INTERPOL</c>. Per-output-sample parallel.
/// </summary>
public static class SilkResamplerDownFirInterpolGpu
{
    /// <summary>
    /// Compute one output sample for the order-24 symmetric 1/2 downsampler.
    /// firCoefs holds 12 Q14 coefficients (the mirrored-pair half of the 24-tap FIR).
    /// </summary>
    public static void ApplyFir1At(
        ArrayView<int> buf, long bufBase,
        ArrayView<short> firCoefs, long coefBase,
        ArrayView<short> output, long outBase,
        int indexIncrementQ16, int outIdx)
    {
        long indexQ16 = (long)outIdx * indexIncrementQ16;
        long bufStart = (indexQ16 >> 16) + bufBase;

        int resQ6 = SmulWB(buf[bufStart + 0] + buf[bufStart + 23], firCoefs[coefBase + 0]);
        for (int k = 1; k < 12; k++)
        {
            resQ6 = SmlaWB(resQ6,
                buf[bufStart + k] + buf[bufStart + 23 - k],
                firCoefs[coefBase + k]);
        }
        output[outBase + outIdx] = Sat16(RShiftRound(resQ6, 6));
    }

    /// <summary>
    /// Compute one output sample for the order-36 symmetric 1/3, 1/4, or 1/6 downsampler.
    /// firCoefs holds 18 Q14 coefficients (the mirrored-pair half of the 36-tap FIR).
    /// </summary>
    public static void ApplyFir2At(
        ArrayView<int> buf, long bufBase,
        ArrayView<short> firCoefs, long coefBase,
        ArrayView<short> output, long outBase,
        int indexIncrementQ16, int outIdx)
    {
        long indexQ16 = (long)outIdx * indexIncrementQ16;
        long bufStart = (indexQ16 >> 16) + bufBase;

        int resQ6 = SmulWB(buf[bufStart + 0] + buf[bufStart + 35], firCoefs[coefBase + 0]);
        for (int k = 1; k < 18; k++)
        {
            resQ6 = SmlaWB(resQ6,
                buf[bufStart + k] + buf[bufStart + 35 - k],
                firCoefs[coefBase + k]);
        }
        output[outBase + outIdx] = Sat16(RShiftRound(resQ6, 6));
    }

    /// <summary>
    /// Compute one output sample for the order-18 polyphase 3/4 or 2/3 downsampler.
    /// firCoefs holds firFracs * 9 Q14 coefficients (rows of 9 polyphase taps each).
    /// firFracs is 3 (3/4 down) or 2 (2/3 down).
    /// </summary>
    public static void ApplyFir0At(
        ArrayView<int> buf, long bufBase,
        ArrayView<short> firCoefs, long coefBase,
        ArrayView<short> output, long outBase,
        int indexIncrementQ16, int firFracs, int outIdx)
    {
        long indexQ16 = (long)outIdx * indexIncrementQ16;
        long bufStart = (indexQ16 >> 16) + bufBase;

        // interpolInd = silk_SMULWB((int)indexQ16 & 0xFFFF, firFracs)
        //             = ((indexQ16 & 0xFFFF) * firFracs) >> 16
        int frac16 = (int)indexQ16 & 0xFFFF;
        int interpolInd = (int)((long)frac16 * (short)firFracs >> 16);
        long interpolStart = coefBase + 9 * (long)interpolInd;

        int resQ6 = SmulWB(buf[bufStart + 0], firCoefs[interpolStart + 0]);
        resQ6 = SmlaWB(resQ6, buf[bufStart + 1], firCoefs[interpolStart + 1]);
        resQ6 = SmlaWB(resQ6, buf[bufStart + 2], firCoefs[interpolStart + 2]);
        resQ6 = SmlaWB(resQ6, buf[bufStart + 3], firCoefs[interpolStart + 3]);
        resQ6 = SmlaWB(resQ6, buf[bufStart + 4], firCoefs[interpolStart + 4]);
        resQ6 = SmlaWB(resQ6, buf[bufStart + 5], firCoefs[interpolStart + 5]);
        resQ6 = SmlaWB(resQ6, buf[bufStart + 6], firCoefs[interpolStart + 6]);
        resQ6 = SmlaWB(resQ6, buf[bufStart + 7], firCoefs[interpolStart + 7]);
        resQ6 = SmlaWB(resQ6, buf[bufStart + 8], firCoefs[interpolStart + 8]);

        long interpolStart2 = coefBase + 9 * (long)(firFracs - 1 - interpolInd);
        resQ6 = SmlaWB(resQ6, buf[bufStart + 17], firCoefs[interpolStart2 + 0]);
        resQ6 = SmlaWB(resQ6, buf[bufStart + 16], firCoefs[interpolStart2 + 1]);
        resQ6 = SmlaWB(resQ6, buf[bufStart + 15], firCoefs[interpolStart2 + 2]);
        resQ6 = SmlaWB(resQ6, buf[bufStart + 14], firCoefs[interpolStart2 + 3]);
        resQ6 = SmlaWB(resQ6, buf[bufStart + 13], firCoefs[interpolStart2 + 4]);
        resQ6 = SmlaWB(resQ6, buf[bufStart + 12], firCoefs[interpolStart2 + 5]);
        resQ6 = SmlaWB(resQ6, buf[bufStart + 11], firCoefs[interpolStart2 + 6]);
        resQ6 = SmlaWB(resQ6, buf[bufStart + 10], firCoefs[interpolStart2 + 7]);
        resQ6 = SmlaWB(resQ6, buf[bufStart + 9], firCoefs[interpolStart2 + 8]);

        output[outBase + outIdx] = Sat16(RShiftRound(resQ6, 6));
    }

    /// <summary>silk_SMULWB(a, b) = (int)((long)a * (short)b >> 16).</summary>
    private static int SmulWB(int a32, short b16) =>
        (int)((long)a32 * b16 >> 16);

    /// <summary>silk_SMLAWB(c, a, b) = c + (int)((long)a * (short)b >> 16).</summary>
    private static int SmlaWB(int c32, int a32, short b16) =>
        c32 + (int)((long)a32 * b16 >> 16);

    /// <summary>silk_RSHIFT_ROUND for shift &gt;= 1.</summary>
    private static int RShiftRound(int a, int shift) =>
        ((a >> (shift - 1)) + 1) >> 1;

    /// <summary>silk_SAT16: saturate int to int16.</summary>
    private static short Sat16(int v)
    {
        if (v > short.MaxValue) return short.MaxValue;
        if (v < short.MinValue) return short.MinValue;
        return (short)v;
    }
}
