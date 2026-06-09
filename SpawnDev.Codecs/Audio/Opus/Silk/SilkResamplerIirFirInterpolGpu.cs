// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable SILK 12-phase fractional FIR interpolator. Mirror of
// silk_resampler_private_IIR_FIR_INTERPOL inside libopus
// silk/resampler_private_IIR_FIR.c. Used as the FIR stage of the
// arbitrary-rate upsampler (after the 2x HQ pre-up).
//
// Per-output-sample independent: each output reads 8 buf shorts and
// 8 FracFir12 shorts (4 from row tableIdx, 4 from mirror row 11-tableIdx).
// One thread per output sample maps cleanly across all 6 ILGPU backends.
//
// All silk macros (SMULBB, SMLABB, SMULWB, RSHIFT_ROUND, SAT16) inlined here.

using ILGPU;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// GPU-callable SILK 12-phase fractional FIR interpolator. Mirror of libopus
/// <c>silk_resampler_private_IIR_FIR_INTERPOL</c>. Per-output-sample parallel.
/// </summary>
public static class SilkResamplerIirFirInterpolGpu
{
    /// <summary>
    /// Compute one output sample at position <paramref name="outIdx"/>:
    /// indexQ16 = outIdx * indexIncrementQ16; bufStart = indexQ16 &gt;&gt; 16;
    /// tableIdx = SMULWB(indexQ16 &amp; 0xFFFF, 12). Reads 8 consecutive buf
    /// shorts and 8 coefs from <paramref name="fracFir12"/> (rowLow + mirrored rowHigh).
    /// </summary>
    public static void ApplyAt(
        ArrayView<short> buf, long bufBase,
        ArrayView<short> fracFir12, long coefBase,
        ArrayView<short> output, long outBase,
        int indexIncrementQ16, int outIdx)
    {
        long indexQ16 = (long)outIdx * indexIncrementQ16;
        long bufStart = (indexQ16 >> 16) + bufBase;

        int frac16 = (int)indexQ16 & 0xFFFF;
        int tableIdx = (int)((long)frac16 * (short)12 >> 16);

        long rowLow = coefBase + 4L * tableIdx;
        long rowHigh = coefBase + 4L * (11 - tableIdx);

        int resQ15 = (int)buf[bufStart + 0] * (int)fracFir12[rowLow + 0];
        resQ15 += (int)buf[bufStart + 1] * (int)fracFir12[rowLow + 1];
        resQ15 += (int)buf[bufStart + 2] * (int)fracFir12[rowLow + 2];
        resQ15 += (int)buf[bufStart + 3] * (int)fracFir12[rowLow + 3];
        resQ15 += (int)buf[bufStart + 4] * (int)fracFir12[rowHigh + 3];
        resQ15 += (int)buf[bufStart + 5] * (int)fracFir12[rowHigh + 2];
        resQ15 += (int)buf[bufStart + 6] * (int)fracFir12[rowHigh + 1];
        resQ15 += (int)buf[bufStart + 7] * (int)fracFir12[rowHigh + 0];

        // RSHIFT_ROUND by 15 + saturate to int16.
        int rounded = ((resQ15 >> 14) + 1) >> 1;
        if (rounded > short.MaxValue) rounded = short.MaxValue;
        if (rounded < short.MinValue) rounded = short.MinValue;
        output[outBase + outIdx] = (short)rounded;
    }
}
