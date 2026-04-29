// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable SILK stereo predictor dequantizer. Mirror of the
// dequantization helper inside SilkStereoDecodePred.DecodePred (libopus
// silk/stereo_decode_pred.c). Takes the entropy-decoded index triples
// (ix0, ix1) and produces the 2 Q13 predictor values fed to
// SilkStereoMsToLrGpu.
//
// The entropy decode runs in OpusRangeCoderGpu (already shipped). This
// primitive picks up where that left off: per-predictor mathematical
// dequant + libopus pre-subtraction.
//
// Single-thread per stream because there are only 2 predictors per
// frame and the second one is subtracted from the first as a final
// fixup.
//
// All silk macros (SMULWB, SMLABB) inlined here.

using ILGPU;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// GPU-callable SILK stereo predictor dequantizer. Mirror of the
/// dequantization helper inside <see cref="SilkStereoDecodePred"/>.
/// </summary>
public static class SilkStereoDecodePredGpu
{
    /// <summary>SILK_FIX_CONST(0.5 / 5, 16) = (int)(0.1 * 65536 + 0.5) = 6554.</summary>
    private const int HALF_OVER_SUB_STEPS_Q16 = 6554;

    /// <summary>
    /// Dequantize the 6 entropy-decoded indices into 2 Q13 predictors.
    /// Bit-exact vs SilkStereoDecodePred.DequantizePredictors. The
    /// stereoPredQuantQ13 input must be the libopus 16-element table
    /// (caller copies SilkStereoDecodePred.StereoPredQuantQ13 to GPU).
    /// </summary>
    /// <param name="predQ13">Output: 2 Q13 predictor values at predBase + 0/1.</param>
    /// <param name="predBase">Base offset.</param>
    /// <param name="stereoPredQuantQ13">Q13 quantization table (length 16).</param>
    /// <param name="tabBase">Base offset.</param>
    /// <param name="ix0_0">First predictor's low3 index.</param>
    /// <param name="ix0_1">First predictor's mid5 index.</param>
    /// <param name="ix0_2">First predictor's high5 index.</param>
    /// <param name="ix1_0">Second predictor's low3 index.</param>
    /// <param name="ix1_1">Second predictor's mid5 index.</param>
    /// <param name="ix1_2">Second predictor's high5 index.</param>
    public static void ApplyAt(
        ArrayView<int> predQ13, long predBase,
        ArrayView<short> stereoPredQuantQ13, long tabBase,
        int ix0_0, int ix0_1, int ix0_2,
        int ix1_0, int ix1_1, int ix1_2)
    {
        // First predictor.
        int idx0 = ix0_0 + 3 * ix0_2;
        int low0 = stereoPredQuantQ13[tabBase + idx0];
        int delta0 = stereoPredQuantQ13[tabBase + idx0 + 1] - low0;
        int step0 = SmulWB(delta0, HALF_OVER_SUB_STEPS_Q16);
        int pred0 = SmlaBB(low0, step0, 2 * ix0_1 + 1);

        // Second predictor.
        int idx1 = ix1_0 + 3 * ix1_2;
        int low1 = stereoPredQuantQ13[tabBase + idx1];
        int delta1 = stereoPredQuantQ13[tabBase + idx1 + 1] - low1;
        int step1 = SmulWB(delta1, HALF_OVER_SUB_STEPS_Q16);
        int pred1 = SmlaBB(low1, step1, 2 * ix1_1 + 1);

        // Pre-subtract second predictor (libopus optimisation).
        predQ13[predBase + 0] = pred0 - pred1;
        predQ13[predBase + 1] = pred1;
    }

    /// <summary>silk_SMULWB(a, b) = (int)((long)a * (short)b >> 16).</summary>
    private static int SmulWB(int a32, int b32) =>
        (int)((long)a32 * (short)b32 >> 16);

    /// <summary>silk_SMLABB(c, a, b) = c + (short)a * (short)b.</summary>
    private static int SmlaBB(int c32, int a32, int b32) =>
        c32 + (int)(short)a32 * (int)(short)b32;
}
