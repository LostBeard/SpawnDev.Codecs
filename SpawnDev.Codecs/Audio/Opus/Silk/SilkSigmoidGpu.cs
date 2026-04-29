// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// SILK fast sigmoid (logistic) approximation, GPU-callable form.
// Bit-exact mirror of SilkSigmoid.silk_sigm_Q15. Inlines the three
// 6-entry LUTs as branches to avoid uploading byte tables for each
// kernel invocation.
//
// Used by SILK VAD / speech-activity estimation paths. First piece
// of the Opus SILK GPU pipeline build-out.

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// GPU-callable SILK <c>silk_sigm_Q15</c>. Pure scalar math (no
/// ArrayView access). Bit-exact mirror of SilkSigmoid.silk_sigm_Q15.
/// </summary>
public static class SilkSigmoidGpu
{
    /// <summary>
    /// Compute sigmoid(in) in Q15 for an input in Q5. Output range
    /// [0, 32767]; clipped at +/- 6*32 to the boundary values.
    /// </summary>
    public static int SigmQ15(int inQ5)
    {
        int ind;
        if (inQ5 < 0)
        {
            inQ5 = -inQ5;
            if (inQ5 >= 6 * 32) return 0;
            ind = inQ5 >> 5;
            return SigmLutNegQ15(ind) - SigmLutSlopeQ10(ind) * (inQ5 & 0x1F);
        }
        else
        {
            if (inQ5 >= 6 * 32) return 32767;
            ind = inQ5 >> 5;
            return SigmLutPosQ15(ind) + SigmLutSlopeQ10(ind) * (inQ5 & 0x1F);
        }
    }

    private static int SigmLutSlopeQ10(int ind)
    {
        // libopus sigm_LUT_slope_Q10: { 237, 153, 73, 30, 12, 7 }.
        if (ind == 0) return 237;
        if (ind == 1) return 153;
        if (ind == 2) return 73;
        if (ind == 3) return 30;
        if (ind == 4) return 12;
        return 7;
    }

    private static int SigmLutPosQ15(int ind)
    {
        // libopus sigm_LUT_pos_Q15: { 16384, 23955, 28861, 31213, 32178, 32548 }.
        if (ind == 0) return 16384;
        if (ind == 1) return 23955;
        if (ind == 2) return 28861;
        if (ind == 3) return 31213;
        if (ind == 4) return 32178;
        return 32548;
    }

    private static int SigmLutNegQ15(int ind)
    {
        // libopus sigm_LUT_neg_Q15: { 16384, 8812, 3906, 1554, 589, 219 }.
        if (ind == 0) return 16384;
        if (ind == 1) return 8812;
        if (ind == 2) return 3906;
        if (ind == 3) return 1554;
        if (ind == 4) return 589;
        return 219;
    }
}
