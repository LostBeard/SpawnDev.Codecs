// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable SILK NLSF residual dequantizer. Mirror of
// SilkNlsfDecoder.ResidualDequant (libopus silk/NLSF_residual_dequant.c).
// Reverse-iterates the NLSF index stream, applying a backward LPC
// predictor + per-stage quantizer step. Output is a Q10 residual stream
// fed to the codebook delta + weighted-add stage.
//
// Sequential per-stream: outQ10 carries forward through iterations
// (each step's outQ10 becomes the next predQ10 input). One-thread-per-
// stream on the GPU. Multiple independent SILK streams parallelize
// across threads.
//
// All silk macros (RSHIFT, SMULBB, LSHIFT, SMLAWB) inlined here.

using ILGPU;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// GPU-callable SILK NLSF residual dequantizer. Mirror of
/// <see cref="SilkNlsfDecoder"/>.ResidualDequant.
/// </summary>
public static class SilkNlsfResidualDequantGpu
{
    private const int NLSF_QUANT_LEVEL_ADJ_Q10 = 102;

    /// <summary>
    /// Reverse-iterate <paramref name="order"/> indices into Q10 residuals.
    /// Bit-exact vs the CPU SilkNlsfDecoder.ResidualDequant.
    /// </summary>
    /// <param name="xQ10">Output residuals in Q10 (length order).</param>
    /// <param name="xBase">Base offset.</param>
    /// <param name="indices">Quantization indices (signed sbyte, length order).</param>
    /// <param name="indicesBase">Base offset.</param>
    /// <param name="predCoefQ8">Backward predictor coefficients in Q8 (length order).</param>
    /// <param name="predBase">Base offset.</param>
    /// <param name="quantStepSizeQ16">Per-codebook quantizer step size in Q16.</param>
    /// <param name="order">Number of input values (NLSF filter order).</param>
    public static void DequantizeAt(
        ArrayView<short> xQ10, long xBase,
        ArrayView<sbyte> indices, long indicesBase,
        ArrayView<byte> predCoefQ8, long predBase,
        int quantStepSizeQ16, int order)
    {
        int outQ10 = 0;
        for (int i = order - 1; i >= 0; i--)
        {
            // predQ10 = (outQ10 * (short)predCoefQ8[i]) >> 8
            // predCoefQ8 is byte (0..255); cast to short keeps it positive (libopus does
            // silk_SMULBB which is a 16x16->32 mul of two shorts).
            short predCoefShort = predCoefQ8[predBase + i];
            int predQ10 = (outQ10 * predCoefShort) >> 8;

            // outQ10 = indices[i] << 10
            outQ10 = (int)indices[indicesBase + i] << 10;

            if (outQ10 > 0) outQ10 -= NLSF_QUANT_LEVEL_ADJ_Q10;
            else if (outQ10 < 0) outQ10 += NLSF_QUANT_LEVEL_ADJ_Q10;

            // SMLAWB(predQ10, outQ10, quantStepSizeQ16)
            //   = predQ10 + ((long)outQ10 * (short)quantStepSizeQ16 >> 16)
            outQ10 = predQ10 + (int)((long)outQ10 * (short)quantStepSizeQ16 >> 16);

            xQ10[xBase + i] = (short)outQ10;
        }
    }
}
