// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Structural port of libopus silk/NLSF_decode.c to clean C#. Contains the
// residual dequantizer helper and the top-level NLSF vector decoder that
// wires together NLSF_unpack + residual_dequant + stabilize.
//
// Upstream Copyright (c) 2006-2011 Skype Limited. BSD 3-Clause. See NOTICE.md.

using static SpawnDev.Codecs.Audio.Opus.Silk.SilkMacros;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// SILK NLSF vector decoder. Decodes a full NLSF (Normalized Line Spectral Frequency)
/// vector from the per-frame quantization indices and the decoder's NLSF codebook.
/// </summary>
internal static class SilkNlsfDecoder
{
    /// <summary>
    /// Predictive dequantizer for NLSF residuals. Walks the index array in reverse,
    /// running a scalar predictor with the per-coefficient predictor coefficients.
    /// Bit-exact port of libopus <c>silk_NLSF_residual_dequant</c>.
    /// </summary>
    /// <param name="xQ10">Output: dequantized residuals in Q10. Length = <paramref name="order"/>.</param>
    /// <param name="indices">Quantization indices (signed).</param>
    /// <param name="predCoefQ8">Backward predictor coefficients in Q8.</param>
    /// <param name="quantStepSizeQ16">Per-codebook quantizer step size in Q16.</param>
    /// <param name="order">Number of input values.</param>
    internal static void ResidualDequant(
        Span<short> xQ10,
        ReadOnlySpan<sbyte> indices,
        ReadOnlySpan<byte> predCoefQ8,
        int quantStepSizeQ16,
        int order)
    {
        int outQ10 = 0;
        for (int i = order - 1; i >= 0; i--)
        {
            int predQ10 = silk_RSHIFT(silk_SMULBB(outQ10, (short)predCoefQ8[i]), 8);
            outQ10 = silk_LSHIFT(indices[i], 10);
            if (outQ10 > 0)
            {
                outQ10 -= SilkConstants.NLSF_QUANT_LEVEL_ADJ_Q10;
            }
            else if (outQ10 < 0)
            {
                outQ10 += SilkConstants.NLSF_QUANT_LEVEL_ADJ_Q10;
            }
            outQ10 = silk_SMLAWB(predQ10, outQ10, quantStepSizeQ16);
            xQ10[i] = (short)outQ10;
        }
    }

    /// <summary>
    /// Decode a full NLSF vector from its codebook path (first-stage index in <c>nlsfIndices[0]</c>
    /// plus per-coefficient residual indices in <c>nlsfIndices[1..order]</c>). Output is in Q15
    /// and is guaranteed monotonically ordered with minimum delta spacing per the codebook.
    /// </summary>
    /// <param name="pNlsfQ15">Output: quantized NLSF vector in Q15. Length = codebook.Order.</param>
    /// <param name="nlsfIndices">Codebook path vector. Length = codebook.Order + 1. Index 0 is
    /// the first-stage codebook index; indices 1..order are the per-coefficient signed residuals.</param>
    /// <param name="codebook">NLSF codebook.</param>
    internal static void Decode(
        Span<short> pNlsfQ15,
        ReadOnlySpan<sbyte> nlsfIndices,
        SilkNlsfCodebook codebook)
    {
        if (codebook is null) throw new ArgumentNullException(nameof(codebook));
        int order = codebook.Order;
        if (pNlsfQ15.Length < order) throw new ArgumentException($"pNlsfQ15 too small (need {order}).", nameof(pNlsfQ15));
        if (nlsfIndices.Length < order + 1) throw new ArgumentException($"nlsfIndices too small (need {order + 1}).", nameof(nlsfIndices));

        // Stack-allocated temporaries sized to MAX_LPC_ORDER.
        Span<byte> predQ8 = stackalloc byte[SilkConstants.MAX_LPC_ORDER];
        Span<short> ecIx = stackalloc short[SilkConstants.MAX_LPC_ORDER];
        Span<short> resQ10 = stackalloc short[SilkConstants.MAX_LPC_ORDER];

        // Unpack entropy table indices and predictor for current first-stage index.
        int cb1Index = nlsfIndices[0];
        SilkNlsfUnpack.Unpack(ecIx, predQ8, codebook, cb1Index);

        // Predictive residual dequantizer on indices[1..].
        ResidualDequant(resQ10, nlsfIndices.Slice(1), predQ8, codebook.QuantStepSizeQ16, (short)order);

        // Apply inverse square-rooted weights to first stage and add residuals.
        int cbBase = cb1Index * order;
        byte[] cb1 = codebook.Cb1NlsfQ8;
        short[] cbWght = codebook.Cb1WghtQ9;
        for (int i = 0; i < order; i++)
        {
            int residual = silk_LSHIFT(resQ10[i], 14);
            int weightedResidual = residual / cbWght[cbBase + i];                 // silk_DIV32_16
            int nlsfQ15Tmp = silk_ADD_LSHIFT32(weightedResidual, (short)cb1[cbBase + i], 7);
            pNlsfQ15[i] = (short)silk_LIMIT_32(nlsfQ15Tmp, 0, 32767);
        }

        // Stabilize: enforce ordering + minimum spacing.
        SilkNlsfStabilize.Stabilize(pNlsfQ15.Slice(0, order), codebook.DeltaMinQ15);
    }
}
