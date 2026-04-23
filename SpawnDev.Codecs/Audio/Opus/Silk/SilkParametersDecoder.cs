// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Structural port of libopus silk/decode_parameters.c to clean C#. Given a
// decoded SilkDecodedIndices, dequantizes every per-frame quantity needed by
// decode_core: gains, NLSFs (plus inter-frame interpolation), LPC coefficients
// per half-frame, pitch lags per subframe, LTP filter taps per subframe, and
// the LTP scale factor. First-frame-after-reset and post-loss BWE handling are
// left for the top-level silk_decode_frame orchestrator to manage by toggling
// the NLSF interpolation factor and applying bandwidth expansion to the LPC
// coefficients after this returns.
//
// Upstream Copyright (c) 2006-2011 Skype Limited. BSD 3-Clause. See NOTICE.md.

using static SpawnDev.Codecs.Audio.Opus.Silk.SilkMacros;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// Per-frame SILK parameter dequantizer. Composes every sub-decoder ported in
/// earlier slices into a single <see cref="Decode"/> call that turns a
/// <see cref="SilkDecodedIndices"/> into a <see cref="SilkDecodedParameters"/>.
/// </summary>
internal static class SilkParametersDecoder
{
    /// <summary>
    /// LTP scale factors in Q14, indexed by <c>indices.LtpScaleIndex</c>.
    /// Matches libopus <c>silk_LTPScales_table_Q14 = { 15565, 12288, 8192 }</c>.
    /// </summary>
    private static readonly short[] LtpScalesQ14 = { 15565, 12288, 8192 };

    /// <summary>
    /// Dequantize all parameters for a single SILK frame.
    /// </summary>
    /// <param name="output">Destination for dequantized values. Every relevant field is populated.</param>
    /// <param name="indices">Side-information indices previously decoded via <see cref="SilkIndicesDecoder"/>.</param>
    /// <param name="codebook">NLSF codebook (NB/MB or WB) matching the frame's sample rate.</param>
    /// <param name="fsKHz">Internal SILK sample rate (8, 12, or 16).</param>
    /// <param name="nbSubfr">Subframe count (2 or 4).</param>
    /// <param name="lastGainIndex">In/out: the previous frame's trailing gain index (required for
    /// conditional / delta gain coding). Updated to the current frame's last index.</param>
    /// <param name="prevNlsfQ15">In/out: previous frame's NLSFs in Q15, used for interpolation in the
    /// first half of the frame. Updated in place to the current frame's NLSFs on exit. Length
    /// &gt;= <c>codebook.Order</c>.</param>
    /// <param name="conditional">0 for independent coding, non-zero for conditional (delta) gain coding.</param>
    internal static void Decode(
        SilkDecodedParameters output,
        SilkDecodedIndices indices,
        SilkNlsfCodebook codebook,
        int fsKHz,
        int nbSubfr,
        ref sbyte lastGainIndex,
        Span<short> prevNlsfQ15,
        int conditional)
    {
        if (output is null) throw new ArgumentNullException(nameof(output));
        if (indices is null) throw new ArgumentNullException(nameof(indices));
        if (codebook is null) throw new ArgumentNullException(nameof(codebook));
        if (fsKHz != 8 && fsKHz != 12 && fsKHz != 16)
            throw new ArgumentException($"Unsupported fs_kHz: {fsKHz}.", nameof(fsKHz));
        if (nbSubfr != 2 && nbSubfr != 4)
            throw new ArgumentException($"nbSubfr must be 2 or 4, got {nbSubfr}.", nameof(nbSubfr));
        int order = codebook.Order;
        if (prevNlsfQ15.Length < order)
            throw new ArgumentException($"prevNlsfQ15 too small (need {order}).", nameof(prevNlsfQ15));

        // 1. Dequantize gains.
        SilkGainDecoder.Dequantize(
            output.GainsQ16.AsSpan(0, nbSubfr),
            indices.GainsIndices.AsSpan(0, nbSubfr),
            ref lastGainIndex,
            conditional: conditional == 0 ? 0 : 1,
            nbSubfr: nbSubfr);

        // 2. Decode NLSF vector.
        Span<short> nlsfQ15 = output.NlsfQ15.AsSpan(0, order);
        SilkNlsfDecoder.Decode(nlsfQ15, indices.NlsfIndices.AsSpan(0, order + 1), codebook);

        // 3. NLSF -> LPC for the SECOND half of the frame.
        Span<short> lpcHalf2 = output.PredCoefQ12.AsSpan(SilkConstants.MAX_LPC_ORDER, order);
        SilkNlsf2A.Compute(lpcHalf2, nlsfQ15, order);

        // 4. First-half LPC: interpolate from prev and current NLSF if interp coef < 4,
        //    otherwise copy the second half.
        Span<short> lpcHalf1 = output.PredCoefQ12.AsSpan(0, order);
        if (indices.NlsfInterpCoefQ2 < 4)
        {
            Span<short> nlsf0Q15 = stackalloc short[SilkConstants.MAX_LPC_ORDER];
            for (int i = 0; i < order; i++)
            {
                int delta = nlsfQ15[i] - prevNlsfQ15[i];
                nlsf0Q15[i] = (short)(prevNlsfQ15[i] + silk_RSHIFT(silk_MUL(indices.NlsfInterpCoefQ2, delta), 2));
            }
            SilkNlsf2A.Compute(lpcHalf1, nlsf0Q15.Slice(0, order), order);
        }
        else
        {
            lpcHalf2.CopyTo(lpcHalf1);
        }

        // 5. Update prev NLSFs for the next frame.
        nlsfQ15.CopyTo(prevNlsfQ15);

        // 6. Voiced-only: pitch + LTP.
        if (indices.SignalType == SilkSideInfoDecoder.TypeVoiced)
        {
            SilkPitchDecoder.ComputeLags(
                output.PitchL.AsSpan(0, nbSubfr),
                indices.LagIndex, indices.ContourIndex, fsKHz, nbSubfr);

            sbyte[] cb = SilkLtpGainTables.Select(indices.PerIndex);
            int vecSize = SilkLtpGainTables.LtpVecSize;
            for (int k = 0; k < nbSubfr; k++)
            {
                int ltpIdx = indices.LtpIndices[k];
                if ((uint)ltpIdx >= (uint)(cb.Length / vecSize))
                    throw new ArgumentOutOfRangeException(
                        nameof(indices), $"indices.LtpIndices[{k}] = {ltpIdx} is out of range.");
                int src = ltpIdx * vecSize;
                int dst = k * vecSize;
                for (int i = 0; i < vecSize; i++)
                {
                    output.LtpCoefQ14[dst + i] = (short)silk_LSHIFT((int)cb[src + i], 7);
                }
            }

            int scaleIdx = indices.LtpScaleIndex;
            if ((uint)scaleIdx >= (uint)LtpScalesQ14.Length)
                throw new ArgumentOutOfRangeException(
                    nameof(indices), $"indices.LtpScaleIndex = {scaleIdx} is out of range.");
            output.LtpScaleQ14 = LtpScalesQ14[scaleIdx];
        }
        else
        {
            for (int k = 0; k < nbSubfr; k++) output.PitchL[k] = 0;
            for (int i = 0; i < nbSubfr * SilkLtpGainTables.LtpVecSize; i++) output.LtpCoefQ14[i] = 0;
            output.LtpScaleQ14 = 0;
        }
    }
}
