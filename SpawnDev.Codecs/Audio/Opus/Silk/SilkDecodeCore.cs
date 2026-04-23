// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Structural port of libopus silk/decode_core.c to clean C#. Given a decoded
// set of SILK parameters and the per-frame pulse sequence, drives the full
// excitation -> LTP state management -> LPC synthesis -> PCM output pipeline,
// updating the persistent channel-decoder state in place.
//
// Upstream Copyright (c) 2006-2011 Skype Limited. BSD 3-Clause. See NOTICE.md.

using static SpawnDev.Codecs.Audio.Opus.Silk.SilkMacros;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// Top-level SILK decode_core. Consumes a dequantized SilkDecodedParameters
/// plus the raw frame pulses and produces int16 PCM samples at the internal
/// SILK sample rate. Packet-loss concealment (PLC) is intentionally out of
/// scope for this port and is expected to be handled by the higher-level
/// silk_decode_frame orchestrator.
/// </summary>
internal static class SilkDecodeCore
{
    /// <summary>LTP filter order. Libopus <c>LTP_ORDER = 5</c>.</summary>
    private const int LtpOrder = 5;

    /// <summary>
    /// Run the SILK synthesis chain for one frame. Writes <paramref name="xqOut"/>
    /// frame_length samples and updates the persistent state buffers
    /// (<see cref="SilkChannelDecoderState.PrevGainQ16"/>, <see cref="SilkChannelDecoderState.SLpcQ14Buf"/>,
    /// excitation buffer). The caller is responsible for pushing <paramref name="xqOut"/>
    /// into <see cref="SilkChannelDecoderState.OutBuf"/> after this returns (that shift
    /// lives in silk_decode_frame in libopus).
    /// </summary>
    /// <param name="state">Persistent channel-decoder state (must be Configure'd).</param>
    /// <param name="parameters">Dequantized parameters from <see cref="SilkParametersDecoder.Decode"/>.</param>
    /// <param name="pulses">Decoded pulse magnitudes. Length &gt;= <see cref="SilkChannelDecoderState.FrameLength"/>.</param>
    /// <param name="signalType">SILK signal type (0 inactive, 1 unvoiced, 2 voiced).</param>
    /// <param name="quantOffsetType">SILK quantizer offset type (0 or 1).</param>
    /// <param name="seed">PRNG seed (from <see cref="SilkDecodedIndices.Seed"/>).</param>
    /// <param name="nlsfInterpolationEnabled">True if NLSF interpolation was active for this frame
    /// (<c>indices.NlsfInterpCoefQ2 &lt; 4</c>). Gates the k==2 LTP re-whitening.</param>
    /// <param name="xqOut">Output PCM buffer. Length &gt;= <see cref="SilkChannelDecoderState.FrameLength"/>.</param>
    internal static void Decode(
        SilkChannelDecoderState state,
        SilkDecodedParameters parameters,
        ReadOnlySpan<short> pulses,
        sbyte signalType,
        sbyte quantOffsetType,
        sbyte seed,
        bool nlsfInterpolationEnabled,
        Span<short> xqOut)
    {
        if (state is null) throw new ArgumentNullException(nameof(state));
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));

        int frameLength = state.FrameLength;
        int subfrLength = state.SubfrLength;
        int ltpMemLength = state.LtpMemLength;
        int lpcOrder = state.LpcOrder;
        int nbSubfr = state.NbSubfr;

        if (frameLength == 0 || subfrLength == 0)
            throw new InvalidOperationException("State not configured; call state.Configure first.");
        if (xqOut.Length < frameLength) throw new ArgumentException($"xqOut too small (need {frameLength}).", nameof(xqOut));
        if (pulses.Length < frameLength) throw new ArgumentException($"pulses too small (need {frameLength}).", nameof(pulses));

        if (state.PrevGainQ16 == 0)
            throw new InvalidOperationException("PrevGainQ16 is 0; must be reset to a non-zero sentinel before first decode.");

        // ---- Step 1: excitation dequantization (writes to state.ExcQ14). ----
        SilkExcitationDequantizer.Dequantize(
            state.ExcQ14.AsSpan(0, frameLength),
            pulses.Slice(0, frameLength),
            signalType, quantOffsetType, seed, frameLength);

        // ---- Step 2: per-subframe synthesis scratch buffers. ----
        // sLPC_Q14: layout [history (MAX_LPC_ORDER) | output samples (subfrLength)].
        //   The first MAX_LPC_ORDER entries are seeded from state.SLpcQ14Buf and
        //   are slid back to the head after each subframe.
        Span<int> sLpcQ14 = stackalloc int[SilkConstants.MAX_LPC_ORDER + SilkConstants.MAX_SUB_FRAME_LENGTH];
        state.SLpcQ14Buf.AsSpan(0, SilkConstants.MAX_LPC_ORDER).CopyTo(sLpcQ14);

        // Voiced-only LTP state scratch. Sized at worst-case (WB 20ms) to keep stackalloc static.
        Span<int> sLtpQ15 = stackalloc int[SilkConstants.MAX_LTP_MEM_LENGTH + SilkConstants.MAX_FRAME_LENGTH];
        Span<short> sLtp = stackalloc short[SilkConstants.MAX_LTP_MEM_LENGTH];
        Span<int> presQ14 = stackalloc int[SilkConstants.MAX_SUB_FRAME_LENGTH];

        int sLtpBufIdx = ltpMemLength;

        // ---- Step 3: subframe loop. ----
        int pexcOff = 0;
        int pxqOff = 0;
        for (int k = 0; k < nbSubfr; k++)
        {
            ReadOnlySpan<short> aQ12 = parameters.PredCoefQ12.AsSpan((k >> 1) * SilkConstants.MAX_LPC_ORDER, lpcOrder);
            ReadOnlySpan<short> bQ14 = parameters.LtpCoefQ14.AsSpan(k * LtpOrder, LtpOrder);

            int gainQ16 = parameters.GainsQ16[k];
            int gainQ10 = silk_RSHIFT(gainQ16, 6);
            int invGainQ31 = silk_INVERSE32_varQ(gainQ16, 47);

            // Gain-adjust the LPC state buffer and return the ratio (for LTP state scaling).
            int gainAdjQ16 = SilkGainAdjust.Apply(sLpcQ14, state.PrevGainQ16, gainQ16);
            state.PrevGainQ16 = gainQ16;

            int lag = 0;

            if (signalType == SilkConstants.TYPE_VOICED)
            {
                lag = parameters.PitchL[k];
                bool rewhitenSubframe = (k == 0) || (k == 2 && nlsfInterpolationEnabled);

                if (rewhitenSubframe)
                {
                    // Re-whitening: apply LPC analysis filter to previous output buffer contents,
                    // writing to sLtp[startIdx..]. Result is the unscaled LTP state for this lag.
                    int startIdx = ltpMemLength - lag - lpcOrder - LtpOrder / 2;
                    if (startIdx <= 0)
                        throw new InvalidOperationException(
                            $"Rewhitening startIdx {startIdx} <= 0 (lag {lag} too large for LTP buffer).");

                    if (k == 2)
                    {
                        // Stage first 2 subframes of the current xq output into outBuf's tail,
                        // so the analysis filter can read them as prediction history.
                        xqOut.Slice(0, 2 * subfrLength).CopyTo(state.OutBuf.AsSpan(ltpMemLength, 2 * subfrLength));
                    }

                    SilkLpcAnalysisFilter.Apply(
                        sLtp.Slice(startIdx),
                        state.OutBuf.AsSpan(startIdx + k * subfrLength),
                        aQ12,
                        ltpMemLength - startIdx,
                        lpcOrder);

                    // After rewhitening, scale sLTP into sLtpQ15 using inv_gain (with LTP_scale adjustment on k==0).
                    int invGainForScale = invGainQ31;
                    if (k == 0)
                    {
                        invGainForScale = silk_LSHIFT(silk_SMULWB(invGainQ31, parameters.LtpScaleQ14), 2);
                    }

                    int numTaps = lag + LtpOrder / 2;
                    for (int i = 0; i < numTaps; i++)
                    {
                        sLtpQ15[sLtpBufIdx - i - 1] = silk_SMULWB(invGainForScale, sLtp[ltpMemLength - i - 1]);
                    }
                }
                else if (gainAdjQ16 != 1 << 16)
                {
                    // Non-rewhitening subframe with gain change: scale existing LTP state entries.
                    int numTaps = lag + LtpOrder / 2;
                    for (int i = 0; i < numTaps; i++)
                    {
                        sLtpQ15[sLtpBufIdx - i - 1] = silk_SMULWW(gainAdjQ16, sLtpQ15[sLtpBufIdx - i - 1]);
                    }
                }
            }

            // ---- Compute residual (pres_Q14). ----
            if (signalType == SilkConstants.TYPE_VOICED)
            {
                // 5-tap LTP prediction + residual write-back + sLtpQ15 state update.
                int predLagPtrOffset = sLtpBufIdx - lag + LtpOrder / 2;
                for (int i = 0; i < subfrLength; i++)
                {
                    int ltpPredQ13 = 2;
                    ltpPredQ13 = silk_SMLAWB(ltpPredQ13, sLtpQ15[predLagPtrOffset + 0], bQ14[0]);
                    ltpPredQ13 = silk_SMLAWB(ltpPredQ13, sLtpQ15[predLagPtrOffset - 1], bQ14[1]);
                    ltpPredQ13 = silk_SMLAWB(ltpPredQ13, sLtpQ15[predLagPtrOffset - 2], bQ14[2]);
                    ltpPredQ13 = silk_SMLAWB(ltpPredQ13, sLtpQ15[predLagPtrOffset - 3], bQ14[3]);
                    ltpPredQ13 = silk_SMLAWB(ltpPredQ13, sLtpQ15[predLagPtrOffset - 4], bQ14[4]);
                    predLagPtrOffset++;

                    presQ14[i] = silk_ADD_LSHIFT32(state.ExcQ14[pexcOff + i], ltpPredQ13, 1);

                    sLtpQ15[sLtpBufIdx] = silk_LSHIFT(presQ14[i], 1);
                    sLtpBufIdx++;
                }
            }
            else
            {
                // Unvoiced / inactive: the residual IS the excitation, no LTP prediction.
                for (int i = 0; i < subfrLength; i++)
                {
                    presQ14[i] = state.ExcQ14[pexcOff + i];
                }
            }

            // ---- LPC synthesis: pres_Q14 -> PCM through the order-N LPC synthesis filter. ----
            SilkLpcSynthesisFilter.Apply(
                sLpcQ14,
                presQ14.Slice(0, subfrLength),
                aQ12,
                gainQ10,
                lpcOrder,
                subfrLength,
                xqOut.Slice(pxqOff, subfrLength));

            // Slide the LPC state: the trailing MAX_LPC_ORDER samples become the history for the next subframe.
            sLpcQ14.Slice(subfrLength, SilkConstants.MAX_LPC_ORDER).CopyTo(sLpcQ14);

            pexcOff += subfrLength;
            pxqOff += subfrLength;
        }

        // ---- Step 4: save the LPC filter state for the next frame. ----
        sLpcQ14.Slice(0, SilkConstants.MAX_LPC_ORDER).CopyTo(state.SLpcQ14Buf);
    }
}
