// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Structural port of libopus silk/decode_frame.c to clean C#. Orchestrates the
// full per-frame SILK decode: indices -> pulses -> parameters -> core, then
// shifts the output buffer and updates stream-level state. Packet-loss
// concealment (silk_PLC), comfort noise (silk_CNG), PLC-glue, OSCE, and DEEP_PLC
// are intentionally out of scope for this port.
//
// Upstream Copyright (c) 2006-2011 Skype Limited. BSD 3-Clause. See NOTICE.md.

using SpawnDev.Codecs.EntropyCoders;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// Top-level per-frame SILK decoder. Drives the full pipeline from a
/// range-coded bitstream to int16 PCM samples, updating the persistent
/// <see cref="SilkChannelDecoderState"/> in place for the next frame.
/// </summary>
internal static class SilkDecodeFrame
{
    /// <summary>
    /// Decode a single SILK frame.
    /// </summary>
    /// <param name="state">Persistent channel-decoder state (must be Configure'd).</param>
    /// <param name="rangeDec">Range decoder positioned at the start of the frame payload.</param>
    /// <param name="pOut">Output PCM buffer. Length &gt;= <see cref="SilkChannelDecoderState.FrameLength"/>.</param>
    /// <param name="vadFlag">VAD flag for the current frame. Controls signal-type iCDF selection.</param>
    /// <param name="conditional">0 for independent coding, non-zero for conditional / delta coding.</param>
    internal static void Decode(
        SilkChannelDecoderState state,
        OpusRangeDecoder rangeDec,
        Span<short> pOut,
        bool vadFlag,
        int conditional)
    {
        if (state is null) throw new ArgumentNullException(nameof(state));
        if (rangeDec is null) throw new ArgumentNullException(nameof(rangeDec));

        int frameLength = state.FrameLength;
        int ltpMemLength = state.LtpMemLength;
        if (frameLength == 0) throw new InvalidOperationException("State not configured; call state.Configure first.");
        if (pOut.Length < frameLength)
            throw new ArgumentException($"pOut too small (need {frameLength}).", nameof(pOut));
        if (ltpMemLength < frameLength)
            throw new InvalidOperationException(
                $"ltpMemLength ({ltpMemLength}) must be >= frameLength ({frameLength}).");

        // Select the NLSF codebook based on the LPC order (set in Configure).
        SilkNlsfCodebook codebook = state.LpcOrder == 16
            ? SilkNlsfCodebookTables.Wb
            : SilkNlsfCodebookTables.NbMb;

        // 1. Decode all side-information indices.
        var indices = new SilkDecodedIndices();
        SilkIndicesDecoder.Decode(
            indices, rangeDec, codebook,
            vadFlag: vadFlag,
            decodeLbrr: false,
            fsKHz: state.FsKHz,
            nbSubfr: state.NbSubfr,
            conditional: conditional,
            prevLagIndex: state.PrevLagIndex,
            prevSignalTypeWasVoiced: state.PrevSignalTypeWasVoiced);

        // First-frame-after-reset suppresses NLSF interpolation (prev NLSFs are garbage).
        if (state.FirstFrameAfterReset)
        {
            indices.NlsfInterpCoefQ2 = 4;
        }

        // 2. Decode the pulse train. Buffer must be aligned up to the shell-coder
        //    frame-length (SHELL_CODEC_FRAME_LENGTH = 16 samples).
        int shellLen = SilkConstants.SHELL_CODEC_FRAME_LENGTH;
        int pulsesLen = (frameLength + shellLen - 1) & ~(shellLen - 1);
        Span<short> pulses = stackalloc short[SilkConstants.MAX_FRAME_LENGTH + SilkConstants.SHELL_CODEC_FRAME_LENGTH];
        pulses = pulses.Slice(0, pulsesLen);
        SilkPulsesDecoder.Decode(
            pulses, rangeDec,
            signalType: indices.SignalType,
            quantOffsetType: indices.QuantOffsetType,
            frameLength: frameLength);

        // 3. Dequantize all parameters (gains, NLSFs + LPC coefs, pitch lags, LTP taps, LTP scale).
        var parameters = new SilkDecodedParameters();
        SilkParametersDecoder.Decode(
            parameters, indices, codebook,
            fsKHz: state.FsKHz,
            nbSubfr: state.NbSubfr,
            lastGainIndex: ref state.LastGainIndex,
            prevNlsfQ15: state.PrevNlsfQ15,
            conditional: conditional);

        // 4. Run the synthesis chain: excitation -> LTP -> LPC -> PCM output.
        bool nlsfInterpolationEnabled = indices.NlsfInterpCoefQ2 < 4;
        SilkDecodeCore.Decode(
            state, parameters, pulses.Slice(0, frameLength),
            signalType: indices.SignalType,
            quantOffsetType: indices.QuantOffsetType,
            seed: indices.Seed,
            nlsfInterpolationEnabled: nlsfInterpolationEnabled,
            xqOut: pOut.Slice(0, frameLength));

        // 5. Shift outBuf: drop the oldest frame_length samples, append the freshly-decoded xq.
        int mvLen = ltpMemLength - frameLength;
        state.OutBuf.AsSpan(frameLength, mvLen).CopyTo(state.OutBuf.AsSpan(0, mvLen));
        pOut.Slice(0, frameLength).CopyTo(state.OutBuf.AsSpan(mvLen, frameLength));

        // 6. Update stream-level state for the next frame.
        state.LossCnt = 0;
        state.PrevSignalType = indices.SignalType;
        state.PrevSignalTypeWasVoiced = indices.SignalType == SilkConstants.TYPE_VOICED;
        state.PrevLagIndex = indices.LagIndex;
        state.FirstFrameAfterReset = false;
        state.LagPrev = parameters.PitchL[state.NbSubfr - 1];
    }
}
