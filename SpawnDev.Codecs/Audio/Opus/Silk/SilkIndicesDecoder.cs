// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Top-level port of libopus silk/decode_indices.c::silk_decode_indices. Drives
// every per-frame side-information decoder (signal type, gains, NLSFs, pitch,
// LTP, seed) in the exact order libopus writes them, populating a caller-provided
// SilkDecodedIndices instance.
//
// Upstream Copyright (c) 2006-2011 Skype Limited. BSD 3-Clause. See NOTICE.md.

using SpawnDev.Codecs.EntropyCoders;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// Top-level silk_decode_indices orchestrator. Reads the complete side-information
/// block for a single SILK frame into a <see cref="SilkDecodedIndices"/>.
/// </summary>
internal static class SilkIndicesDecoder
{
    /// <summary>
    /// Decode the full SILK side-information block for a single frame.
    /// </summary>
    /// <param name="indices">Output: caller-allocated indices struct. All relevant fields are filled.</param>
    /// <param name="rangeDec">Range decoder positioned at the start of the side-info block.</param>
    /// <param name="codebook">NLSF codebook (NB/MB or WB, selected by caller based on fs_kHz).</param>
    /// <param name="vadFlag">VAD flag for the current frame (controls signal-type table selection).</param>
    /// <param name="decodeLbrr">True when decoding an LBRR frame (forces use of the VAD signal-type table).</param>
    /// <param name="fsKHz">Internal SILK sample rate in kHz (8, 12, or 16).</param>
    /// <param name="nbSubfr">Subframe count (2 for 10 ms frames, 4 for 20 ms frames).</param>
    /// <param name="conditional">0 for independent coding, non-zero for conditional (delta) coding.</param>
    /// <param name="prevLagIndex">Previous frame's pitch lag (used when conditional &amp; prev voiced).</param>
    /// <param name="prevSignalTypeWasVoiced">Whether the previous frame was voiced (enables delta-lag coding).</param>
    internal static void Decode(
        SilkDecodedIndices indices,
        OpusRangeDecoder rangeDec,
        SilkNlsfCodebook codebook,
        bool vadFlag,
        bool decodeLbrr,
        int fsKHz,
        int nbSubfr,
        int conditional,
        short prevLagIndex,
        bool prevSignalTypeWasVoiced)
    {
        if (indices is null) throw new ArgumentNullException(nameof(indices));
        if (rangeDec is null) throw new ArgumentNullException(nameof(rangeDec));
        if (codebook is null) throw new ArgumentNullException(nameof(codebook));

        // 1. Signal type + quantizer offset.
        var sig = SilkSideInfoDecoder.DecodeSignalType(rangeDec, useVadTable: vadFlag || decodeLbrr);
        indices.SignalType = sig.SignalType;
        indices.QuantOffsetType = sig.QuantOffsetType;

        // 2. Gains.
        SilkGainDecoder.DecodeIndices(
            indices.GainsIndices.AsSpan(0, nbSubfr),
            rangeDec,
            signalType: sig.SignalType,
            conditional: conditional,
            nbSubfr: nbSubfr);

        // 3. NLSFs (codebook index + per-coefficient residuals + interpolation factor).
        int order = codebook.Order;
        int interp = SilkNlsfDecoder.DecodeIndices(
            indices.NlsfIndices.AsSpan(0, order + 1),
            rangeDec,
            codebook,
            signalType: sig.SignalType,
            nbSubfr: nbSubfr);
        indices.NlsfInterpCoefQ2 = (sbyte)interp;

        // 4. Voiced-only: pitch + LTP.
        if (sig.SignalType == SilkSideInfoDecoder.TypeVoiced)
        {
            var pitch = SilkPitchDecoder.DecodeIndices(
                rangeDec, fsKHz, nbSubfr, prevLagIndex, prevSignalTypeWasVoiced, conditional);
            indices.LagIndex = pitch.LagIndex;
            indices.ContourIndex = pitch.ContourIndex;

            SilkLtpDecoder.DecodeIndices(
                indices.LtpIndices.AsSpan(0, nbSubfr),
                rangeDec,
                conditional: conditional,
                nbSubfr: nbSubfr,
                out sbyte perIdx,
                out sbyte scaleIdx);
            indices.PerIndex = perIdx;
            indices.LtpScaleIndex = scaleIdx;
        }
        else
        {
            // Zero out voiced-only fields for determinism.
            indices.LagIndex = 0;
            indices.ContourIndex = 0;
            indices.PerIndex = 0;
            indices.LtpScaleIndex = 0;
            for (int k = 0; k < nbSubfr; k++) indices.LtpIndices[k] = 0;
        }

        // 5. Seed.
        indices.Seed = SilkSideInfoDecoder.DecodeSeed(rangeDec);
    }
}
