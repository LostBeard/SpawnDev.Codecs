// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable orchestrator port of SilkIndicesDecoder.Decode (libopus
// silk/decode_indices.c). Drives every per-frame side-information decoder
// in the exact order libopus writes them, populating a single int output
// buffer with all decoded indices for downstream consumption.
//
// Pipeline (matches libopus silk_decode_indices order):
//   1. Signal type + quantizer offset (SilkSideInfoDecoderGpu).
//   2. Gain indices (SilkGainIndicesDecoderGpu).
//   3. NLSF indices + Q2 interpolation factor (SilkNlsfIndicesDecoderGpu).
//   4. Voiced-only: pitch indices + LTP indices.
//   5. PRNG seed (SilkSideInfoDecoderGpu).
//
// Sequential per-stream because every stage shares the range decoder
// state. One thread per stream; multi-channel decode parallelizes across
// threads.
//
// Upstream Copyright (c) 2006-2011 Skype Limited. BSD 3-Clause.

using ILGPU;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// Output layout for SilkIndicesDecoderGpu.Decode. All ints in one
/// flat ArrayView; offsets defined as constants here so callers can
/// extract individual fields after readback.
/// </summary>
public static class SilkDecodedIndicesLayout
{
    /// <summary>SignalType (0 inactive, 1 unvoiced, 2 voiced).</summary>
    public const int SignalTypeOffset = 0;
    /// <summary>QuantOffsetType (0 or 1).</summary>
    public const int QuantOffsetTypeOffset = 1;
    /// <summary>Q2 NLSF interpolation factor (0..4).</summary>
    public const int NlsfInterpCoefQ2Offset = 2;
    /// <summary>Pitch lag index (0 if unvoiced).</summary>
    public const int LagIndexOffset = 3;
    /// <summary>Pitch contour index (0 if unvoiced).</summary>
    public const int ContourIndexOffset = 4;
    /// <summary>LTP periodicity index (0 if unvoiced).</summary>
    public const int PerIndexOffset = 5;
    /// <summary>LTP scale index (0 if unvoiced or conditional).</summary>
    public const int LtpScaleIndexOffset = 6;
    /// <summary>PRNG seed (0..3).</summary>
    public const int SeedOffset = 7;
    /// <summary>Per-subframe gain indices [Offset .. Offset + nbSubfr).</summary>
    public const int GainsIndicesOffset = 8;
    /// <summary>NLSF cb1 + per-coef residual indices [Offset .. Offset + order + 1).
    /// Worst case order=16 -> 17 entries occupying [12..28].</summary>
    public const int NlsfIndicesOffset = 12;
    /// <summary>Per-subframe LTP gain indices [Offset .. Offset + nbSubfr).
    /// Slots [30, 31] are scratch used by the orchestrator to capture the
    /// LTP perIndex + scaleIndex from `SilkLtpIndicesDecoderGpu` before
    /// they're copied to the named PerIndexOffset / LtpScaleIndexOffset
    /// slots. They overlap intentionally; do NOT consume slot 30/31 for
    /// other state.</summary>
    public const int LtpIndicesOffset = 32;

    /// <summary>Total int slots needed for the output buffer (worst case order=16, nbSubfr=4).</summary>
    public const int TotalSlots = 36;
}

/// <summary>
/// GPU-callable orchestrator that decodes the full SILK side-information
/// block for a single frame. Mirror of `SilkIndicesDecoder.Decode`.
/// </summary>
public static class SilkIndicesDecoderGpu
{
    /// <summary>SILK signal type: voiced.</summary>
    public const int TypeVoiced = 2;

    /// <summary>
    /// Decode the full SILK side-information block. Output layout per
    /// <see cref="SilkDecodedIndicesLayout"/>.
    /// </summary>
    /// <param name="state">Range decoder state (advanced in place).</param>
    /// <param name="buf">Encoded packet buffer.</param>
    /// <param name="bufStart">Offset of the packet in <paramref name="buf"/>.</param>
    /// <param name="storage">Length of the packet in bytes.</param>
    /// <param name="inputs">The iCDF tables + scratch, packed (see <see cref="SilkIndicesInputs"/>).</param>
    /// <param name="nVectors">codebook.NVectors.</param>
    /// <param name="order">codebook.Order.</param>
    /// <param name="nbSubfr">Subframe count (2 or 4).</param>
    /// <param name="fsKHz">Internal SILK sample rate (8, 12, or 16).</param>
    /// <param name="vadFlag">VAD flag (1 to use VAD signal-type table).</param>
    /// <param name="decodeLbrr">Decoding LBRR frame (1 to force VAD signal-type table).</param>
    /// <param name="conditional">0 for independent coding, non-zero for conditional.</param>
    /// <param name="prevLagIndex">Previous frame's pitch lag (delta-coded path).</param>
    /// <param name="prevSignalTypeWasVoiced">1 if prev frame voiced, 0 otherwise.</param>
    /// <param name="firstFrameAfterReset">1 if first frame after reset (suppresses NLSF interpolation).</param>
    /// <param name="skipLtpScale">1 for libopus CODE_INDEPENDENTLY_NO_LTP_SCALING (the side channel right after a
    /// mid-only frame): independent gains, but no LTP scale index.</param>
    /// <param name="output">Output ArrayView&lt;int&gt; of length &gt;= SilkDecodedIndicesLayout.TotalSlots.</param>
    /// <param name="outputBase">Offset into <paramref name="output"/>.</param>
    public static void Decode(
        ref OpusRangeDecoderGpuState state,
        ArrayView<byte> buf, int bufStart, uint storage,
        SilkIndicesInputs inputs,
        int nVectors, int order, int nbSubfr, int fsKHz,
        int vadFlag, int decodeLbrr, int conditional,
        int prevLagIndex, int prevSignalTypeWasVoiced,
        int firstFrameAfterReset,
        ArrayView<int> output, long outputBase,
        int skipLtpScale = 0)
    {
        // Step 1: Signal type + quantizer offset.
        bool useVadTable = (vadFlag != 0) || (decodeLbrr != 0);
        SilkSideInfoDecoderGpu.DecodeSignalType(
            ref state, buf, bufStart, storage,
            inputs.Bytes, inputs.TypeOffsetVadIcdfAt,
            inputs.Bytes, inputs.TypeOffsetNoVadIcdfAt,
            useVadTable,
            out int signalType, out int quantOffsetType);
        output[outputBase + SilkDecodedIndicesLayout.SignalTypeOffset] = signalType;
        output[outputBase + SilkDecodedIndicesLayout.QuantOffsetTypeOffset] = quantOffsetType;

        // Step 2: Gain indices.
        SilkGainIndicesDecoderGpu.DecodeIndices(
            ref state, buf, bufStart, storage,
            inputs.Bytes, inputs.GainIcdfAt,
            inputs.Bytes, inputs.DeltaGainIcdfAt,
            inputs.Bytes, inputs.Uniform8IcdfAt,
            signalType, conditional, nbSubfr,
            output, outputBase + SilkDecodedIndicesLayout.GainsIndicesOffset);

        // Step 3: NLSF indices + interpolation factor.
        // cb1IcdfBaseOffset = (signalType >> 1) * nVectors, computed dynamically here.
        int dynamicCb1IcdfBaseOffset = (signalType >> 1) * nVectors;
        int interp = SilkNlsfIndicesDecoderGpu.DecodeIndices(
            ref state, buf, bufStart, storage,
            inputs.Bytes, inputs.Cb1IcdfAt + dynamicCb1IcdfBaseOffset,
            inputs.Bytes, inputs.EcIcdfAt,
            inputs.Bytes, inputs.EcSelAt,
            inputs.Bytes, inputs.PredQ8SourceAt,
            inputs.Bytes, inputs.NlsfExtIcdfAt,
            inputs.Bytes, inputs.NlsfInterpolationFactorIcdfAt,
            inputs.EcIxScratch, 0,
            inputs.Bytes, inputs.PredQ8ScratchAt,
            order, nbSubfr,
            output, outputBase + SilkDecodedIndicesLayout.NlsfIndicesOffset);
        // First-frame-after-reset suppresses NLSF interpolation (prev NLSFs are garbage).
        if (firstFrameAfterReset != 0) interp = 4;
        output[outputBase + SilkDecodedIndicesLayout.NlsfInterpCoefQ2Offset] = interp;

        // Step 4: Voiced-only: pitch + LTP.
        if (signalType == TypeVoiced)
        {
            // Pitch indices.
            // Reuse NlsfIndicesOffset+order+1 area is wrong; pitch goes to its dedicated
            // (LagIndex, Contour) slots below.
            int pitchLag, pitchContour;
            // SilkPitchIndicesDecoderGpu writes to a 2-int block; use a tiny temporary
            // by routing through output's LagIndex/ContourIndex pair directly.
            SilkPitchIndicesDecoderGpu.DecodeIndices(
                ref state, buf, bufStart, storage,
                inputs.Bytes, inputs.PitchDeltaIcdfAt,
                inputs.Bytes, inputs.PitchLagIcdfAt,
                inputs.Bytes, inputs.LagLowBitsIcdfAt,
                inputs.Bytes, inputs.ContourIcdfAt,
                fsKHz, prevLagIndex, prevSignalTypeWasVoiced, conditional,
                output, outputBase + SilkDecodedIndicesLayout.LagIndexOffset);

            pitchLag = output[outputBase + SilkDecodedIndicesLayout.LagIndexOffset];
            pitchContour = output[outputBase + SilkDecodedIndicesLayout.ContourIndexOffset];

            // LTP indices. Output writes perIndex/scale/gainIndices to a temporary
            // 6-int region that we rewrite into the layout's separate slots.
            // SilkLtpIndicesDecoderGpu writes [perIndex, scaleIndex, gain[0..nbSubfr]]
            // into output[base..base+2+nbSubfr]. We route gain indices directly into
            // LtpIndicesOffset by passing outputBase + (LtpIndicesOffset - 2) — that
            // way the per/scale go just before the gain indices block, and we then
            // copy them into the named slots.
            int ltpTempBase = (int)(outputBase + SilkDecodedIndicesLayout.LtpIndicesOffset - 2);
            SilkLtpIndicesDecoderGpu.DecodeIndices(
                ref state, buf, bufStart, storage,
                inputs.Bytes, inputs.LtpPerIndexIcdfAt,
                inputs.Bytes, inputs.LtpGainIcdfFlatAt,
                inputs.LtpGainOffsets, 0,
                inputs.Bytes, inputs.LtpScaleIcdfAt,
                conditional != 0 || skipLtpScale != 0 ? 1 : 0, nbSubfr,
                output, ltpTempBase);
            int perIndex = output[ltpTempBase + 0];
            int ltpScaleIndex = output[ltpTempBase + 1];
            output[outputBase + SilkDecodedIndicesLayout.PerIndexOffset] = perIndex;
            output[outputBase + SilkDecodedIndicesLayout.LtpScaleIndexOffset] = ltpScaleIndex;
            // ltpTempBase + 2 .. ltpTempBase + 2 + nbSubfr is now LtpIndicesOffset onwards (gain indices).
        }
        else
        {
            // Zero out voiced-only fields.
            output[outputBase + SilkDecodedIndicesLayout.LagIndexOffset] = 0;
            output[outputBase + SilkDecodedIndicesLayout.ContourIndexOffset] = 0;
            output[outputBase + SilkDecodedIndicesLayout.PerIndexOffset] = 0;
            output[outputBase + SilkDecodedIndicesLayout.LtpScaleIndexOffset] = 0;
            for (int k = 0; k < nbSubfr; k++)
                output[outputBase + SilkDecodedIndicesLayout.LtpIndicesOffset + k] = 0;
        }

        // Step 5: PRNG seed.
        int seed = SilkSideInfoDecoderGpu.DecodeSeed(
            ref state, buf, bufStart, storage,
            inputs.Bytes, inputs.Uniform4IcdfAt);
        output[outputBase + SilkDecodedIndicesLayout.SeedOffset] = seed;
    }
}
