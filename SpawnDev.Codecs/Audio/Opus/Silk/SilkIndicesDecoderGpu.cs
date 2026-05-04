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
    /// <param name="typeOffsetVadIcdf">silk_type_offset_VAD_iCDF (4 entries).</param>
    /// <param name="typeOffsetNoVadIcdf">silk_type_offset_no_VAD_iCDF (2 entries).</param>
    /// <param name="uniform4Icdf">silk_uniform4_iCDF (4 entries; used for seed).</param>
    /// <param name="gainIcdf">silk_gain_iCDF flat (24 entries: 3 signal types × 8).</param>
    /// <param name="deltaGainIcdf">silk_delta_gain_iCDF (41 entries).</param>
    /// <param name="uniform8Icdf">silk_uniform8_iCDF (8 entries; used for first-subframe gain LSB).</param>
    /// <param name="cb1Icdf">NLSF codebook Cb1Icdf, length 2 * NVectors.</param>
    /// <param name="ecIcdf">NLSF codebook EcIcdf.</param>
    /// <param name="ecSel">NLSF codebook EcSel bytes.</param>
    /// <param name="predQ8Source">NLSF codebook PredQ8 source.</param>
    /// <param name="nlsfExtIcdf">silk_NLSF_EXT_iCDF (7 entries).</param>
    /// <param name="nlsfInterpolationFactorIcdf">silk_NLSF_interpolation_factor_iCDF (5 entries).</param>
    /// <param name="pitchDeltaIcdf">silk_pitch_delta_iCDF (21 entries).</param>
    /// <param name="pitchLagIcdf">silk_pitch_lag_iCDF (32 entries).</param>
    /// <param name="lagLowBitsIcdf">fs_kHz-resolved Uniform4/6/8.</param>
    /// <param name="contourIcdf">(fs_kHz, nbSubfr)-resolved contour iCDF.</param>
    /// <param name="ltpPerIndexIcdf">silk_LTP_per_index_iCDF (3 entries).</param>
    /// <param name="ltpGainIcdfFlat">Flat-packed LtpGain0+1+2 (56 entries).</param>
    /// <param name="ltpGainOffsets">[0, 8, 24] - offsets into ltpGainIcdfFlat per perIndex.</param>
    /// <param name="ltpScaleIcdf">silk_LTP_scale_iCDF (3 entries).</param>
    /// <param name="ecIxScratch">Scratch buffer (length >= MaxLpcOrder=16).</param>
    /// <param name="predQ8Scratch">Scratch buffer (length >= MaxLpcOrder=16).</param>
    /// <param name="cb1IcdfBaseOffset">Pre-computed (signalType >> 1) * nVectors offset for the NLSF
    /// step. For SILK, signalType is the OUTPUT of step 1 not an input — but the offset is computed
    /// AFTER step 1 inside the kernel. So this parameter is unused here; included for parity with
    /// the standalone NLSF decoder where the host pre-computes it. Kernel computes it dynamically.
    /// Unused.</param>
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
    /// <param name="output">Output ArrayView&lt;int&gt; of length &gt;= SilkDecodedIndicesLayout.TotalSlots.</param>
    /// <param name="outputBase">Offset into <paramref name="output"/>.</param>
    public static void Decode(
        ref OpusRangeDecoderGpuState state,
        ArrayView<byte> buf, int bufStart, uint storage,
        ArrayView<byte> typeOffsetVadIcdf,
        ArrayView<byte> typeOffsetNoVadIcdf,
        ArrayView<byte> uniform4Icdf,
        ArrayView<byte> gainIcdf,
        ArrayView<byte> deltaGainIcdf,
        ArrayView<byte> uniform8Icdf,
        ArrayView<byte> cb1Icdf,
        ArrayView<byte> ecIcdf,
        ArrayView<byte> ecSel,
        ArrayView<byte> predQ8Source,
        ArrayView<byte> nlsfExtIcdf,
        ArrayView<byte> nlsfInterpolationFactorIcdf,
        ArrayView<byte> pitchDeltaIcdf,
        ArrayView<byte> pitchLagIcdf,
        ArrayView<byte> lagLowBitsIcdf,
        ArrayView<byte> contourIcdf,
        ArrayView<byte> ltpPerIndexIcdf,
        ArrayView<byte> ltpGainIcdfFlat,
        ArrayView<int> ltpGainOffsets,
        ArrayView<byte> ltpScaleIcdf,
        ArrayView<short> ecIxScratch,
        ArrayView<byte> predQ8Scratch,
        int nVectors, int order, int nbSubfr, int fsKHz,
        int vadFlag, int decodeLbrr, int conditional,
        int prevLagIndex, int prevSignalTypeWasVoiced,
        int firstFrameAfterReset,
        ArrayView<int> output, long outputBase)
    {
        // Step 1: Signal type + quantizer offset.
        bool useVadTable = (vadFlag != 0) || (decodeLbrr != 0);
        SilkSideInfoDecoderGpu.DecodeSignalType(
            ref state, buf, bufStart, storage,
            typeOffsetVadIcdf, 0,
            typeOffsetNoVadIcdf, 0,
            useVadTable,
            out int signalType, out int quantOffsetType);
        output[outputBase + SilkDecodedIndicesLayout.SignalTypeOffset] = signalType;
        output[outputBase + SilkDecodedIndicesLayout.QuantOffsetTypeOffset] = quantOffsetType;

        // Step 2: Gain indices.
        SilkGainIndicesDecoderGpu.DecodeIndices(
            ref state, buf, bufStart, storage,
            gainIcdf, 0,
            deltaGainIcdf, 0,
            uniform8Icdf, 0,
            signalType, conditional, nbSubfr,
            output, outputBase + SilkDecodedIndicesLayout.GainsIndicesOffset);

        // Step 3: NLSF indices + interpolation factor.
        // cb1IcdfBaseOffset = (signalType >> 1) * nVectors, computed dynamically here.
        int dynamicCb1IcdfBaseOffset = (signalType >> 1) * nVectors;
        int interp = SilkNlsfIndicesDecoderGpu.DecodeIndices(
            ref state, buf, bufStart, storage,
            cb1Icdf, dynamicCb1IcdfBaseOffset,
            ecIcdf, 0,
            ecSel, 0,
            predQ8Source, 0,
            nlsfExtIcdf, 0,
            nlsfInterpolationFactorIcdf, 0,
            ecIxScratch, 0,
            predQ8Scratch, 0,
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
                pitchDeltaIcdf, 0,
                pitchLagIcdf, 0,
                lagLowBitsIcdf, 0,
                contourIcdf, 0,
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
                ltpPerIndexIcdf, 0,
                ltpGainIcdfFlat, 0,
                ltpGainOffsets, 0,
                ltpScaleIcdf, 0,
                conditional, nbSubfr,
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
            uniform4Icdf, 0);
        output[outputBase + SilkDecodedIndicesLayout.SeedOffset] = seed;
    }
}
