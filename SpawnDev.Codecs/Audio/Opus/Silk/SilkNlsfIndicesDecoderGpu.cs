// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable port of SilkNlsfDecoder.DecodeIndices. Reads the NLSF
// index block from the libopus range-coded bitstream:
//   - First-stage codebook index (cb1) from the signal-type-selected
//     half of the codebook's Cb1Icdf table.
//   - Per-coefficient signed residual indices via the codebook's EcIcdf
//     table indexed by SilkNlsfUnpack-derived ecIx[i].
//   - Rail-extension symbols when the index hits 0 or 2*MAX_AMPLITUDE
//     (NlsfExt table).
//   - Q2 NLSF interpolation factor when nbSubfr == MAX_NB_SUBFR (20ms
//     frames) via NlsfInterpolationFactor table.
//
// Sequential per-stream because the range decoder advances stateful per
// symbol read. Multi-channel decode parallelizes across threads.
//
// Upstream Copyright (c) 2006-2011 Skype Limited. BSD 3-Clause.

using ILGPU;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// GPU-callable decoder for SILK NLSF indices. Mirror of
/// `SilkNlsfDecoder.DecodeIndices` (CPU reference in
/// SpawnDev.Codecs.References).
/// </summary>
public static class SilkNlsfIndicesDecoderGpu
{
    /// <summary>2 * NLSF_QUANT_MAX_AMPLITUDE = 8 (rail-top symbol).</summary>
    public const int RailTopSymbol = 8;
    /// <summary>NLSF_QUANT_MAX_AMPLITUDE = 4 (residual offset).</summary>
    public const int NlsfQuantMaxAmplitude = 4;
    /// <summary>SilkConstants.MAX_NB_SUBFR.</summary>
    public const int MaxNbSubfr = 4;
    /// <summary>SilkConstants.MAX_LPC_ORDER.</summary>
    public const int MaxLpcOrder = 16;

    /// <summary>
    /// Decode the NLSF index block. Writes <c>order + 1</c> entries into
    /// <paramref name="nlsfIndicesOut"/>: index 0 is the first-stage
    /// codebook index (sbyte range, here packed as int); indices [1..order]
    /// are signed residuals in <c>[-MAX_AMPLITUDE - 6, +MAX_AMPLITUDE + 6]</c>.
    /// Returns the Q2 interpolation factor (always 4 for nbSubfr != MAX_NB_SUBFR).
    /// </summary>
    /// <param name="state">Range decoder state (advanced in place).</param>
    /// <param name="buf">Encoded packet buffer.</param>
    /// <param name="bufStart">Offset of the packet in <paramref name="buf"/>.</param>
    /// <param name="storage">Length of the packet in bytes.</param>
    /// <param name="cb1Icdf">Codebook Cb1Icdf, length 2 * NVectors.</param>
    /// <param name="cb1IcdfBase">Offset into <paramref name="cb1Icdf"/>.</param>
    /// <param name="nVectors">codebook.NVectors.</param>
    /// <param name="ecIcdf">Codebook EcIcdf, layout: 9 entries per ecIx slot.</param>
    /// <param name="ecIcdfBase">Offset into <paramref name="ecIcdf"/>.</param>
    /// <param name="ecSel">Codebook EcSel bytes (length nVectors * order / 2).</param>
    /// <param name="ecSelBase">Offset into <paramref name="ecSel"/>.</param>
    /// <param name="predQ8Source">Codebook PredQ8 array (length 2 * (order - 1)).</param>
    /// <param name="predQ8SrcBase">Offset into <paramref name="predQ8Source"/>.</param>
    /// <param name="nlsfExtIcdf">SilkIcdfTables.NlsfExt (7 entries).</param>
    /// <param name="nlsfExtBase">Offset into <paramref name="nlsfExtIcdf"/>.</param>
    /// <param name="nlsfInterpolationFactorIcdf">SilkIcdfTables.NlsfInterpolationFactor (5 entries).</param>
    /// <param name="nlsfInterpolationFactorBase">Offset into <paramref name="nlsfInterpolationFactorIcdf"/>.</param>
    /// <param name="ecIxScratch">Scratch buffer for ecIx[order] (length >= MaxLpcOrder).</param>
    /// <param name="ecIxScratchBase">Offset.</param>
    /// <param name="predQ8Scratch">Scratch buffer for predQ8[order] (length >= MaxLpcOrder).</param>
    /// <param name="predQ8ScratchBase">Offset.</param>
    /// <param name="signalType">SILK signal type (0/1/2). (signalType >> 1) selects the cb1 iCDF half.</param>
    /// <param name="order">codebook.Order (10 or 16). Must be even.</param>
    /// <param name="nbSubfr">Subframe count (2 or 4).</param>
    /// <param name="nlsfIndicesOut">Output: order+1 ints.</param>
    /// <param name="nlsfIndicesOutBase">Offset.</param>
    /// <returns>Q2 NLSF interpolation coefficient in [0, 4].</returns>
    public static int DecodeIndices(
        ref OpusRangeDecoderGpuState state,
        ArrayView<byte> buf, int bufStart, uint storage,
        ArrayView<byte> cb1Icdf, long cb1IcdfBase,
        ArrayView<byte> ecIcdf, long ecIcdfBase,
        ArrayView<byte> ecSel, long ecSelBase,
        ArrayView<byte> predQ8Source, long predQ8SrcBase,
        ArrayView<byte> nlsfExtIcdf, long nlsfExtBase,
        ArrayView<byte> nlsfInterpolationFactorIcdf, long nlsfInterpolationFactorBase,
        ArrayView<short> ecIxScratch, long ecIxScratchBase,
        ArrayView<byte> predQ8Scratch, long predQ8ScratchBase,
        int order, int nbSubfr,
        ArrayView<int> nlsfIndicesOut, long nlsfIndicesOutBase)
    {
        // Step 1: first-stage codebook index. Caller pre-computes
        // cb1IcdfBase = origBase + (signalType >> 1) * nVectors so the
        // kernel signature stays under ILGPU's Action<16> ceiling.
        int cb1Index = OpusRangeDecoderGpu.DecodeIcdf(
            ref state, buf, bufStart, storage,
            cb1Icdf, cb1IcdfBase, 8);
        nlsfIndicesOut[nlsfIndicesOutBase + 0] = cb1Index;

        // Step 2: unpack ecIx + predQ8 for the chosen cb1Index.
        // SilkNlsfUnpackGpu.UnpackPairAt is per-pair; loop pairs.
        long ecSelEntryBase = ecSelBase + (long)cb1Index * (order / 2);
        for (int pairIdx = 0; pairIdx < order / 2; pairIdx++)
        {
            SilkNlsfUnpackGpu.UnpackPairAt(
                ecIxScratch, ecIxScratchBase,
                predQ8Scratch, predQ8ScratchBase,
                ecSel, ecSelEntryBase,
                predQ8Source, predQ8SrcBase,
                order, pairIdx);
        }

        // Step 3: per-coefficient residual indices with rail extension.
        for (int i = 0; i < order; i++)
        {
            int ecIxValue = ecIxScratch[ecIxScratchBase + i];
            int ix = OpusRangeDecoderGpu.DecodeIcdf(
                ref state, buf, bufStart, storage,
                ecIcdf, ecIcdfBase + ecIxValue, 8);
            if (ix == 0)
            {
                ix -= OpusRangeDecoderGpu.DecodeIcdf(
                    ref state, buf, bufStart, storage,
                    nlsfExtIcdf, nlsfExtBase, 8);
            }
            else if (ix == RailTopSymbol)
            {
                ix += OpusRangeDecoderGpu.DecodeIcdf(
                    ref state, buf, bufStart, storage,
                    nlsfExtIcdf, nlsfExtBase, 8);
            }
            nlsfIndicesOut[nlsfIndicesOutBase + 1 + i] = ix - NlsfQuantMaxAmplitude;
        }

        // Step 4: Q2 interpolation factor for 20ms frames.
        if (nbSubfr == MaxNbSubfr)
        {
            return OpusRangeDecoderGpu.DecodeIcdf(
                ref state, buf, bufStart, storage,
                nlsfInterpolationFactorIcdf, nlsfInterpolationFactorBase, 8);
        }
        return 4;
    }
}
