// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable port of SilkLtpDecoder.DecodeIndices. Reads voiced-frame
// LTP indices from the libopus range-coded bitstream:
//   1. perIndex (3-symbol iCDF) selects one of three LTP gain codebooks.
//   2. Per-subframe LTP gain index from the perIndex-selected codebook
//      (8 / 16 / 32 entries for perIndex 0 / 1 / 2 respectively).
//   3. LTP scale index (3-symbol iCDF), only when conditional == 0.
//
// The three gain iCDFs are flat-packed into one ArrayView<byte> with an
// offsets table so the kernel can index dynamically after reading perIndex.
//
// Upstream Copyright (c) 2006-2011 Skype Limited. BSD 3-Clause.

using ILGPU;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// GPU-callable decoder for SILK LTP indices. Mirror of
/// `SilkLtpDecoder.DecodeIndices` (CPU reference in
/// SpawnDev.Codecs.References).
/// </summary>
public static class SilkLtpIndicesDecoderGpu
{
    /// <summary>
    /// Number of LTP gain codebooks (perIndex selects one).
    /// </summary>
    public const int NumLtpGainCodebooks = 3;

    /// <summary>
    /// Decode the LTP index block. Output layout in <paramref name="output"/>:
    ///   [0]              = perIndex (0, 1, or 2)
    ///   [1]              = ltpScaleIndex (0..2; 0 when conditional != 0)
    ///   [2..2+nbSubfr]   = per-subframe LTP gain indices
    /// </summary>
    /// <param name="state">Range decoder state (advanced in place).</param>
    /// <param name="buf">Encoded packet buffer.</param>
    /// <param name="bufStart">Offset of the packet in <paramref name="buf"/>.</param>
    /// <param name="storage">Length of the packet in bytes.</param>
    /// <param name="ltpPerIndexIcdf">silk_LTP_per_index_iCDF (3 entries).</param>
    /// <param name="ltpPerIndexBase">Offset.</param>
    /// <param name="ltpGainIcdfFlat">Flat-packed LtpGain0+1+2 (8 + 16 + 32 = 56 entries).</param>
    /// <param name="ltpGainIcdfFlatBase">Offset.</param>
    /// <param name="ltpGainOffsets">Offsets into <paramref name="ltpGainIcdfFlat"/> for each
    /// perIndex: [0, 8, 24]. Length >= NumLtpGainCodebooks.</param>
    /// <param name="ltpGainOffsetsBase">Offset.</param>
    /// <param name="ltpScaleIcdf">silk_LTP_scale_iCDF (3 entries).</param>
    /// <param name="ltpScaleBase">Offset.</param>
    /// <param name="conditional">0 = independent (read LTP scale); non-zero = conditional (scale=0).</param>
    /// <param name="nbSubfr">Subframe count (2 or 4).</param>
    /// <param name="output">Output ArrayView&lt;int&gt; of length &gt;= 2 + nbSubfr.</param>
    /// <param name="outputBase">Offset.</param>
    public static void DecodeIndices(
        ref OpusRangeDecoderGpuState state,
        ArrayView<byte> buf, int bufStart, uint storage,
        ArrayView<byte> ltpPerIndexIcdf, long ltpPerIndexBase,
        ArrayView<byte> ltpGainIcdfFlat, long ltpGainIcdfFlatBase,
        ArrayView<int> ltpGainOffsets, long ltpGainOffsetsBase,
        ArrayView<byte> ltpScaleIcdf, long ltpScaleBase,
        int conditional, int nbSubfr,
        ArrayView<int> output, long outputBase)
    {
        // 1. perIndex.
        int perIndex = OpusRangeDecoderGpu.DecodeIcdf(
            ref state, buf, bufStart, storage,
            ltpPerIndexIcdf, ltpPerIndexBase, 8);
        output[outputBase + 0] = perIndex;

        // 2. Per-subframe LTP gain indices using the perIndex-selected iCDF.
        int gainIcdfOffset = ltpGainOffsets[ltpGainOffsetsBase + perIndex];
        for (int k = 0; k < nbSubfr; k++)
        {
            int gainIdx = OpusRangeDecoderGpu.DecodeIcdf(
                ref state, buf, bufStart, storage,
                ltpGainIcdfFlat, ltpGainIcdfFlatBase + gainIcdfOffset, 8);
            output[outputBase + 2 + k] = gainIdx;
        }

        // 3. LTP scale index (conditional gate).
        int ltpScaleIndex = 0;
        if (conditional == 0)
        {
            ltpScaleIndex = OpusRangeDecoderGpu.DecodeIcdf(
                ref state, buf, bufStart, storage,
                ltpScaleIcdf, ltpScaleBase, 8);
        }
        output[outputBase + 1] = ltpScaleIndex;
    }
}
