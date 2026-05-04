// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable port of SilkPitchDecoder.DecodeIndices. Reads voiced-frame
// pitch lag + contour indices from the libopus range-coded bitstream:
//   - When (conditional && prevSignalTypeWasVoiced), tries a delta-coded
//     lag via PitchDelta iCDF; raw 0 falls through to absolute coding.
//   - Absolute coding: coarse lag from PitchLag iCDF + low bits from a
//     fs_kHz-specific Uniform iCDF (4/6/8 entries).
//   - Contour from a (fs_kHz, nbSubfr)-specific iCDF.
//
// Caller resolves the fs_kHz-specific lag-LSB iCDF + the (fs_kHz, nbSubfr)
// contour iCDF on the host and passes them as ArrayView<byte> parameters.
//
// Upstream Copyright (c) 2006-2011 Skype Limited. BSD 3-Clause.

using ILGPU;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// GPU-callable decoder for SILK pitch indices (lag + contour).
/// Mirror of `SilkPitchDecoder.DecodeIndices` (CPU reference in
/// SpawnDev.Codecs.References).
/// </summary>
public static class SilkPitchIndicesDecoderGpu
{
    /// <summary>
    /// Decode (lagIndex, contourIndex) for a voiced SILK frame. Output
    /// layout in <paramref name="output"/>: [0]=lagIndex, [1]=contourIndex.
    /// </summary>
    /// <param name="state">Range decoder state (advanced in place).</param>
    /// <param name="buf">Encoded packet buffer.</param>
    /// <param name="bufStart">Offset of the packet in <paramref name="buf"/>.</param>
    /// <param name="storage">Length of the packet in bytes.</param>
    /// <param name="pitchDeltaIcdf">silk_pitch_delta_iCDF (21 entries).</param>
    /// <param name="pitchDeltaBase">Offset.</param>
    /// <param name="pitchLagIcdf">silk_pitch_lag_iCDF (32 entries).</param>
    /// <param name="pitchLagBase">Offset.</param>
    /// <param name="lagLowBitsIcdf">Caller-resolved fs_kHz-specific Uniform4/6/8 iCDF.</param>
    /// <param name="lagLowBitsBase">Offset.</param>
    /// <param name="contourIcdf">Caller-resolved (fs_kHz, nbSubfr)-specific contour iCDF.</param>
    /// <param name="contourBase">Offset.</param>
    /// <param name="fsKHz">Internal SILK sample rate (8, 12, or 16).</param>
    /// <param name="prevLagIndex">Previous frame's pitch lag (used for delta-coded path).</param>
    /// <param name="prevSignalTypeWasVoiced">1 if previous frame was voiced, 0 otherwise.</param>
    /// <param name="conditional">0 = independent coding (absolute lag); non-zero = conditional (try delta).</param>
    /// <param name="output">Output ArrayView&lt;int&gt; of length &gt;= 2.
    /// [0]=lagIndex, [1]=contourIndex.</param>
    /// <param name="outputBase">Offset into <paramref name="output"/>.</param>
    public static void DecodeIndices(
        ref OpusRangeDecoderGpuState state,
        ArrayView<byte> buf, int bufStart, uint storage,
        ArrayView<byte> pitchDeltaIcdf, long pitchDeltaBase,
        ArrayView<byte> pitchLagIcdf, long pitchLagBase,
        ArrayView<byte> lagLowBitsIcdf, long lagLowBitsBase,
        ArrayView<byte> contourIcdf, long contourBase,
        int fsKHz, int prevLagIndex, int prevSignalTypeWasVoiced,
        int conditional,
        ArrayView<int> output, long outputBase)
    {
        int lagIndex = 0;
        int decodeAbsolute = 1;

        if (conditional != 0 && prevSignalTypeWasVoiced != 0)
        {
            int rawDelta = OpusRangeDecoderGpu.DecodeIcdf(
                ref state, buf, bufStart, storage,
                pitchDeltaIcdf, pitchDeltaBase, 8);
            if (rawDelta > 0)
            {
                int delta = rawDelta - 9;
                lagIndex = prevLagIndex + delta;
                decodeAbsolute = 0;
            }
        }

        if (decodeAbsolute != 0)
        {
            int coarse = OpusRangeDecoderGpu.DecodeIcdf(
                ref state, buf, bufStart, storage,
                pitchLagIcdf, pitchLagBase, 8);
            int lsb = OpusRangeDecoderGpu.DecodeIcdf(
                ref state, buf, bufStart, storage,
                lagLowBitsIcdf, lagLowBitsBase, 8);
            lagIndex = coarse * (fsKHz >> 1) + lsb;
        }

        int contour = OpusRangeDecoderGpu.DecodeIcdf(
            ref state, buf, bufStart, storage,
            contourIcdf, contourBase, 8);

        output[outputBase + 0] = lagIndex;
        output[outputBase + 1] = contour;
    }
}
