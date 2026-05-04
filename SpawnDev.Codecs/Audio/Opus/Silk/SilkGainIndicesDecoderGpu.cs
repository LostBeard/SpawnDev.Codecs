// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable port of SilkGainDecoder.DecodeIndices. Reads per-subframe
// gain indices from the libopus range-coded bitstream via
// OpusRangeDecoderGpu.DecodeIcdf.
//
// Sequential per-stream because each subframe's index decode advances the
// shared range decoder state. One thread per stream; multi-channel decode
// parallelizes across threads.
//
// Upstream Copyright (c) 2006-2011 Skype Limited. BSD 3-Clause.

using ILGPU;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// GPU-callable decoder for SILK gain indices. Mirror of
/// `SilkGainDecoder.DecodeIndices` (CPU reference in
/// SpawnDev.Codecs.References).
/// </summary>
public static class SilkGainIndicesDecoderGpu
{
    /// <summary>Number of rows in the per-signal-type Gain iCDF table.</summary>
    public const int GainIcdfNumTypes = 3;
    /// <summary>Entries per row in the Gain iCDF table.</summary>
    public const int GainIcdfEntriesPerType = 8;

    /// <summary>
    /// Decode <paramref name="nbSubfr"/> per-subframe gain indices.
    /// Writes into <paramref name="indicesOut"/>[0..nbSubfr).
    /// </summary>
    /// <param name="state">Range decoder state (advanced in place).</param>
    /// <param name="buf">Encoded packet buffer.</param>
    /// <param name="bufStart">Offset of the packet in <paramref name="buf"/>.</param>
    /// <param name="storage">Length of the packet in bytes.</param>
    /// <param name="gainIcdf">Flat 24-entry `silk_gain_iCDF` (3 signal types × 8 entries each).
    /// Caller passes the full table; the offset is computed via
    /// `signalType * GainIcdfEntriesPerType`.</param>
    /// <param name="gainIcdfBase">Offset into <paramref name="gainIcdf"/>.</param>
    /// <param name="deltaGainIcdf">41-entry `silk_delta_gain_iCDF`.</param>
    /// <param name="deltaGainBase">Offset into <paramref name="deltaGainIcdf"/>.</param>
    /// <param name="uniform8Icdf">8-symbol `silk_uniform8_iCDF` (used for LSB on independent
    /// gain coding of the first subframe).</param>
    /// <param name="uniform8Base">Offset into <paramref name="uniform8Icdf"/>.</param>
    /// <param name="signalType">SILK signal type (0 inactive, 1 unvoiced, 2 voiced). Only used
    /// when <paramref name="conditional"/> is 0.</param>
    /// <param name="conditional">0 for independent coding; non-zero for conditional (delta) coding.</param>
    /// <param name="nbSubfr">Subframe count (2 for 10ms frames, 4 for 20ms frames).</param>
    /// <param name="indicesOut">Output buffer for the decoded indices (sbyte values
    /// in [0,63] per libopus, packed into ints here for ILGPU compatibility).
    /// Length must be >= <paramref name="nbSubfr"/>.</param>
    /// <param name="indicesOutBase">Offset into <paramref name="indicesOut"/>.</param>
    public static void DecodeIndices(
        ref OpusRangeDecoderGpuState state,
        ArrayView<byte> buf, int bufStart, uint storage,
        ArrayView<byte> gainIcdf, long gainIcdfBase,
        ArrayView<byte> deltaGainIcdf, long deltaGainBase,
        ArrayView<byte> uniform8Icdf, long uniform8Base,
        int signalType, int conditional, int nbSubfr,
        ArrayView<int> indicesOut, long indicesOutBase)
    {
        int first;
        if (conditional != 0)
        {
            first = OpusRangeDecoderGpu.DecodeIcdf(
                ref state, buf, bufStart, storage,
                deltaGainIcdf, deltaGainBase, 8);
        }
        else
        {
            long signalTypeOffset = (long)signalType * GainIcdfEntriesPerType;
            int msb = OpusRangeDecoderGpu.DecodeIcdf(
                ref state, buf, bufStart, storage,
                gainIcdf, gainIcdfBase + signalTypeOffset, 8);
            int lsb = OpusRangeDecoderGpu.DecodeIcdf(
                ref state, buf, bufStart, storage,
                uniform8Icdf, uniform8Base, 8);
            first = (msb << 3) + lsb;
        }
        indicesOut[indicesOutBase + 0] = first;

        for (int i = 1; i < nbSubfr; i++)
        {
            indicesOut[indicesOutBase + i] = OpusRangeDecoderGpu.DecodeIcdf(
                ref state, buf, bufStart, storage,
                deltaGainIcdf, deltaGainBase, 8);
        }
    }
}
