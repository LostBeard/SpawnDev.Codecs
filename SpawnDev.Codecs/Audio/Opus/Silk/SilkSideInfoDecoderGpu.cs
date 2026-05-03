// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable port of the small scalar side-information blocks of
// libopus silk/decode_indices.c. Mirrors the CPU SilkSideInfoDecoder
// (SpawnDev.Codecs.References/Audio/Opus/Silk/SilkSideInfoDecoder.cs)
// using OpusRangeDecoderGpu.DecodeIcdf. Used by the SILK decode
// integration kernel for the per-frame signal-type / quantizer-offset
// + PRNG seed reads.
//
// Upstream Copyright (c) 2006-2011 Skype Limited. BSD 3-Clause.

using ILGPU;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// GPU-callable decoders for the small scalar side-information fields
/// in a SILK frame: signal type + quantizer offset, and the PRNG seed.
/// Mirrors <see cref="SpawnDev.Codecs.Audio.Opus.Silk"/>'s CPU helper
/// `SilkSideInfoDecoder` bit-for-bit.
/// </summary>
public static class SilkSideInfoDecoderGpu
{
    /// <summary>SILK signal type: inactive.</summary>
    public const int TypeInactive = 0;
    /// <summary>SILK signal type: unvoiced.</summary>
    public const int TypeUnvoiced = 1;
    /// <summary>SILK signal type: voiced.</summary>
    public const int TypeVoiced = 2;

    /// <summary>
    /// Decode the combined signal-type / quantizer-offset index. When
    /// <paramref name="useVadTable"/> is true (VAD flag set, or LBRR
    /// decoding), reads from the 4-symbol VAD iCDF and adds 2 to the raw
    /// symbol (mapping it into signalType 1 or 2). Otherwise reads from
    /// the 2-symbol no-VAD iCDF (signalType 0).
    /// </summary>
    /// <param name="state">Range decoder state, advanced in place.</param>
    /// <param name="buf">Encoded packet buffer.</param>
    /// <param name="bufStart">Offset of the packet in <paramref name="buf"/>.</param>
    /// <param name="storage">Length of the packet in bytes.</param>
    /// <param name="typeOffsetVadIcdf">4-symbol VAD iCDF (libopus
    /// <c>silk_type_offset_VAD_iCDF</c>).</param>
    /// <param name="typeOffsetVadBase">Offset into
    /// <paramref name="typeOffsetVadIcdf"/>.</param>
    /// <param name="typeOffsetNoVadIcdf">2-symbol no-VAD iCDF (libopus
    /// <c>silk_type_offset_no_VAD_iCDF</c>).</param>
    /// <param name="typeOffsetNoVadBase">Offset into
    /// <paramref name="typeOffsetNoVadIcdf"/>.</param>
    /// <param name="useVadTable">Whether to read the 4-symbol VAD iCDF
    /// (true) or the 2-symbol no-VAD iCDF (false).</param>
    /// <param name="signalTypeOut">Receives the decoded signal type
    /// (0, 1, or 2).</param>
    /// <param name="quantOffsetTypeOut">Receives the decoded quantizer
    /// offset type (0 or 1).</param>
    public static void DecodeSignalType(
        ref OpusRangeDecoderGpuState state,
        ArrayView<byte> buf, int bufStart, uint storage,
        ArrayView<byte> typeOffsetVadIcdf, long typeOffsetVadBase,
        ArrayView<byte> typeOffsetNoVadIcdf, long typeOffsetNoVadBase,
        bool useVadTable,
        out int signalTypeOut,
        out int quantOffsetTypeOut)
    {
        int ix;
        if (useVadTable)
        {
            ix = OpusRangeDecoderGpu.DecodeIcdf(
                ref state, buf, bufStart, storage,
                typeOffsetVadIcdf, typeOffsetVadBase, 8) + 2;
        }
        else
        {
            ix = OpusRangeDecoderGpu.DecodeIcdf(
                ref state, buf, bufStart, storage,
                typeOffsetNoVadIcdf, typeOffsetNoVadBase, 8);
        }
        signalTypeOut = ix >> 1;
        quantOffsetTypeOut = ix & 1;
    }

    /// <summary>
    /// Decode the 2-bit PRNG seed used by <c>silk_decode_core</c> to
    /// drive the sign-scrambling of the unsigned pulse magnitudes.
    /// Reads a single symbol from the 4-symbol uniform iCDF.
    /// </summary>
    /// <param name="state">Range decoder state, advanced in place.</param>
    /// <param name="buf">Encoded packet buffer.</param>
    /// <param name="bufStart">Offset of the packet in <paramref name="buf"/>.</param>
    /// <param name="storage">Length of the packet in bytes.</param>
    /// <param name="uniform4Icdf">4-symbol uniform iCDF (libopus
    /// <c>silk_uniform4_iCDF</c>).</param>
    /// <param name="uniform4Base">Offset into <paramref name="uniform4Icdf"/>.</param>
    /// <returns>Seed in <c>[0, 3]</c>.</returns>
    public static int DecodeSeed(
        ref OpusRangeDecoderGpuState state,
        ArrayView<byte> buf, int bufStart, uint storage,
        ArrayView<byte> uniform4Icdf, long uniform4Base)
    {
        return OpusRangeDecoderGpu.DecodeIcdf(
            ref state, buf, bufStart, storage,
            uniform4Icdf, uniform4Base, 8);
    }
}
