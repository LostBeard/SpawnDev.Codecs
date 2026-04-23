// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Structural port of the scalar side-information blocks of libopus
// silk/decode_indices.c to clean C#. The larger blocks (gain indices,
// NLSF indices) live on their dedicated decoder classes; this file holds
// the small stateless scalars (signal type + quantizer offset, PRNG seed).
//
// Upstream Copyright (c) 2006-2011 Skype Limited. BSD 3-Clause. See NOTICE.md.

using SpawnDev.Codecs.EntropyCoders;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>Decoded signal-type + quantizer-offset pair for a single SILK frame.</summary>
internal readonly record struct SilkSignalTypeOffset(sbyte SignalType, sbyte QuantOffsetType);

/// <summary>
/// Decoders for the small scalar side-information fields in a SILK frame:
/// signal type + quantizer offset, and the PRNG seed.
/// </summary>
internal static class SilkSideInfoDecoder
{
    /// <summary>SILK signal type: inactive.</summary>
    internal const sbyte TypeInactive = 0;
    /// <summary>SILK signal type: unvoiced.</summary>
    internal const sbyte TypeUnvoiced = 1;
    /// <summary>SILK signal type: voiced.</summary>
    internal const sbyte TypeVoiced = 2;

    /// <summary>
    /// Decode the combined signal-type / quantizer-offset index.
    /// When <paramref name="useVadTable"/> is true (VAD flag set, or decoding LBRR),
    /// reads from <see cref="SilkIcdfTables.TypeOffsetVad"/> and adds 2 to the raw
    /// symbol (mapping it into signalType 1 or 2). Otherwise reads from
    /// <see cref="SilkIcdfTables.TypeOffsetNoVad"/> (signalType 0).
    /// </summary>
    /// <param name="rangeDec">Range decoder positioned at the signal-type field.</param>
    /// <param name="useVadTable">Whether to read the 4-symbol VAD iCDF or the 2-symbol no-VAD iCDF.</param>
    /// <returns>Decoded signal type (0, 1, or 2) and quantizer offset type (0 or 1).</returns>
    internal static SilkSignalTypeOffset DecodeSignalType(OpusRangeDecoder rangeDec, bool useVadTable)
    {
        if (rangeDec is null) throw new ArgumentNullException(nameof(rangeDec));

        int ix;
        if (useVadTable)
        {
            ix = rangeDec.DecodeIcdf(SilkIcdfTables.TypeOffsetVad, 8) + 2;
        }
        else
        {
            ix = rangeDec.DecodeIcdf(SilkIcdfTables.TypeOffsetNoVad, 8);
        }
        return new SilkSignalTypeOffset((sbyte)(ix >> 1), (sbyte)(ix & 1));
    }

    /// <summary>
    /// Decode the 2-bit PRNG seed used by <c>silk_decode_core</c> to drive the
    /// sign-scrambling of the unsigned pulse magnitudes. Reads a single symbol
    /// from <see cref="SilkIcdfTables.Uniform4"/>.
    /// </summary>
    /// <param name="rangeDec">Range decoder positioned at the seed field.</param>
    /// <returns>Seed in <c>[0, 3]</c>.</returns>
    internal static sbyte DecodeSeed(OpusRangeDecoder rangeDec)
    {
        if (rangeDec is null) throw new ArgumentNullException(nameof(rangeDec));
        return (sbyte)rangeDec.DecodeIcdf(SilkIcdfTables.Uniform4, 8);
    }
}
