// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Structural port of the pitch-decode block of libopus silk/decode_indices.c to
// clean C#. Handles both delta-coded (relative to previous frame) and absolute
// pitch lag indices, plus the pitch-contour index.
//
// Upstream Copyright (c) 2006-2011 Skype Limited. BSD 3-Clause. See NOTICE.md.

using SpawnDev.Codecs.EntropyCoders;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>Decoded pitch parameters for a voiced SILK frame.</summary>
internal readonly record struct SilkPitchIndices(short LagIndex, sbyte ContourIndex);

/// <summary>
/// Decodes pitch indices (lag + contour) for a voiced SILK frame. Called only
/// when <c>signalType == TYPE_VOICED</c>. Matches the voiced-pitch block in
/// libopus <c>silk_decode_indices</c>.
/// </summary>
internal static class SilkPitchDecoder
{
    /// <summary>
    /// Read the pitch lag and contour indices from the bitstream.
    /// <para>
    /// When <paramref name="conditional"/> is non-zero AND
    /// <paramref name="prevSignalTypeWasVoiced"/> is true, the decoder first tries a
    /// delta-coded lag: it reads <see cref="SilkIcdfTables.PitchDelta"/>; a raw value
    /// of 0 falls through to absolute coding, otherwise <c>delta = raw - 9</c> is
    /// applied to <paramref name="prevLagIndex"/>.
    /// </para>
    /// <para>
    /// Absolute coding reads a coarse lag from <see cref="SilkIcdfTables.PitchLag"/>
    /// (multiplied by <c>fs_kHz / 2</c>) plus a sample-rate-dependent LSB from
    /// <see cref="SilkIcdfTables.SelectPitchLagLowBits"/>. The contour is then read
    /// from <see cref="SilkIcdfTables.SelectPitchContour"/>.
    /// </para>
    /// </summary>
    internal static SilkPitchIndices DecodeIndices(
        OpusRangeDecoder rangeDec,
        int fsKHz,
        int nbSubfr,
        short prevLagIndex,
        bool prevSignalTypeWasVoiced,
        int conditional)
    {
        if (rangeDec is null) throw new ArgumentNullException(nameof(rangeDec));
        if (fsKHz != 8 && fsKHz != 12 && fsKHz != 16)
            throw new ArgumentException($"Unsupported fs_kHz: {fsKHz}.", nameof(fsKHz));
        if (nbSubfr != 2 && nbSubfr != 4)
            throw new ArgumentException($"nbSubfr must be 2 or 4, got {nbSubfr}.", nameof(nbSubfr));

        short lagIndex;
        bool decodeAbsolute = true;

        if (conditional != 0 && prevSignalTypeWasVoiced)
        {
            int rawDelta = rangeDec.DecodeIcdf(SilkIcdfTables.PitchDelta, 8);
            if (rawDelta > 0)
            {
                int delta = rawDelta - 9;
                lagIndex = (short)(prevLagIndex + delta);
                decodeAbsolute = false;
            }
            else
            {
                // raw == 0 signals "switch to absolute"; fall through.
                lagIndex = 0;
            }
        }
        else
        {
            lagIndex = 0;
        }

        if (decodeAbsolute)
        {
            int coarse = rangeDec.DecodeIcdf(SilkIcdfTables.PitchLag, 8);
            int lsb = rangeDec.DecodeIcdf(SilkIcdfTables.SelectPitchLagLowBits(fsKHz), 8);
            lagIndex = (short)(coarse * (fsKHz >> 1) + lsb);
        }

        sbyte contour = (sbyte)rangeDec.DecodeIcdf(
            SilkIcdfTables.SelectPitchContour(fsKHz, nbSubfr), 8);

        return new SilkPitchIndices(lagIndex, contour);
    }
}
