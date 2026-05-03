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

    /// <summary>
    /// Expand the decoded <c>(lagIndex, contourIndex)</c> pair into <paramref name="nbSubfr"/>
    /// per-subframe pitch lags, matching libopus <c>silk_decode_pitch</c>. The resulting
    /// lags are clamped to <c>[PE_MIN_LAG_MS * fsKHz, PE_MAX_LAG_MS * fsKHz]</c>.
    /// </summary>
    /// <param name="pitchLags">Output: <paramref name="nbSubfr"/> pitch lag values in samples.</param>
    /// <param name="lagIndex">Decoded coarse lag index (from <see cref="SilkPitchIndices.LagIndex"/>).</param>
    /// <param name="contourIndex">Decoded contour index (from <see cref="SilkPitchIndices.ContourIndex"/>).</param>
    /// <param name="fsKHz">Internal SILK sample rate (8, 12, or 16).</param>
    /// <param name="nbSubfr">Subframe count (2 or 4).</param>
    internal static void ComputeLags(
        Span<int> pitchLags,
        short lagIndex,
        sbyte contourIndex,
        int fsKHz,
        int nbSubfr)
    {
        if (fsKHz != 8 && fsKHz != 12 && fsKHz != 16)
            throw new ArgumentException($"Unsupported fs_kHz: {fsKHz}.", nameof(fsKHz));
        if (nbSubfr != 2 && nbSubfr != 4)
            throw new ArgumentException($"nbSubfr must be 2 or 4, got {nbSubfr}.", nameof(nbSubfr));
        if (pitchLags.Length < nbSubfr)
            throw new ArgumentException($"pitchLags too small (need {nbSubfr}).", nameof(pitchLags));

        sbyte[] cb;
        int cbSize;
        if (fsKHz == 8)
        {
            if (nbSubfr == SilkConstants.PE_MAX_NB_SUBFR)
            {
                cb = SilkPitchContourTables.Stage2;
                cbSize = SilkConstants.PE_NB_CBKS_STAGE2_EXT;
            }
            else
            {
                cb = SilkPitchContourTables.Stage210Ms;
                cbSize = SilkConstants.PE_NB_CBKS_STAGE2_10MS;
            }
        }
        else
        {
            if (nbSubfr == SilkConstants.PE_MAX_NB_SUBFR)
            {
                cb = SilkPitchContourTables.Stage3;
                cbSize = SilkConstants.PE_NB_CBKS_STAGE3_MAX;
            }
            else
            {
                cb = SilkPitchContourTables.Stage310Ms;
                cbSize = SilkConstants.PE_NB_CBKS_STAGE3_10MS;
            }
        }

        if ((uint)contourIndex >= (uint)cbSize)
            throw new ArgumentOutOfRangeException(nameof(contourIndex),
                $"contourIndex {contourIndex} out of range [0, {cbSize}) for fsKHz={fsKHz}, nbSubfr={nbSubfr}.");

        int minLag = SilkConstants.PE_MIN_LAG_MS * fsKHz;
        int maxLag = SilkConstants.PE_MAX_LAG_MS * fsKHz;
        int baseLag = minLag + lagIndex;

        for (int k = 0; k < nbSubfr; k++)
        {
            int lag = baseLag + cb[k * cbSize + contourIndex];
            if (lag < minLag) lag = minLag;
            else if (lag > maxLag) lag = maxLag;
            pitchLags[k] = lag;
        }
    }
}
