// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Structural port of the LTP-index block of libopus silk/decode_indices.c to
// clean C#. Decodes the periodicity (codebook-select) index, per-subframe LTP
// gain indices, and the conditional-coding-gated LTP scale index.
//
// Upstream Copyright (c) 2006-2011 Skype Limited. BSD 3-Clause. See NOTICE.md.

using SpawnDev.Codecs.EntropyCoders;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// Decodes the LTP index block for a voiced SILK frame. Called after the pitch
/// indices (lag + contour), only when the frame is voiced.
/// </summary>
internal static class SilkLtpDecoder
{
    /// <summary>
    /// Read the LTP indices from the bitstream.
    /// <list type="number">
    /// <item>PERIndex: a 3-symbol code selecting one of the three LTP gain codebooks.</item>
    /// <item>Per-subframe LTP gain index: codebook-dependent (8, 16, or 32 symbols) for each of <paramref name="nbSubfr"/> subframes.</item>
    /// <item>LTP scale index: 3-symbol iCDF, only when <paramref name="conditional"/> is 0. Otherwise the scale index is fixed to 0.</item>
    /// </list>
    /// </summary>
    /// <param name="ltpIndices">Output: per-subframe LTP gain indices. Length &gt;= <paramref name="nbSubfr"/>.</param>
    /// <param name="rangeDec">Range decoder positioned at the LTP block.</param>
    /// <param name="conditional">0 for independent coding (enables LTP scale read), non-zero for conditional.</param>
    /// <param name="nbSubfr">Subframe count (2 or 4).</param>
    /// <param name="perIndex">Out: the PERIndex (0, 1, or 2).</param>
    /// <param name="ltpScaleIndex">Out: the LTP scale index (0..2). 0 when <paramref name="conditional"/> != 0.</param>
    internal static void DecodeIndices(
        Span<sbyte> ltpIndices,
        OpusRangeDecoder rangeDec,
        int conditional,
        int nbSubfr,
        out sbyte perIndex,
        out sbyte ltpScaleIndex)
    {
        if (rangeDec is null) throw new ArgumentNullException(nameof(rangeDec));
        if (nbSubfr != 2 && nbSubfr != 4)
            throw new ArgumentException($"nbSubfr must be 2 or 4, got {nbSubfr}.", nameof(nbSubfr));
        if (ltpIndices.Length < nbSubfr)
            throw new ArgumentException($"ltpIndices too small (need {nbSubfr}).", nameof(ltpIndices));

        perIndex = (sbyte)rangeDec.DecodeIcdf(SilkIcdfTables.LtpPerIndex, 8);

        byte[] gainIcdf = SilkIcdfTables.SelectLtpGain(perIndex);
        for (int k = 0; k < nbSubfr; k++)
        {
            ltpIndices[k] = (sbyte)rangeDec.DecodeIcdf(gainIcdf, 8);
        }

        if (conditional == 0)
        {
            ltpScaleIndex = (sbyte)rangeDec.DecodeIcdf(SilkIcdfTables.LtpScale, 8);
        }
        else
        {
            ltpScaleIndex = 0;
        }
    }

    /// <summary>
    /// Retrieve the 5-tap Q7 LTP gain vector for a given <paramref name="perIndex"/> +
    /// <paramref name="ltpIndex"/> pair, writing it into <paramref name="taps"/>.
    /// Matches the per-subframe LTP filter lookup in libopus <c>silk_decode_parameters</c>.
    /// </summary>
    /// <param name="taps">Output: 5 signed Q7 taps.</param>
    /// <param name="perIndex">Periodicity index selecting which codebook (0, 1, or 2).</param>
    /// <param name="ltpIndex">Entry index within the selected codebook (0..cb_size-1).</param>
    internal static void GetGainVector(Span<sbyte> taps, int perIndex, int ltpIndex)
    {
        if (taps.Length < SilkLtpGainTables.LtpVecSize)
            throw new ArgumentException($"taps too small (need {SilkLtpGainTables.LtpVecSize}).", nameof(taps));

        sbyte[] cb = SilkLtpGainTables.Select(perIndex);
        int cbSize = cb.Length / SilkLtpGainTables.LtpVecSize;
        if ((uint)ltpIndex >= (uint)cbSize)
            throw new ArgumentOutOfRangeException(nameof(ltpIndex),
                $"ltpIndex {ltpIndex} out of range [0, {cbSize}) for perIndex {perIndex}.");

        int offset = ltpIndex * SilkLtpGainTables.LtpVecSize;
        for (int i = 0; i < SilkLtpGainTables.LtpVecSize; i++)
        {
            taps[i] = cb[offset + i];
        }
    }
}
