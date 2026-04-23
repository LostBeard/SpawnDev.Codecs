// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Structural port of libopus silk/gain_quant.c::silk_gains_dequant and the
// gain-index decoding portion of silk/decode_indices.c to clean C#.
//
// Upstream Copyright (c) 2006-2011 Skype Limited. BSD 3-Clause. See NOTICE.md.

using SpawnDev.Codecs.EntropyCoders;
using static SpawnDev.Codecs.Audio.Opus.Silk.SilkMacros;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// SILK scalar gain dequantization. Converts per-subframe gain indices recovered
/// from the range-coded bitstream into linear gains in Q16 format.
///
/// Operates on a delta-coded index stream with hysteresis: the first subframe's
/// index may be fully coded (conditional == 0) or delta-coded from the previous
/// frame's last index. Subsequent subframes in the same frame are always delta.
/// </summary>
internal static class SilkGainDecoder
{
    /// <summary>
    /// Dequantize <paramref name="nbSubfr"/> gain indices from <paramref name="ind"/>
    /// into linear gains in <paramref name="gainQ16"/> (Q16 format).
    /// </summary>
    /// <param name="gainQ16">Output buffer, must have length >= <paramref name="nbSubfr"/>.</param>
    /// <param name="ind">Input gain indices (as decoded from the bitstream), length >= <paramref name="nbSubfr"/>.</param>
    /// <param name="prevInd">In/out: last index from the previous frame. Updated to the last index produced by this call.</param>
    /// <param name="conditional">If 1, the first gain is delta-coded from <paramref name="prevInd"/>; if 0, it is a full index.</param>
    /// <param name="nbSubfr">Number of subframes to dequantize (typically <c>MAX_NB_SUBFR</c> = 4 or <c>MIN_NB_SUBFR</c> = 2).</param>
    internal static void Dequantize(
        Span<int> gainQ16,
        ReadOnlySpan<sbyte> ind,
        ref sbyte prevInd,
        int conditional,
        int nbSubfr)
    {
        if (nbSubfr <= 0 || nbSubfr > SilkConstants.MAX_NB_SUBFR)
            throw new ArgumentOutOfRangeException(nameof(nbSubfr));
        if (gainQ16.Length < nbSubfr) throw new ArgumentException("gainQ16 too small.", nameof(gainQ16));
        if (ind.Length < nbSubfr) throw new ArgumentException("ind too small.", nameof(ind));

        for (int k = 0; k < nbSubfr; k++)
        {
            if (k == 0 && conditional == 0)
            {
                // Gain index is not allowed to go down more than 16 steps (~21.8 dB).
                prevInd = (sbyte)silk_max_int(ind[k], prevInd - 16);
            }
            else
            {
                // Delta index.
                int indTmp = ind[k] + SilkConstants.MIN_DELTA_GAIN_QUANT;

                // Accumulate deltas.
                int doubleStepSizeThreshold =
                    2 * SilkConstants.MAX_DELTA_GAIN_QUANT - SilkConstants.N_LEVELS_QGAIN + prevInd;

                if (indTmp > doubleStepSizeThreshold)
                {
                    prevInd = (sbyte)(prevInd + silk_LSHIFT(indTmp, 1) - doubleStepSizeThreshold);
                }
                else
                {
                    prevInd = (sbyte)(prevInd + indTmp);
                }
            }

            prevInd = (sbyte)silk_LIMIT_int(prevInd, 0, SilkConstants.N_LEVELS_QGAIN - 1);

            // Scale and convert to linear scale via log2lin.
            int inLogQ7 = silk_min_32(
                silk_SMULWB(SilkConstants.GAIN_INV_SCALE_Q16, prevInd) + SilkConstants.GAIN_OFFSET_Q7,
                SilkConstants.GAIN_LOG_CLAMP_HIGH_Q7);
            gainQ16[k] = SilkLog2.silk_log2lin(inLogQ7);
        }
    }

    /// <summary>
    /// Read <paramref name="nbSubfr"/> gain indices from <paramref name="rangeDec"/>, matching
    /// the gain-index block in libopus <c>silk_decode_indices</c>. The first index is either
    /// independent (8-symbol MSB <c>silk_gain_iCDF[signalType]</c> shifted left by 3, plus an
    /// 8-symbol uniform LSB) when <paramref name="conditional"/> == 0, or delta-coded when it
    /// is non-zero. Subsequent indices are always delta-coded from the 41-symbol
    /// <c>silk_delta_gain_iCDF</c>.
    /// </summary>
    /// <param name="indices">Output buffer for gain indices; length must be &gt;= <paramref name="nbSubfr"/>.</param>
    /// <param name="rangeDec">Range decoder positioned at the start of the gain-index block.</param>
    /// <param name="signalType">SILK signal type (0 inactive, 1 unvoiced, 2 voiced). Only used when
    /// <paramref name="conditional"/> is 0.</param>
    /// <param name="conditional">0 for independent coding (first frame of a packet, or after a VAD
    /// boundary), non-zero for conditional (delta) coding.</param>
    /// <param name="nbSubfr">Subframe count - 2 for 10 ms frames, 4 for 20 ms frames.</param>
    internal static void DecodeIndices(
        Span<sbyte> indices,
        OpusRangeDecoder rangeDec,
        int signalType,
        int conditional,
        int nbSubfr)
    {
        if (rangeDec is null) throw new ArgumentNullException(nameof(rangeDec));
        if (nbSubfr <= 0 || nbSubfr > SilkConstants.MAX_NB_SUBFR)
            throw new ArgumentOutOfRangeException(nameof(nbSubfr));
        if (indices.Length < nbSubfr) throw new ArgumentException("indices too small.", nameof(indices));
        if ((uint)signalType >= SilkIcdfTables.GainIcdfNumTypes)
            throw new ArgumentOutOfRangeException(nameof(signalType), $"signalType must be in [0, {SilkIcdfTables.GainIcdfNumTypes - 1}].");

        int first;
        if (conditional != 0)
        {
            first = rangeDec.DecodeIcdf(SilkIcdfTables.DeltaGain, 8);
        }
        else
        {
            int gainIcdfStart = SilkIcdfTables.GainIcdfOffset(signalType);
            int msb = rangeDec.DecodeIcdf(
                SilkIcdfTables.Gain.AsSpan(gainIcdfStart, SilkIcdfTables.GainIcdfEntriesPerType),
                8);
            int lsb = rangeDec.DecodeIcdf(SilkIcdfTables.Uniform8, 8);
            first = (msb << 3) + lsb;
        }
        indices[0] = (sbyte)first;

        for (int i = 1; i < nbSubfr; i++)
        {
            indices[i] = (sbyte)rangeDec.DecodeIcdf(SilkIcdfTables.DeltaGain, 8);
        }
    }
}
