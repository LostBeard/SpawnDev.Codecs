// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Structural port of libopus silk/gain_quant.c::silk_gains_dequant to clean C#.
//
// Upstream Copyright (c) 2006-2011 Skype Limited. BSD 3-Clause. See NOTICE.md.

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
}
