// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Structural port of libopus silk/log2lin.c and silk/lin2log.c to clean C#.
//
// Upstream Copyright (c) 2006-2011 Skype Limited. BSD 3-Clause. See NOTICE.md.
//
// Both functions use a piece-wise parabolic approximation per libopus; they are
// near-inverses of each other (small error due to the approximation, not exact).
// Used throughout SILK for gain dequantization, LTP scale encoding, and energy
// manipulation.

using static SpawnDev.Codecs.Audio.Opus.Silk.SilkMacros;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// SILK logarithm/linear conversions. Near-inverse pair for log2-based scaling
/// used internally by SILK gain and LTP processing.
/// </summary>
internal static class SilkLog2
{
    /// <summary>
    /// Approximation of <c>2^()</c>. Very close inverse of <see cref="silk_lin2log"/>.
    /// Converts a Q7 log-scale value to a linear 32-bit value.
    /// </summary>
    /// <param name="inLogQ7">Input on log scale (Q7 format).</param>
    /// <returns>
    /// Linear value. Returns 0 for negative input, <see cref="SilkMacros.silk_int32_MAX"/>
    /// when input >= 3967 (31 in Q7).
    /// </returns>
    internal static int silk_log2lin(int inLogQ7)
    {
        if (inLogQ7 < 0) return 0;
        if (inLogQ7 >= SilkConstants.GAIN_LOG_CLAMP_HIGH_Q7) return silk_int32_MAX;

        int @out = silk_LSHIFT(1, silk_RSHIFT(inLogQ7, 7));
        int fracQ7 = inLogQ7 & 0x7F;

        if (inLogQ7 < 2048)
        {
            // Piece-wise parabolic approximation (low range).
            @out = silk_ADD_RSHIFT32(
                @out,
                silk_MUL(@out, silk_SMLAWB(fracQ7, silk_SMULBB(fracQ7, 128 - fracQ7), -174)),
                7);
        }
        else
        {
            // Piece-wise parabolic approximation (high range).
            @out = silk_MLA(
                @out,
                silk_RSHIFT(@out, 7),
                silk_SMLAWB(fracQ7, silk_SMULBB(fracQ7, 128 - fracQ7), -174));
        }
        return @out;
    }

    /// <summary>
    /// Approximation of <c>128 * log2()</c>. Very close inverse of <see cref="silk_log2lin"/>.
    /// Converts a linear 32-bit value to a Q7 log-scale value.
    /// </summary>
    /// <param name="inLin">Input in linear scale.</param>
    /// <returns>Log value in Q7 format.</returns>
    internal static int silk_lin2log(int inLin)
    {
        silk_CLZ_FRAC(inLin, out int lz, out int fracQ7);
        // Piece-wise parabolic approximation.
        return silk_ADD_LSHIFT32(
            silk_SMLAWB(fracQ7, silk_MUL(fracQ7, 128 - fracQ7), 179),
            31 - lz,
            7);
    }
}
