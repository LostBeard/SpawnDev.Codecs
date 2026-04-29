// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// SILK log2/lin conversions, GPU-callable form. Bit-exact mirror of
// SilkLog2.silk_log2lin / silk_lin2log. Inlines the SILK macros
// (silk_LSHIFT, silk_RSHIFT, silk_MUL, silk_SMLAWB, silk_SMULBB,
// silk_MLA, silk_ADD_RSHIFT32, silk_ADD_LSHIFT32, silk_CLZ_FRAC,
// silk_CLZ32) as scalar math so the helper has no SILK-specific
// dependencies.
//
// Used by SILK gain dequantization, LTP scale encoding, and energy
// manipulation. Pure scalar math - no ArrayView access, no allocations,
// no exceptions.

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// GPU-callable SILK <c>silk_log2lin</c> + <c>silk_lin2log</c>. Pure
/// scalar math. Bit-exact mirror of <see cref="SilkLog2"/>.
/// </summary>
public static class SilkLog2Gpu
{
    private const int Int32Max = int.MaxValue;
    private const int GainLogClampHighQ7 = 3967; // SilkConstants.GAIN_LOG_CLAMP_HIGH_Q7

    /// <summary>
    /// Approximation of <c>2^()</c>. Near-inverse of
    /// <see cref="Lin2LogQ7"/>. Converts a Q7 log-scale value to a
    /// linear 32-bit value. Returns 0 for negative input;
    /// <see cref="int.MaxValue"/> when input >= 3967 (31 in Q7).
    /// </summary>
    public static int Log2LinQ7(int inLogQ7)
    {
        if (inLogQ7 < 0) return 0;
        if (inLogQ7 >= GainLogClampHighQ7) return Int32Max;

        // out = 1 << (inLogQ7 >> 7) - the integer part.
        int @out = 1 << (inLogQ7 >> 7);
        int fracQ7 = inLogQ7 & 0x7F;

        if (inLogQ7 < 2048)
        {
            // Piece-wise parabolic approximation (low range).
            // out = out + ((out * (fracQ7 + ((fracQ7 * (128 - fracQ7) >> 0) * -174 >> 16))) >> 7)
            // libopus: silk_ADD_RSHIFT32(out, silk_MUL(out, silk_SMLAWB(fracQ7, silk_SMULBB(fracQ7, 128 - fracQ7), -174)), 7)
            int smulbb = (short)fracQ7 * (short)(128 - fracQ7);
            // silk_SMLAWB(a, b, c) = a + (int)((long)b * (short)c >> 16)
            int smlawb = fracQ7 + (int)((long)smulbb * (short)(-174) >> 16);
            // silk_MUL(out, smlawb)
            int mul = @out * smlawb;
            // silk_ADD_RSHIFT32(out, mul, 7)
            @out = @out + (mul >> 7);
        }
        else
        {
            // Piece-wise parabolic approximation (high range).
            // libopus: silk_MLA(out, silk_RSHIFT(out, 7), silk_SMLAWB(fracQ7, silk_SMULBB(fracQ7, 128 - fracQ7), -174))
            int smulbb = (short)fracQ7 * (short)(128 - fracQ7);
            int smlawb = fracQ7 + (int)((long)smulbb * (short)(-174) >> 16);
            // silk_MLA(a, b, c) = a + b * c
            @out = @out + (@out >> 7) * smlawb;
        }
        return @out;
    }

    /// <summary>
    /// Approximation of <c>128 * log2()</c>. Near-inverse of
    /// <see cref="Log2LinQ7"/>. Converts a linear 32-bit value to
    /// a Q7 log-scale value.
    /// </summary>
    public static int Lin2LogQ7(int inLin)
    {
        // silk_CLZ_FRAC(inVal, out lz, out fracQ7)
        int lz = LeadingZeroCount32((uint)inLin);
        // Shift inVal left by lz so MSB is bit 31, then take bits 30..24
        // (the 7 bits just below the MSB).
        int fracQ7 = (int)(((uint)inLin << lz >> 24) & 0x7F);

        // Piece-wise parabolic approximation.
        // libopus: silk_ADD_LSHIFT32(silk_SMLAWB(fracQ7, silk_MUL(fracQ7, 128 - fracQ7), 179), 31 - lz, 7)
        int smlawb = fracQ7 + (int)((long)(fracQ7 * (128 - fracQ7)) * (short)179 >> 16);
        // silk_ADD_LSHIFT32(a, b, shift) = a + (b << shift)
        return smlawb + ((31 - lz) << 7);
    }

    /// <summary>
    /// Count leading zeros of a 32-bit unsigned value. Returns 32 for
    /// input 0 (matches BitOperations.LeadingZeroCount + libopus
    /// silk_CLZ32 behavior).
    /// </summary>
    private static int LeadingZeroCount32(uint x)
    {
        if (x == 0) return 32;
        int n = 0;
        if ((x & 0xFFFF0000u) == 0) { n += 16; x <<= 16; }
        if ((x & 0xFF000000u) == 0) { n += 8; x <<= 8; }
        if ((x & 0xF0000000u) == 0) { n += 4; x <<= 4; }
        if ((x & 0xC0000000u) == 0) { n += 2; x <<= 2; }
        if ((x & 0x80000000u) == 0) { n += 1; }
        return n;
    }
}
