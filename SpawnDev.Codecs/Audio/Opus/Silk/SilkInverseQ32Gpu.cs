// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable silk_INVERSE32_varQ. Mirror of SilkMacros.silk_INVERSE32_varQ
// (libopus silk/Inlines.h). Newton-like 32-bit integer reciprocal in
// variable-Q format, used by silk_LPC_inverse_pred_gain_QA inside the
// SILK NLSF stability check + LPC fit paths.
//
// All silk macros (CLZ32, abs, LSHIFT, RSHIFT, DIV32_16, SMULWB,
// SMLAWW, LSHIFT_SAT32) are inlined to avoid host-helper dependencies
// inside the kernel.

using ILGPU;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// GPU-callable SILK 32-bit integer reciprocal in variable-Q format.
/// Mirror of <see cref="SilkMacros"/>.silk_INVERSE32_varQ.
/// </summary>
public static class SilkInverseQ32Gpu
{
    /// <summary>
    /// Compute (1 / b32) in Q(<paramref name="Qres"/>) using two
    /// Newton-like refinement steps. Caller must ensure b32 != 0 and
    /// Qres &gt; 0. Bit-exact mirror of the CPU
    /// <see cref="SilkMacros"/>.silk_INVERSE32_varQ.
    /// </summary>
    public static int Compute(int b32, int Qres)
    {
        int absB = b32 < 0 ? -b32 : b32;
        int bHeadrm = Clz32(absB) - 1;
        int b32Nrm = b32 << bHeadrm;

        // silk_DIV32_16: int / short
        int divisor = b32Nrm >> 16;
        int b32Inv = (int.MaxValue >> 2) / divisor;

        int result = b32Inv << 16;

        // silk_SMULWB: (a * (short)b) >> 16
        int errQ32_inner = (int)((long)b32Nrm * (short)b32Inv >> 16);
        int errQ32 = ((1 << 29) - errQ32_inner) << 3;

        // silk_SMLAWW(result, errQ32, b32Inv) = result + silk_SMULWW(errQ32, b32Inv)
        // silk_SMULWW(a, b) = SMULWB(a, b) + a * RSHIFT_ROUND(b, 16)
        // SMULWB(a, b) = (long)a * (short)b >> 16
        // RSHIFT_ROUND(b, 16) = (b + (1 << 15)) >> 16
        long smulwbInner = (long)errQ32 * (short)b32Inv;
        int smulwb = (int)(smulwbInner >> 16);
        int rshiftRound = (b32Inv + (1 << 15)) >> 16;
        result = result + smulwb + errQ32 * rshiftRound;

        int lshift = 61 - bHeadrm - Qres;
        if (lshift <= 0)
        {
            // silk_LSHIFT_SAT32: saturated left shift.
            int neg = -lshift;
            if (neg <= 0) return result;
            // Saturate if shifting would overflow.
            if (neg >= 32) return result < 0 ? int.MinValue : int.MaxValue;
            int max = int.MaxValue >> neg;
            int min = int.MinValue >> neg;
            if (result > max) return int.MaxValue;
            if (result < min) return int.MinValue;
            return result << neg;
        }
        if (lshift < 32) return result >> lshift;
        return 0;
    }

    /// <summary>silk_CLZ32: count leading zeros of a 32-bit signed int. Returns 32 for 0.</summary>
    private static int Clz32(int x)
    {
        if (x == 0) return 32;
        uint u = (uint)x;
        int n = 0;
        if ((u & 0xFFFF0000u) == 0) { n += 16; u <<= 16; }
        if ((u & 0xFF000000u) == 0) { n += 8; u <<= 8; }
        if ((u & 0xF0000000u) == 0) { n += 4; u <<= 4; }
        if ((u & 0xC0000000u) == 0) { n += 2; u <<= 2; }
        if ((u & 0x80000000u) == 0) { n += 1; }
        return n;
    }
}
