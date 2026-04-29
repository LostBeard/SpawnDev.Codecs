// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable silk_DIV32_varQ. Mirror of SilkMacros.silk_DIV32_varQ
// (libopus silk/Inlines.h). Variable-Q 32-bit division: returns a
// Q-Qres approximation to a32 / b32. Used widely throughout SILK
// (gain adjust, LTP scale, NLSF normalisation).
//
// All silk macros (CLZ32, abs, LSHIFT, RSHIFT, DIV32_16, SMULWB,
// SMMUL, SMLAWB, LSHIFT_SAT32, LSHIFT_ovflw, SUB32_ovflw) inlined
// here.

using ILGPU;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// GPU-callable SILK variable-Q 32-bit division. Mirror of
/// <see cref="SilkMacros"/>.silk_DIV32_varQ.
/// </summary>
public static class SilkDivVarQGpu
{
    /// <summary>
    /// Compute a32 / b32 in Q(<paramref name="Qres"/>). Caller must ensure
    /// b32 != 0 and Qres &gt;= 0. Bit-exact mirror of the CPU
    /// <see cref="SilkMacros"/>.silk_DIV32_varQ.
    /// </summary>
    public static int Compute(int a32, int b32, int Qres)
    {
        int aHeadrm = Clz32(Abs(a32)) - 1;
        int a32Nrm = unchecked(a32 << aHeadrm);

        int bHeadrm = Clz32(Abs(b32)) - 1;
        int b32Nrm = unchecked(b32 << bHeadrm);

        // silk_DIV32_16: int / short
        int divisor = b32Nrm >> 16;
        int b32Inv = (int.MaxValue >> 2) / divisor;

        // silk_SMULWB(a, b) = (int)((long)a * (short)b >> 16)
        int result = (int)((long)a32Nrm * (short)b32Inv >> 16);

        // a32Nrm = SUB32_ovflw(a32Nrm, LSHIFT_ovflw(SMMUL(b32Nrm, result), 3))
        // SMMUL(a, b) = ((long)a * b) >> 32
        long smmul = ((long)b32Nrm * result) >> 32;
        a32Nrm = unchecked(a32Nrm - unchecked((int)smmul << 3));

        // result = SMLAWB(result, a32Nrm, b32Inv)
        result = result + (int)((long)a32Nrm * (short)b32Inv >> 16);

        int lshift = 29 + aHeadrm - bHeadrm - Qres;
        if (lshift < 0)
        {
            int neg = -lshift;
            if (neg >= 32) return result < 0 ? int.MinValue : (result > 0 ? int.MaxValue : 0);
            int max = int.MaxValue >> neg;
            int min = int.MinValue >> neg;
            if (result > max) return int.MaxValue;
            if (result < min) return int.MinValue;
            return result << neg;
        }
        if (lshift < 32) return result >> lshift;
        return 0;
    }

    /// <summary>silk_abs: kernel-safe int absolute value (handles int.MinValue).</summary>
    private static int Abs(int x) => x < 0 ? unchecked(-x) : x;

    /// <summary>silk_CLZ32: count leading zeros of int. Returns 32 for 0.</summary>
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
