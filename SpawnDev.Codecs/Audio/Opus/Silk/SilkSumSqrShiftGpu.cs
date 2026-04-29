// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable silk_sum_sqr_shift. Mirror of SilkSumSqrShift.Compute
// for in-kernel use. Computes sum-of-squares with a dynamically chosen
// right-shift so the result fits in int32 with 2 bits of headroom.
// Used in SILK analysis paths for energy estimation.
//
// All required SILK macros (CLZ32, SMULBB, SMLABB_ovflw,
// ADD_RSHIFT_uint, max_32) are inlined to avoid host-helper
// dependencies inside the kernel.

using ILGPU;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// GPU-callable SILK sum-of-squares with dynamic right-shift.
/// Bit-exact mirror of <see cref="SilkSumSqrShift"/>.Compute.
/// </summary>
public static class SilkSumSqrShiftGpu
{
    /// <summary>
    /// Computes sum(x[i]^2) >> shift for the input vector, choosing
    /// shift automatically so the result fits in int32 with 2 bits of
    /// headroom.
    /// </summary>
    /// <param name="x">Input vector of int16 samples.</param>
    /// <param name="xBase">Starting offset in <paramref name="x"/>.</param>
    /// <param name="len">Number of samples.</param>
    /// <param name="energy">Output: accumulated sum-of-squares.</param>
    /// <param name="shift">Output: number of bits the sum was right-shifted by.</param>
    public static void Compute(
        ArrayView<short> x, long xBase, int len,
        out int energy, out int shift)
    {
        int shft = 31 - Clz32(len);
        uint nrg = (uint)len;

        int i;
        for (i = 0; i < len - 1; i += 2)
        {
            int s0 = x[xBase + i];
            int s1 = x[xBase + i + 1];
            uint nrgTmp = (uint)(s0 * s0);
            nrgTmp = (uint)((int)nrgTmp + s1 * s1);
            nrg = AddRShiftUint(nrg, nrgTmp, shft);
        }
        if (i < len)
        {
            int s0 = x[xBase + i];
            uint nrgTmp = (uint)(s0 * s0);
            nrg = AddRShiftUint(nrg, nrgTmp, shft);
        }

        int nrgSigned = (int)nrg;
        int max32 = shft + 3 - Clz32(nrgSigned);
        shft = max32 > 0 ? max32 : 0;

        nrg = 0;
        for (i = 0; i < len - 1; i += 2)
        {
            int s0 = x[xBase + i];
            int s1 = x[xBase + i + 1];
            uint nrgTmp = (uint)(s0 * s0);
            nrgTmp = (uint)((int)nrgTmp + s1 * s1);
            nrg = AddRShiftUint(nrg, nrgTmp, shft);
        }
        if (i < len)
        {
            int s0 = x[xBase + i];
            uint nrgTmp = (uint)(s0 * s0);
            nrg = AddRShiftUint(nrg, nrgTmp, shft);
        }

        shift = shft;
        energy = (int)nrg;
    }

    /// <summary>silk_ADD_RSHIFT_uint: a + (b >> shift).</summary>
    private static uint AddRShiftUint(uint a, uint b, int shift)
    {
        return a + (b >> shift);
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
