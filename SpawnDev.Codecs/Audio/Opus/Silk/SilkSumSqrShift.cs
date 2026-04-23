// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Structural port of libopus silk/sum_sqr_shift.c to clean C#.
//
// Upstream Copyright (c) 2006-2011 Skype Limited. BSD 3-Clause. See NOTICE.md.
//
// Computes sum-of-squares with dynamic right-shift so the result fits in a
// 32-bit signed integer with two bits of headroom. The shift chosen depends
// on both the input length and the actual magnitude of the accumulated sum,
// discovered in two passes. Used in SILK analysis paths for energy estimation.

using static SpawnDev.Codecs.Audio.Opus.Silk.SilkMacros;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// Computes the sum of squares of a 16-bit integer vector with an automatically-
/// chosen right-shift that keeps the accumulator fitting in int32 with headroom.
/// </summary>
internal static class SilkSumSqrShift
{
    /// <summary>
    /// Computes <c>sum(x[i]^2) &gt;&gt; shift</c> for the input vector, choosing shift
    /// automatically so the result fits in int32 with 2 bits of headroom.
    /// </summary>
    /// <param name="x">Input vector.</param>
    /// <param name="energy">Output: accumulated sum of squares, after applying the chosen shift.</param>
    /// <param name="shift">Output: number of bits the sum of squares was right-shifted by.</param>
    internal static void Compute(ReadOnlySpan<short> x, out int energy, out int shift)
    {
        int len = x.Length;

        // Conservative starting shift based on input length alone.
        int shft = 31 - silk_CLZ32(len);
        // Seed the accumulator with `len` so the initial shift stays conservative.
        uint nrg = (uint)len;

        int i;
        for (i = 0; i < len - 1; i += 2)
        {
            uint nrgTmp = (uint)silk_SMULBB(x[i], x[i]);
            nrgTmp = (uint)silk_SMLABB_ovflw((int)nrgTmp, x[i + 1], x[i + 1]);
            nrg = silk_ADD_RSHIFT_uint(nrg, nrgTmp, shft);
        }
        if (i < len)
        {
            // One sample left.
            uint nrgTmp = (uint)silk_SMULBB(x[i], x[i]);
            nrg = silk_ADD_RSHIFT_uint(nrg, nrgTmp, shft);
        }

        // Now nrg >= 0 (cast back to signed). Tighten the shift to fit with 2 bits of headroom.
        int nrgSigned = (int)nrg;
        shft = silk_max_32(0, shft + 3 - silk_CLZ32(nrgSigned));

        // Second pass with the refined shift.
        nrg = 0;
        for (i = 0; i < len - 1; i += 2)
        {
            uint nrgTmp = (uint)silk_SMULBB(x[i], x[i]);
            nrgTmp = (uint)silk_SMLABB_ovflw((int)nrgTmp, x[i + 1], x[i + 1]);
            nrg = silk_ADD_RSHIFT_uint(nrg, nrgTmp, shft);
        }
        if (i < len)
        {
            uint nrgTmp = (uint)silk_SMULBB(x[i], x[i]);
            nrg = silk_ADD_RSHIFT_uint(nrg, nrgTmp, shft);
        }

        shift = shft;
        energy = (int)nrg;
    }
}
