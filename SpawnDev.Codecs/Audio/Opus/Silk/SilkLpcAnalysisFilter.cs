// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Structural port of libopus silk/LPC_analysis_filter.c to clean C#. Applies
// the MA prediction-error filter (inverse of the LPC synthesis filter) to an
// input signal. Used during voiced-subframe re-whitening inside decode_core.
//
// Upstream Copyright (c) 2006-2011 Skype Limited. BSD 3-Clause. See NOTICE.md.

using static SpawnDev.Codecs.Audio.Opus.Silk.SilkMacros;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// LPC analysis filter: computes <c>out[n] = in[n+d] - sum_{k=0..d-1} B[k] * in[n+d-1-k] / 2^12</c>
/// for each sample after the first <c>d</c> samples. Output samples <c>[0..d)</c>
/// are zeroed (no prediction history available yet).
/// </summary>
internal static class SilkLpcAnalysisFilter
{
    /// <summary>
    /// Apply the LPC analysis filter to <paramref name="inSignal"/>, writing results to
    /// <paramref name="outSignal"/>. Matches libopus <c>silk_LPC_analysis_filter</c>.
    /// </summary>
    /// <param name="outSignal">Output signal. Length &gt;= <paramref name="len"/>.</param>
    /// <param name="inSignal">Input signal. Length &gt;= <paramref name="len"/>.</param>
    /// <param name="bQ12">MA prediction coefficients in Q12. Length &gt;= <paramref name="d"/>.</param>
    /// <param name="len">Input signal length.</param>
    /// <param name="d">Filter order (must be even, &gt;= 6, and &lt;= <paramref name="len"/>).</param>
    internal static void Apply(
        Span<short> outSignal,
        ReadOnlySpan<short> inSignal,
        ReadOnlySpan<short> bQ12,
        int len,
        int d)
    {
        if (d < 6 || (d & 1) != 0) throw new ArgumentException($"d ({d}) must be even and >= 6.", nameof(d));
        if (d > len) throw new ArgumentException($"d ({d}) must be <= len ({len}).", nameof(d));
        if (inSignal.Length < len) throw new ArgumentException($"inSignal too small (need {len}).", nameof(inSignal));
        if (outSignal.Length < len) throw new ArgumentException($"outSignal too small (need {len}).", nameof(outSignal));
        if (bQ12.Length < d) throw new ArgumentException($"bQ12 too small (need {d}).", nameof(bQ12));

        for (int ix = d; ix < len; ix++)
        {
            // in_ptr points to inSignal[ix - 1]; accesses in_ptr[0], in_ptr[-1], ..., in_ptr[-(d-1)],
            // plus in_ptr[1] = inSignal[ix] as the desired sample to subtract the prediction from.
            int baseIdx = ix - 1;

            int out32Q12 = silk_SMULBB(inSignal[baseIdx], bQ12[0]);
            out32Q12 = silk_SMLABB_ovflw(out32Q12, inSignal[baseIdx - 1], bQ12[1]);
            out32Q12 = silk_SMLABB_ovflw(out32Q12, inSignal[baseIdx - 2], bQ12[2]);
            out32Q12 = silk_SMLABB_ovflw(out32Q12, inSignal[baseIdx - 3], bQ12[3]);
            out32Q12 = silk_SMLABB_ovflw(out32Q12, inSignal[baseIdx - 4], bQ12[4]);
            out32Q12 = silk_SMLABB_ovflw(out32Q12, inSignal[baseIdx - 5], bQ12[5]);
            for (int j = 6; j < d; j += 2)
            {
                out32Q12 = silk_SMLABB_ovflw(out32Q12, inSignal[baseIdx - j], bQ12[j]);
                out32Q12 = silk_SMLABB_ovflw(out32Q12, inSignal[baseIdx - j - 1], bQ12[j + 1]);
            }

            // Subtract the accumulated prediction from the delayed input in Q12 domain.
            out32Q12 = silk_SUB32_ovflw(silk_LSHIFT(inSignal[ix], 12), out32Q12);

            int out32 = silk_RSHIFT_ROUND(out32Q12, 12);
            outSignal[ix] = silk_SAT16(out32);
        }

        // First d output samples are undefined (no prediction history), zero them.
        for (int i = 0; i < d; i++) outSignal[i] = 0;
    }
}
