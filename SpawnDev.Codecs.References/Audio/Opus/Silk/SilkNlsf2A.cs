// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Structural port of libopus silk/NLSF2A.c to clean C#. Converts normalized
// line spectral frequencies (NLSFs) into Q12 monic whitening filter coefficients,
// applying bandwidth-expansion iterations to guarantee a stable LPC filter.
//
// Upstream Copyright (c) 2006-2011 Skype Limited. BSD 3-Clause. See NOTICE.md.

using static SpawnDev.Codecs.Audio.Opus.Silk.SilkConstants;
using static SpawnDev.Codecs.Audio.Opus.Silk.SilkMacros;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// NLSF-to-LPC conversion. Given normalized LSFs (in <c>Q15</c>) this produces Q12 monic
/// whitening filter coefficients and, if the resulting filter is unstable, iterates
/// bandwidth expansion (up to <see cref="SilkConstants.MAX_LPC_STABILIZE_ITERATIONS"/>
/// times) until <see cref="SilkLpcInvPredGain.Compute"/> accepts it.
/// <para>
/// The LSF-to-cos mapping uses the <see cref="SilkLsfCosTab.Q12"/> piecewise linear
/// approximation: this is the standard SILK approach from libopus (see the comment in
/// the upstream <c>NLSF2A.c</c>). The per-coefficient ordering during the even/odd
/// polynomial construction matches the empirically-derived orderings chosen upstream
/// to maximize numerical accuracy.
/// </para>
/// </summary>
internal static class SilkNlsf2A
{
    /// <summary>Internal QA format used during polynomial construction (libopus <c>QA = 16</c>, distinct from the QA in LPC_inv_pred_gain).</summary>
    private const int QA = 16;

    /// <summary>Ordering permutation for order-16 filters.</summary>
    private static readonly byte[] Ordering16 = { 0, 15, 8, 7, 4, 11, 12, 3, 2, 13, 10, 5, 6, 9, 14, 1 };

    /// <summary>Ordering permutation for order-10 filters.</summary>
    private static readonly byte[] Ordering10 = { 0, 9, 6, 3, 4, 5, 8, 1, 2, 7 };

    /// <summary>
    /// Build an intermediate <c>QA</c> polynomial from a vector of interleaved
    /// <c>2*cos(LSF)</c> values. Matches libopus <c>silk_NLSF2A_find_poly</c>.
    /// </summary>
    /// <param name="outPoly">Output polynomial in Q<see cref="QA"/>. Length <c>dd+1</c>.</param>
    /// <param name="cLSF">Interleaved <c>2*cos(LSF)</c> values in Q<see cref="QA"/>.</param>
    /// <param name="cLSFStride">Stride between interleaved entries (always 2 in the current caller).</param>
    /// <param name="dd">Polynomial order (half the filter order).</param>
    private static void FindPoly(Span<int> outPoly, ReadOnlySpan<int> cLSF, int cLSFStride, int dd)
    {
        outPoly[0] = silk_LSHIFT(1, QA);
        outPoly[1] = -cLSF[0];
        for (int k = 1; k < dd; k++)
        {
            int ftmp = cLSF[cLSFStride * k];
            outPoly[k + 1] = silk_LSHIFT(outPoly[k - 1], 1)
                - (int)silk_RSHIFT_ROUND64(silk_SMULL(ftmp, outPoly[k]), QA);
            for (int n = k; n > 1; n--)
            {
                outPoly[n] += outPoly[n - 2]
                    - (int)silk_RSHIFT_ROUND64(silk_SMULL(ftmp, outPoly[n - 1]), QA);
            }
            outPoly[1] -= ftmp;
        }
    }

    /// <summary>
    /// Convert normalized LSFs in Q15 to Q12 monic whitening filter coefficients.
    /// Applies bandwidth expansion to guarantee stability, up to
    /// <see cref="SilkConstants.MAX_LPC_STABILIZE_ITERATIONS"/> iterations.
    /// </summary>
    /// <param name="aQ12">Output: Q12 LPC coefficients. Length <paramref name="d"/>.</param>
    /// <param name="nlsf">Input: NLSFs in Q15. Length <paramref name="d"/>.</param>
    /// <param name="d">Filter order; must be 10 or 16.</param>
    internal static void Compute(Span<short> aQ12, ReadOnlySpan<short> nlsf, int d)
    {
        if (d != 10 && d != 16) throw new ArgumentException($"d must be 10 or 16, got {d}.", nameof(d));
        if (aQ12.Length < d) throw new ArgumentException($"aQ12 too small (need {d}).", nameof(aQ12));
        if (nlsf.Length < d) throw new ArgumentException($"nlsf too small (need {d}).", nameof(nlsf));

        byte[] ordering = d == 16 ? Ordering16 : Ordering10;

        Span<int> cosLsfQA = stackalloc int[MAX_LPC_ORDER];
        Span<int> p = stackalloc int[MAX_LPC_ORDER / 2 + 1];
        Span<int> q = stackalloc int[MAX_LPC_ORDER / 2 + 1];
        Span<int> a32QA1 = stackalloc int[MAX_LPC_ORDER];

        for (int k = 0; k < d; k++)
        {
            // f_int: integer part of NLSF in Q7 range [0, 127] (rounded down via right-shift by 8).
            int fInt = silk_RSHIFT(nlsf[k], 15 - 7);
            // f_frac: remainder in [0, 255].
            int fFrac = nlsf[k] - silk_LSHIFT(fInt, 15 - 7);

            int cosVal = SilkLsfCosTab.Q12[fInt];             // Q12
            int delta = SilkLsfCosTab.Q12[fInt + 1] - cosVal; // Q12, range 0..200

            // Linear interpolation, then promote to Q(QA) = Q16.
            cosLsfQA[ordering[k]] = silk_RSHIFT_ROUND(
                silk_LSHIFT(cosVal, 8) + silk_MUL(delta, fFrac),
                20 - QA);
        }

        int dd = silk_RSHIFT(d, 1);

        // Even polynomial uses cos_LSF_QA[0], cos_LSF_QA[2], cos_LSF_QA[4], ...
        // Odd polynomial uses cos_LSF_QA[1], cos_LSF_QA[3], cos_LSF_QA[5], ...
        FindPoly(p, cosLsfQA, 2, dd);
        FindPoly(q, cosLsfQA.Slice(1), 2, dd);

        // Fold even (P) and odd (Q) polynomials back into the full filter in Q(QA+1).
        for (int k = 0; k < dd; k++)
        {
            int pTmp = p[k + 1] + p[k];
            int qTmp = q[k + 1] - q[k];
            a32QA1[k] = -qTmp - pTmp;
            a32QA1[d - k - 1] = qTmp - pTmp;
        }

        // Convert to Q12 int16 with iterative LPC_fit-driven saturation handling.
        SilkLpcFit.Fit(aQ12, a32QA1.Slice(0, d), 12, QA + 1, d);

        // If the result is still unstable, bandwidth-expand the unscaled coefficients
        // and refit. Each iteration shrinks the filter's poles a little further.
        for (int i = 0;
             SilkLpcInvPredGain.Compute(aQ12, d) == 0 && i < MAX_LPC_STABILIZE_ITERATIONS;
             i++)
        {
            // chirp = 1.0 - (2^(i+1)) / 2^16, so each iteration halves the remaining
            // margin to 1.0. Matches libopus: 65536 - silk_LSHIFT(2, i).
            SilkBwexpander.Expand32(a32QA1.Slice(0, d), 65536 - silk_LSHIFT(2, i));
            for (int k = 0; k < d; k++)
            {
                aQ12[k] = (short)silk_RSHIFT_ROUND(a32QA1[k], QA + 1 - 12);
            }
        }
    }
}
