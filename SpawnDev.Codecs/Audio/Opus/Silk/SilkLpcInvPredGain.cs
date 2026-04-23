// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Structural port of libopus silk/LPC_inv_pred_gain.c to clean C#. Computes the
// inverse LPC prediction gain and tests filter stability (all poles within the
// unit circle) via a reverse Levinson-Durbin recursion on reflection coefficients.
//
// Upstream Copyright (c) 2006-2011 Skype Limited. BSD 3-Clause. See NOTICE.md.

using static SpawnDev.Codecs.Audio.Opus.Silk.SilkMacros;
using static SpawnDev.Codecs.Audio.Opus.Silk.SilkConstants;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// Computes the inverse LPC prediction gain and validates filter stability.
/// <para>
/// The LPC synthesis filter <c>1 / A(z)</c> is stable iff all of its poles lie inside
/// the unit circle. This function runs the standard reverse Levinson-Durbin step: at
/// each order it extracts the reflection coefficient <c>rc_k = -a_k</c>, checks that
/// |rc_k| &lt; 1 (with a small margin against fixed-point edge cases), and accumulates
/// the inverse prediction gain <c>prod_k(1 - rc_k^2)</c>. If any reflection coefficient
/// is out of range, or if the accumulated inverse gain drops below
/// <c>1 / MAX_PREDICTION_POWER_GAIN</c>, the filter is rejected and 0 is returned.
/// </para>
/// <para>
/// Internally we work in a <c>QA = 24</c>-bit Q-format for precision, but the public
/// entry point accepts the standard SILK-native Q12 coefficients. Output is Q30.
/// </para>
/// </summary>
internal static class SilkLpcInvPredGain
{
    /// <summary>Internal high-precision Q-format used by the Levinson-Durbin recursion (libopus <c>QA = 24</c>).</summary>
    private const int QA = 24;

    /// <summary>
    /// Stability threshold in Q<see cref="QA"/>: <c>SILK_FIX_CONST(0.99975, 24) = (int)(0.99975 * 2^24 + 0.5) = 16773022</c>.
    /// </summary>
    private const int A_LIMIT = 16773022;

    /// <summary>
    /// Compute the inverse LPC prediction gain from Q12 coefficients, returning the
    /// result in Q30. If the filter is unstable or its gain would exceed
    /// <see cref="SilkConstants.MAX_PREDICTION_POWER_GAIN"/>, returns 0.
    /// </summary>
    /// <param name="aQ12">LPC coefficients in Q12. Length <paramref name="order"/>.</param>
    /// <param name="order">LPC prediction order.</param>
    /// <returns>Inverse prediction gain in Q30, or 0 if unstable.</returns>
    internal static int Compute(ReadOnlySpan<short> aQ12, int order)
    {
        if (aQ12.Length < order) throw new ArgumentException($"aQ12 too small (need {order}).", nameof(aQ12));

        Span<int> aTmpQA = stackalloc int[order];
        int dcResp = 0;

        for (int k = 0; k < order; k++)
        {
            dcResp += aQ12[k];
            aTmpQA[k] = silk_LSHIFT32(aQ12[k], QA - 12);
        }

        // If the DC gain alone is already unstable, skip the full recursion.
        if (dcResp >= 4096) return 0;

        return LpcInversePredGainQA(aTmpQA, order);
    }

    /// <summary>
    /// Core reverse-Levinson recursion on Q<see cref="QA"/> coefficients. Matches
    /// libopus <c>LPC_inverse_pred_gain_QA_c</c>. The coefficient array is modified in place.
    /// </summary>
    /// <param name="aQA">In/out LPC coefficients in Q<see cref="QA"/>. Length <paramref name="order"/>.</param>
    /// <param name="order">LPC prediction order.</param>
    /// <returns>Inverse prediction gain in Q30, or 0 if unstable.</returns>
    private static int LpcInversePredGainQA(Span<int> aQA, int order)
    {
        int invGainQ30 = 1 << 30;

        int k;
        for (k = order - 1; k > 0; k--)
        {
            // Stability check on the current highest-order coefficient.
            if (aQA[k] > A_LIMIT || aQA[k] < -A_LIMIT) return 0;

            // rc_Q31 = -A_QA[k] shifted from QA to Q31. Range: (-2^30, +2^30) given A_LIMIT.
            int rcQ31 = -silk_LSHIFT(aQA[k], 31 - QA);

            // rc_mult1_Q30 = 1 - rc^2 in Q30. Range: [1, 2^30].
            int rcMult1Q30 = silk_SUB32(1 << 30, silk_SMMUL(rcQ31, rcQ31));

            // Update accumulated inverse gain.
            invGainQ30 = silk_LSHIFT(silk_SMMUL(invGainQ30, rcMult1Q30), 2);
            if (invGainQ30 < INV_GAIN_Q30_MIN) return 0;

            // rc_mult2 = 1 / rc_mult1, in variable-Q.
            int mult2Q = 32 - silk_CLZ32(silk_abs(rcMult1Q30));
            int rcMult2 = silk_INVERSE32_varQ(rcMult1Q30, mult2Q + 30);

            // Reverse-Levinson update: a_new[n] = (a[n] - a[k-1-n] * rc) / (1 - rc^2)
            int halfK = (k + 1) >> 1;
            for (int n = 0; n < halfK; n++)
            {
                int tmp1 = aQA[n];
                int tmp2 = aQA[k - n - 1];

                long tmp64 = silk_RSHIFT_ROUND64(
                    silk_SMULL(silk_SUB_SAT32(tmp1, silk_MUL32_FRAC_Q(tmp2, rcQ31, 31)), rcMult2),
                    mult2Q);
                if (tmp64 > silk_int32_MAX || tmp64 < silk_int32_MIN) return 0;
                aQA[n] = (int)tmp64;

                tmp64 = silk_RSHIFT_ROUND64(
                    silk_SMULL(silk_SUB_SAT32(tmp2, silk_MUL32_FRAC_Q(tmp1, rcQ31, 31)), rcMult2),
                    mult2Q);
                if (tmp64 > silk_int32_MAX || tmp64 < silk_int32_MIN) return 0;
                aQA[k - n - 1] = (int)tmp64;
            }
        }

        // Final iteration: k == 0. Stability + gain check on the last coefficient.
        if (aQA[k] > A_LIMIT || aQA[k] < -A_LIMIT) return 0;

        int rcQ31Last = -silk_LSHIFT(aQA[0], 31 - QA);
        int rcMult1Q30Last = silk_SUB32(1 << 30, silk_SMMUL(rcQ31Last, rcQ31Last));

        invGainQ30 = silk_LSHIFT(silk_SMMUL(invGainQ30, rcMult1Q30Last), 2);
        if (invGainQ30 < INV_GAIN_Q30_MIN) return 0;

        return invGainQ30;
    }
}
