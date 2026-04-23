// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Structural port of the LPC-synthesis inner loop of libopus silk/decode_core.c
// to clean C#. Given a Q14 residual signal, Q12 LPC coefficients, and a running
// Q14 state buffer, produces PCM output samples scaled by a Q10 gain.
//
// Upstream Copyright (c) 2006-2011 Skype Limited. BSD 3-Clause. See NOTICE.md.

using static SpawnDev.Codecs.Audio.Opus.Silk.SilkMacros;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// LPC synthesis filter. For each input sample <c>pres_Q14[i]</c>:
/// <list type="number">
/// <item>Compute the LPC prediction from the last <c>order</c> state values.</item>
/// <item>Add the residual: <c>state[i] = pres[i] + (pred &lt;&lt; 4)</c>, saturating.</item>
/// <item>Scale by gain and round to int16 PCM.</item>
/// </list>
/// <para>
/// The state buffer is layout-compatible with libopus' <c>sLPC_Q14</c>: the first
/// <see cref="SilkConstants.MAX_LPC_ORDER"/> entries are filter history carried
/// between subframes; samples 0..subfrLen-1 are written to positions
/// <c>MAX_LPC_ORDER + i</c>. After processing, callers must copy the trailing
/// <c>MAX_LPC_ORDER</c> entries back to the beginning to preserve the history for
/// the next subframe (this class does NOT do that slide - it's the caller's
/// responsibility, matching libopus memcpy behavior).
/// </para>
/// </summary>
internal static class SilkLpcSynthesisFilter
{
    /// <summary>
    /// Apply the LPC synthesis filter over <paramref name="subfrLen"/> samples.
    /// </summary>
    /// <param name="stateQ14">In/out state buffer. Length &gt;= <see cref="SilkConstants.MAX_LPC_ORDER"/> + <paramref name="subfrLen"/>.
    /// History is at indices <c>[0, MAX_LPC_ORDER)</c>; output samples are written to
    /// <c>[MAX_LPC_ORDER, MAX_LPC_ORDER + subfrLen)</c>.</param>
    /// <param name="presQ14">Residual (post-LTP or excitation) signal in Q14. Length &gt;= <paramref name="subfrLen"/>.</param>
    /// <param name="aQ12">LPC coefficients in Q12. Length &gt;= <paramref name="order"/>.</param>
    /// <param name="gainQ10">Gain in Q10 (= Gain_Q16 &gt;&gt; 6).</param>
    /// <param name="order">LPC order (10 or 16).</param>
    /// <param name="subfrLen">Subframe length in samples.</param>
    /// <param name="pcmOut">Output PCM samples (int16). Length &gt;= <paramref name="subfrLen"/>.</param>
    internal static void Apply(
        Span<int> stateQ14,
        ReadOnlySpan<int> presQ14,
        ReadOnlySpan<short> aQ12,
        int gainQ10,
        int order,
        int subfrLen,
        Span<short> pcmOut)
    {
        if (order != 10 && order != 16)
            throw new ArgumentException($"order must be 10 or 16, got {order}.", nameof(order));
        if (subfrLen <= 0) throw new ArgumentOutOfRangeException(nameof(subfrLen));
        if (stateQ14.Length < SilkConstants.MAX_LPC_ORDER + subfrLen)
            throw new ArgumentException($"stateQ14 too small (need {SilkConstants.MAX_LPC_ORDER + subfrLen}).", nameof(stateQ14));
        if (presQ14.Length < subfrLen) throw new ArgumentException($"presQ14 too small (need {subfrLen}).", nameof(presQ14));
        if (aQ12.Length < order) throw new ArgumentException($"aQ12 too small (need {order}).", nameof(aQ12));
        if (pcmOut.Length < subfrLen) throw new ArgumentException($"pcmOut too small (need {subfrLen}).", nameof(pcmOut));

        int maxOrder = SilkConstants.MAX_LPC_ORDER;

        for (int i = 0; i < subfrLen; i++)
        {
            // Short-term prediction: LPC_pred_Q10 = order/2 (rounding bias) + sum_k A[k] * state[MAX_LPC_ORDER + i - 1 - k].
            int lpcPredQ10 = silk_RSHIFT(order, 1);
            lpcPredQ10 = silk_SMLAWB(lpcPredQ10, stateQ14[maxOrder + i - 1], aQ12[0]);
            lpcPredQ10 = silk_SMLAWB(lpcPredQ10, stateQ14[maxOrder + i - 2], aQ12[1]);
            lpcPredQ10 = silk_SMLAWB(lpcPredQ10, stateQ14[maxOrder + i - 3], aQ12[2]);
            lpcPredQ10 = silk_SMLAWB(lpcPredQ10, stateQ14[maxOrder + i - 4], aQ12[3]);
            lpcPredQ10 = silk_SMLAWB(lpcPredQ10, stateQ14[maxOrder + i - 5], aQ12[4]);
            lpcPredQ10 = silk_SMLAWB(lpcPredQ10, stateQ14[maxOrder + i - 6], aQ12[5]);
            lpcPredQ10 = silk_SMLAWB(lpcPredQ10, stateQ14[maxOrder + i - 7], aQ12[6]);
            lpcPredQ10 = silk_SMLAWB(lpcPredQ10, stateQ14[maxOrder + i - 8], aQ12[7]);
            lpcPredQ10 = silk_SMLAWB(lpcPredQ10, stateQ14[maxOrder + i - 9], aQ12[8]);
            lpcPredQ10 = silk_SMLAWB(lpcPredQ10, stateQ14[maxOrder + i - 10], aQ12[9]);
            if (order == 16)
            {
                lpcPredQ10 = silk_SMLAWB(lpcPredQ10, stateQ14[maxOrder + i - 11], aQ12[10]);
                lpcPredQ10 = silk_SMLAWB(lpcPredQ10, stateQ14[maxOrder + i - 12], aQ12[11]);
                lpcPredQ10 = silk_SMLAWB(lpcPredQ10, stateQ14[maxOrder + i - 13], aQ12[12]);
                lpcPredQ10 = silk_SMLAWB(lpcPredQ10, stateQ14[maxOrder + i - 14], aQ12[13]);
                lpcPredQ10 = silk_SMLAWB(lpcPredQ10, stateQ14[maxOrder + i - 15], aQ12[14]);
                lpcPredQ10 = silk_SMLAWB(lpcPredQ10, stateQ14[maxOrder + i - 16], aQ12[15]);
            }

            // Add residual (saturating) and write new state sample.
            stateQ14[maxOrder + i] = silk_ADD_SAT32(presQ14[i], silk_LSHIFT_SAT32(lpcPredQ10, 4));

            // Scale with gain and round to int16.
            pcmOut[i] = silk_SAT16(silk_RSHIFT_ROUND(silk_SMULWW(stateQ14[maxOrder + i], gainQ10), 8));
        }
    }
}
