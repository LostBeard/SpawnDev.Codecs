// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Structural port of the gain-adjustment block inside libopus silk/decode_core.c.
// When the current subframe's gain differs from the previous subframe's gain, the
// LPC state buffer must be scaled by the gain ratio to keep the synthesis
// filter's output consistent across the gain step.
//
// Upstream Copyright (c) 2006-2011 Skype Limited. BSD 3-Clause. See NOTICE.md.

using static SpawnDev.Codecs.Audio.Opus.Silk.SilkMacros;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// Gain-adjustment helper called between SILK subframes during synthesis.
/// Updates the LPC state buffer in place to account for a change in the gain
/// applied to subsequent samples, and returns the Q16 gain-ratio for use by
/// any downstream state (e.g. LTP state rescaling in decode_core).
/// </summary>
internal static class SilkGainAdjust
{
    /// <summary>
    /// Rescale the LPC state buffer to compensate for a gain change between
    /// subframes. Matches the gain-adjust step inside libopus decode_core.
    /// </summary>
    /// <param name="stateQ14">LPC state buffer (first <see cref="SilkConstants.MAX_LPC_ORDER"/>
    /// entries are filter history). Scaled in place.</param>
    /// <param name="prevGainQ16">Previous subframe's gain in Q16.</param>
    /// <param name="curGainQ16">Current subframe's gain in Q16.</param>
    /// <returns>Gain ratio in Q16 (= <c>prevGainQ16 / curGainQ16</c>), or <c>1&lt;&lt;16</c>
    /// when the gains are equal. Callers use this to scale LTP state similarly.</returns>
    internal static int Apply(Span<int> stateQ14, int prevGainQ16, int curGainQ16)
    {
        if (stateQ14.Length < SilkConstants.MAX_LPC_ORDER)
            throw new ArgumentException(
                $"stateQ14 too small (need {SilkConstants.MAX_LPC_ORDER}).", nameof(stateQ14));
        if (curGainQ16 == 0) throw new ArgumentException("curGainQ16 must be non-zero.", nameof(curGainQ16));

        if (prevGainQ16 == curGainQ16)
        {
            return 1 << 16;
        }

        int gainAdjQ16 = silk_DIV32_varQ(prevGainQ16, curGainQ16, 16);

        // Apply the gain ratio to the first MAX_LPC_ORDER state samples (filter history).
        for (int i = 0; i < SilkConstants.MAX_LPC_ORDER; i++)
        {
            stateQ14[i] = silk_SMULWW(gainAdjQ16, stateQ14[i]);
        }

        return gainAdjQ16;
    }
}
