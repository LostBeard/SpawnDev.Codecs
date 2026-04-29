// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable SILK gain-adjust step. Mirror of SilkGainAdjust.Apply
// (libopus silk/decode_core.c gain-adjust block). Rescales the first
// MAX_LPC_ORDER (16) entries of the LPC state buffer when the gain
// changes between subframes.
//
// Single-thread on GPU because the work is tiny (compute one Q16 ratio,
// then 16 multiplies). Alternatively a single thread computes the ratio
// and broadcasts via shared memory; for cross-backend portability +
// simplicity we keep it as a one-thread primitive.
//
// Uses SilkDivVarQGpu for the Q16 gain ratio. All silk macros (SMULWW,
// SMULWB) inlined here.

using ILGPU;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// GPU-callable SILK gain-adjust step. Mirror of
/// <see cref="SilkGainAdjust"/>.Apply.
/// </summary>
public static class SilkGainAdjustGpu
{
    private const int MAX_LPC_ORDER = 16;

    /// <summary>
    /// Rescale the LPC state buffer to compensate for a gain change between
    /// subframes. Writes the gain ratio (Q16) into <paramref name="gainAdjOut"/>
    /// at <paramref name="gainAdjBase"/>. State entries [0..MAX_LPC_ORDER) are
    /// scaled in place. Bit-exact vs the CPU SilkGainAdjust.Apply.
    /// </summary>
    /// <param name="stateQ14">LPC state buffer (first MAX_LPC_ORDER entries scaled).</param>
    /// <param name="stateBase">Base offset into <paramref name="stateQ14"/>.</param>
    /// <param name="prevGainQ16">Previous subframe's gain in Q16.</param>
    /// <param name="curGainQ16">Current subframe's gain in Q16. Must be non-zero.</param>
    /// <param name="gainAdjOut">Output buffer for the Q16 gain ratio (single int).</param>
    /// <param name="gainAdjBase">Base offset into <paramref name="gainAdjOut"/>.</param>
    public static void ApplyAt(
        ArrayView<int> stateQ14, long stateBase,
        int prevGainQ16, int curGainQ16,
        ArrayView<int> gainAdjOut, long gainAdjBase)
    {
        if (prevGainQ16 == curGainQ16)
        {
            gainAdjOut[gainAdjBase] = 1 << 16;
            return;
        }

        int gainAdjQ16 = SilkDivVarQGpu.Compute(prevGainQ16, curGainQ16, 16);
        gainAdjOut[gainAdjBase] = gainAdjQ16;

        for (int i = 0; i < MAX_LPC_ORDER; i++)
        {
            stateQ14[stateBase + i] = SmulWW(gainAdjQ16, stateQ14[stateBase + i]);
        }
    }

    /// <summary>silk_SMULWW(a, b) = SMULWB(a, b) + a * RSHIFT_ROUND(b, 16).</summary>
    private static int SmulWW(int a32, int b32)
    {
        int smulwb = (int)((long)a32 * (short)b32 >> 16);
        int rshiftRound = (b32 + (1 << 15)) >> 16;
        return smulwb + a32 * rshiftRound;
    }
}
