using SpawnDev.Codecs.Audio.Opus.Silk;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for <see cref="SilkGainAdjust.Apply"/> - the between-subframe gain
/// adjustment helper that keeps the LPC state buffer consistent when a gain
/// step is applied in decode_core.
/// </summary>
public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void GainAdjust_SameGain_ReturnsOneAndLeavesStateUnchanged()
    {
        int[] state = new int[SilkConstants.MAX_LPC_ORDER];
        for (int i = 0; i < state.Length; i++) state[i] = (i + 1) * 10000;
        int[] original = new int[SilkConstants.MAX_LPC_ORDER];
        state.AsSpan().CopyTo(original);

        int ratio = SilkGainAdjust.Apply(state, prevGainQ16: 65536, curGainQ16: 65536);

        Equal(1 << 16, ratio);
        for (int i = 0; i < state.Length; i++) Equal(original[i], state[i], $"state[{i}]");
    }

    [TestMethod]
    public void GainAdjust_GainIncreasedByTwo_ScalesStateDownByHalf()
    {
        // prevGain = 1.0 Q16, curGain = 2.0 Q16. Ratio = 0.5 Q16 = 32768.
        int[] state = new int[SilkConstants.MAX_LPC_ORDER];
        for (int i = 0; i < state.Length; i++) state[i] = 100000;

        int ratio = SilkGainAdjust.Apply(state, prevGainQ16: 65536, curGainQ16: 131072);

        // Ratio approximates 0.5 in Q16 = 32768 (within Newton-refinement error).
        True(ratio > 30000 && ratio < 35000, $"expected ~32768, got {ratio}");

        // Each state sample should be approximately halved.
        for (int i = 0; i < state.Length; i++)
        {
            True(state[i] > 40000 && state[i] < 60000, $"state[{i}] = {state[i]} should be ~50000");
        }
    }

    [TestMethod]
    public void GainAdjust_GainHalved_ScalesStateUpByTwo()
    {
        // prevGain = 2.0 Q16, curGain = 1.0 Q16. Ratio = 2.0 Q16 = 131072.
        int[] state = new int[SilkConstants.MAX_LPC_ORDER];
        for (int i = 0; i < state.Length; i++) state[i] = 50000;

        int ratio = SilkGainAdjust.Apply(state, prevGainQ16: 131072, curGainQ16: 65536);

        True(ratio > 125000 && ratio < 140000, $"expected ~131072, got {ratio}");

        // Each state sample should be approximately doubled.
        for (int i = 0; i < state.Length; i++)
        {
            True(state[i] > 95000 && state[i] < 115000, $"state[{i}] = {state[i]} should be ~100000");
        }
    }

    [TestMethod]
    public void GainAdjust_OnlyScalesFirstMaxLpcOrderSamples()
    {
        // State buffer of length MAX_LPC_ORDER + 10. The extra 10 samples should NOT
        // be touched (those are the "current subframe output slots", not history).
        int[] state = new int[SilkConstants.MAX_LPC_ORDER + 10];
        for (int i = 0; i < state.Length; i++) state[i] = 200000;
        int[] expectedTail = new int[10];
        for (int i = 0; i < 10; i++) expectedTail[i] = 200000;

        SilkGainAdjust.Apply(state, prevGainQ16: 65536, curGainQ16: 131072);

        for (int i = 0; i < 10; i++)
        {
            Equal(expectedTail[i], state[SilkConstants.MAX_LPC_ORDER + i],
                $"state[{SilkConstants.MAX_LPC_ORDER + i}] should be untouched");
        }
    }

    // -------- Arg validation --------

    [TestMethod]
    public void GainAdjust_ZeroCurGain_Throws()
    {
        int[] state = new int[SilkConstants.MAX_LPC_ORDER];
        Throws<ArgumentException>(() =>
            SilkGainAdjust.Apply(state, prevGainQ16: 65536, curGainQ16: 0));
    }

    [TestMethod]
    public void GainAdjust_StateTooSmall_Throws()
    {
        int[] state = new int[SilkConstants.MAX_LPC_ORDER - 1];
        Throws<ArgumentException>(() =>
            SilkGainAdjust.Apply(state, prevGainQ16: 65536, curGainQ16: 131072));
    }
}
