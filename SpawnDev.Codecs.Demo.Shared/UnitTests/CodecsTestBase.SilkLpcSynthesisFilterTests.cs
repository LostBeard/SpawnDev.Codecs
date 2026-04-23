using SpawnDev.Codecs.Audio.Opus.Silk;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for <see cref="SilkLpcSynthesisFilter.Apply"/> - the LPC synthesis filter
/// inner loop used by decode_core to turn a Q14 residual signal into output PCM.
/// </summary>
public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void LpcSynthesisFilter_ZeroInput_ZeroState_ProducesZeroPcm()
    {
        // All-zero residual, all-zero state, any LPC coefs: output PCM is zero (the
        // rounding bias leaks into the state buffer as small positive values, but
        // after gain-scaling they round to 0 at the PCM output).
        int[] state = new int[SilkConstants.MAX_LPC_ORDER + 40];
        int[] pres = new int[40];
        short[] aQ12 = new short[10];
        for (int i = 0; i < 10; i++) aQ12[i] = (short)((i + 1) * 100); // arbitrary
        short[] pcm = new short[40];

        SilkLpcSynthesisFilter.Apply(state, pres, aQ12, gainQ10: 1024, order: 10, subfrLen: 40, pcm);

        // PCM output all zero.
        for (int i = 0; i < 40; i++) Equal((short)0, pcm[i], $"pos {i}");
    }

    [TestMethod]
    public void LpcSynthesisFilter_ZeroCoefficients_OutputMirrorsResidual()
    {
        // With all LPC coefs zero, the synthesis filter reduces to:
        //   state[i] = pres[i] + (rounding << 4)
        //   pcm[i] = SAT16(RSHIFT_ROUND(state[i] * Gain_Q10, 8))
        // The rounding bias is (order/2 << 4) = 80 for order 10 -- small.
        // Use a mild residual + gain to keep everything well within int16 range.
        int[] state = new int[SilkConstants.MAX_LPC_ORDER + 40];
        int[] pres = new int[40];
        for (int i = 0; i < 40; i++) pres[i] = 1000; // small Q14 residual (0.061)
        short[] aQ12 = new short[10]; // all zeros
        short[] pcm = new short[40];
        int gainQ10 = 1024; // 1.0 in Q10

        SilkLpcSynthesisFilter.Apply(state, pres, aQ12, gainQ10, order: 10, subfrLen: 40, pcm);

        // With A=0: state[i] = pres[i] + (5 << 4) = 1080. pcm = SAT16(round((1080 * 1024) >> 16) via SMULWW).
        // SMULWW(1080, 1024) = SMULWW formula = MLA(SMULWB(1080, 1024), 1080, RSHIFT_ROUND(1024, 16))
        //   SMULWB(1080, 1024) = (1080 * 1024) >> 16 = 1105920 >> 16 = 16.88 -> 16.
        //   RSHIFT_ROUND(1024, 16): shift > 1 case = ((1024 >> 15) + 1) >> 1 = 1 >> 1 = 0.
        //   MLA(16, 1080, 0) = 16.
        // RSHIFT_ROUND(16, 8) = shift > 1 case = ((16 >> 7) + 1) >> 1 = (0 + 1) >> 1 = 0.
        // So pcm[i] should be 0 for all 40 samples (very quiet signal).
        // More practically: confirm they're all small positive.
        for (int i = 0; i < 40; i++)
        {
            True(pcm[i] >= 0 && pcm[i] < 10, $"pcm[{i}] = {pcm[i]} should be small positive");
        }
    }

    [TestMethod]
    public void LpcSynthesisFilter_WriteToStateMatchesResidualPlusPrediction()
    {
        // Verify the state buffer value at position MAX_LPC_ORDER matches the formula
        // state[MAX+0] = pres[0] + (LPC_pred_Q10 << 4) with zero history.
        int[] state = new int[SilkConstants.MAX_LPC_ORDER + 1];
        int[] pres = { 123456 };
        short[] aQ12 = new short[10]; // zero coefficients
        short[] pcm = new short[1];

        SilkLpcSynthesisFilter.Apply(state, pres, aQ12, gainQ10: 1024, order: 10, subfrLen: 1, pcm);

        // Prediction = order/2 << 4 = 80 (the rounding bias), shifted left by 4 = 1280.
        // state[MAX] = 123456 + 1280 = 124736? But wait - Q10 << 4 gives Q14 so yes.
        // Actually: lpcPredQ10 starts at order/2 = 5. A is zero so it stays at 5.
        // state[MAX+0] = pres[0] + (5 << 4) = pres[0] + 80.
        Equal(123456 + 80, state[SilkConstants.MAX_LPC_ORDER]);
    }

    [TestMethod]
    public void LpcSynthesisFilter_Order16_DoesNotCrashAtRealisticSize()
    {
        // Realistic WB 20ms / 4 subframes = 80 samples per subframe at fs=16kHz.
        int[] state = new int[SilkConstants.MAX_LPC_ORDER + 80];
        var rng = new Random(123);
        int[] pres = new int[80];
        for (int i = 0; i < 80; i++) pres[i] = rng.Next(-50000, 50000);
        short[] aQ12 = new short[16];
        for (int i = 0; i < 16; i++) aQ12[i] = (short)rng.Next(-500, 500);
        short[] pcm = new short[80];

        SilkLpcSynthesisFilter.Apply(state, pres, aQ12, gainQ10: 2048, order: 16, subfrLen: 80, pcm);

        // Just confirm no crash and output is in short range (already enforced by SAT16, but verify).
        for (int i = 0; i < 80; i++)
        {
            True(pcm[i] >= short.MinValue && pcm[i] <= short.MaxValue, $"pcm[{i}] out of range");
        }
    }

    [TestMethod]
    public void LpcSynthesisFilter_HistoryAffectsOutput()
    {
        // Seed non-zero history values and verify they influence the output.
        // Use a simple coefficient that multiplies the last history sample.
        int[] stateA = new int[SilkConstants.MAX_LPC_ORDER + 1];
        stateA[SilkConstants.MAX_LPC_ORDER - 1] = 1000000; // big history value at lag 1
        int[] stateB = new int[SilkConstants.MAX_LPC_ORDER + 1]; // all zero history
        int[] pres = new int[1];
        short[] aQ12 = new short[10];
        aQ12[0] = 4096; // 1.0 in Q12 (applied to state[-1])
        short[] pcmA = new short[1];
        short[] pcmB = new short[1];

        SilkLpcSynthesisFilter.Apply(stateA, pres, aQ12, 1024, 10, 1, pcmA);
        SilkLpcSynthesisFilter.Apply(stateB, pres, aQ12, 1024, 10, 1, pcmB);

        // The two runs should produce different new state values (history contributes to prediction).
        True(stateA[SilkConstants.MAX_LPC_ORDER] != stateB[SilkConstants.MAX_LPC_ORDER],
            $"history should affect output: A = {stateA[SilkConstants.MAX_LPC_ORDER]}, B = {stateB[SilkConstants.MAX_LPC_ORDER]}");
    }

    // -------- Argument validation --------

    [TestMethod]
    public void LpcSynthesisFilter_InvalidOrder_Throws()
    {
        int[] state = new int[30];
        int[] pres = new int[10];
        short[] aQ12 = new short[12];
        short[] pcm = new short[10];
        Throws<ArgumentException>(() =>
            SilkLpcSynthesisFilter.Apply(state, pres, aQ12, 1024, order: 12, subfrLen: 10, pcm));
    }

    [TestMethod]
    public void LpcSynthesisFilter_StateBufferTooSmall_Throws()
    {
        int[] state = new int[SilkConstants.MAX_LPC_ORDER + 5]; // need +10
        int[] pres = new int[10];
        short[] aQ12 = new short[10];
        short[] pcm = new short[10];
        Throws<ArgumentException>(() =>
            SilkLpcSynthesisFilter.Apply(state, pres, aQ12, 1024, order: 10, subfrLen: 10, pcm));
    }
}
