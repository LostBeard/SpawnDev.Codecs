using SpawnDev.Codecs.Audio.Opus.Silk;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for <see cref="SilkSigmoid"/> - the Q15 sigmoid approximation used by SILK.
/// Reference points are hand-derived from the lookup tables in libopus sigm_Q15.c;
/// interpolation behavior is verified by checking that fractional inputs fall
/// between adjacent LUT entries.
/// </summary>
public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Sigmoid_InputZero_ReturnsHalf()
    {
        // Input 0 in Q5 -> index 0 -> sigm_LUT_pos_Q15[0] = 16384 (= 0.5 in Q15).
        Equal(16384, SilkSigmoid.silk_sigm_Q15(0));
    }

    [TestMethod]
    public void Sigmoid_ExactLutPoints_MatchTable()
    {
        // Positive branch, exact LUT indices (inQ5 = 0, 32, 64, 96, 128, 160).
        int[] expectedPos = { 16384, 23955, 28861, 31213, 32178, 32548 };
        for (int i = 0; i < 6; i++)
        {
            int inQ5 = i * 32;
            Equal(expectedPos[i], SilkSigmoid.silk_sigm_Q15(inQ5), $"inQ5={inQ5}");
        }

        // Negative branch, exact LUT indices (inQ5 = 0, -32, -64, -96, -128, -160).
        int[] expectedNeg = { 16384, 8812, 3906, 1554, 589, 219 };
        for (int i = 0; i < 6; i++)
        {
            int inQ5 = -i * 32;
            Equal(expectedNeg[i], SilkSigmoid.silk_sigm_Q15(inQ5), $"inQ5={inQ5}");
        }
    }

    [TestMethod]
    public void Sigmoid_LargePositive_ClipsTo32767()
    {
        Equal(32767, SilkSigmoid.silk_sigm_Q15(6 * 32));
        Equal(32767, SilkSigmoid.silk_sigm_Q15(1000));
        Equal(32767, SilkSigmoid.silk_sigm_Q15(int.MaxValue));
    }

    [TestMethod]
    public void Sigmoid_LargeNegative_ClipsToZero()
    {
        Equal(0, SilkSigmoid.silk_sigm_Q15(-6 * 32));
        Equal(0, SilkSigmoid.silk_sigm_Q15(-1000));
        Equal(0, SilkSigmoid.silk_sigm_Q15(int.MinValue + 1));
    }

    [TestMethod]
    public void Sigmoid_NearlyMonotonic_AcrossRange()
    {
        // The libopus LUT approximation is NOT strictly monotonic at LUT-entry transitions
        // because each LUT value is independently rounded. However the function is
        // monotonic WITHIN each 32-sample LUT segment, and the overall envelope is
        // increasing. Verify: any decrease across neighbors is bounded by a small constant.
        int prev = 0;
        const int MaxLocalDecrease = 5; // empirical bound across LUT-transition jumps
        for (int inQ5 = -6 * 32; inQ5 < 6 * 32; inQ5++)
        {
            int cur = SilkSigmoid.silk_sigm_Q15(inQ5);
            if (prev - cur > MaxLocalDecrease)
                throw new Exception($"Large non-monotonic jump at inQ5={inQ5}: {prev} -> {cur}");
            prev = cur;
        }
    }

    [TestMethod]
    public void Sigmoid_MidpointPositive_ExactInterpolation()
    {
        // inQ5 = 16 (half-step between 0 and 32). Expected:
        //   sigm_LUT_pos_Q15[0] + slope[0] * (16 & 0x1F)
        //   = 16384 + 237 * 16
        //   = 16384 + 3792 = 20176.
        Equal(16384 + 237 * 16, SilkSigmoid.silk_sigm_Q15(16));
    }

    [TestMethod]
    public void Sigmoid_MidpointNegative_ExactInterpolation()
    {
        // inQ5 = -16 -> in = 16 (abs), index 0, neg branch:
        //   sigm_LUT_neg_Q15[0] - slope[0] * 16
        //   = 16384 - 237 * 16
        //   = 16384 - 3792 = 12592.
        Equal(16384 - 237 * 16, SilkSigmoid.silk_sigm_Q15(-16));
    }

    [TestMethod]
    public void Sigmoid_ApproximateSymmetryAroundOrigin()
    {
        // sigmoid(x) + sigmoid(-x) == 1 in continuous math. In this Q15 LUT approximation,
        // pos[i] and neg[i] are rounded independently, so the sum at non-zero exact LUT
        // points is 32767 (off by 1 from the ideal 32768). At the origin it's exactly
        // 32768 (16384 + 16384).
        for (int i = 0; i < 6; i++)
        {
            int inQ5 = i * 32;
            int sum = SilkSigmoid.silk_sigm_Q15(inQ5) + SilkSigmoid.silk_sigm_Q15(-inQ5);
            int diff = Math.Abs(sum - 32768);
            if (diff > 1)
                throw new Exception($"Symmetry diff too large at inQ5={inQ5}: sum={sum} (expected 32767 or 32768)");
        }
    }
}
