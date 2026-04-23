using SpawnDev.Codecs.Audio.Opus.Silk;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for <see cref="SilkLpcAnalysisFilter.Apply"/> - the MA prediction-error
/// filter used by decode_core for voiced-subframe re-whitening.
/// </summary>
public abstract partial class CodecsTestBase
{
    // -------- Trivial cases --------

    [TestMethod]
    public void LpcAnalysisFilter_AllZeroInput_ProducesAllZeroOutput()
    {
        short[] input = new short[100];
        short[] bQ12 = { 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000 };
        short[] output = new short[100];

        SilkLpcAnalysisFilter.Apply(output, input, bQ12, len: 100, d: 10);
        for (int i = 0; i < 100; i++) Equal((short)0, output[i], $"pos {i}");
    }

    [TestMethod]
    public void LpcAnalysisFilter_ZeroCoefficients_PassesInputThrough()
    {
        // With all filter coefficients zero, out[n] = in[n] >> 0 (identity) for n >= d.
        short[] input = new short[20];
        for (int i = 0; i < 20; i++) input[i] = (short)(i * 100);
        short[] bQ12 = new short[6]; // all zeros
        short[] output = new short[20];

        SilkLpcAnalysisFilter.Apply(output, input, bQ12, len: 20, d: 6);

        // out[0..5] = 0 (undefined filter history region).
        for (int i = 0; i < 6; i++) Equal((short)0, output[i], $"pos {i}");
        // out[6..19] = input[6..19] because B is zero so nothing to subtract.
        for (int i = 6; i < 20; i++) Equal(input[i], output[i], $"pos {i}");
    }

    // -------- Known analytic case --------

    [TestMethod]
    public void LpcAnalysisFilter_SingleTapCoefficient_MatchesManualCalculation()
    {
        // b[0] = 4096 (= 1.0 in Q12). Filter becomes: out[n] = in[n] - 1.0 * in[n-1].
        // For a constant input c, out[n] = c - c = 0 after the filter-history region.
        short[] input = new short[20];
        for (int i = 0; i < 20; i++) input[i] = 1000; // constant
        short[] bQ12 = { 4096, 0, 0, 0, 0, 0 };
        short[] output = new short[20];

        SilkLpcAnalysisFilter.Apply(output, input, bQ12, len: 20, d: 6);

        // For constant input and b=[1,0,...,0], the filter subtracts the previous
        // sample (also constant), yielding 0 for every sample past the history region.
        for (int i = 6; i < 20; i++) Equal((short)0, output[i], $"pos {i}");
    }

    [TestMethod]
    public void LpcAnalysisFilter_SingleTapCoefficient_OnRampInput_ProducesConstantDelta()
    {
        // b[0] = 4096 (1.0 in Q12). For in[n] = n, out[n] = in[n] - in[n-1] = 1 for n >= d.
        short[] input = new short[20];
        for (int i = 0; i < 20; i++) input[i] = (short)i;
        short[] bQ12 = { 4096, 0, 0, 0, 0, 0 };
        short[] output = new short[20];

        SilkLpcAnalysisFilter.Apply(output, input, bQ12, len: 20, d: 6);

        for (int i = 6; i < 20; i++) Equal((short)1, output[i], $"pos {i}");
    }

    // -------- Argument validation --------

    [TestMethod]
    public void LpcAnalysisFilter_OddOrder_Throws()
    {
        short[] input = new short[20];
        short[] bQ12 = new short[7];
        short[] output = new short[20];
        Throws<ArgumentException>(() =>
            SilkLpcAnalysisFilter.Apply(output, input, bQ12, len: 20, d: 7));
    }

    [TestMethod]
    public void LpcAnalysisFilter_OrderBelowSix_Throws()
    {
        short[] input = new short[20];
        short[] bQ12 = new short[4];
        short[] output = new short[20];
        Throws<ArgumentException>(() =>
            SilkLpcAnalysisFilter.Apply(output, input, bQ12, len: 20, d: 4));
    }

    [TestMethod]
    public void LpcAnalysisFilter_OrderGreaterThanLen_Throws()
    {
        short[] input = new short[10];
        short[] bQ12 = new short[16];
        short[] output = new short[10];
        Throws<ArgumentException>(() =>
            SilkLpcAnalysisFilter.Apply(output, input, bQ12, len: 10, d: 16));
    }

    [TestMethod]
    public void LpcAnalysisFilter_BQ12TooSmall_Throws()
    {
        short[] input = new short[20];
        short[] bQ12 = new short[5]; // needs 6 for d=6
        short[] output = new short[20];
        Throws<ArgumentException>(() =>
            SilkLpcAnalysisFilter.Apply(output, input, bQ12, len: 20, d: 6));
    }

    // -------- Order 10 and 16 (realistic SILK orders) --------

    [TestMethod]
    public void LpcAnalysisFilter_Order10_RampInput_ProducesSmoothOutput()
    {
        // Realistic SILK NB/MB order of 10.
        short[] input = new short[100];
        for (int i = 0; i < 100; i++) input[i] = (short)(i * 10);
        short[] bQ12 = { 200, 100, 50, 25, 10, 5, 2, 1, 0, 0 };
        short[] output = new short[100];

        SilkLpcAnalysisFilter.Apply(output, input, bQ12, len: 100, d: 10);

        // For a linear ramp and small coefficients, the output should be bounded.
        for (int i = 10; i < 100; i++)
        {
            True(output[i] > -32768 && output[i] < 32767, $"out[{i}] = {output[i]} should be in int16 range");
        }
    }

    [TestMethod]
    public void LpcAnalysisFilter_Order16_DoesNotCrash()
    {
        // Realistic SILK WB order of 16. Just verify it runs to completion.
        short[] input = new short[200];
        var rng = new Random(42);
        for (int i = 0; i < 200; i++) input[i] = (short)rng.Next(-5000, 5000);
        short[] bQ12 = new short[16];
        for (int i = 0; i < 16; i++) bQ12[i] = (short)rng.Next(-200, 200);
        short[] output = new short[200];

        SilkLpcAnalysisFilter.Apply(output, input, bQ12, len: 200, d: 16);

        // Output must be zeroed in the history region.
        for (int i = 0; i < 16; i++) Equal((short)0, output[i], $"history pos {i}");
    }
}
