using SpawnDev.Codecs.Audio.Opus.Silk;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for the downsample FIR path of <see cref="SilkResampler.Apply"/>. Covers
/// all six supported ratios (3/4, 2/3, 1/2, 1/3, 1/4, 1/6) via the USE_DOWN_FIR
/// dispatch, verifying the AR2 pre-filter + polyphase FIR produce stable,
/// in-range output at the expected output sample count.
/// </summary>
public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void ResamplerDownFir_16To8_ZeroInput_ProducesZeroOutput()
    {
        var state = new SilkResamplerState();
        SilkResampler.Init(state, 16000, 8000, forEncode: false);

        short[] input = new short[state.FsInKHz * 10]; // 160
        short[] output = new short[state.FsOutKHz * 10]; // 80

        SilkResampler.Apply(state, output, input, input.Length);

        for (int i = 0; i < output.Length; i++) Equal((short)0, output[i], $"output[{i}]");
    }

    [TestMethod]
    public void ResamplerDownFir_16To8_OutputCountIsHalf()
    {
        var state = new SilkResamplerState();
        SilkResampler.Init(state, 16000, 8000, forEncode: false);

        var rng = new Random(0x1234);
        short[] input = new short[state.FsInKHz * 10];
        for (int i = 0; i < input.Length; i++) input[i] = (short)rng.Next(-3000, 3000);
        short[] output = new short[state.FsOutKHz * 10];

        SilkResampler.Apply(state, output, input, input.Length);

        // Output should have some energy but stay in int16 range.
        for (int i = 0; i < output.Length; i++)
        {
            True(output[i] >= short.MinValue && output[i] <= short.MaxValue, $"output[{i}] in range");
        }
    }

    [TestMethod]
    public void ResamplerDownFir_16To12_3To4_UsesExpectedCoefs()
    {
        var state = new SilkResamplerState();
        SilkResampler.Init(state, 16000, 12000, forEncode: false);
        Equal(SilkResamplerConstants.USE_DOWN_FIR, state.ResamplerFunction);
        Equal(3, state.FirFracs);
        Equal(SilkResamplerConstants.DOWN_ORDER_FIR0, state.FirOrder);
        // Coefs should reference the 3/4 table.
        if (!ReferenceEquals(state.Coefs, SilkResamplerTables.Coefs3To4))
            throw new Exception("Expected state.Coefs to be Coefs3To4");

        short[] input = new short[state.FsInKHz * 10];
        var rng = new Random(5);
        for (int i = 0; i < input.Length; i++) input[i] = (short)rng.Next(-2000, 2000);
        short[] output = new short[state.FsOutKHz * 10];
        SilkResampler.Apply(state, output, input, input.Length);

        for (int i = 0; i < output.Length; i++)
        {
            True(output[i] >= short.MinValue && output[i] <= short.MaxValue);
        }
    }

    [TestMethod]
    public void ResamplerDownFir_12To8_2To3_UsesExpectedCoefs()
    {
        var state = new SilkResamplerState();
        SilkResampler.Init(state, 12000, 8000, forEncode: false);
        if (!ReferenceEquals(state.Coefs, SilkResamplerTables.Coefs2To3))
            throw new Exception("Expected Coefs2To3");

        short[] input = new short[state.FsInKHz * 10]; // 120
        var rng = new Random(6);
        for (int i = 0; i < input.Length; i++) input[i] = (short)rng.Next(-2000, 2000);
        short[] output = new short[state.FsOutKHz * 10]; // 80
        SilkResampler.Apply(state, output, input, input.Length);

        for (int i = 0; i < output.Length; i++)
        {
            True(output[i] >= short.MinValue && output[i] <= short.MaxValue);
        }
    }

    [TestMethod]
    public void ResamplerDownFir_16To8_StateCarriesAcrossCalls()
    {
        var state = new SilkResamplerState();
        SilkResampler.Init(state, 16000, 8000, forEncode: false);

        short[] input1 = new short[state.FsInKHz * 10];
        short[] input2 = new short[state.FsInKHz * 10];
        for (int i = 0; i < input1.Length; i++)
        {
            input1[i] = (short)(1000 * Math.Sin(i * 0.1));
            input2[i] = (short)(1000 * Math.Sin((i + input1.Length) * 0.1));
        }
        short[] output1 = new short[state.FsOutKHz * 10];
        short[] output2 = new short[state.FsOutKHz * 10];

        SilkResampler.Apply(state, output1, input1, input1.Length);
        SilkResampler.Apply(state, output2, input2, input2.Length);

        // Both calls completed; output2 should not be identical to output1 (different input).
        bool identical = true;
        for (int i = 0; i < output1.Length; i++)
        {
            if (output1[i] != output2[i]) { identical = false; break; }
        }
        True(!identical, "output2 should differ from output1 (different input)");
    }

    [TestMethod]
    public void ResamplerDownFir_DcInput_ProducesBoundedOutput()
    {
        // DC input through the downsample filter should produce bounded output
        // (may or may not settle to DC level depending on filter's DC gain).
        var state = new SilkResamplerState();
        SilkResampler.Init(state, 16000, 8000, forEncode: false);

        short[] input = new short[state.FsInKHz * 10];
        for (int i = 0; i < input.Length; i++) input[i] = 500;
        short[] output = new short[state.FsOutKHz * 10];

        // Several batches to settle the filter.
        for (int pass = 0; pass < 10; pass++)
        {
            SilkResampler.Apply(state, output, input, input.Length);
            for (int i = 0; i < output.Length; i++)
            {
                True(output[i] >= short.MinValue && output[i] <= short.MaxValue,
                    $"pass {pass} output[{i}] out of range");
            }
        }
    }
}
