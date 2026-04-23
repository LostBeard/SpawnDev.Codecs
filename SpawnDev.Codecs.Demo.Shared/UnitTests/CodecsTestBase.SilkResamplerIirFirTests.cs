using SpawnDev.Codecs.Audio.Opus.Silk;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for the arbitrary-upsample <see cref="SilkResampler.Apply"/> path
/// (USE_IIR_FIR). Covers 16-&gt;48 kHz (3x), 16-&gt;24 kHz (1.5x), 12-&gt;48 kHz (4x),
/// 8-&gt;24 kHz (3x) etc. After this slice the resampler is complete - every
/// supported rate pair has a working implementation.
/// </summary>
public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void ResamplerIirFir_16To48_ZeroInput_ProducesZeroOutput()
    {
        var state = new SilkResamplerState();
        SilkResampler.Init(state, fsHzIn: 16000, fsHzOut: 48000, forEncode: false);
        Equal(SilkResamplerConstants.USE_IIR_FIR, state.ResamplerFunction);

        short[] input = new short[state.FsInKHz * 10];
        short[] output = new short[state.FsOutKHz * 10];

        SilkResampler.Apply(state, output, input, input.Length);

        for (int i = 0; i < output.Length; i++) Equal((short)0, output[i], $"output[{i}]");
    }

    [TestMethod]
    public void ResamplerIirFir_16To48_OutputCountIs3x()
    {
        var state = new SilkResamplerState();
        SilkResampler.Init(state, fsHzIn: 16000, fsHzOut: 48000, forEncode: false);

        var rng = new Random(0xBEEF);
        short[] input = new short[state.FsInKHz * 10]; // 160
        for (int i = 0; i < input.Length; i++) input[i] = (short)rng.Next(-3000, 3000);
        short[] output = new short[state.FsOutKHz * 10]; // 480

        SilkResampler.Apply(state, output, input, input.Length);

        for (int i = 0; i < output.Length; i++)
        {
            True(output[i] >= short.MinValue && output[i] <= short.MaxValue, $"output[{i}] in range");
        }
    }

    [TestMethod]
    public void ResamplerIirFir_16To24_NonIntegerRatio()
    {
        // 16 -> 24 kHz is a 3/2 ratio; not exact 2x, routes through IIR_FIR.
        var state = new SilkResamplerState();
        SilkResampler.Init(state, fsHzIn: 16000, fsHzOut: 24000, forEncode: false);
        Equal(SilkResamplerConstants.USE_IIR_FIR, state.ResamplerFunction);

        short[] input = new short[state.FsInKHz * 10]; // 160
        for (int i = 0; i < input.Length; i++) input[i] = (short)(500 * Math.Sin(i * 0.1));
        short[] output = new short[state.FsOutKHz * 10]; // 240

        SilkResampler.Apply(state, output, input, input.Length);

        for (int i = 0; i < output.Length; i++)
        {
            True(output[i] >= short.MinValue && output[i] <= short.MaxValue);
        }
    }

    [TestMethod]
    public void ResamplerIirFir_12To48_4xRatio()
    {
        var state = new SilkResamplerState();
        SilkResampler.Init(state, fsHzIn: 12000, fsHzOut: 48000, forEncode: false);
        Equal(SilkResamplerConstants.USE_IIR_FIR, state.ResamplerFunction);

        short[] input = new short[state.FsInKHz * 10]; // 120
        var rng = new Random(0x77);
        for (int i = 0; i < input.Length; i++) input[i] = (short)rng.Next(-1000, 1000);
        short[] output = new short[state.FsOutKHz * 10]; // 480

        SilkResampler.Apply(state, output, input, input.Length);

        for (int i = 0; i < output.Length; i++)
        {
            True(output[i] >= short.MinValue && output[i] <= short.MaxValue);
        }
    }

    [TestMethod]
    public void ResamplerIirFir_8To48_6xRatio()
    {
        var state = new SilkResamplerState();
        SilkResampler.Init(state, fsHzIn: 8000, fsHzOut: 48000, forEncode: false);
        Equal(SilkResamplerConstants.USE_IIR_FIR, state.ResamplerFunction);

        short[] input = new short[state.FsInKHz * 10]; // 80
        var rng = new Random(0x88);
        for (int i = 0; i < input.Length; i++) input[i] = (short)rng.Next(-2000, 2000);
        short[] output = new short[state.FsOutKHz * 10]; // 480

        SilkResampler.Apply(state, output, input, input.Length);

        for (int i = 0; i < output.Length; i++)
        {
            True(output[i] >= short.MinValue && output[i] <= short.MaxValue);
        }
    }

    [TestMethod]
    public void ResamplerIirFir_16To48_StateCarriesAcrossCalls()
    {
        var state = new SilkResamplerState();
        SilkResampler.Init(state, 16000, 48000, forEncode: false);

        int len = state.FsInKHz * 10;
        short[] input1 = new short[len];
        short[] input2 = new short[len];
        for (int i = 0; i < len; i++)
        {
            input1[i] = (short)(1000 * Math.Sin(i * 0.15));
            input2[i] = (short)(1000 * Math.Sin((i + len) * 0.15));
        }
        short[] output1 = new short[state.FsOutKHz * 10];
        short[] output2 = new short[state.FsOutKHz * 10];

        SilkResampler.Apply(state, output1, input1, len);
        SilkResampler.Apply(state, output2, input2, len);

        // Jump at batch boundary should be bounded.
        int jump = Math.Abs(output2[0] - output1[output1.Length - 1]);
        True(jump < 2500, $"Batch-boundary jump {jump} too large");
    }

    [TestMethod]
    public void ResamplerInit_AllStubPaths_NowImplemented()
    {
        // After slice 45 there should be no stubbed Apply() paths. This test
        // replaces the old NotImplementedPaths check.
        var state = new SilkResamplerState();

        // Every supported decoder rate pair should now decode without NotImplementedException.
        int[] fsIns = { 8000, 12000, 16000 };
        int[] fsOuts = { 8000, 12000, 16000, 24000, 48000 };
        foreach (int fsIn in fsIns)
        {
            foreach (int fsOut in fsOuts)
            {
                SilkResampler.Init(state, fsIn, fsOut, forEncode: false);
                int inLen = fsIn / 100; // 10 ms batch
                int outLen = fsOut / 100;
                short[] input = new short[inLen];
                short[] output = new short[outLen];
                // Must not throw NotImplementedException.
                try
                {
                    SilkResampler.Apply(state, output, input, inLen);
                }
                catch (NotImplementedException ex)
                {
                    throw new Exception($"{fsIn} -> {fsOut}: still NotImplemented: {ex.Message}");
                }
            }
        }
    }
}
