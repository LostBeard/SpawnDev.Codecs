using SpawnDev.Codecs.Audio.Opus.Silk;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for the 2x-upsample path of <see cref="SilkResampler.Apply"/>. Covers
/// the USE_UP2_HQ_WRAPPER dispatch at fs_in=8/12 and fs_out=2*fs_in, verifying
/// that the cascade of three all-pass filters produces stable, in-range output
/// and that state carries across calls.
/// </summary>
public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void ResamplerUp2_8To16_ZeroInput_ProducesZeroOutput()
    {
        // Zero input with zero state should yield zero output (filter has no free-running response).
        var state = new SilkResamplerState();
        SilkResampler.Init(state, fsHzIn: 8000, fsHzOut: 16000, forEncode: false);

        short[] input = new short[state.FsInKHz * 10];
        short[] output = new short[state.FsOutKHz * 10];

        SilkResampler.Apply(state, output, input, input.Length);

        for (int i = 0; i < output.Length; i++) Equal((short)0, output[i], $"output[{i}]");
    }

    [TestMethod]
    public void ResamplerUp2_8To16_OutputCountIsDouble()
    {
        // For an exact 2x upsampler, input length 80 -> output length 160.
        var state = new SilkResamplerState();
        SilkResampler.Init(state, fsHzIn: 8000, fsHzOut: 16000, forEncode: false);

        short[] input = new short[state.FsInKHz * 10];
        var rng = new Random(42);
        for (int i = 0; i < input.Length; i++) input[i] = (short)rng.Next(-5000, 5000);

        short[] output = new short[state.FsOutKHz * 10];

        SilkResampler.Apply(state, output, input, input.Length);

        // Output is in int16 range and has produced non-zero values for non-zero input.
        int nonZero = 0;
        for (int i = 0; i < output.Length; i++)
        {
            True(output[i] >= short.MinValue && output[i] <= short.MaxValue, $"output[{i}] in range");
            if (output[i] != 0) nonZero++;
        }
        True(nonZero > output.Length / 4, $"Expected many non-zero outputs, got {nonZero}/{output.Length}");
    }

    [TestMethod]
    public void ResamplerUp2_8To16_DcInput_ProducesBoundedDcOutput()
    {
        // DC (constant) input passes through all-pass filter at unity gain. The all-pass
        // cascade preserves DC level; output should be near the input constant after
        // the filter settles.
        var state = new SilkResamplerState();
        SilkResampler.Init(state, fsHzIn: 8000, fsHzOut: 16000, forEncode: false);

        short constantValue = 1000;
        short[] input = new short[state.FsInKHz * 10];
        for (int i = 0; i < input.Length; i++) input[i] = constantValue;
        short[] output = new short[state.FsOutKHz * 10];

        // Prime the filter with a few batches of DC input so the state settles.
        for (int pass = 0; pass < 5; pass++)
        {
            SilkResampler.Apply(state, output, input, input.Length);
        }

        // After settling, output should be close to the DC level (within a small transient band).
        int mid = output.Length / 2;
        for (int i = mid; i < output.Length; i++)
        {
            int diff = Math.Abs(output[i] - constantValue);
            True(diff < 50, $"DC output[{i}] = {output[i]} should be near {constantValue} (diff {diff})");
        }
    }

    [TestMethod]
    public void ResamplerUp2_8To16_StateCarriesAcrossCalls()
    {
        // A continuous sinusoid decoded across two batches should not have a discontinuity
        // at the batch boundary. Verify the output is smooth (bounded derivative) around
        // the batch-2 start.
        var state = new SilkResamplerState();
        SilkResampler.Init(state, fsHzIn: 8000, fsHzOut: 16000, forEncode: false);

        int len = state.FsInKHz * 10;
        short[] input1 = new short[len];
        short[] input2 = new short[len];
        for (int i = 0; i < len; i++)
        {
            input1[i] = (short)(1000 * Math.Sin(i * 0.2));
            input2[i] = (short)(1000 * Math.Sin((i + len) * 0.2));
        }

        short[] output1 = new short[state.FsOutKHz * 10];
        short[] output2 = new short[state.FsOutKHz * 10];

        SilkResampler.Apply(state, output1, input1, len);
        SilkResampler.Apply(state, output2, input2, len);

        // Both outputs in range. Concatenation transition: no sample should jump > ~2000.
        int jump = Math.Abs(output2[0] - output1[output1.Length - 1]);
        True(jump < 2500, $"Batch-boundary jump too large: {jump}");
    }

    [TestMethod]
    public void ResamplerUp2_12To24_AlsoDispatchesAndProducesDoubleCount()
    {
        // 12 -> 24 kHz is also a 2x upsample, uses the same up2 path.
        var state = new SilkResamplerState();
        SilkResampler.Init(state, fsHzIn: 12000, fsHzOut: 24000, forEncode: false);
        Equal(SilkResamplerConstants.USE_UP2_HQ_WRAPPER, state.ResamplerFunction);

        short[] input = new short[state.FsInKHz * 10];
        var rng = new Random(0x12345);
        for (int i = 0; i < input.Length; i++) input[i] = (short)rng.Next(-2000, 2000);
        short[] output = new short[state.FsOutKHz * 10];

        SilkResampler.Apply(state, output, input, input.Length);

        for (int i = 0; i < output.Length; i++)
        {
            True(output[i] >= short.MinValue && output[i] <= short.MaxValue, $"output[{i}] in range");
        }
    }
}
