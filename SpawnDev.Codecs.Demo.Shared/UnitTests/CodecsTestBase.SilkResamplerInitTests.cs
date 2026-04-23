using SpawnDev.Codecs.Audio.Opus.Silk;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for <see cref="SilkResampler.Init"/> dispatch logic and the pass-through
/// <see cref="SilkResampler.Apply"/> path (identity rate). Upsample / downsample
/// variants are stubbed in this slice and throw <see cref="NotImplementedException"/>;
/// their tests will land alongside the implementation slices.
/// </summary>
public abstract partial class CodecsTestBase
{
    // -------- Init: rate routing --------

    [TestMethod]
    public void ResamplerInit_Decoder_IdentityRate_SelectsCopyPath()
    {
        var state = new SilkResamplerState();
        SilkResampler.Init(state, fsHzIn: 16000, fsHzOut: 16000, forEncode: false);
        Equal(SilkResamplerConstants.USE_COPY, state.ResamplerFunction);
        Equal(16, state.FsInKHz);
        Equal(16, state.FsOutKHz);
    }

    [TestMethod]
    public void ResamplerInit_Decoder_8To16_SelectsUp2Path()
    {
        var state = new SilkResamplerState();
        SilkResampler.Init(state, fsHzIn: 8000, fsHzOut: 16000, forEncode: false);
        Equal(SilkResamplerConstants.USE_UP2_HQ_WRAPPER, state.ResamplerFunction);
    }

    [TestMethod]
    public void ResamplerInit_Decoder_16To48_SelectsIirFirPath()
    {
        var state = new SilkResamplerState();
        SilkResampler.Init(state, fsHzIn: 16000, fsHzOut: 48000, forEncode: false);
        Equal(SilkResamplerConstants.USE_IIR_FIR, state.ResamplerFunction);
    }

    [TestMethod]
    public void ResamplerInit_Decoder_16To24_SelectsIirFirPath()
    {
        // 16 -> 24 kHz is a 3/2 upsample, not exact 2x, so it uses IIR_FIR.
        var state = new SilkResamplerState();
        SilkResampler.Init(state, fsHzIn: 16000, fsHzOut: 24000, forEncode: false);
        Equal(SilkResamplerConstants.USE_IIR_FIR, state.ResamplerFunction);
    }

    [TestMethod]
    public void ResamplerInit_Decoder_16To8_SelectsDownFirPath()
    {
        var state = new SilkResamplerState();
        SilkResampler.Init(state, fsHzIn: 16000, fsHzOut: 8000, forEncode: false);
        Equal(SilkResamplerConstants.USE_DOWN_FIR, state.ResamplerFunction);
        Equal(1, state.FirFracs);
        Equal(SilkResamplerConstants.DOWN_ORDER_FIR1, state.FirOrder);
    }

    [TestMethod]
    public void ResamplerInit_Decoder_12To8_SelectsDownFirPath_3Over4()
    {
        // 12 * 4 == 8 * 6, hmm. Actually silk's condition is: fs_out * 4 == fs_in * 3
        // (i.e. 3/4 downsample): 8 * 4 = 32, 12 * 3 = 36. Nope. Try other:
        // fs_out * 3 == fs_in * 2: 8 * 3 = 24, 12 * 2 = 24. YES => 2/3 downsample.
        var state = new SilkResamplerState();
        SilkResampler.Init(state, fsHzIn: 12000, fsHzOut: 8000, forEncode: false);
        Equal(SilkResamplerConstants.USE_DOWN_FIR, state.ResamplerFunction);
        Equal(2, state.FirFracs);
        Equal(SilkResamplerConstants.DOWN_ORDER_FIR0, state.FirOrder);
    }

    // -------- Init: delay matrix --------

    [TestMethod]
    public void ResamplerInit_DelayMatrix_Decoder_MatchesLibopus()
    {
        var state = new SilkResamplerState();

        // Decoder row for fs_in=16: {0, 3, 12, 7, 7, 7} for outputs {8, 12, 16, 24, 48, 96}.
        SilkResampler.Init(state, 16000, 8000, forEncode: false);
        Equal(0, state.InputDelay);

        SilkResampler.Init(state, 16000, 12000, forEncode: false);
        Equal(3, state.InputDelay);

        SilkResampler.Init(state, 16000, 16000, forEncode: false);
        Equal(12, state.InputDelay);

        SilkResampler.Init(state, 16000, 24000, forEncode: false);
        Equal(7, state.InputDelay);

        SilkResampler.Init(state, 16000, 48000, forEncode: false);
        Equal(7, state.InputDelay);
    }

    [TestMethod]
    public void ResamplerInit_DelayMatrix_Decoder_Nb_MatchesLibopus()
    {
        var state = new SilkResamplerState();

        // Decoder row for fs_in=8: {4, 0, 2, 0, 0, 0}.
        SilkResampler.Init(state, 8000, 8000, forEncode: false);
        Equal(4, state.InputDelay);

        SilkResampler.Init(state, 8000, 16000, forEncode: false);
        Equal(2, state.InputDelay);
    }

    // -------- Init: unsupported rates --------

    [TestMethod]
    public void ResamplerInit_Decoder_InvalidInputRate_Throws()
    {
        var state = new SilkResamplerState();
        Throws<ArgumentException>(() =>
            SilkResampler.Init(state, 24000, 48000, forEncode: false));
        Throws<ArgumentException>(() =>
            SilkResampler.Init(state, 11000, 8000, forEncode: false));
    }

    [TestMethod]
    public void ResamplerInit_Encoder_InvalidOutputRate_Throws()
    {
        var state = new SilkResamplerState();
        Throws<ArgumentException>(() =>
            SilkResampler.Init(state, 48000, 48000, forEncode: true));
    }

    // -------- Apply: identity / pass-through --------

    [TestMethod]
    public void ResamplerApply_IdentityRate_PassesThroughOverOneBatch()
    {
        // 16 kHz in, 16 kHz out, 10 ms batch = 160 samples.
        var state = new SilkResamplerState();
        SilkResampler.Init(state, 16000, 16000, forEncode: false);
        int fsInKHz = state.FsInKHz;

        // Input: simple ramp. Output should be the same ramp minus the initial-delay offset.
        short[] input = new short[fsInKHz * 10];
        for (int i = 0; i < input.Length; i++) input[i] = (short)(i + 1);
        short[] output = new short[fsInKHz * 10];

        SilkResampler.Apply(state, output, input, input.Length);

        // After apply:
        //   output[0..FsInKHz) = DelayBuf pre-population (zeros + first FsInKHz-InputDelay input samples)
        //   output[FsInKHz..inLen) = input[FsInKHz-InputDelay..inLen-InputDelay)
        // Since the first time we apply, DelayBuf was zero before input was staged in.
        // Output[0..InputDelay) should be zero (the initial DelayBuf zeros).
        for (int i = 0; i < state.InputDelay; i++)
        {
            Equal((short)0, output[i], $"output[{i}] should be 0 from initial DelayBuf");
        }

        // Output[InputDelay..FsInKHz) = input[0..FsInKHz-InputDelay) = 1..FsInKHz-InputDelay (the ramp).
        for (int i = state.InputDelay; i < fsInKHz; i++)
        {
            short expected = (short)(i - state.InputDelay + 1);
            Equal(expected, output[i], $"output[{i}] should be ramp offset");
        }
    }

    [TestMethod]
    public void ResamplerApply_IdentityRate_StateCarriesAcrossCalls()
    {
        // Two back-to-back identity-rate calls. The second call's first samples should
        // pick up from the first call's delay buffer.
        var state = new SilkResamplerState();
        SilkResampler.Init(state, 8000, 8000, forEncode: false);
        int fsIn = state.FsInKHz;

        short[] input1 = new short[fsIn * 10];
        short[] input2 = new short[fsIn * 10];
        for (int i = 0; i < input1.Length; i++)
        {
            input1[i] = (short)(100 + i);
            input2[i] = (short)(200 + i);
        }
        short[] output1 = new short[fsIn * 10];
        short[] output2 = new short[fsIn * 10];

        SilkResampler.Apply(state, output1, input1, input1.Length);
        SilkResampler.Apply(state, output2, input2, input2.Length);

        // output2's first InputDelay samples should be the tail of input1 (InputDelay samples).
        for (int i = 0; i < state.InputDelay; i++)
        {
            short expected = input1[input1.Length - state.InputDelay + i];
            Equal(expected, output2[i], $"output2[{i}] should carry over from input1's tail");
        }
    }

    [TestMethod]
    public void ResamplerApply_NotImplementedPaths_Throw()
    {
        // Verify the stub paths throw NotImplementedException with a clear message.
        var state = new SilkResamplerState();
        SilkResampler.Init(state, 8000, 16000, forEncode: false); // up2 path
        short[] input = new short[80];
        short[] output = new short[160];
        Throws<NotImplementedException>(() =>
            SilkResampler.Apply(state, output, input, input.Length));

        SilkResampler.Init(state, 16000, 48000, forEncode: false); // iir_fir path
        short[] input2 = new short[160];
        short[] output2 = new short[480];
        Throws<NotImplementedException>(() =>
            SilkResampler.Apply(state, output2, input2, input2.Length));

        SilkResampler.Init(state, 16000, 8000, forEncode: false); // down_fir path
        short[] input3 = new short[160];
        short[] output3 = new short[80];
        Throws<NotImplementedException>(() =>
            SilkResampler.Apply(state, output3, input3, input3.Length));
    }
}
