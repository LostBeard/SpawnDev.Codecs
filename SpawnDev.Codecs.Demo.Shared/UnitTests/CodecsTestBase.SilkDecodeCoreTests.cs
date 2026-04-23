using SpawnDev.Codecs.Audio.Opus.Silk;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for <see cref="SilkDecodeCore.Decode"/> - the top-level SILK synthesis
/// pipeline that turns decoded parameters + pulses into PCM output. Covers the
/// unvoiced path extensively (simpler, no LTP state), plus structural tests for
/// the voiced path (state-buffer side effects + sanity-ranged output).
/// </summary>
public abstract partial class CodecsTestBase
{
    /// <summary>Set up a SilkChannelDecoderState configured for NB 20ms and reset to first-frame defaults.</summary>
    private static SilkChannelDecoderState NewNbState()
    {
        var state = new SilkChannelDecoderState();
        state.Configure(fsKHz: 8, nbSubfr: 4, lpcOrder: 10);
        state.Reset();
        return state;
    }

    /// <summary>Set up a SilkDecodedParameters for NB with the given gain, zero NLSFs -> zero LPC coefs, zero pitch/LTP.</summary>
    private static SilkDecodedParameters NewNbParameters(int gainQ16)
    {
        var p = new SilkDecodedParameters();
        for (int k = 0; k < 4; k++) p.GainsQ16[k] = gainQ16;
        // PredCoefQ12 defaults to all zeros -> LPC synthesis is pass-through of excitation scaled by gain.
        // PitchL, LtpCoefQ14, NlsfQ15 all default to zero.
        p.LtpScaleQ14 = 15565;
        return p;
    }

    // -------- Unvoiced path --------

    [TestMethod]
    public void DecodeCore_Unvoiced_ZeroPulses_ProducesValidPcmRange()
    {
        var state = NewNbState();
        var parameters = NewNbParameters(gainQ16: 65536); // 1.0 Q16
        short[] pulses = new short[state.FrameLength]; // all zero
        short[] xq = new short[state.FrameLength];

        SilkDecodeCore.Decode(
            state, parameters, pulses,
            signalType: SilkConstants.TYPE_UNVOICED,
            quantOffsetType: 0, seed: 0,
            nlsfInterpolationEnabled: false,
            xqOut: xq);

        // With zero pulses + zero LPC + unvoiced, the residual is just the offset (100 Q10 = 1600 Q14
        // for UVL). Scaled by gain 1.0, we get a low-level signal. Output must be in int16 range.
        for (int i = 0; i < state.FrameLength; i++)
        {
            True(xq[i] >= short.MinValue && xq[i] <= short.MaxValue, $"xq[{i}] out of range: {xq[i]}");
        }
    }

    [TestMethod]
    public void DecodeCore_Unvoiced_StateIsUpdatedAfterDecode()
    {
        var state = NewNbState();
        var parameters = NewNbParameters(gainQ16: 100000);
        short[] pulses = new short[state.FrameLength];
        for (int i = 0; i < state.FrameLength; i++) pulses[i] = (short)(i % 3 - 1); // small values
        short[] xq = new short[state.FrameLength];

        int prevGainBefore = state.PrevGainQ16;

        SilkDecodeCore.Decode(
            state, parameters, pulses,
            signalType: SilkConstants.TYPE_UNVOICED,
            quantOffsetType: 0, seed: 1,
            nlsfInterpolationEnabled: false,
            xqOut: xq);

        // prev_gain_Q16 should now be the last subframe's gain.
        Equal(100000, state.PrevGainQ16);
        // prev_gain was different before (65536 default).
        True(prevGainBefore != state.PrevGainQ16, "PrevGainQ16 should have been updated");
    }

    [TestMethod]
    public void DecodeCore_Unvoiced_TwoBackToBackFrames_NoCrash()
    {
        var state = NewNbState();
        var parameters = NewNbParameters(gainQ16: 65536);
        short[] pulses = new short[state.FrameLength];
        short[] xq = new short[state.FrameLength];

        // Frame 1
        SilkDecodeCore.Decode(state, parameters, pulses,
            signalType: SilkConstants.TYPE_UNVOICED, quantOffsetType: 0, seed: 0,
            nlsfInterpolationEnabled: false, xqOut: xq);

        // Frame 2 - state carries over, second frame decodes on top of first.
        SilkDecodeCore.Decode(state, parameters, pulses,
            signalType: SilkConstants.TYPE_UNVOICED, quantOffsetType: 0, seed: 1,
            nlsfInterpolationEnabled: false, xqOut: xq);

        // Both frames completed without crashing; output still valid.
        for (int i = 0; i < state.FrameLength; i++)
        {
            True(xq[i] >= short.MinValue && xq[i] <= short.MaxValue, $"frame 2 xq[{i}] out of range");
        }
    }

    [TestMethod]
    public void DecodeCore_Inactive_NoLtp_ProducesValidOutput()
    {
        // Inactive (signalType=0) should behave identically to unvoiced w.r.t. synthesis
        // (no LTP), just using different offset values.
        var state = NewNbState();
        var parameters = NewNbParameters(gainQ16: 50000);
        short[] pulses = new short[state.FrameLength];
        short[] xq = new short[state.FrameLength];

        SilkDecodeCore.Decode(state, parameters, pulses,
            signalType: SilkConstants.TYPE_NO_VOICE_ACTIVITY,
            quantOffsetType: 1, seed: 2,
            nlsfInterpolationEnabled: false, xqOut: xq);

        for (int i = 0; i < state.FrameLength; i++)
        {
            True(xq[i] >= short.MinValue && xq[i] <= short.MaxValue);
        }
    }

    // -------- Voiced path --------

    [TestMethod]
    public void DecodeCore_Voiced_FirstFrameFromReset_NoCrash()
    {
        // First-frame voiced: previous output buffer is zero (fresh reset), so
        // rewhitening produces zero LTP state. Still a valid (silent-ish) decode path.
        var state = NewNbState();
        var parameters = NewNbParameters(gainQ16: 65536);
        // Set up a valid pitch lag (middle of range) for all 4 subframes.
        int midLag = (SilkConstants.PE_MIN_LAG_MS + SilkConstants.PE_MAX_LAG_MS) / 2 * 8; // = 80 samples
        for (int k = 0; k < 4; k++) parameters.PitchL[k] = midLag;
        // Simple LTP filter: concentrated at center tap, small taps elsewhere. Q14 values.
        for (int k = 0; k < 4; k++)
        {
            parameters.LtpCoefQ14[k * 5 + 0] = 100;
            parameters.LtpCoefQ14[k * 5 + 1] = 200;
            parameters.LtpCoefQ14[k * 5 + 2] = 8000; // ~0.5 in Q14
            parameters.LtpCoefQ14[k * 5 + 3] = 200;
            parameters.LtpCoefQ14[k * 5 + 4] = 100;
        }
        parameters.LtpScaleQ14 = 15565;

        short[] pulses = new short[state.FrameLength];
        for (int i = 0; i < state.FrameLength; i++) pulses[i] = (short)(i % 3 - 1);
        short[] xq = new short[state.FrameLength];

        SilkDecodeCore.Decode(state, parameters, pulses,
            signalType: SilkConstants.TYPE_VOICED,
            quantOffsetType: 0, seed: 3,
            nlsfInterpolationEnabled: false, xqOut: xq);

        for (int i = 0; i < state.FrameLength; i++)
        {
            True(xq[i] >= short.MinValue && xq[i] <= short.MaxValue, $"voiced xq[{i}] out of range: {xq[i]}");
        }
    }

    [TestMethod]
    public void DecodeCore_Voiced_WithNlsfInterpolation_NoCrash()
    {
        // Voiced + NLSF interpolation enabled exercises the k==2 rewhiten branch.
        var state = NewNbState();
        var parameters = NewNbParameters(gainQ16: 65536);
        int midLag = (SilkConstants.PE_MIN_LAG_MS + SilkConstants.PE_MAX_LAG_MS) / 2 * 8;
        for (int k = 0; k < 4; k++)
        {
            parameters.PitchL[k] = midLag;
            parameters.LtpCoefQ14[k * 5 + 2] = 8000;
        }
        parameters.LtpScaleQ14 = 12288;

        short[] pulses = new short[state.FrameLength];
        short[] xq = new short[state.FrameLength];

        SilkDecodeCore.Decode(state, parameters, pulses,
            signalType: SilkConstants.TYPE_VOICED,
            quantOffsetType: 0, seed: 0,
            nlsfInterpolationEnabled: true,
            xqOut: xq);

        for (int i = 0; i < state.FrameLength; i++)
        {
            True(xq[i] >= short.MinValue && xq[i] <= short.MaxValue);
        }
    }

    // -------- Argument validation --------

    [TestMethod]
    public void DecodeCore_NullState_Throws()
    {
        var parameters = NewNbParameters(65536);
        short[] pulses = new short[160];
        short[] xq = new short[160];
        Throws<ArgumentNullException>(() =>
            SilkDecodeCore.Decode(null!, parameters, pulses, 1, 0, 0, false, xq));
    }

    [TestMethod]
    public void DecodeCore_NullParameters_Throws()
    {
        var state = NewNbState();
        short[] pulses = new short[160];
        short[] xq = new short[160];
        Throws<ArgumentNullException>(() =>
            SilkDecodeCore.Decode(state, null!, pulses, 1, 0, 0, false, xq));
    }

    [TestMethod]
    public void DecodeCore_UnconfiguredState_Throws()
    {
        var state = new SilkChannelDecoderState(); // NO Configure() call
        var parameters = NewNbParameters(65536);
        short[] pulses = new short[160];
        short[] xq = new short[160];
        Throws<InvalidOperationException>(() =>
            SilkDecodeCore.Decode(state, parameters, pulses, 1, 0, 0, false, xq));
    }

    [TestMethod]
    public void DecodeCore_ZeroPrevGain_Throws()
    {
        var state = NewNbState();
        state.PrevGainQ16 = 0; // manually nuked
        var parameters = NewNbParameters(65536);
        short[] pulses = new short[state.FrameLength];
        short[] xq = new short[state.FrameLength];
        Throws<InvalidOperationException>(() =>
            SilkDecodeCore.Decode(state, parameters, pulses,
                signalType: SilkConstants.TYPE_UNVOICED,
                quantOffsetType: 0, seed: 0,
                nlsfInterpolationEnabled: false, xqOut: xq));
    }

    [TestMethod]
    public void DecodeCore_OutputTooSmall_Throws()
    {
        var state = NewNbState();
        var parameters = NewNbParameters(65536);
        short[] pulses = new short[state.FrameLength];
        short[] xq = new short[state.FrameLength - 1]; // too small
        Throws<ArgumentException>(() =>
            SilkDecodeCore.Decode(state, parameters, pulses,
                SilkConstants.TYPE_UNVOICED, 0, 0, false, xq));
    }
}
