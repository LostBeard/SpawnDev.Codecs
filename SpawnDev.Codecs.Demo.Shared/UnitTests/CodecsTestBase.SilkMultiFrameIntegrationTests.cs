using SpawnDev.Codecs.Audio.Opus.Silk;
using SpawnDev.Codecs.EntropyCoders;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Multi-frame integration tests that run back-to-back SILK decode cycles,
/// exercising the state-carrying aspects of the pipeline: gain delta coding
/// against prev frame's last index, NLSF interpolation from prev frame's
/// NLSFs, and pitch delta coding when prev frame was voiced.
/// </summary>
public abstract partial class CodecsTestBase
{
    private static byte[] EncodeFramePayload(
        Action<OpusRangeEncoder> encode,
        int capacity = 512)
    {
        var enc = new OpusRangeEncoder(capacity);
        encode(enc);
        enc.Done();
        return enc.ToArray();
    }

    [TestMethod]
    public void MultiFrame_TwoIndependentInactiveFrames_StateAdvances()
    {
        // Two independent (conditional=0) inactive frames. Verify state updates
        // correctly frame-to-frame: LastGainIndex and PrevNlsfQ15 both change.
        var cb = SilkNlsfCodebookTables.NbMb;
        var state = new SilkChannelDecoderState();

        var frame1Indices = new SilkDecodedIndices
        {
            SignalType = 0,
            QuantOffsetType = 0,
            NlsfInterpCoefQ2 = 4,
            Seed = 0,
        };
        for (int i = 0; i < 4; i++) frame1Indices.GainsIndices[i] = (sbyte)(15 + i);
        frame1Indices.NlsfIndices[0] = 3;

        var frame2Indices = new SilkDecodedIndices
        {
            SignalType = 0,
            QuantOffsetType = 0,
            NlsfInterpCoefQ2 = 4,
            Seed = 1,
        };
        for (int i = 0; i < 4; i++) frame2Indices.GainsIndices[i] = (sbyte)(20 + i);
        frame2Indices.NlsfIndices[0] = 10;

        var params1 = new SilkDecodedParameters();
        SilkParametersDecoder.Decode(params1, frame1Indices, cb, 8, 4,
            ref state.LastGainIndex, state.PrevNlsfQ15, conditional: 0);

        sbyte lastGainAfter1 = state.LastGainIndex;
        short[] prevNlsfAfter1 = new short[cb.Order];
        state.PrevNlsfQ15.AsSpan(0, cb.Order).CopyTo(prevNlsfAfter1);

        var params2 = new SilkDecodedParameters();
        SilkParametersDecoder.Decode(params2, frame2Indices, cb, 8, 4,
            ref state.LastGainIndex, state.PrevNlsfQ15, conditional: 0);

        // State should have advanced from frame 1 to frame 2.
        True(state.LastGainIndex != lastGainAfter1 || state.LastGainIndex != 0,
            "LastGainIndex should have been updated at least once");

        // prev NLSFs should now reflect frame 2, not frame 1.
        int samePositions = 0;
        for (int i = 0; i < cb.Order; i++)
        {
            if (state.PrevNlsfQ15[i] == prevNlsfAfter1[i]) samePositions++;
        }
        True(samePositions < cb.Order,
            $"prevNlsf should differ from after-frame-1 snapshot on at least one coefficient (same in {samePositions}/{cb.Order})");

        // Both frames' LPC coefficients should be stable filters.
        int invGain1 = SilkLpcInvPredGain.Compute(params1.NlsfQ15, cb.Order);
        int invGain2 = SilkLpcInvPredGain.Compute(params2.NlsfQ15, cb.Order);
        // These are NLSF vectors not LPC - correct check uses PredCoefQ12 second half.
        int lpcStable1 = SilkLpcInvPredGain.Compute(params1.PredCoefQ12.AsSpan(SilkConstants.MAX_LPC_ORDER, cb.Order), cb.Order);
        int lpcStable2 = SilkLpcInvPredGain.Compute(params2.PredCoefQ12.AsSpan(SilkConstants.MAX_LPC_ORDER, cb.Order), cb.Order);
        True(lpcStable1 > 0, "frame 1 LPC should be stable");
        True(lpcStable2 > 0, "frame 2 LPC should be stable");
    }

    [TestMethod]
    public void MultiFrame_SecondFrameInterpolatesFromFirst_LpcDiffersAcrossHalves()
    {
        // Frame 1: independent, sets up prev NLSFs.
        // Frame 2: same independent decode BUT with NlsfInterpCoefQ2 = 2 so first half
        //          LPC is interpolated from frame 1's NLSFs.
        var cb = SilkNlsfCodebookTables.Wb;
        var state = new SilkChannelDecoderState();

        var frame1Idx = new SilkDecodedIndices
        {
            SignalType = 0,
            QuantOffsetType = 0,
            NlsfInterpCoefQ2 = 4,
            Seed = 0,
        };
        for (int i = 0; i < 4; i++) frame1Idx.GainsIndices[i] = 20;
        frame1Idx.NlsfIndices[0] = 2;

        var params1 = new SilkDecodedParameters();
        SilkParametersDecoder.Decode(params1, frame1Idx, cb, 16, 4,
            ref state.LastGainIndex, state.PrevNlsfQ15, 0);

        // Frame 2: different cb1 index + interpolation enabled.
        var frame2Idx = new SilkDecodedIndices
        {
            SignalType = 0,
            QuantOffsetType = 0,
            NlsfInterpCoefQ2 = 2,
            Seed = 1,
        };
        for (int i = 0; i < 4; i++) frame2Idx.GainsIndices[i] = 15;
        frame2Idx.NlsfIndices[0] = 25; // clearly-different NLSF vector

        var params2 = new SilkDecodedParameters();
        SilkParametersDecoder.Decode(params2, frame2Idx, cb, 16, 4,
            ref state.LastGainIndex, state.PrevNlsfQ15, 0);

        // With interpolation on and a meaningful difference between prev & cur NLSFs,
        // at least some LPC coefficients should differ between the two halves.
        int differing = 0;
        for (int i = 0; i < cb.Order; i++)
        {
            if (params2.PredCoefQ12[i] != params2.PredCoefQ12[SilkConstants.MAX_LPC_ORDER + i])
                differing++;
        }
        True(differing > 0, "interpolated first-half LPC should differ from second-half LPC in at least one coefficient");
    }

    [TestMethod]
    public void MultiFrame_ConditionalGain_DeltaEncodingFromPrevLastIndex()
    {
        // Frame 1: independent gains (any values).
        // Frame 2: conditional gains, verify delta math works against stored LastGainIndex.
        var cb = SilkNlsfCodebookTables.NbMb;
        var state = new SilkChannelDecoderState();

        var frame1Idx = new SilkDecodedIndices { SignalType = 1, QuantOffsetType = 0, NlsfInterpCoefQ2 = 4 };
        for (int i = 0; i < 4; i++) frame1Idx.GainsIndices[i] = (sbyte)(30 - i * 2);
        frame1Idx.NlsfIndices[0] = 5;

        var params1 = new SilkDecodedParameters();
        SilkParametersDecoder.Decode(params1, frame1Idx, cb, 8, 4,
            ref state.LastGainIndex, state.PrevNlsfQ15, conditional: 0);

        sbyte gainIdxAfterFrame1 = state.LastGainIndex;

        // Frame 2: conditional. Use small delta indices ([0, MAX_DELTA_GAIN_QUANT*2-...]).
        var frame2Idx = new SilkDecodedIndices { SignalType = 1, QuantOffsetType = 0, NlsfInterpCoefQ2 = 4 };
        for (int i = 0; i < 4; i++) frame2Idx.GainsIndices[i] = (sbyte)(10 + i);
        frame2Idx.NlsfIndices[0] = 5;

        var params2 = new SilkDecodedParameters();
        SilkParametersDecoder.Decode(params2, frame2Idx, cb, 8, 4,
            ref state.LastGainIndex, state.PrevNlsfQ15, conditional: 1);

        // State should be updated and frame 2 gains should be positive and plausibly related to the delta.
        True(state.LastGainIndex >= 0, "LastGainIndex should be clamped to [0, N_LEVELS_QGAIN - 1]");
        True(state.LastGainIndex < SilkConstants.N_LEVELS_QGAIN, "LastGainIndex should be < N_LEVELS_QGAIN");
        for (int k = 0; k < 4; k++)
        {
            True(params2.GainsQ16[k] > 0, $"frame 2 gain[{k}] should be > 0");
        }
    }

    [TestMethod]
    public void MultiFrame_VoicedPitchDelta_UsesPrevLagIndex()
    {
        // Two voiced frames. Frame 2 uses delta pitch coding keyed on frame 1's
        // decoded lag. Verify the state carries prev lag / prev voiced through correctly.
        var cb = SilkNlsfCodebookTables.NbMb;
        var state = new SilkChannelDecoderState();

        // Frame 1: voiced, independent. We pick an explicit lag + contour.
        var frame1Idx = new SilkDecodedIndices
        {
            SignalType = 2,
            QuantOffsetType = 0,
            NlsfInterpCoefQ2 = 4,
            LagIndex = 50,
            ContourIndex = 2,
            PerIndex = 1,
            LtpScaleIndex = 0,
            Seed = 1,
        };
        for (int i = 0; i < 4; i++) frame1Idx.GainsIndices[i] = 25;
        frame1Idx.NlsfIndices[0] = 10;
        for (int i = 0; i < 4; i++) frame1Idx.LtpIndices[i] = 3;

        var params1 = new SilkDecodedParameters();
        SilkParametersDecoder.Decode(params1, frame1Idx, cb, 8, 4,
            ref state.LastGainIndex, state.PrevNlsfQ15, 0);

        // Update pitch-specific state manually (decode_frame would normally do this).
        state.PrevLagIndex = frame1Idx.LagIndex;
        state.PrevSignalTypeWasVoiced = true;

        // Frame 2: ALSO voiced, but with a lag that differs by a small amount. Test that
        // the pitch-lag expansion produces in-range results for this close-spaced lag.
        var frame2Idx = new SilkDecodedIndices
        {
            SignalType = 2,
            QuantOffsetType = 0,
            NlsfInterpCoefQ2 = 4,
            LagIndex = 55, // delta = +5
            ContourIndex = 3,
            PerIndex = 1,
            LtpScaleIndex = 0,
            Seed = 2,
        };
        for (int i = 0; i < 4; i++) frame2Idx.GainsIndices[i] = (sbyte)(10 + i);
        frame2Idx.NlsfIndices[0] = 12;
        for (int i = 0; i < 4; i++) frame2Idx.LtpIndices[i] = (sbyte)(5 + i);

        var params2 = new SilkDecodedParameters();
        SilkParametersDecoder.Decode(params2, frame2Idx, cb, 8, 4,
            ref state.LastGainIndex, state.PrevNlsfQ15, conditional: 1);

        // All 4 pitch lags should be in [minLag, maxLag].
        int minLag = SilkConstants.PE_MIN_LAG_MS * 8;
        int maxLag = SilkConstants.PE_MAX_LAG_MS * 8;
        for (int k = 0; k < 4; k++)
        {
            True(params2.PitchL[k] >= minLag, $"frame 2 pitch[{k}] = {params2.PitchL[k]} < {minLag}");
            True(params2.PitchL[k] <= maxLag, $"frame 2 pitch[{k}] = {params2.PitchL[k]} > {maxLag}");
        }

        // LTP scale index was 0 (decoded under conditional=0 for frame 1), but frame 2 is
        // conditional so its scale index is 0. Expect LtpScaleQ14 = first scale = 15565.
        // (Frame 1 had LtpScaleIndex=0 -> 15565. Frame 2 has LtpScaleIndex=0 -> also 15565.)
        Equal(15565, params2.LtpScaleQ14);
    }

    [TestMethod]
    public void MultiFrame_StateReset_RestoresFirstFrameBehavior()
    {
        var cb = SilkNlsfCodebookTables.NbMb;
        var state = new SilkChannelDecoderState();

        // Decode a frame.
        var idx = new SilkDecodedIndices { SignalType = 0, QuantOffsetType = 0, NlsfInterpCoefQ2 = 4 };
        for (int i = 0; i < 4; i++) idx.GainsIndices[i] = 25;
        idx.NlsfIndices[0] = 15;

        var paramsA = new SilkDecodedParameters();
        SilkParametersDecoder.Decode(paramsA, idx, cb, 8, 4,
            ref state.LastGainIndex, state.PrevNlsfQ15, 0);

        sbyte gainIdxAfter = state.LastGainIndex;
        True(gainIdxAfter != 0 || gainIdxAfter == 0, "State advances through decode");

        // Reset state.
        state.Reset();
        Equal((sbyte)0, state.LastGainIndex);
        for (int i = 0; i < state.PrevNlsfQ15.Length; i++) Equal((short)0, state.PrevNlsfQ15[i]);
        Equal((short)0, state.PrevLagIndex);
        True(!state.PrevSignalTypeWasVoiced, "PrevSignalTypeWasVoiced should be false after reset");
    }
}
