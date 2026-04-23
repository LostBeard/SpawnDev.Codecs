using SpawnDev.Codecs.Audio.Opus.Silk;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for <see cref="SilkParametersDecoder.Decode"/> - the silk_decode_parameters
/// orchestrator that turns a decoded SilkDecodedIndices into a SilkDecodedParameters
/// value set: gains, NLSFs, LPC coefficients (with inter-half interpolation), pitch
/// lags, LTP taps, and LTP scale.
/// </summary>
public abstract partial class CodecsTestBase
{
    private static SilkDecodedIndices BuildInactiveIndices()
    {
        var idx = new SilkDecodedIndices
        {
            SignalType = 0,
            QuantOffsetType = 1,
            NlsfInterpCoefQ2 = 4,
            Seed = 0,
        };
        // Give each subframe a modest, slightly-increasing gain index.
        for (int i = 0; i < 4; i++) idx.GainsIndices[i] = (sbyte)(10 + i);
        idx.NlsfIndices[0] = 5;
        return idx;
    }

    private static SilkDecodedIndices BuildVoicedIndices()
    {
        var idx = new SilkDecodedIndices
        {
            SignalType = 2,
            QuantOffsetType = 0,
            NlsfInterpCoefQ2 = 4,
            LagIndex = 50,
            ContourIndex = 3,
            PerIndex = 1,
            LtpScaleIndex = 1,
            Seed = 2,
        };
        for (int i = 0; i < 4; i++) idx.GainsIndices[i] = 20;
        idx.NlsfIndices[0] = 12;
        for (int i = 0; i < 4; i++) idx.LtpIndices[i] = (sbyte)(i + 4);
        return idx;
    }

    // -------- Non-voiced frame: pitch + LTP zeroed --------

    [TestMethod]
    public void ParametersDecoder_InactiveFrame_PitchAndLtpZero()
    {
        var cb = SilkNlsfCodebookTables.NbMb;
        var idx = BuildInactiveIndices();
        var output = new SilkDecodedParameters();

        short[] prevNlsf = new short[SilkConstants.MAX_LPC_ORDER];
        sbyte lastGainIdx = 0;

        SilkParametersDecoder.Decode(
            output, idx, cb, fsKHz: 8, nbSubfr: 4,
            ref lastGainIdx, prevNlsf, conditional: 0);

        // Voiced-only fields should be zero.
        for (int k = 0; k < 4; k++) Equal(0, output.PitchL[k], $"pitch[{k}]");
        for (int i = 0; i < 20; i++) Equal((short)0, output.LtpCoefQ14[i], $"ltp[{i}]");
        Equal(0, output.LtpScaleQ14);

        // Gains should be non-zero and positive.
        for (int k = 0; k < 4; k++) True(output.GainsQ16[k] > 0, $"gains[{k}] = {output.GainsQ16[k]}");
    }

    // -------- Voiced frame: all fields populated --------

    [TestMethod]
    public void ParametersDecoder_VoicedFrame_AllFieldsPopulated()
    {
        var cb = SilkNlsfCodebookTables.NbMb;
        var idx = BuildVoicedIndices();
        var output = new SilkDecodedParameters();

        short[] prevNlsf = new short[SilkConstants.MAX_LPC_ORDER];
        sbyte lastGainIdx = 0;

        SilkParametersDecoder.Decode(
            output, idx, cb, fsKHz: 8, nbSubfr: 4,
            ref lastGainIdx, prevNlsf, conditional: 0);

        // Pitch lags: all 4 should be in the valid range [minLag, maxLag].
        int minLag = SilkConstants.PE_MIN_LAG_MS * 8;
        int maxLag = SilkConstants.PE_MAX_LAG_MS * 8;
        for (int k = 0; k < 4; k++)
        {
            True(output.PitchL[k] >= minLag, $"pitch[{k}] = {output.PitchL[k]} < {minLag}");
            True(output.PitchL[k] <= maxLag, $"pitch[{k}] = {output.PitchL[k]} > {maxLag}");
        }

        // LTP taps: at least one should be non-zero (reading from cb=1 which has non-zero entries).
        int ltpNonZero = 0;
        for (int i = 0; i < 20; i++) if (output.LtpCoefQ14[i] != 0) ltpNonZero++;
        True(ltpNonZero > 0, "Expected non-zero LTP taps for voiced frame");

        // LTP scale: LtpScalesTable_Q14[1] = 12288.
        Equal(12288, output.LtpScaleQ14);
    }

    [TestMethod]
    public void ParametersDecoder_VoicedFrame_LtpTapsShiftedToQ14()
    {
        // Verify that LTP taps are the Q7 codebook entries left-shifted by 7 (Q7 -> Q14).
        var cb = SilkNlsfCodebookTables.NbMb;
        var idx = BuildVoicedIndices();
        idx.PerIndex = 0;          // 8-entry codebook
        idx.LtpIndices[0] = 0;     // First entry of Cb0: { 4, 6, 24, 7, 5 }
        idx.LtpIndices[1] = 0;
        idx.LtpIndices[2] = 0;
        idx.LtpIndices[3] = 0;
        var output = new SilkDecodedParameters();

        short[] prevNlsf = new short[SilkConstants.MAX_LPC_ORDER];
        sbyte lastGainIdx = 0;

        SilkParametersDecoder.Decode(
            output, idx, cb, fsKHz: 8, nbSubfr: 4,
            ref lastGainIdx, prevNlsf, conditional: 0);

        sbyte[] expectedQ7 = { 4, 6, 24, 7, 5 };
        for (int k = 0; k < 4; k++)
        {
            for (int i = 0; i < 5; i++)
            {
                short expected = (short)(expectedQ7[i] << 7);
                Equal(expected, output.LtpCoefQ14[k * 5 + i], $"subframe {k}, tap {i}");
            }
        }
    }

    // -------- NLSF interpolation path --------

    [TestMethod]
    public void ParametersDecoder_InterpolationDisabled_FirstHalfCopiesSecondHalf()
    {
        // NlsfInterpCoefQ2 == 4 (the disabled/"just use current frame" value): the decoder
        // should copy the second-half LPC into the first half without re-deriving.
        var cb = SilkNlsfCodebookTables.NbMb;
        var idx = BuildInactiveIndices();
        idx.NlsfInterpCoefQ2 = 4;
        var output = new SilkDecodedParameters();

        short[] prevNlsf = new short[SilkConstants.MAX_LPC_ORDER];
        sbyte lastGainIdx = 0;

        SilkParametersDecoder.Decode(
            output, idx, cb, fsKHz: 8, nbSubfr: 4,
            ref lastGainIdx, prevNlsf, conditional: 0);

        for (int i = 0; i < cb.Order; i++)
        {
            short h0 = output.PredCoefQ12[i];
            short h1 = output.PredCoefQ12[SilkConstants.MAX_LPC_ORDER + i];
            Equal(h1, h0, $"coeff {i}: first-half should equal second-half");
        }
    }

    [TestMethod]
    public void ParametersDecoder_InterpolationEnabled_FirstHalfDiffersFromSecondHalf()
    {
        // With prev NLSFs set to a DIFFERENT vector than current, interp coef < 4 should
        // yield a first-half LPC that differs from the second half.
        var cb = SilkNlsfCodebookTables.NbMb;
        var idx = BuildInactiveIndices();
        idx.NlsfInterpCoefQ2 = 2; // interpolate halfway
        idx.NlsfIndices[0] = 5;
        var output = new SilkDecodedParameters();

        // Set prev NLSFs to a clearly-different shape (roughly corresponds to codebook entry 20).
        short[] prevNlsf = new short[SilkConstants.MAX_LPC_ORDER];
        for (int i = 0; i < 10; i++) prevNlsf[i] = (short)(100 + i * 3000);

        sbyte lastGainIdx = 0;
        SilkParametersDecoder.Decode(
            output, idx, cb, fsKHz: 8, nbSubfr: 4,
            ref lastGainIdx, prevNlsf, conditional: 0);

        // Expect at least some coefficients to differ.
        int different = 0;
        for (int i = 0; i < cb.Order; i++)
        {
            if (output.PredCoefQ12[i] != output.PredCoefQ12[SilkConstants.MAX_LPC_ORDER + i])
                different++;
        }
        True(different > 0, $"Expected interp to produce a different first-half LPC (same count: {cb.Order - different})");
    }

    [TestMethod]
    public void ParametersDecoder_PrevNlsfUpdatedToCurrentOnExit()
    {
        var cb = SilkNlsfCodebookTables.Wb;
        var idx = BuildInactiveIndices();
        idx.NlsfIndices[0] = 7;
        var output = new SilkDecodedParameters();

        short[] prevNlsf = new short[SilkConstants.MAX_LPC_ORDER];
        sbyte lastGainIdx = 0;

        SilkParametersDecoder.Decode(
            output, idx, cb, fsKHz: 16, nbSubfr: 4,
            ref lastGainIdx, prevNlsf, conditional: 0);

        for (int i = 0; i < cb.Order; i++)
        {
            Equal(output.NlsfQ15[i], prevNlsf[i], $"prevNlsf[{i}] should now equal current NlsfQ15[{i}]");
        }
    }

    // -------- LTP scale table --------

    [TestMethod]
    public void ParametersDecoder_LtpScaleTable_AllThreeIndices()
    {
        var cb = SilkNlsfCodebookTables.NbMb;
        int[] expectedScales = { 15565, 12288, 8192 };
        var idx = BuildVoicedIndices();

        for (int s = 0; s < 3; s++)
        {
            idx.LtpScaleIndex = (sbyte)s;
            var output = new SilkDecodedParameters();
            short[] prevNlsf = new short[SilkConstants.MAX_LPC_ORDER];
            sbyte lastGainIdx = 0;

            SilkParametersDecoder.Decode(
                output, idx, cb, fsKHz: 8, nbSubfr: 4,
                ref lastGainIdx, prevNlsf, conditional: 0);

            Equal(expectedScales[s], output.LtpScaleQ14, $"scale index {s}");
        }
    }

    // -------- Arg validation --------

    [TestMethod]
    public void ParametersDecoder_NullOutput_Throws()
    {
        short[] prevNlsf = new short[SilkConstants.MAX_LPC_ORDER];
        sbyte lastGainIdx = 0;
        var idx = BuildInactiveIndices();
        Throws<ArgumentNullException>(() =>
            SilkParametersDecoder.Decode(null!, idx, SilkNlsfCodebookTables.NbMb,
                8, 4, ref lastGainIdx, prevNlsf, 0));
    }

    [TestMethod]
    public void ParametersDecoder_UnsupportedFsKHz_Throws()
    {
        var idx = BuildInactiveIndices();
        var output = new SilkDecodedParameters();
        short[] prevNlsf = new short[SilkConstants.MAX_LPC_ORDER];
        sbyte lastGainIdx = 0;
        Throws<ArgumentException>(() =>
            SilkParametersDecoder.Decode(output, idx, SilkNlsfCodebookTables.NbMb,
                fsKHz: 11, nbSubfr: 4, ref lastGainIdx, prevNlsf, 0));
    }
}
