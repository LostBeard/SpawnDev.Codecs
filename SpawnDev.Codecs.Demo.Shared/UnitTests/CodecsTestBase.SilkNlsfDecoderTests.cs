using SpawnDev.Codecs.Audio.Opus.Silk;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for <see cref="SilkNlsfDecoder"/>: the predictive residual dequantizer
/// helper and the top-level NLSF decode entry point that stitches unpack +
/// residual + stabilize.
/// </summary>
public abstract partial class CodecsTestBase
{
    // -------- ResidualDequant (static helper) --------

    [TestMethod]
    public void ResidualDequant_AllZeros_ProducesAllZeros()
    {
        // indices = 0, predictor = anything. out_Q10 starts at 0; each iter multiplies 0 by
        // pred and writes 0 back. Result: all zeros.
        int order = 10;
        var indices = new sbyte[order];
        var pred = new byte[order];
        for (int i = 0; i < order; i++) pred[i] = 100;

        Span<short> xQ10 = stackalloc short[order];
        SilkNlsfDecoder.ResidualDequant(xQ10, indices, pred, quantStepSizeQ16: 65536, order: order);

        for (int i = 0; i < order; i++) Equal((short)0, xQ10[i], $"xQ10[{i}]");
    }

    [TestMethod]
    public void ResidualDequant_SingleNonZeroIndex_ProducesExpectedValue()
    {
        // Indices = [0, 0, ..., 0, 1] (only last set). Predictor = all zeros.
        // For i=order-1: outQ10 = 1 << 10 = 1024. 1024 > 0 so subtract 102 -> 922.
        //                outQ10 = silk_SMLAWB(0, 922, 65536) = 0 + (922 * 65536 high-word of low-16).
        //                Low 16 of 65536 is 0, so result is 0 + 0 = 0. Hmm that seems off.
        // Let me reread: silk_SMLAWB(a, b, c) = a + ((b * (short)c) >> 16).
        // (short)65536 = 0 (truncation of the low 16 bits). So yes, result is 0.
        // So for quantStepSize=65536 (exactly 1.0 in Q16), the result is effectively 0 due to
        // truncation of the low 16 bits. Use a step size that's representable in low 16.

        int order = 10;
        var indices = new sbyte[order];
        indices[order - 1] = 1;
        var pred = new byte[order]; // all zero predictor

        // Use a step size whose low 16 bits are significant: 0x8000 (half Q16).
        Span<short> xQ10 = stackalloc short[order];
        SilkNlsfDecoder.ResidualDequant(xQ10, indices, pred, quantStepSizeQ16: 0x8000, order: order);

        // For i=9: predQ10 = 0 (out_Q10=0). out_Q10 = 1<<10 = 1024. Subtract 102 -> 922.
        //          final out_Q10 = silk_SMLAWB(0, 922, 0x8000) = 0 + ((922 * 0x8000) >> 16).
        //          Actually (short)0x8000 = -32768 (sign-extended). So 922 * -32768 = -30212096.
        //          -30212096 >> 16 = -462 (signed arithmetic shift).
        // Check.
        // (short)0x8000 == -32768 (sign-extended from low 16 bits).
        int expected = (922 * -32768) >> 16;
        Equal((short)expected, xQ10[order - 1], $"last xQ10");
    }

    [TestMethod]
    public void ResidualDequant_NegativeIndex_AddsLevelAdj()
    {
        // Single negative index at end.
        int order = 10;
        var indices = new sbyte[order];
        indices[order - 1] = -1;
        var pred = new byte[order];

        Span<short> xQ10 = stackalloc short[order];
        SilkNlsfDecoder.ResidualDequant(xQ10, indices, pred, quantStepSizeQ16: 0x8000, order: order);

        // For index=-1: out_Q10 = -1 << 10 = -1024. out_Q10 < 0 so add 102 -> -922.
        // final = silk_SMLAWB(0, -922, 0x8000) = 0 + ((-922 * -32768) >> 16) = 30212096 >> 16 = 460 (with rounding).
        int expected = (-922 * -32768) >> 16;
        Equal((short)expected, xQ10[order - 1], $"negative index last xQ10");
    }

    // -------- Decode (top-level NLSF vector decoder) --------

    private static SilkNlsfCodebook BuildDecoderTestCodebook()
    {
        // Single-vector synthetic codebook, order 10.
        // CB1_NLSF_Q8 for vector 0: monotonically-increasing values evenly spaced in Q8 range
        // (0..255 mapped into Q15 via the decoder math below). Choose values far apart so the
        // stabilizer leaves them alone.
        short order = 10;
        var cb1NlsfQ8 = new byte[order]; // 1 vector * 10 = 10 bytes
        for (int i = 0; i < order; i++) cb1NlsfQ8[i] = (byte)((i + 1) * 20); // 20, 40, 60, ..., 200

        // CB1_Wght_Q9: use uniform weights so division yields predictable residuals.
        var cb1WghtQ9 = new short[order];
        for (int i = 0; i < order; i++) cb1WghtQ9[i] = 2048; // 4.0 in Q9

        // ec_sel all zeros -> unpack produces zero entropy indices + lower-half predictors.
        var ecSel = new byte[order / 2]; // 5 bytes for 1 vector of order 10

        // Predictor table large enough for all ec_sel patterns (synthetic, see NLSF_unpack tests).
        var predQ8 = new byte[2 * order]; // 20 bytes, all zero

        // Delta-min spacing small enough to not trigger stabilization on our test vectors.
        var deltaMin = new short[order + 1];
        for (int i = 0; i <= order; i++) deltaMin[i] = 50;

        return new SilkNlsfCodebook
        {
            NVectors = 1,
            Order = order,
            QuantStepSizeQ16 = 0x2000, // small step size
            InvQuantStepSizeQ6 = 0,
            Cb1NlsfQ8 = cb1NlsfQ8,
            Cb1WghtQ9 = cb1WghtQ9,
            Cb1Icdf = Array.Empty<byte>(),
            PredQ8 = predQ8,
            EcSel = ecSel,
            EcIcdf = Array.Empty<byte>(),
            EcRatesQ5 = Array.Empty<byte>(),
            DeltaMinQ15 = deltaMin,
        };
    }

    [TestMethod]
    public void NlsfDecode_ZeroResiduals_ReturnsCodebookEntry()
    {
        // indices = [0, 0, 0, ..., 0]: first-stage=0, residuals all zero.
        // With zero residuals, pNLSF_Q15[i] = silk_ADD_LSHIFT32(0, cb1[i], 7) = cb1[i] << 7.
        // Then stabilize checks ordering / spacing.
        var cb = BuildDecoderTestCodebook();
        var indices = new sbyte[cb.Order + 1]; // all zero

        Span<short> nlsfQ15 = stackalloc short[cb.Order];
        SilkNlsfDecoder.Decode(nlsfQ15, indices, cb);

        // Expected: cb1[i] << 7 = (i+1)*20 << 7 = (i+1) * 2560.
        for (int i = 0; i < cb.Order; i++)
        {
            int expected = (i + 1) * 20 * 128;
            Equal((short)expected, nlsfQ15[i], $"nlsf[{i}]");
        }
    }

    [TestMethod]
    public void NlsfDecode_Output_IsOrderedAfterStabilize()
    {
        // Even with unusual residuals, after stabilize the output must be monotonically ordered
        // with respect to the codebook's deltaMin_Q15 spacing constraint.
        var cb = BuildDecoderTestCodebook();
        var indices = new sbyte[cb.Order + 1];
        // Use small residuals that perturb but keep order reasonable after stabilize.
        for (int i = 1; i <= cb.Order; i++) indices[i] = (sbyte)((i % 3) - 1); // -1, 0, 1, -1, 0, 1, ...

        Span<short> nlsfQ15 = stackalloc short[cb.Order];
        SilkNlsfDecoder.Decode(nlsfQ15, indices, cb);

        // Post-stabilize, verify ordering + spacing.
        for (int i = 1; i < cb.Order; i++)
        {
            if (nlsfQ15[i] < nlsfQ15[i - 1] + cb.DeltaMinQ15[i])
                throw new Exception($"ordering/spacing broken at i={i}: {nlsfQ15[i - 1]} -> {nlsfQ15[i]} (min delta {cb.DeltaMinQ15[i]})");
        }
    }

    [TestMethod]
    public void NlsfDecode_Output_WithinQ15Range()
    {
        var cb = BuildDecoderTestCodebook();
        var indices = new sbyte[cb.Order + 1];
        for (int i = 1; i <= cb.Order; i++) indices[i] = (sbyte)(i - 5); // -4..5 range

        Span<short> nlsfQ15 = stackalloc short[cb.Order];
        SilkNlsfDecoder.Decode(nlsfQ15, indices, cb);

        for (int i = 0; i < cb.Order; i++)
        {
            True(nlsfQ15[i] >= 0, $"nlsf[{i}] = {nlsfQ15[i]} should be >= 0");
            True(nlsfQ15[i] <= 32767, $"nlsf[{i}] = {nlsfQ15[i]} should be <= 32767");
        }
    }

    [TestMethod]
    public void NlsfDecode_OutputTooSmall_Throws()
    {
        var cb = BuildDecoderTestCodebook();
        short[] small = new short[5];
        sbyte[] indices = new sbyte[cb.Order + 1];
        Throws<ArgumentException>(() => SilkNlsfDecoder.Decode(small, indices, cb));
    }

    [TestMethod]
    public void NlsfDecode_IndicesTooShort_Throws()
    {
        var cb = BuildDecoderTestCodebook();
        short[] nlsfQ15 = new short[cb.Order];
        sbyte[] tooShort = new sbyte[cb.Order]; // needs order+1
        Throws<ArgumentException>(() => SilkNlsfDecoder.Decode(nlsfQ15, tooShort, cb));
    }

    [TestMethod]
    public void NlsfDecode_NullCodebook_Throws()
    {
        short[] nlsfQ15 = new short[10];
        sbyte[] indices = new sbyte[11];
        Throws<ArgumentNullException>(() => SilkNlsfDecoder.Decode(nlsfQ15, indices, null!));
    }
}
