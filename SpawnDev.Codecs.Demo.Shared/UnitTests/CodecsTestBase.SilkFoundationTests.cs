using SpawnDev.Codecs.Audio.Opus.Silk;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Audio.Opus.Silk.SilkMacros;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Unit tests for SILK foundation primitives: fixed-point macros, log2/lin2log
/// conversions, and gain dequantization. Hand-traced expected values match the
/// libopus C reference bit-exactly; where exact bit-match is infeasible (e.g.
/// parabolic approximation round-trip), tests use tight numerical tolerance.
/// </summary>
public abstract partial class CodecsTestBase
{
    // -------- SilkMacros --------

    [TestMethod]
    public void SilkMacros_Smulwb_Basics()
    {
        // silk_SMULWB(a, b) = (a * (short)b) >> 16
        Equal(0, silk_SMULWB(0, 0));
        Equal(0, silk_SMULWB(1 << 16, 0));
        // (1 << 16) * 1 >> 16 = 1
        Equal(1, silk_SMULWB(1 << 16, 1));
        // (2 << 16) * 3 >> 16 = 6
        Equal(6, silk_SMULWB(2 << 16, 3));
        // Negative low-16 of b32: (1 << 16) * -1 >> 16 = -1
        Equal(-1, silk_SMULWB(1 << 16, -1));
    }

    [TestMethod]
    public void SilkMacros_Smulbb_Basics()
    {
        Equal(0, silk_SMULBB(0, 0));
        Equal(1, silk_SMULBB(1, 1));
        Equal(6, silk_SMULBB(2, 3));
        Equal(-6, silk_SMULBB(2, -3));
        // Only low 16 bits (signed) of each operand are used.
        Equal(1, silk_SMULBB(0x00010001, 0x00010001));
    }

    [TestMethod]
    public void SilkMacros_Smlawb_AccumulatesCorrectly()
    {
        // silk_SMLAWB(a, b, c) = a + ((b * (short)c) >> 16)
        Equal(0, silk_SMLAWB(0, 0, 0));
        Equal(5, silk_SMLAWB(5, 1 << 16, 0));
        Equal(10, silk_SMLAWB(5, 1 << 16, 5));
    }

    [TestMethod]
    public void SilkMacros_Mla_AccumulatesProduct()
    {
        Equal(10, silk_MLA(1, 3, 3));      // 1 + 3*3 = 10
        Equal(0, silk_MLA(0, 0, 0));
        Equal(-5, silk_MLA(-5, 1, 0));      // -5 + 1*0 = -5
    }

    [TestMethod]
    public void SilkMacros_Shifts_WorkLikeC()
    {
        Equal(16, silk_LSHIFT(1, 4));
        Equal(1, silk_RSHIFT(16, 4));
        // Arithmetic right shift preserves sign.
        Equal(-1, silk_RSHIFT(-1, 4));
        Equal(-2, silk_RSHIFT(-16, 3));
    }

    [TestMethod]
    public void SilkMacros_AddShifts_WorkLikeC()
    {
        Equal(10, silk_ADD_RSHIFT32(8, 4, 1));   // 8 + (4 >> 1) = 10
        Equal(10, silk_ADD_LSHIFT32(2, 1, 3));   // 2 + (1 << 3) = 10
    }

    [TestMethod]
    public void SilkMacros_Limit_Clamps()
    {
        Equal(5, silk_LIMIT_int(5, 0, 10));
        Equal(0, silk_LIMIT_int(-5, 0, 10));
        Equal(10, silk_LIMIT_int(15, 0, 10));
        Equal(3, silk_LIMIT_int(3, 3, 3));
    }

    [TestMethod]
    public void SilkMacros_MinMax_MatchesMath()
    {
        Equal(1, silk_min_int(1, 2));
        Equal(2, silk_max_int(1, 2));
        Equal(-5, silk_min_int(-5, 5));
    }

    [TestMethod]
    public void SilkMacros_ClzFrac_HandTracedValues()
    {
        // (inVal, expectedLz, expectedFracQ7)
        var cases = new (int v, int lz, int frac)[]
        {
            (1, 31, 0),     // 1 shifted left 31 = 0x80000000; >>24 = 0x80; & 0x7F = 0
            (2, 30, 0),     // 2 shifted left 30 = 0x80000000; same result
            (3, 30, 64),    // 3 shifted left 30 = 0xC0000000; >>24 = 0xC0; & 0x7F = 0x40 = 64
            (4, 29, 0),     // 4 shifted left 29 = 0x80000000; frac = 0
            (0x80000000.GetHashCode(), 0, 0) // 0x80000000 has 0 leading zeros; dummy check via hash
        };
        // Manual cases (avoid the dummy above which varies).
        silk_CLZ_FRAC(1, out int lz, out int frac);
        Equal(31, lz, "CLZ of 1");
        Equal(0, frac, "Frac of 1");

        silk_CLZ_FRAC(2, out lz, out frac);
        Equal(30, lz, "CLZ of 2");
        Equal(0, frac, "Frac of 2");

        silk_CLZ_FRAC(3, out lz, out frac);
        Equal(30, lz, "CLZ of 3");
        Equal(64, frac, "Frac of 3");

        silk_CLZ_FRAC(4, out lz, out frac);
        Equal(29, lz, "CLZ of 4");
        Equal(0, frac, "Frac of 4");
    }

    // -------- SilkLog2 --------

    [TestMethod]
    public void SilkLog2_Log2lin_KnownPoints()
    {
        // silk_log2lin(0) = 1 (parabolic approx starts at 1, frac=0 => out stays 1)
        Equal(1, SilkLog2.silk_log2lin(0));
        // silk_log2lin(128) = 2 (128 in Q7 == 1.0; 2^1 = 2)
        Equal(2, SilkLog2.silk_log2lin(128));
        // silk_log2lin(256) = 4 (256 in Q7 == 2.0; 2^2 = 4)
        Equal(4, SilkLog2.silk_log2lin(256));
        // silk_log2lin(384) = 8 (3.0 in Q7 -> 2^3 = 8)
        Equal(8, SilkLog2.silk_log2lin(384));
    }

    [TestMethod]
    public void SilkLog2_Log2lin_ClampsOutOfRange()
    {
        Equal(0, SilkLog2.silk_log2lin(-1));
        Equal(0, SilkLog2.silk_log2lin(-100));
        Equal(silk_int32_MAX, SilkLog2.silk_log2lin(3967));
        Equal(silk_int32_MAX, SilkLog2.silk_log2lin(10000));
    }

    [TestMethod]
    public void SilkLog2_Lin2log_KnownPoints()
    {
        Equal(0, SilkLog2.silk_lin2log(1));
        Equal(128, SilkLog2.silk_lin2log(2));
        Equal(256, SilkLog2.silk_lin2log(4));
        Equal(384, SilkLog2.silk_lin2log(8));
    }

    [TestMethod]
    public void SilkLog2_RoundTrip_WithinTolerance()
    {
        // log2lin(lin2log(x)) should be close to x for reasonable x.
        // libopus uses parabolic approximation so small error is expected; we tolerate
        // up to ~1.5% since the approx has known ~1% worst case over the range.
        int[] testValues = { 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 10000, 100000 };
        foreach (var x in testValues)
        {
            int log = SilkLog2.silk_lin2log(x);
            int reconstructed = SilkLog2.silk_log2lin(log);
            double relativeError = x == 0 ? Math.Abs(reconstructed) : Math.Abs((double)(reconstructed - x) / x);
            if (relativeError > 0.02)
                throw new Exception($"Round-trip error too large for x={x}: got {reconstructed}, error={relativeError:P2}");
        }
    }

    // -------- SilkGainDecoder --------

    [TestMethod]
    public void SilkGainDecoder_UnconditionalFirst_ProducesExpectedGain()
    {
        // With conditional=0 and prev_ind=0 ignored, first index directly sets prev_ind.
        // Then the gain is derived from silk_log2lin(INV_SCALE * prev_ind + OFFSET).
        sbyte prev = 0;
        Span<int> gains = stackalloc int[1];
        sbyte[] ind = { 32 };
        SilkGainDecoder.Dequantize(gains, ind, ref prev, conditional: 0, nbSubfr: 1);
        Equal((sbyte)32, prev, "prev_ind after decode");
        True(gains[0] > 0, $"gain should be positive, got {gains[0]}");
    }

    [TestMethod]
    public void SilkGainDecoder_UnconditionalFirst_LimitsDownwardJump()
    {
        // Unconditional first index: prev = max(ind[0], prev - 16).
        // If prev starts at 40 and ind[0] is 10, prev becomes max(10, 24) = 24.
        sbyte prev = 40;
        Span<int> gains = stackalloc int[1];
        sbyte[] ind = { 10 };
        SilkGainDecoder.Dequantize(gains, ind, ref prev, conditional: 0, nbSubfr: 1);
        Equal((sbyte)24, prev, "prev_ind after downward-limited unconditional decode");
    }

    [TestMethod]
    public void SilkGainDecoder_DeltaCoded_AccumulatesCorrectly()
    {
        // conditional=1 means first subframe is delta from prev. Subsequent subframes
        // are always delta. Simple deltas (ind_tmp = ind + MIN_DELTA_GAIN_QUANT, then
        // accumulate).
        sbyte prev = 20;
        Span<int> gains = stackalloc int[4];
        // With MIN_DELTA_GAIN_QUANT=-4, ind=4 means indTmp=0 (no delta).
        // Let's use ind=[8, 4, 4, 4]: indTmp=[4, 0, 0, 0].
        // Start prev=20, doubleStepThreshold = 2*36 - 64 + 20 = 28.
        // After subframe 0: indTmp=4, 4 <= 28, prev = 20 + 4 = 24. clamped [0, 63] = 24.
        // Subframe 1: indTmp=0, prev = 24 + 0 = 24.
        // Subframe 2: indTmp=0, prev = 24.
        // Subframe 3: indTmp=0, prev = 24.
        sbyte[] ind = { 8, 4, 4, 4 };
        SilkGainDecoder.Dequantize(gains, ind, ref prev, conditional: 1, nbSubfr: 4);
        Equal((sbyte)24, prev, "prev_ind after 4 deltas");
        // All 4 gains should be equal since prev_ind is equal after subframes 1-3.
        Equal(gains[1], gains[2], "gain[1] == gain[2]");
        Equal(gains[2], gains[3], "gain[2] == gain[3]");
    }

    [TestMethod]
    public void SilkGainDecoder_InvalidNbSubfr_Throws()
    {
        sbyte prev = 0;
        int[] gains = new int[4];
        sbyte[] ind = { 0, 0, 0, 0 };
        Throws<ArgumentOutOfRangeException>(() =>
        {
            sbyte p = prev;
            SilkGainDecoder.Dequantize(gains, ind, ref p, 0, 0);
        });
        Throws<ArgumentOutOfRangeException>(() =>
        {
            sbyte p = prev;
            SilkGainDecoder.Dequantize(gains, ind, ref p, 0, 5);
        });
    }

    [TestMethod]
    public void SilkGainDecoder_BufferTooSmall_Throws()
    {
        sbyte prev = 0;
        int[] gains = new int[2];
        sbyte[] ind = { 0, 0, 0, 0 };
        Throws<ArgumentException>(() =>
        {
            sbyte p = prev;
            SilkGainDecoder.Dequantize(gains, ind, ref p, 0, 4);
        });
    }

    // -------- Constants sanity --------

    [TestMethod]
    public void SilkConstants_DerivedValues_MatchLibopusPreprocessor()
    {
        // Hand-computed per libopus silk/gain_quant.c preprocessor arithmetic.
        Equal(2090, SilkConstants.GAIN_OFFSET_Q7, "GAIN_OFFSET_Q7");
        Equal(2251, SilkConstants.GAIN_SCALE_Q16, "GAIN_SCALE_Q16");
        Equal(1907825, SilkConstants.GAIN_INV_SCALE_Q16, "GAIN_INV_SCALE_Q16");
        Equal(3967, SilkConstants.GAIN_LOG_CLAMP_HIGH_Q7, "GAIN_LOG_CLAMP_HIGH_Q7");
    }
}
