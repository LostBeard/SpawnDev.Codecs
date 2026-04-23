using SpawnDev.Codecs.Audio.Opus.Silk;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;
using static SpawnDev.Codecs.Audio.Opus.Silk.SilkMacros;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for the decode_core support helpers: silk_MLA_ovflw, silk_RAND
/// (pseudo-random generator), silk_ADD_SAT32, silk_SUB_LSHIFT32,
/// silk_LSHIFT_ovflw, silk_DIV32_varQ, and the quantization offsets LUT.
/// Each verified against either libopus reference semantics or direct
/// arithmetic.
/// </summary>
public abstract partial class CodecsTestBase
{
    // -------- Quantization offset table --------

    [TestMethod]
    public void DecodeCoreHelpers_QuantizationOffsets_MatchLibopus()
    {
        // Row 0 (non-voiced): OFFSET_UVL = 100, OFFSET_UVH = 240.
        Equal((short)100, SilkConstants.QUANTIZATION_OFFSETS_Q10[0, 0]);
        Equal((short)240, SilkConstants.QUANTIZATION_OFFSETS_Q10[0, 1]);
        // Row 1 (voiced): OFFSET_VL = 32, OFFSET_VH = 100.
        Equal((short)32, SilkConstants.QUANTIZATION_OFFSETS_Q10[1, 0]);
        Equal((short)100, SilkConstants.QUANTIZATION_OFFSETS_Q10[1, 1]);
    }

    [TestMethod]
    public void DecodeCoreHelpers_Constants_MatchLibopus()
    {
        Equal(80, SilkConstants.QUANT_LEVEL_ADJUST_Q10);
        Equal(907633515, SilkConstants.RAND_INCREMENT);
        Equal(196314165, SilkConstants.RAND_MULTIPLIER);
    }

    // -------- silk_MLA_ovflw --------

    [TestMethod]
    public void MlaOvflw_BasicAddAndMultiply_MatchesExpectedValue()
    {
        // Non-overflowing case: 10 + 3 * 5 = 25.
        Equal(25, silk_MLA_ovflw(10, 3, 5));
        Equal(-10, silk_MLA_ovflw(0, 5, -2)); // 0 + 5*-2 = -10 (via unsigned wrap of -2 giving 0xFFFFFFFE, *5 = ...)
                                               // With unsigned math: 5 * (uint)-2 = 5 * 4294967294 = overflow
                                               // Let's compute: (uint)-2 = 0xFFFFFFFE. 5 * 0xFFFFFFFE = 21474836470 mod 2^32.
                                               // 21474836470 mod 4294967296 = 21474836470 - 4*4294967296 = 21474836470 - 17179869184 = 4294967286.
                                               // Signed interpretation: 4294967286 - 4294967296 = -10. Confirmed.
    }

    [TestMethod]
    public void MlaOvflw_Overflow_WrapsAroundCleanly()
    {
        // Force 32-bit overflow. b*c that wraps to a specific residue.
        // (uint)0x80000000 * 2 = 0x100000000 mod 2^32 = 0. So 100 + 0 = 100.
        Equal(100, silk_MLA_ovflw(100, int.MinValue, 2));
    }

    // -------- silk_RAND --------

    [TestMethod]
    public void Rand_FromSeedZero_MatchesManualComputation()
    {
        // silk_RAND(0) = silk_MLA_ovflw(RAND_INCREMENT, 0, RAND_MULTIPLIER) = RAND_INCREMENT + 0 = 907633515.
        Equal(907633515, silk_RAND(0));
    }

    [TestMethod]
    public void Rand_FromSeedOne_MatchesManualComputation()
    {
        // silk_RAND(1) = 907633515 + 196314165 = 1103947680.
        Equal(1103947680, silk_RAND(1));
    }

    [TestMethod]
    public void Rand_SequenceIsDeterministic()
    {
        // Run the PRNG five times from seed 0; verify each step's state matches
        // manual application of the formula.
        int seed = 0;
        int[] expected = new int[5];
        long cur = 0;
        for (int i = 0; i < 5; i++)
        {
            cur = ((long)cur * 196314165 + 907633515) & 0xFFFFFFFFL;
            expected[i] = (int)cur; // signed interpretation of lower 32 bits
        }

        for (int i = 0; i < 5; i++)
        {
            seed = silk_RAND(seed);
            Equal(expected[i], seed, $"iter {i}");
        }
    }

    [TestMethod]
    public void Rand_ProducesDiverseValues()
    {
        // 100 iterations should produce many distinct outputs (trivially bounded but useful sanity).
        var set = new HashSet<int>();
        int seed = 42;
        for (int i = 0; i < 100; i++)
        {
            seed = silk_RAND(seed);
            set.Add(seed);
        }
        True(set.Count > 90, $"Expected > 90 distinct values in 100 iters, got {set.Count}");
    }

    // -------- silk_ADD_SAT32 --------

    [TestMethod]
    public void AddSat32_NonOverflowing_EqualsRegularAdd()
    {
        Equal(5, silk_ADD_SAT32(2, 3));
        Equal(-5, silk_ADD_SAT32(-2, -3));
        // int.MinValue + int.MaxValue = -1 exactly (no overflow).
        Equal(-1, silk_ADD_SAT32(int.MinValue, int.MaxValue));
    }

    [TestMethod]
    public void AddSat32_OverflowPositive_ClampsToMaxValue()
    {
        Equal(int.MaxValue, silk_ADD_SAT32(int.MaxValue, 1));
        Equal(int.MaxValue, silk_ADD_SAT32(int.MaxValue - 10, 100));
    }

    [TestMethod]
    public void AddSat32_OverflowNegative_ClampsToMinValue()
    {
        Equal(int.MinValue, silk_ADD_SAT32(int.MinValue, -1));
        Equal(int.MinValue, silk_ADD_SAT32(int.MinValue + 10, -100));
    }

    // -------- silk_SUB_LSHIFT32 --------

    [TestMethod]
    public void SubLshift32_MatchesDefinition()
    {
        // a - (b << shift).
        Equal(100 - (3 << 2), silk_SUB_LSHIFT32(100, 3, 2));
        Equal(0 - (5 << 0), silk_SUB_LSHIFT32(0, 5, 0));
        Equal(-10 - (-2 << 1), silk_SUB_LSHIFT32(-10, -2, 1));
    }

    // -------- silk_LSHIFT_ovflw --------

    [TestMethod]
    public void LshiftOvflw_MatchesUnsignedShift()
    {
        // (uint)a << shift, reinterpreted as signed.
        Equal(8, silk_LSHIFT_ovflw(2, 2));
        // Overflow case: large positive shifts past the sign bit.
        Equal(unchecked((int)(((uint)0x40000000) << 1)), silk_LSHIFT_ovflw(0x40000000, 1)); // should be 0x80000000 as int -> int.MinValue
        Equal(int.MinValue, silk_LSHIFT_ovflw(0x40000000, 1));
    }

    // -------- silk_DIV32_varQ --------

    [TestMethod]
    public void Div32VarQ_Q16GainAdjustment_MatchesExpectedRatio()
    {
        // prev = 100.0 in Q16 = 6553600. cur = 50.0 in Q16 = 3276800. Ratio = 2.0 in Q16 = 131072.
        int res = silk_DIV32_varQ(6553600, 3276800, 16);
        double relErr = Math.Abs(res - 131072.0) / 131072.0;
        True(relErr < 0.001, $"expected ~131072 (2.0 in Q16), got {res} (rel err {relErr:P})");
    }

    [TestMethod]
    public void Div32VarQ_IdenticalInputs_EqualsOne()
    {
        // N / N at Q16 should be 1.0 in Q16 = 65536.
        int gainQ16 = 1234567;
        int res = silk_DIV32_varQ(gainQ16, gainQ16, 16);
        double relErr = Math.Abs(res - 65536.0) / 65536.0;
        True(relErr < 0.001, $"expected ~65536 (1.0 in Q16), got {res} (rel err {relErr:P})");
    }

    [TestMethod]
    public void Div32VarQ_NegativeNumerator_ProducesNegativeResult()
    {
        int prev = -6553600;
        int cur = 3276800;
        int res = silk_DIV32_varQ(prev, cur, 16);
        True(res < 0, $"-2.0 / 1.0 should be negative, got {res}");
        double relErr = Math.Abs(res - (-131072.0)) / 131072.0;
        True(relErr < 0.001, $"expected ~-131072 (-2.0 in Q16), got {res} (rel err {relErr:P})");
    }

    [TestMethod]
    public void Div32VarQ_SignMatchesInputSign()
    {
        True(silk_DIV32_varQ(100_000, 50, 4) > 0, "+ / + should be >=0");
        True(silk_DIV32_varQ(-100_000, 50, 4) < 0, "- / + should be <0");
        True(silk_DIV32_varQ(100_000, -50, 4) < 0, "+ / - should be <0");
        True(silk_DIV32_varQ(-100_000, -50, 4) > 0, "- / - should be >=0");
    }
}
