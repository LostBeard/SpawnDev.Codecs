using SpawnDev.Codecs.Audio.Opus.Silk;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Audio.Opus.Silk.SilkMacros;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for <see cref="SilkSumSqrShift"/> (energy-with-auto-shift) and the macros it
/// depends on (silk_CLZ32, silk_max_32, silk_ADD32_ovflw, silk_SMLABB_ovflw,
/// silk_ADD_RSHIFT_uint).
/// </summary>
public abstract partial class CodecsTestBase
{
    // -------- New SilkMacros helpers --------

    [TestMethod]
    public void SilkMacros_Clz32_HandTracedValues()
    {
        Equal(32, silk_CLZ32(0));
        Equal(31, silk_CLZ32(1));
        Equal(30, silk_CLZ32(2));
        Equal(30, silk_CLZ32(3));
        Equal(29, silk_CLZ32(4));
        Equal(24, silk_CLZ32(0xFF));
        Equal(16, silk_CLZ32(0xFFFF));
        Equal(1, silk_CLZ32(0x40000000));
        Equal(0, silk_CLZ32(unchecked((int)0x80000000))); // sign bit set
    }

    [TestMethod]
    public void SilkMacros_Max32_ReturnsLargerValue()
    {
        Equal(5, silk_max_32(3, 5));
        Equal(5, silk_max_32(5, 3));
        Equal(0, silk_max_32(-10, 0));
        Equal(int.MaxValue, silk_max_32(int.MaxValue, 0));
    }

    [TestMethod]
    public void SilkMacros_Add32Ovflw_WrapsAroundInt32()
    {
        Equal(5, silk_ADD32_ovflw(2, 3));
        // Positive overflow wraps.
        Equal(int.MinValue, silk_ADD32_ovflw(int.MaxValue, 1));
        // Negative overflow wraps.
        Equal(int.MaxValue, silk_ADD32_ovflw(int.MinValue, -1));
    }

    [TestMethod]
    public void SilkMacros_Smlabb_Ovflw_BasicAccumulate()
    {
        Equal(10, silk_SMLABB_ovflw(1, 3, 3)); // 1 + 3*3 = 10
        Equal(0, silk_SMLABB_ovflw(0, 0, 0));
        // Overflow-wrapping behavior on the addition.
        Equal(int.MinValue, silk_SMLABB_ovflw(int.MaxValue, 1, 1));
    }

    [TestMethod]
    public void SilkMacros_AddRshiftUint_UnsignedSemantics()
    {
        Equal(5u, silk_ADD_RSHIFT_uint(3u, 4u, 1));     // 3 + (4>>1) = 3 + 2 = 5
        Equal(0u, silk_ADD_RSHIFT_uint(0u, 0u, 0));
        Equal(uint.MaxValue, silk_ADD_RSHIFT_uint(uint.MaxValue, 0u, 0));
        // Large unsigned b without sign-extension issues.
        Equal(1u + (0x80000000u >> 1), silk_ADD_RSHIFT_uint(1u, 0x80000000u, 1));
    }

    // -------- SilkSumSqrShift --------

    [TestMethod]
    public void SumSqrShift_ZeroVector_ZeroEnergy()
    {
        var x = new short[128];
        SilkSumSqrShift.Compute(x, out int energy, out int shift);
        Equal(0, energy, "energy of all zeros");
        True(shift >= 0, $"shift should be non-negative, got {shift}");
    }

    [TestMethod]
    public void SumSqrShift_SingleNonzeroSample_MatchesSquare()
    {
        // For a length-1 vector [5], sum(x^2) = 25. With small-length small-value input,
        // the auto-shift should be 0 (plenty of int32 headroom).
        var x = new short[] { 5 };
        SilkSumSqrShift.Compute(x, out int energy, out int shift);
        // With shift=0, energy should equal 25 + len (because of conservative starting seed? No,
        // second pass starts nrg=0). Let me recompute.
        // First pass: shft = 31 - CLZ32(1) = 31 - 31 = 0. nrg = 1. Iter i=0: nrg_tmp = 25. nrg = 1 + (25>>0) = 26.
        // Then shft = max(0, 0+3 - CLZ32(26)) = max(0, 3 - 27) = 0.
        // Second pass: nrg = 0. Iter i=0: nrg_tmp = 25. nrg = 0 + 25 = 25.
        // energy = 25, shift = 0.
        Equal(25, energy);
        Equal(0, shift);
    }

    [TestMethod]
    public void SumSqrShift_KnownVector_ExactValue()
    {
        // x = [1, 2, 3, 4]. Sum of squares = 1+4+9+16 = 30.
        var x = new short[] { 1, 2, 3, 4 };
        SilkSumSqrShift.Compute(x, out int energy, out int shift);
        // Length 4: shft = 31 - CLZ32(4) = 31 - 29 = 2. Then recompute...
        // First pass: nrg seeded at 4. Pair(1,2): nrg_tmp = 1*1 = 1, then 1 + 2*2 = 5. nrg = 4 + (5>>2) = 4+1 = 5.
        //    Pair(3,4): nrg_tmp = 9 + 16 = 25. nrg = 5 + (25>>2) = 5+6 = 11.
        // shft_new = max(0, 2+3 - CLZ32(11)) = max(0, 5 - 28) = 0.
        // Second pass with shft=0: nrg=0. Pair(1,2): nrg_tmp=1+4=5. nrg=0+5=5.
        //    Pair(3,4): nrg_tmp=9+16=25. nrg=5+25=30.
        // So energy = 30 >> 0 = 30, shift = 0.
        Equal(30, energy);
        Equal(0, shift);
    }

    [TestMethod]
    public void SumSqrShift_LargeValues_ShiftsRightToFit()
    {
        // Vector of max-value samples (32767); length 256. Sum of squares = 256 * 32767^2 ≈ 2.75e11,
        // way larger than int32. Expect a nonzero shift so energy fits.
        var x = new short[256];
        for (int i = 0; i < x.Length; i++) x[i] = short.MaxValue;

        SilkSumSqrShift.Compute(x, out int energy, out int shift);

        // Reconstruct an approximate energy: energy * 2^shift should be close to 256 * 32767^2.
        long reconstructed = (long)energy << shift;
        long expected = 256L * 32767 * 32767;
        double ratio = (double)reconstructed / expected;
        if (ratio < 0.99 || ratio > 1.01)
            throw new Exception($"Reconstructed energy {reconstructed} vs expected {expected}, ratio={ratio:F4}");
        True(shift > 0, $"Large input should produce shift > 0, got {shift}");
        True(energy >= 0, $"Energy must be non-negative after shift, got {energy}");
    }

    [TestMethod]
    public void SumSqrShift_OddLength_HandlesTrailingSample()
    {
        // Length 5: 4 paired + 1 trailing.
        var x = new short[] { 1, 2, 3, 4, 5 };
        SilkSumSqrShift.Compute(x, out int energy, out int shift);
        // Expected sum of squares = 1+4+9+16+25 = 55.
        // Auto-shift logic: shft = 31-CLZ32(5) = 31-29 = 2 initially. Second pass re-computes.
        // With small values, final shift should be 0 and energy = 55.
        Equal(55, energy);
        Equal(0, shift);
    }

    [TestMethod]
    public void SumSqrShift_MixedSignInput_OnlyAbsMatters()
    {
        var positive = new short[] { 10, 20, 30 };
        var negative = new short[] { -10, -20, -30 };

        SilkSumSqrShift.Compute(positive, out int posEnergy, out int posShift);
        SilkSumSqrShift.Compute(negative, out int negEnergy, out int negShift);

        Equal(posEnergy, negEnergy, "energy identical for sign-flipped input");
        Equal(posShift, negShift, "shift identical for sign-flipped input");
    }
}
