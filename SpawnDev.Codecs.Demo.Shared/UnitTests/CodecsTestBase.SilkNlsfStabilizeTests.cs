using SpawnDev.Codecs.Audio.Opus.Silk;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Audio.Opus.Silk.SilkMacros;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for SILK NLSF stabilization + the helper macros it depends on
/// (silk_RSHIFT_ROUND, silk_LIMIT_32, silk_ADD_SAT16, silk_insertion_sort_...).
/// </summary>
public abstract partial class CodecsTestBase
{
    // -------- New SilkMacros helpers --------

    [TestMethod]
    public void SilkMacros_RShiftRound_HandTracedValues()
    {
        // shift == 1: (a >> 1) + (a & 1)
        Equal(1, silk_RSHIFT_ROUND(1, 1));   // (0) + 1 = 1
        Equal(1, silk_RSHIFT_ROUND(2, 1));   // 1 + 0 = 1
        Equal(2, silk_RSHIFT_ROUND(3, 1));   // 1 + 1 = 2
        Equal(0, silk_RSHIFT_ROUND(0, 1));

        // shift > 1: ((a >> (shift-1)) + 1) >> 1
        Equal(1, silk_RSHIFT_ROUND(3, 2));   // ((3 >> 1) + 1) >> 1 = (1 + 1) >> 1 = 1
        Equal(1, silk_RSHIFT_ROUND(4, 2));   // ((4 >> 1) + 1) >> 1 = (2 + 1) >> 1 = 1
        Equal(2, silk_RSHIFT_ROUND(6, 2));   // ((6 >> 1) + 1) >> 1 = (3 + 1) >> 1 = 2
        Equal(2, silk_RSHIFT_ROUND(8, 2));   // ((8 >> 1) + 1) >> 1 = (4 + 1) >> 1 = 2
    }

    [TestMethod]
    public void SilkMacros_Limit32_ClampsWithinBounds()
    {
        Equal(5, silk_LIMIT_32(5, 0, 10));
        Equal(0, silk_LIMIT_32(-100, 0, 10));
        Equal(10, silk_LIMIT_32(100, 0, 10));
        Equal(int.MaxValue, silk_LIMIT_32(int.MaxValue, 0, int.MaxValue));
        Equal(int.MinValue, silk_LIMIT_32(int.MinValue, int.MinValue, 0));
    }

    [TestMethod]
    public void SilkMacros_AddSat16_SaturatesCorrectly()
    {
        Equal((short)5, silk_ADD_SAT16(2, 3));
        Equal((short)-5, silk_ADD_SAT16(-2, -3));
        Equal(short.MaxValue, silk_ADD_SAT16(short.MaxValue, 100));
        Equal(short.MaxValue, silk_ADD_SAT16(short.MaxValue, 1));
        Equal(short.MinValue, silk_ADD_SAT16(short.MinValue, -1));
        Equal(short.MinValue, silk_ADD_SAT16(short.MinValue, short.MinValue));
        // Boundary: exactly MaxValue without saturation.
        Equal(short.MaxValue, silk_ADD_SAT16(short.MaxValue, 0));
    }

    [TestMethod]
    public void SilkMacros_InsertionSort_SortsAscending()
    {
        // Already sorted.
        Span<short> a = stackalloc short[] { 1, 2, 3, 4, 5 };
        silk_insertion_sort_increasing_all_values_int16(a);
        for (int i = 0; i < a.Length; i++) Equal((short)(i + 1), a[i]);

        // Reverse sorted (worst case).
        Span<short> b = stackalloc short[] { 5, 4, 3, 2, 1 };
        silk_insertion_sort_increasing_all_values_int16(b);
        for (int i = 0; i < b.Length; i++) Equal((short)(i + 1), b[i]);

        // Random.
        Span<short> c = stackalloc short[] { 3, 1, 4, 1, 5, 9, 2, 6 };
        silk_insertion_sort_increasing_all_values_int16(c);
        short[] expected = { 1, 1, 2, 3, 4, 5, 6, 9 };
        for (int i = 0; i < c.Length; i++) Equal(expected[i], c[i]);

        // Single element + empty are no-ops.
        Span<short> single = stackalloc short[] { 42 };
        silk_insertion_sort_increasing_all_values_int16(single);
        Equal((short)42, single[0]);

        Span<short> empty = Span<short>.Empty;
        silk_insertion_sort_increasing_all_values_int16(empty); // should not throw
    }

    [TestMethod]
    public void SilkMacros_InsertionSort_HandlesNegatives()
    {
        Span<short> a = stackalloc short[] { 3, -5, 0, 100, -1 };
        silk_insertion_sort_increasing_all_values_int16(a);
        short[] expected = { -5, -1, 0, 3, 100 };
        for (int i = 0; i < a.Length; i++) Equal(expected[i], a[i]);
    }

    // -------- SilkNlsfStabilize --------

    [TestMethod]
    public void NlsfStabilize_AlreadyStable_Unchanged()
    {
        // NLSF sorted with plenty of spacing.
        Span<short> nlsf = stackalloc short[] { 1000, 5000, 10000, 20000, 30000 };
        short[] original = nlsf.ToArray();
        short[] deltaMin = { 500, 500, 500, 500, 500, 500 }; // L+1 = 6 entries
        SilkNlsfStabilize.Stabilize(nlsf, deltaMin);
        for (int i = 0; i < nlsf.Length; i++)
        {
            Equal(original[i], nlsf[i], $"NLSF[{i}] changed unexpectedly");
        }
    }

    [TestMethod]
    public void NlsfStabilize_TooCloseMiddle_SpacesApart()
    {
        // NLSF vector with two middle entries too close.
        Span<short> nlsf = stackalloc short[] { 1000, 5000, 5100, 20000, 30000 };
        short[] deltaMin = { 500, 500, 500, 500, 500, 500 };
        SilkNlsfStabilize.Stabilize(nlsf, deltaMin);

        // Verify ordering + spacing post-stabilize.
        for (int i = 1; i < nlsf.Length; i++)
        {
            if (nlsf[i] < nlsf[i - 1] + deltaMin[i])
                throw new Exception($"After stabilize: NLSF[{i}]={nlsf[i]} too close to NLSF[{i - 1}]={nlsf[i - 1]} (min delta {deltaMin[i]})");
        }
    }

    [TestMethod]
    public void NlsfStabilize_FirstBelowBoundary_ClampsUp()
    {
        // First NLSF below its minimum; should be raised to deltaMin[0].
        Span<short> nlsf = stackalloc short[] { 100, 5000, 10000, 20000, 30000 };
        short[] deltaMin = { 500, 500, 500, 500, 500, 500 };
        SilkNlsfStabilize.Stabilize(nlsf, deltaMin);
        True(nlsf[0] >= deltaMin[0], $"nlsf[0]={nlsf[0]} should be >= {deltaMin[0]}");
    }

    [TestMethod]
    public void NlsfStabilize_LastAboveBoundary_ClampsDown()
    {
        // Last NLSF too close to the upper bound (1 << 15 = 32768).
        Span<short> nlsf = stackalloc short[] { 1000, 5000, 10000, 20000, 32700 };
        short[] deltaMin = { 500, 500, 500, 500, 500, 500 };
        SilkNlsfStabilize.Stabilize(nlsf, deltaMin);
        int upperBound = (1 << 15) - deltaMin[5];
        True(nlsf[nlsf.Length - 1] <= upperBound, $"last NLSF {nlsf[nlsf.Length - 1]} should be <= {upperBound}");
    }

    [TestMethod]
    public void NlsfStabilize_ReversedInput_FallbackSortsCorrectly()
    {
        // A maliciously reversed input requires the fallback path in libopus; our port
        // is expected to produce a valid ordered output regardless of which path it takes.
        Span<short> nlsf = stackalloc short[] { 30000, 20000, 10000, 5000, 1000 };
        short[] deltaMin = { 500, 500, 500, 500, 500, 500 };
        SilkNlsfStabilize.Stabilize(nlsf, deltaMin);
        for (int i = 1; i < nlsf.Length; i++)
        {
            if (nlsf[i] < nlsf[i - 1] + deltaMin[i])
                throw new Exception($"Post-stabilize ordering broken at i={i}: {nlsf[i - 1]} -> {nlsf[i]} (min delta {deltaMin[i]})");
        }
    }

    [TestMethod]
    public void NlsfStabilize_DeltaMinTooSmall_Throws()
    {
        short[] nlsf = new short[5];
        short[] deltaMin = new short[5]; // Missing the L+1 entry.
        Throws<ArgumentException>(() => SilkNlsfStabilize.Stabilize(nlsf, deltaMin));
    }

    [TestMethod]
    public void NlsfStabilize_LastDeltaZero_Throws()
    {
        short[] nlsf = new short[5];
        short[] deltaMin = { 500, 500, 500, 500, 500, 0 }; // deltaMin[L] must be >= 1
        Throws<ArgumentException>(() => SilkNlsfStabilize.Stabilize(nlsf, deltaMin));
    }
}
