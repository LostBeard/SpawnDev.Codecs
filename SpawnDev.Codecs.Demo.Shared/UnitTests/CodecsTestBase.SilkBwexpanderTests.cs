using SpawnDev.Codecs.Audio.Opus.Silk;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Audio.Opus.Silk.SilkMacros;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for SILK bandwidth expander (AR filter chirp) and the <see cref="silk_SMULWW"/>
/// macro it depends on.
/// </summary>
public abstract partial class CodecsTestBase
{
    // -------- silk_SMULWW --------

    [TestMethod]
    public void SilkMacros_Smulww_ProducesExpectedProduct()
    {
        // silk_SMULWW(65536, x) == x (chirp of 1.0 identity)
        Equal(100, silk_SMULWW(65536, 100));
        Equal(-100, silk_SMULWW(65536, -100));
        Equal(0, silk_SMULWW(0, 100));

        // silk_SMULWW(0x8000, 100) == scaling by 0.5 (32768/65536)
        // The algorithm: silk_SMULWB(a, b) + a * silk_RSHIFT_ROUND(b, 16).
        // For a=0x8000, b=100: silk_SMULWB(0x8000, 100) = (0x8000 * 100) >> 16 = (3276800) >> 16 = 50.
        // silk_RSHIFT_ROUND(100, 16) = ((100 >> 15) + 1) >> 1 = (0 + 1) >> 1 = 0.
        // So result = 50 + 0x8000 * 0 = 50.
        Equal(50, silk_SMULWW(0x8000, 100));
    }

    [TestMethod]
    public void SilkMacros_Smulww_HandlesLargeValues()
    {
        // chirp of 1.0 preserves large values without loss in the bottom 16 bits.
        Equal(1 << 20, silk_SMULWW(65536, 1 << 20));
    }

    // -------- SilkBwexpander --------

    [TestMethod]
    public void Bwexpander16_ChirpOne_Identity()
    {
        // chirp == 65536 (1.0 in Q16) should be very close to identity for 16-bit AR.
        Span<short> ar = stackalloc short[] { 1000, 2000, 3000, 4000 };
        short[] original = ar.ToArray();
        SilkBwexpander.Expand16(ar, 65536);

        // At chirp=1.0, each coefficient is multiplied by chirp^(i+1) = 1.0; output equals input
        // up to rounding in silk_RSHIFT_ROUND(x, 16), which is 0 for x in typical AR range.
        for (int i = 0; i < ar.Length; i++)
        {
            int diff = Math.Abs(ar[i] - original[i]);
            if (diff > 1) throw new Exception($"ar[{i}] changed by {diff} under chirp=1.0 (expected ~0)");
        }
    }

    [TestMethod]
    public void Bwexpander16_ChirpZero_Zeroes()
    {
        // chirp == 0 multiplies everything by 0. Should produce all zeros (with some rounding).
        Span<short> ar = stackalloc short[] { 1000, 2000, 3000, 4000 };
        SilkBwexpander.Expand16(ar, 0);
        for (int i = 0; i < ar.Length; i++)
        {
            Equal((short)0, ar[i], $"ar[{i}]");
        }
    }

    [TestMethod]
    public void Bwexpander16_ChirpHalf_HalvesFirstCoeff()
    {
        // chirp == 32768 (0.5 in Q16). First coefficient should be approximately halved.
        Span<short> ar = stackalloc short[] { 10000, 0, 0, 0 };
        SilkBwexpander.Expand16(ar, 32768);
        // First coefficient: 10000 * 0.5 = 5000
        int diff = Math.Abs(ar[0] - 5000);
        if (diff > 1) throw new Exception($"ar[0] = {ar[0]}, expected ~5000");
    }

    [TestMethod]
    public void Bwexpander16_EmptyArray_NoOp()
    {
        Span<short> ar = Span<short>.Empty;
        SilkBwexpander.Expand16(ar, 32768); // should not throw
    }

    [TestMethod]
    public void Bwexpander32_ChirpOne_Identity()
    {
        Span<int> ar = stackalloc int[] { 1000, 2000, 3000, 4000 };
        int[] original = ar.ToArray();
        SilkBwexpander.Expand32(ar, 65536);

        // Same expectation as 16-bit: chirp=1.0 is near-identity.
        for (int i = 0; i < ar.Length; i++)
        {
            int diff = Math.Abs(ar[i] - original[i]);
            if (diff > 1) throw new Exception($"ar[{i}] changed by {diff} under chirp=1.0 (expected ~0)");
        }
    }

    [TestMethod]
    public void Bwexpander32_ChirpZero_Zeroes()
    {
        Span<int> ar = stackalloc int[] { 1 << 20, 1 << 20, 1 << 20, 1 << 20 };
        SilkBwexpander.Expand32(ar, 0);
        for (int i = 0; i < ar.Length; i++)
        {
            Equal(0, ar[i], $"ar[{i}]");
        }
    }

    [TestMethod]
    public void Bwexpander32_ChirpDecaysAlongCoefficients()
    {
        // With chirp = 0.9 (in Q16: 58982), later coefficients get smaller: coef[i] scaled by
        // chirp^(i+1). ar[3] / ar[0] should be roughly chirp^3 of the original proportionality.
        Span<int> ar = stackalloc int[] { 1 << 20, 1 << 20, 1 << 20, 1 << 20 };
        SilkBwexpander.Expand32(ar, 58982); // ~0.9 in Q16

        // Ratios should be monotonically decreasing.
        if (ar[0] <= ar[1] || ar[1] <= ar[2] || ar[2] <= ar[3])
            throw new Exception($"Chirp decay broken: [{ar[0]}, {ar[1]}, {ar[2]}, {ar[3]}]");

        // Sanity: ar[3] should be roughly ar[0] * 0.9^3 = ar[0] * 0.729.
        double ratio = (double)ar[3] / ar[0];
        if (ratio < 0.70 || ratio > 0.76)
            throw new Exception($"ar[3]/ar[0] = {ratio:F3}, expected near 0.729");
    }

    [TestMethod]
    public void Bwexpander32_EmptyArray_NoOp()
    {
        Span<int> ar = Span<int>.Empty;
        SilkBwexpander.Expand32(ar, 32768);
    }
}
