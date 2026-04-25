// Tests for Vp9CoefProbs.DefaultCoefProbs{8x8,16x16,32x32} (slice 143).
// Same pattern as slice 142's 4x4 tests: length, first-triple pin
// against libvpx, and the band-0 zero-padding invariant.

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private static void AssertProbTableShape(byte[] probs, string label)
    {
        Equal(432, probs.Length);
        // Band 0 zero-padding: 9 zero bytes per (plane, ref) combination,
        // 36 zero bytes total at predictable offsets.
        for (int plane = 0; plane < 2; plane++)
        for (int refT = 0; refT < 2; refT++)
        for (int ctx = 3; ctx < 6; ctx++)
        for (int node = 0; node < 3; node++)
        {
            int i = Vp9CoefProbs.Index4x4(plane, refT, 0, ctx, node);
            Equal((byte)0, probs[i]);
        }
    }

    [TestMethod]
    public void Vp9DefaultCoefProbs8x8_HasCorrectShapeAndPadding()
    {
        AssertProbTableShape(Vp9CoefProbs.DefaultCoefProbs8x8, "DefaultCoefProbs8x8");
        // First triple from libvpx 8x8 Y/Intra/Band 0 ctx 0: { 125, 34, 187 }.
        Equal((byte)125, Vp9CoefProbs.DefaultCoefProbs8x8[Vp9CoefProbs.Index4x4(0, 0, 0, 0, 0)]);
        Equal((byte)34,  Vp9CoefProbs.DefaultCoefProbs8x8[Vp9CoefProbs.Index4x4(0, 0, 0, 0, 1)]);
        Equal((byte)187, Vp9CoefProbs.DefaultCoefProbs8x8[Vp9CoefProbs.Index4x4(0, 0, 0, 0, 2)]);
    }

    [TestMethod]
    public void Vp9DefaultCoefProbs16x16_HasCorrectShapeAndPadding()
    {
        AssertProbTableShape(Vp9CoefProbs.DefaultCoefProbs16x16, "DefaultCoefProbs16x16");
        // First triple from libvpx 16x16 Y/Intra/Band 0 ctx 0:
        // libvpx default_coef_probs_16x16[0][0][0][0] = { 7, 27, 153 }
        // (verified directly from vp9_entropy.c).
        Equal((byte)7,   Vp9CoefProbs.DefaultCoefProbs16x16[Vp9CoefProbs.Index4x4(0, 0, 0, 0, 0)]);
        Equal((byte)27,  Vp9CoefProbs.DefaultCoefProbs16x16[Vp9CoefProbs.Index4x4(0, 0, 0, 0, 1)]);
        Equal((byte)153, Vp9CoefProbs.DefaultCoefProbs16x16[Vp9CoefProbs.Index4x4(0, 0, 0, 0, 2)]);
    }

    [TestMethod]
    public void Vp9DefaultCoefProbs32x32_HasCorrectShapeAndPadding()
    {
        AssertProbTableShape(Vp9CoefProbs.DefaultCoefProbs32x32, "DefaultCoefProbs32x32");
        // First triple from libvpx 32x32 Y/Intra/Band 0 ctx 0:
        // libvpx default_coef_probs_32x32[0][0][0][0] = { 17, 38, 140 }
        // (verified directly from vp9_entropy.c).
        Equal((byte)17,  Vp9CoefProbs.DefaultCoefProbs32x32[Vp9CoefProbs.Index4x4(0, 0, 0, 0, 0)]);
        Equal((byte)38,  Vp9CoefProbs.DefaultCoefProbs32x32[Vp9CoefProbs.Index4x4(0, 0, 0, 0, 1)]);
        Equal((byte)140, Vp9CoefProbs.DefaultCoefProbs32x32[Vp9CoefProbs.Index4x4(0, 0, 0, 0, 2)]);
    }

    [TestMethod]
    public void Vp9DefaultCoefProbs_DispatcherReturnsCorrectArray()
    {
        True(ReferenceEquals(Vp9CoefProbs.DefaultCoefProbs4x4,   Vp9CoefProbs.DefaultCoefProbsFor(Vp9TxSize.Tx4x4)));
        True(ReferenceEquals(Vp9CoefProbs.DefaultCoefProbs8x8,   Vp9CoefProbs.DefaultCoefProbsFor(Vp9TxSize.Tx8x8)));
        True(ReferenceEquals(Vp9CoefProbs.DefaultCoefProbs16x16, Vp9CoefProbs.DefaultCoefProbsFor(Vp9TxSize.Tx16x16)));
        True(ReferenceEquals(Vp9CoefProbs.DefaultCoefProbs32x32, Vp9CoefProbs.DefaultCoefProbsFor(Vp9TxSize.Tx32x32)));
    }

    [TestMethod]
    public void Vp9DefaultCoefProbs_AllNonPadEntries_AreInValidRange()
    {
        // VP9 probabilities live in [1, 255]; 0 is reserved for "dead
        // prob tree" (libvpx MIN_PROB = 1). Verify across all four prob
        // tables that every NON-PADDING entry satisfies the invariant.
        // Padding slots (band 0 ctx 3..5) are intentionally zero and
        // must be excluded from this check.
        var tables = new[]
        {
            Vp9CoefProbs.DefaultCoefProbs4x4,
            Vp9CoefProbs.DefaultCoefProbs8x8,
            Vp9CoefProbs.DefaultCoefProbs16x16,
            Vp9CoefProbs.DefaultCoefProbs32x32,
        };
        foreach (var table in tables)
        {
            for (int plane = 0; plane < 2; plane++)
            for (int refT = 0; refT < 2; refT++)
            for (int band = 0; band < 6; band++)
            for (int ctx = 0; ctx < 6; ctx++)
            for (int node = 0; node < 3; node++)
            {
                bool isPad = (band == 0 && ctx >= 3);
                if (isPad) continue;
                int i = Vp9CoefProbs.Index4x4(plane, refT, band, ctx, node);
                True(table[i] >= 1, $"table[{i}] = {table[i]} below MIN_PROB");
            }
        }
    }
}
