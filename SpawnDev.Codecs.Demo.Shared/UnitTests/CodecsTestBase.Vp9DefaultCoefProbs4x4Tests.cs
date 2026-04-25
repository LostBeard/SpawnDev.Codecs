// Tests for Vp9CoefProbs.DefaultCoefProbs4x4 (slice 142). Length and
// pinned values against libvpx, plus the band-0 zero-padding sanity
// check (ctx 3..5 of band 0 must be all zero in the flat layout).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9DefaultCoefProbs4x4_HasCorrect432EntryLength()
    {
        Equal(432, Vp9CoefProbs.DefaultCoefProbs4x4.Length);
    }

    [TestMethod]
    public void Vp9DefaultCoefProbs4x4_PinnedFirstEntries_MatchLibvpx()
    {
        // From libvpx default_coef_probs_4x4 [Y plane][Intra][Band 0]:
        //   { 195, 29, 183 }
        //   { 84, 49, 136 }
        //   { 8, 42, 71 }
        var p = Vp9CoefProbs.DefaultCoefProbs4x4;
        Equal((byte)195, p[Vp9CoefProbs.Index4x4(0, 0, 0, 0, 0)]);
        Equal((byte)29,  p[Vp9CoefProbs.Index4x4(0, 0, 0, 0, 1)]);
        Equal((byte)183, p[Vp9CoefProbs.Index4x4(0, 0, 0, 0, 2)]);
        Equal((byte)84,  p[Vp9CoefProbs.Index4x4(0, 0, 0, 1, 0)]);
        Equal((byte)49,  p[Vp9CoefProbs.Index4x4(0, 0, 0, 1, 1)]);
        Equal((byte)136, p[Vp9CoefProbs.Index4x4(0, 0, 0, 1, 2)]);
        Equal((byte)8,   p[Vp9CoefProbs.Index4x4(0, 0, 0, 2, 0)]);
        Equal((byte)42,  p[Vp9CoefProbs.Index4x4(0, 0, 0, 2, 1)]);
        Equal((byte)71,  p[Vp9CoefProbs.Index4x4(0, 0, 0, 2, 2)]);
    }

    [TestMethod]
    public void Vp9DefaultCoefProbs4x4_Band0Ctx3To5_AreZeroPadded()
    {
        // Band 0 only has 3 actual contexts in libvpx; the rectangular
        // flat layout pads ctx 3..5 with zero across all 4 (plane, ref)
        // combinations. Verify the 36 padded entries are all zero.
        var p = Vp9CoefProbs.DefaultCoefProbs4x4;
        for (int plane = 0; plane < 2; plane++)
        for (int refT = 0; refT < 2; refT++)
        for (int ctx = 3; ctx < 6; ctx++)
        for (int node = 0; node < 3; node++)
        {
            int i = Vp9CoefProbs.Index4x4(plane, refT, 0, ctx, node);
            Equal((byte)0, p[i]);
        }
    }

    [TestMethod]
    public void Vp9DefaultCoefProbs4x4_PinnedBand1FirstContext_MatchesLibvpx()
    {
        // Band 1's first context starts right after band 0's 18 entries
        // (3 real + 3 zero-padded). libvpx Y/Intra/Band 1[0] = { 31, 107, 169 }.
        var p = Vp9CoefProbs.DefaultCoefProbs4x4;
        Equal((byte)31,  p[Vp9CoefProbs.Index4x4(0, 0, 1, 0, 0)]);
        Equal((byte)107, p[Vp9CoefProbs.Index4x4(0, 0, 1, 0, 1)]);
        Equal((byte)169, p[Vp9CoefProbs.Index4x4(0, 0, 1, 0, 2)]);
    }

    [TestMethod]
    public void Vp9DefaultCoefProbs4x4_PinnedUVInterBand5Last_MatchesLibvpx()
    {
        // Last triple of the entire 4x4 table: UV plane / Inter ref /
        // Band 5 / ctx 5 = libvpx { 8, 23, 61 }. Pinning the very last
        // entry catches a class of off-by-one errors in the flat extraction.
        var p = Vp9CoefProbs.DefaultCoefProbs4x4;
        Equal((byte)8,  p[Vp9CoefProbs.Index4x4(1, 1, 5, 5, 0)]);
        Equal((byte)23, p[Vp9CoefProbs.Index4x4(1, 1, 5, 5, 1)]);
        Equal((byte)61, p[Vp9CoefProbs.Index4x4(1, 1, 5, 5, 2)]);
        // And the very last flat slot.
        Equal((byte)61, p[431]);
    }

    [TestMethod]
    public void Vp9DefaultCoefProbs4x4_Index4x4_IsConsistentWithRowMajorMath()
    {
        // Sanity-check the index helper round-trips. Every
        // (plane, ref, band, ctx, node) tuple should produce the same
        // index as the manual flat formula.
        for (int plane = 0; plane < 2; plane++)
        for (int refT = 0; refT < 2; refT++)
        for (int band = 0; band < 6; band++)
        for (int ctx = 0; ctx < 6; ctx++)
        for (int node = 0; node < 3; node++)
        {
            int got = Vp9CoefProbs.Index4x4(plane, refT, band, ctx, node);
            int expected = ((((plane * 2 + refT) * 6 + band) * 6 + ctx) * 3 + node);
            Equal(expected, got);
        }
    }
}
