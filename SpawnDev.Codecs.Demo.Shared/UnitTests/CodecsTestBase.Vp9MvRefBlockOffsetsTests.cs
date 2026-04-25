// Tests for Vp9MvRefBlockOffsets (slice 276).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9MvRefBlockOffsets_Constants_MatchLibvpx()
    {
        Equal(8, Vp9MvRefBlockOffsets.Neighbours);
        Equal(13, Vp9MvRefBlockOffsets.Lookup.GetLength(0));
        Equal(8, Vp9MvRefBlockOffsets.Lookup.GetLength(1));
        Equal(2, Vp9MvRefBlockOffsets.Lookup.GetLength(2));
    }

    [TestMethod]
    public void Vp9MvRefBlockOffsets_AllSubEightSizes_Identical()
    {
        // Block4x4 / 4x8 / 8x4 / 8x8 all share the same 8-neighbor pattern.
        for (int n = 0; n < 8; n++)
        {
            var (r0, c0) = Vp9MvRefBlockOffsets.GetOffset(Vp9BlockSize.Block4x4, n);
            Equal(r0, Vp9MvRefBlockOffsets.Lookup[(int)Vp9BlockSize.Block4x8, n, 0]);
            Equal(c0, Vp9MvRefBlockOffsets.Lookup[(int)Vp9BlockSize.Block4x8, n, 1]);
            Equal(r0, Vp9MvRefBlockOffsets.Lookup[(int)Vp9BlockSize.Block8x4, n, 0]);
            Equal(c0, Vp9MvRefBlockOffsets.Lookup[(int)Vp9BlockSize.Block8x4, n, 1]);
            Equal(r0, Vp9MvRefBlockOffsets.Lookup[(int)Vp9BlockSize.Block8x8, n, 0]);
            Equal(c0, Vp9MvRefBlockOffsets.Lookup[(int)Vp9BlockSize.Block8x8, n, 1]);
        }
    }

    [TestMethod]
    public void Vp9MvRefBlockOffsets_Block16x16_FirstNeighbor_IsAbove()
    {
        var (r, c) = Vp9MvRefBlockOffsets.GetOffset(Vp9BlockSize.Block16x16, 0);
        Equal((sbyte)-1, r);
        Equal((sbyte)0, c);
    }

    [TestMethod]
    public void Vp9MvRefBlockOffsets_Block64x64_FarBelow()
    {
        // 64x64 has unusually large positive offsets to reach into the
        // already-decoded right-of-the-block neighbors.
        var (r, c) = Vp9MvRefBlockOffsets.GetOffset(Vp9BlockSize.Block64x64, 7);
        Equal((sbyte)-1, r);
        Equal((sbyte)6, c);
    }

    [TestMethod]
    public void Vp9MvRefBlockOffsets_Rejects_OutOfRangeBlockSize()
    {
        Throws<ArgumentOutOfRangeException>(() =>
            Vp9MvRefBlockOffsets.GetOffset((Vp9BlockSize)99, 0));
    }

    [TestMethod]
    public void Vp9MvRefBlockOffsets_Rejects_OutOfRangeNeighbor()
    {
        Throws<ArgumentOutOfRangeException>(() =>
            Vp9MvRefBlockOffsets.GetOffset(Vp9BlockSize.Block16x16, 8));
        Throws<ArgumentOutOfRangeException>(() =>
            Vp9MvRefBlockOffsets.GetOffset(Vp9BlockSize.Block16x16, -1));
    }
}
