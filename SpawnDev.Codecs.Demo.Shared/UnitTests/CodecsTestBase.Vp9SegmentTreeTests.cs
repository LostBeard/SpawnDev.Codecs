// Tests for Vp9SegmentTree (slice 256).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9SegmentTree_Constants_MatchLibvpx()
    {
        Equal(8, Vp9SegmentTree.MaxSegments);
        Equal(7, Vp9SegmentTree.TreeProbs);
        Equal(14, Vp9SegmentTree.Tree.Length);
    }

    [TestMethod]
    public void Vp9SegmentTree_Decode_AllZeros_PicksSegment0()
    {
        // 0, 0, 0 -> -0 leaf.
        int callIdx = 0;
        int[] bits = { 0, 0, 0 };
        Equal(0, Vp9SegmentTree.Decode(p => bits[callIdx++], BuildSegProbs()));
    }

    [TestMethod]
    public void Vp9SegmentTree_Decode_PicksSegment7()
    {
        // 1, 1, 1 -> ROOT.right=4, i4.right=12, i12.right=-7 leaf.
        int callIdx = 0;
        int[] bits = { 1, 1, 1 };
        Equal(7, Vp9SegmentTree.Decode(p => bits[callIdx++], BuildSegProbs()));
    }

    [TestMethod]
    public void Vp9SegmentTree_Decode_PicksSegment3()
    {
        // 0, 1, 1 -> ROOT.left=2, i2.right=8, i8.right=-3 leaf.
        int callIdx = 0;
        int[] bits = { 0, 1, 1 };
        Equal(3, Vp9SegmentTree.Decode(p => bits[callIdx++], BuildSegProbs()));
    }

    [TestMethod]
    public void Vp9SegmentTree_Decode_PicksSegment4()
    {
        // 1, 0, 0 -> ROOT.right=4, i4.left=10, i10.left=-4 leaf.
        int callIdx = 0;
        int[] bits = { 1, 0, 0 };
        Equal(4, Vp9SegmentTree.Decode(p => bits[callIdx++], BuildSegProbs()));
    }

    [TestMethod]
    public void Vp9SegmentTree_Decode_RejectsShortProbs()
    {
        Throws<ArgumentException>(() =>
            Vp9SegmentTree.Decode(p => 0, new byte[6]));
    }

    [TestMethod]
    public void Vp9SegmentTree_Decode_RejectsNullReader()
    {
        Throws<ArgumentNullException>(() =>
            Vp9SegmentTree.Decode(null!, BuildSegProbs()));
    }

    private static byte[] BuildSegProbs()
    {
        var p = new byte[Vp9SegmentTree.TreeProbs];
        for (int i = 0; i < p.Length; i++) p[i] = 128;
        return p;
    }
}
