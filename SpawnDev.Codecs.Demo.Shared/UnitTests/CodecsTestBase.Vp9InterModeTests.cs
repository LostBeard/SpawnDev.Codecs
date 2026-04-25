// Tests for Vp9InterMode + Vp9InterModeTree (slice 158).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9InterMode_OffsetValuesMatchLibvpx()
    {
        // libvpx INTER_OFFSET: NEARESTMV=0, NEARMV=1, ZEROMV=2, NEWMV=3.
        Equal((byte)0, (byte)Vp9InterMode.NearestMv);
        Equal((byte)1, (byte)Vp9InterMode.NearMv);
        Equal((byte)2, (byte)Vp9InterMode.ZeroMv);
        Equal((byte)3, (byte)Vp9InterMode.NewMv);
        Equal(4, Vp9InterModeTree.InterModes);
    }

    [TestMethod]
    public void Vp9InterModeTree_HasCorrectShape()
    {
        // 6 entries = 3 internal nodes x 2 branches.
        Equal(6, Vp9InterModeTree.Tree.Length);
        Equal((sbyte)(-2), Vp9InterModeTree.Tree[0]); // -ZeroMv = -2
        Equal((sbyte)2,    Vp9InterModeTree.Tree[1]); // -> NEAREST_NM
        Equal((sbyte)0,    Vp9InterModeTree.Tree[2]); // -NearestMv = 0 (zero-leaf)
        Equal((sbyte)4,    Vp9InterModeTree.Tree[3]); // -> NEAR_OR_NEW
        Equal((sbyte)(-1), Vp9InterModeTree.Tree[4]); // -NearMv = -1
        Equal((sbyte)(-3), Vp9InterModeTree.Tree[5]); // -NewMv = -3
    }

    [TestMethod]
    public void Vp9InterModeTree_DecodesZeroMvOnFirstZeroBit()
    {
        // Tree slot 0 = -ZeroMv = -2, leaf.
        var bits = new int[] { 0 };
        int idx = 0;
        var m = Vp9InterModeTree.Decode(_ => bits[idx++], stackalloc byte[3]);
        Equal(Vp9InterMode.ZeroMv, m);
    }

    [TestMethod]
    public void Vp9InterModeTree_DecodesNearestMvOnPath_1_0()
    {
        // i=0 bit 1 -> i=2 (NEAREST_NM), bit 0 -> -NearestMv = 0 leaf.
        var bits = new int[] { 1, 0 };
        int idx = 0;
        var m = Vp9InterModeTree.Decode(_ => bits[idx++], stackalloc byte[3]);
        Equal(Vp9InterMode.NearestMv, m);
    }

    [TestMethod]
    public void Vp9InterModeTree_DecodesNearMvOnPath_1_1_0()
    {
        // i=0 bit 1 -> i=2, bit 1 -> i=4 (NEAR_OR_NEW), bit 0 -> -NearMv = -1.
        var bits = new int[] { 1, 1, 0 };
        int idx = 0;
        var m = Vp9InterModeTree.Decode(_ => bits[idx++], stackalloc byte[3]);
        Equal(Vp9InterMode.NearMv, m);
    }

    [TestMethod]
    public void Vp9InterModeTree_DecodesNewMvOnPath_1_1_1()
    {
        // ...bit 1 -> -NewMv = -3.
        var bits = new int[] { 1, 1, 1 };
        int idx = 0;
        var m = Vp9InterModeTree.Decode(_ => bits[idx++], stackalloc byte[3]);
        Equal(Vp9InterMode.NewMv, m);
    }

    [TestMethod]
    public void Vp9InterModeTree_RejectsUndersizedProbs()
    {
        Throws<ArgumentException>(() => Vp9InterModeTree.Decode(_ => 0, new byte[2]));
    }
}
