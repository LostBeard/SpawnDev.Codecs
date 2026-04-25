// Tests for Vp9IntraMode + Vp9IntraModeTree (slice 153).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9IntraMode_EnumValuesMatchLibvpx()
    {
        Equal((byte)0, (byte)Vp9IntraMode.DcPred);
        Equal((byte)1, (byte)Vp9IntraMode.VPred);
        Equal((byte)2, (byte)Vp9IntraMode.HPred);
        Equal((byte)3, (byte)Vp9IntraMode.D45Pred);
        Equal((byte)4, (byte)Vp9IntraMode.D135Pred);
        Equal((byte)5, (byte)Vp9IntraMode.D117Pred);
        Equal((byte)6, (byte)Vp9IntraMode.D153Pred);
        Equal((byte)7, (byte)Vp9IntraMode.D207Pred);
        Equal((byte)8, (byte)Vp9IntraMode.D63Pred);
        Equal((byte)9, (byte)Vp9IntraMode.TmPred);
        Equal(10, Vp9IntraModeTree.IntraModes);
    }

    [TestMethod]
    public void Vp9IntraModeTree_HasCorrectShape()
    {
        // 18 entries (9 internal nodes x 2 branches).
        Equal(18, Vp9IntraModeTree.Tree.Length);
        // Spot-check the pinned topology.
        Equal((sbyte)0,  Vp9IntraModeTree.Tree[0]);  // -DcPred = 0
        Equal((sbyte)2,  Vp9IntraModeTree.Tree[1]);  // -> TM_NODE
        Equal((sbyte)(-9), Vp9IntraModeTree.Tree[2]); // -TmPred = -9
        Equal((sbyte)8,  Vp9IntraModeTree.Tree[6]);  // COM_NODE -> H_NODE
        Equal((sbyte)12, Vp9IntraModeTree.Tree[7]);  // COM_NODE -> D45_NODE
        Equal((sbyte)(-7), Vp9IntraModeTree.Tree[17]); // -D207Pred = -7
    }

    [TestMethod]
    public void Vp9IntraModeTree_DecodesDcPredOnFirstZeroBit()
    {
        // Tree slot 0 = -DcPred = 0; with bit 0 read first, walker
        // returns DcPred immediately.
        var bits = new int[] { 0 };
        int idx = 0;
        var m = Vp9IntraModeTree.Decode(_ => bits[idx++], stackalloc byte[9]);
        Equal(Vp9IntraMode.DcPred, m);
    }

    [TestMethod]
    public void Vp9IntraModeTree_DecodesTmPredOnPath_1_0()
    {
        // i=0 bit 1 -> i=2 (TM_NODE), bit 0 -> -TmPred leaf.
        var bits = new int[] { 1, 0 };
        int idx = 0;
        var m = Vp9IntraModeTree.Decode(_ => bits[idx++], stackalloc byte[9]);
        Equal(Vp9IntraMode.TmPred, m);
    }

    [TestMethod]
    public void Vp9IntraModeTree_DecodesVPredOnPath_1_1_0()
    {
        // i=0 bit 1 -> i=2, bit 1 -> i=4 (V_NODE), bit 0 -> -VPred.
        var bits = new int[] { 1, 1, 0 };
        int idx = 0;
        var m = Vp9IntraModeTree.Decode(_ => bits[idx++], stackalloc byte[9]);
        Equal(Vp9IntraMode.VPred, m);
    }

    [TestMethod]
    public void Vp9IntraModeTree_DecodesHPredOnPath_1_1_1_0_0()
    {
        // i=0 bit 1 -> i=2, bit 1 -> i=4, bit 1 -> i=6 (COM_NODE), bit 0 -> i=8 (H_NODE), bit 0 -> -HPred.
        var bits = new int[] { 1, 1, 1, 0, 0 };
        int idx = 0;
        var m = Vp9IntraModeTree.Decode(_ => bits[idx++], stackalloc byte[9]);
        Equal(Vp9IntraMode.HPred, m);
    }

    [TestMethod]
    public void Vp9IntraModeTree_DecodesD135PredOnPath_1_1_1_0_1_0()
    {
        // ...COM_NODE bit 0 -> H_NODE, bit 1 -> i=10 (D135_NODE), bit 0 -> -D135Pred.
        var bits = new int[] { 1, 1, 1, 0, 1, 0 };
        int idx = 0;
        var m = Vp9IntraModeTree.Decode(_ => bits[idx++], stackalloc byte[9]);
        Equal(Vp9IntraMode.D135Pred, m);
    }

    [TestMethod]
    public void Vp9IntraModeTree_DecodesD207PredOnPath_AllOnesIntoD153_Last1()
    {
        // i=0 bit 1, i=2 bit 1, i=4 bit 1 -> i=6 (COM_NODE),
        // bit 1 -> i=12 (D45_NODE), bit 1 -> i=14 (D63_NODE),
        // bit 1 -> i=16 (D153_NODE), bit 1 -> -D207Pred.
        var bits = new int[] { 1, 1, 1, 1, 1, 1, 1 };
        int idx = 0;
        var m = Vp9IntraModeTree.Decode(_ => bits[idx++], stackalloc byte[9]);
        Equal(Vp9IntraMode.D207Pred, m);
    }

    [TestMethod]
    public void Vp9IntraModeTree_RejectsUndersizedProbs()
    {
        Throws<ArgumentException>(() => Vp9IntraModeTree.Decode(_ => 0, new byte[8]));
    }
}
