// Tests for Vp9PartitionType + Vp9PartitionTree (slice 151).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9PartitionType_EnumValuesMatchLibvpxOrdering()
    {
        Equal((byte)0, (byte)Vp9PartitionType.None);
        Equal((byte)1, (byte)Vp9PartitionType.Horz);
        Equal((byte)2, (byte)Vp9PartitionType.Vert);
        Equal((byte)3, (byte)Vp9PartitionType.Split);
    }

    [TestMethod]
    public void Vp9PartitionTree_HasCorrectShape()
    {
        // 6 entries = 3 internal nodes x 2 branches.
        Equal(6, Vp9PartitionTree.Tree.Length);
        Equal((sbyte)0,  Vp9PartitionTree.Tree[0]);   // -None = 0
        Equal((sbyte)2,  Vp9PartitionTree.Tree[1]);   // -> H_OR_V_OR_S
        Equal((sbyte)(-1), Vp9PartitionTree.Tree[2]); // -Horz = -1
        Equal((sbyte)4,  Vp9PartitionTree.Tree[3]);   // -> V_OR_S
        Equal((sbyte)(-2), Vp9PartitionTree.Tree[4]); // -Vert = -2
        Equal((sbyte)(-3), Vp9PartitionTree.Tree[5]); // -Split = -3
    }

    [TestMethod]
    public void Vp9PartitionTree_DecodesNoneOnFirstZeroBit()
    {
        var bits = new int[] { 0 };
        int idx = 0;
        var p = Vp9PartitionTree.Decode(_ => bits[idx++], stackalloc byte[3] { 128, 128, 128 });
        Equal(Vp9PartitionType.None, p);
    }

    [TestMethod]
    public void Vp9PartitionTree_DecodesHorzOnSecondBitZero()
    {
        // ROOT: bit 1 -> i=2, then bit 0 at probs[1] -> -Horz.
        var bits = new int[] { 1, 0 };
        int idx = 0;
        var p = Vp9PartitionTree.Decode(_ => bits[idx++], stackalloc byte[3]);
        Equal(Vp9PartitionType.Horz, p);
    }

    [TestMethod]
    public void Vp9PartitionTree_DecodesVertOnSecondBitOneThirdBitZero()
    {
        // ROOT: 1 -> i=2, 1 -> i=4, 0 -> -Vert.
        var bits = new int[] { 1, 1, 0 };
        int idx = 0;
        var p = Vp9PartitionTree.Decode(_ => bits[idx++], stackalloc byte[3]);
        Equal(Vp9PartitionType.Vert, p);
    }

    [TestMethod]
    public void Vp9PartitionTree_DecodesSplitOnAllBitsOne()
    {
        // ROOT: 1 -> i=2, 1 -> i=4, 1 -> -Split.
        var bits = new int[] { 1, 1, 1 };
        int idx = 0;
        var p = Vp9PartitionTree.Decode(_ => bits[idx++], stackalloc byte[3]);
        Equal(Vp9PartitionType.Split, p);
    }

    [TestMethod]
    public void Vp9PartitionTree_RejectsUndersizedProbs()
    {
        Throws<ArgumentException>(() => Vp9PartitionTree.Decode(_ => 0, new byte[2]));
    }
}
