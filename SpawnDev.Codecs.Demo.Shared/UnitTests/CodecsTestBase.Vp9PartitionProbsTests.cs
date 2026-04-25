// Tests for Vp9PartitionProbs (slice 152).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9PartitionProbs_BothTablesAre48Bytes()
    {
        Equal(48, Vp9PartitionProbs.KfPartitionProbs.Length);
        Equal(48, Vp9PartitionProbs.DefaultPartitionProbs.Length);
    }

    [TestMethod]
    public void Vp9PartitionProbs_KfFirstContextMatchesLibvpx()
    {
        // 8x8->4x4 a/l both not split: { 158, 97, 94 }.
        Equal((byte)158, Vp9PartitionProbs.KfPartitionProbs[0]);
        Equal((byte)97,  Vp9PartitionProbs.KfPartitionProbs[1]);
        Equal((byte)94,  Vp9PartitionProbs.KfPartitionProbs[2]);
    }

    [TestMethod]
    public void Vp9PartitionProbs_KfLastContextMatchesLibvpx()
    {
        // 64x64->32x32 a/l both split: { 12, 3, 3 }.
        Equal((byte)12, Vp9PartitionProbs.KfPartitionProbs[45]);
        Equal((byte)3,  Vp9PartitionProbs.KfPartitionProbs[46]);
        Equal((byte)3,  Vp9PartitionProbs.KfPartitionProbs[47]);
    }

    [TestMethod]
    public void Vp9PartitionProbs_DefaultFirstAndLastContextMatchLibvpx()
    {
        // First: 8x8->4x4 a/l both not split = { 199, 122, 141 }.
        Equal((byte)199, Vp9PartitionProbs.DefaultPartitionProbs[0]);
        Equal((byte)122, Vp9PartitionProbs.DefaultPartitionProbs[1]);
        Equal((byte)141, Vp9PartitionProbs.DefaultPartitionProbs[2]);
        // Last: 64x64->32x32 a/l both split = { 10, 7, 6 }.
        Equal((byte)10, Vp9PartitionProbs.DefaultPartitionProbs[45]);
        Equal((byte)7,  Vp9PartitionProbs.DefaultPartitionProbs[46]);
        Equal((byte)6,  Vp9PartitionProbs.DefaultPartitionProbs[47]);
    }

    [TestMethod]
    public void Vp9PartitionProbs_Index_ProducesCorrectFlatOffset()
    {
        // Hand-verify a few index values.
        Equal(0,  Vp9PartitionProbs.Index(0, 0, 0));
        Equal(2,  Vp9PartitionProbs.Index(0, 0, 2));
        Equal(3,  Vp9PartitionProbs.Index(0, 1, 0));
        Equal(12, Vp9PartitionProbs.Index(1, 0, 0));
        Equal(45, Vp9PartitionProbs.Index(3, 3, 0));
        Equal(47, Vp9PartitionProbs.Index(3, 3, 2));
    }

    [TestMethod]
    public void Vp9PartitionProbs_Index_RejectsOutOfRangeArgs()
    {
        Throws<ArgumentOutOfRangeException>(() => Vp9PartitionProbs.Index(4, 0, 0));
        Throws<ArgumentOutOfRangeException>(() => Vp9PartitionProbs.Index(0, 4, 0));
        Throws<ArgumentOutOfRangeException>(() => Vp9PartitionProbs.Index(0, 0, 3));
        Throws<ArgumentOutOfRangeException>(() => Vp9PartitionProbs.Index(-1, 0, 0));
    }

    [TestMethod]
    public void Vp9PartitionProbs_KeyframeProbsHelper_ReturnsExpectedSlice()
    {
        // 32x32->16x16 a unsplit + l split (sizeIdx=2, splitState=2)
        // = { 67, 33, 11 } per libvpx.
        var probs = Vp9PartitionProbs.KeyframeProbs(2, 2);
        Equal(3, probs.Length);
        Equal((byte)67, probs[0]);
        Equal((byte)33, probs[1]);
        Equal((byte)11, probs[2]);
    }

    [TestMethod]
    public void Vp9PartitionProbs_DefaultProbsHelper_ReturnsExpectedSlice()
    {
        // 16x16->8x8 a/l both not split (sizeIdx=1, splitState=0)
        // = { 174, 73, 87 } per libvpx.
        var probs = Vp9PartitionProbs.DefaultProbs(1, 0);
        Equal(3, probs.Length);
        Equal((byte)174, probs[0]);
        Equal((byte)73,  probs[1]);
        Equal((byte)87,  probs[2]);
    }

    [TestMethod]
    public void Vp9PartitionProbs_DriveProbabilityIntoPartitionTree_DecodesNoneAtFirstZeroBit()
    {
        // End-to-end: pull a real probability slice and run it through
        // slice 151's Vp9PartitionTree.Decode. With first bit = 0,
        // result must be None regardless of which slice we picked.
        var probs = Vp9PartitionProbs.KeyframeProbs(0, 0);
        var bits = new int[] { 0 };
        int idx = 0;
        var p = Vp9PartitionTree.Decode(_ => bits[idx++], probs);
        Equal(Vp9PartitionType.None, p);
    }
}
