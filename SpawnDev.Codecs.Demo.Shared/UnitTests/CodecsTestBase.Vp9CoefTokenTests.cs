// Tests for Vp9CoefToken (slice 145). Verifies the token enum values
// match libvpx, the tree array shape is correct, and DecodeConToken
// reaches every category leaf via deterministic bit sequences.

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9CoefToken_ValuesMatchLibvpxDefines()
    {
        // libvpx vp9_entropy.h: ZERO_TOKEN=0, ONE_TOKEN=1, ..., EOB_TOKEN=11.
        Equal((byte)0,  (byte)Vp9CoefToken.Zero);
        Equal((byte)1,  (byte)Vp9CoefToken.One);
        Equal((byte)2,  (byte)Vp9CoefToken.Two);
        Equal((byte)3,  (byte)Vp9CoefToken.Three);
        Equal((byte)4,  (byte)Vp9CoefToken.Four);
        Equal((byte)5,  (byte)Vp9CoefToken.Category1);
        Equal((byte)6,  (byte)Vp9CoefToken.Category2);
        Equal((byte)7,  (byte)Vp9CoefToken.Category3);
        Equal((byte)8,  (byte)Vp9CoefToken.Category4);
        Equal((byte)9,  (byte)Vp9CoefToken.Category5);
        Equal((byte)10, (byte)Vp9CoefToken.Category6);
        Equal((byte)11, (byte)Vp9CoefToken.Eob);
    }

    [TestMethod]
    public void Vp9CoefTrees_CoefConTree_HasCorrectShape()
    {
        // 16 entries = 8 internal nodes x 2 branches.
        Equal(16, Vp9CoefTrees.CoefConTree.Length);
        // Spot-check pinned entries against libvpx vp9_coef_con_tree.
        Equal((sbyte)2,  Vp9CoefTrees.CoefConTree[0]);   // LOW_VAL -> TWO
        Equal((sbyte)6,  Vp9CoefTrees.CoefConTree[1]);   // LOW_VAL -> HIGH_LOW
        Equal((sbyte)(-2), Vp9CoefTrees.CoefConTree[2]); // TWO leaf -> -Two
        Equal((sbyte)4,  Vp9CoefTrees.CoefConTree[3]);   // TWO -> THREE
        Equal((sbyte)(-3), Vp9CoefTrees.CoefConTree[4]); // THREE leaf -> -Three
        Equal((sbyte)(-4), Vp9CoefTrees.CoefConTree[5]); // THREE leaf -> -Four
        Equal((sbyte)(-9), Vp9CoefTrees.CoefConTree[14]);  // CAT5_token = 9
        Equal((sbyte)(-10), Vp9CoefTrees.CoefConTree[15]); // CAT6_token = 10
    }

    [TestMethod]
    public void Vp9CoefTrees_DecodeConToken_BitSequenceLeadsToTwoToken()
    {
        // From the tree: i=0 (LOW_VAL), bit 0 -> i=2 (TWO), bit 0 -> -Two.
        // So bits [0, 0] should decode TWO_TOKEN.
        var bits = new int[] { 0, 0 };
        int idx = 0;
        var tok = Vp9CoefTrees.DecodeConToken(_ => bits[idx++], stackalloc byte[8] { 1, 1, 1, 1, 1, 1, 1, 1 });
        Equal(Vp9CoefToken.Two, tok);
    }

    [TestMethod]
    public void Vp9CoefTrees_DecodeConToken_BitSequenceLeadsToThreeToken()
    {
        // i=0 -> bit 0 -> i=2 (TWO) -> bit 1 -> i=4 (THREE) -> bit 0 -> -Three.
        var bits = new int[] { 0, 1, 0 };
        int idx = 0;
        var tok = Vp9CoefTrees.DecodeConToken(_ => bits[idx++], stackalloc byte[8]);
        Equal(Vp9CoefToken.Three, tok);
    }

    [TestMethod]
    public void Vp9CoefTrees_DecodeConToken_BitSequenceLeadsToFourToken()
    {
        // i=0 -> bit 0 -> i=2 -> bit 1 -> i=4 -> bit 1 -> -Four.
        var bits = new int[] { 0, 1, 1 };
        int idx = 0;
        var tok = Vp9CoefTrees.DecodeConToken(_ => bits[idx++], stackalloc byte[8]);
        Equal(Vp9CoefToken.Four, tok);
    }

    [TestMethod]
    public void Vp9CoefTrees_DecodeConToken_BitSequenceLeadsToCategory1()
    {
        // i=0 -> bit 1 -> i=6 (HIGH_LOW) -> bit 0 -> i=8 (CAT_ONE) -> bit 0 -> -Cat1.
        var bits = new int[] { 1, 0, 0 };
        int idx = 0;
        var tok = Vp9CoefTrees.DecodeConToken(_ => bits[idx++], stackalloc byte[8]);
        Equal(Vp9CoefToken.Category1, tok);
    }

    [TestMethod]
    public void Vp9CoefTrees_DecodeConToken_BitSequenceLeadsToCategory6()
    {
        // i=0 -> 1 -> i=6 -> 1 -> i=10 (CAT_THREEFOUR) -> 1 -> i=14 (CAT_FIVE) -> 1 -> -Cat6.
        var bits = new int[] { 1, 1, 1, 1 };
        int idx = 0;
        var tok = Vp9CoefTrees.DecodeConToken(_ => bits[idx++], stackalloc byte[8]);
        Equal(Vp9CoefToken.Category6, tok);
    }

    [TestMethod]
    public void Vp9CoefTrees_DecodeConToken_RejectsUndersizedProbs()
    {
        Throws<ArgumentException>(() => Vp9CoefTrees.DecodeConToken(_ => 0, new byte[7]));
    }
}
