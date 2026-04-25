// Tests for Vp9MvClassTree (slice 236).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9MvClassTree_Constants_MatchLibvpx()
    {
        Equal(11, Vp9MvClassTree.Classes);
        Equal(20, Vp9MvClassTree.Tree.Length);
    }

    [TestMethod]
    public void Vp9MvClassTree_Decode_AllZero_PicksClass0()
    {
        // First bit 0 -> ROOT.left = -Class0.
        Equal(Vp9MvClassType.Class0,
            Vp9MvClassTree.Decode(p => 0, BuildProbs()));
    }

    [TestMethod]
    public void Vp9MvClassTree_Decode_BitsOneZero_PicksClass1()
    {
        // 1 -> i=2; 0 -> i2.left = -Class1.
        int callIdx = 0;
        int[] bits = { 1, 0 };
        Equal(Vp9MvClassType.Class1,
            Vp9MvClassTree.Decode(p => bits[callIdx++], BuildProbs()));
    }

    [TestMethod]
    public void Vp9MvClassTree_Decode_BitsOnesThenZeros_PicksClass2()
    {
        // 1, 1, 0, 0 -> ROOT.right=2 -> i2.right=4 -> i4.left=6 -> i6.left=-Class2.
        int callIdx = 0;
        int[] bits = { 1, 1, 0, 0 };
        Equal(Vp9MvClassType.Class2,
            Vp9MvClassTree.Decode(p => bits[callIdx++], BuildProbs()));
    }

    [TestMethod]
    public void Vp9MvClassTree_Decode_AllOnes_PicksClass10()
    {
        // ROOT.right=2 -> i2.right=4 -> i4.right=8 -> i8.right=12
        // -> i12.right=14 -> i14.right=18 -> i18.right=-Class10.
        // 7 ones to reach the deepest leaf.
        int callIdx = 0;
        int[] bits = { 1, 1, 1, 1, 1, 1, 1 };
        Equal(Vp9MvClassType.Class10,
            Vp9MvClassTree.Decode(p => bits[callIdx++], BuildProbs()));
    }

    [TestMethod]
    public void Vp9MvClassTree_Decode_PicksClass6()
    {
        // 1, 1, 1, 1, 0 -> i12.left = -Class6.
        int callIdx = 0;
        int[] bits = { 1, 1, 1, 1, 0 };
        Equal(Vp9MvClassType.Class6,
            Vp9MvClassTree.Decode(p => bits[callIdx++], BuildProbs()));
    }

    [TestMethod]
    public void Vp9MvClassTree_Decode_RejectsShortProbs()
    {
        Throws<ArgumentException>(() =>
            Vp9MvClassTree.Decode(p => 0, new byte[] { 128 }));
    }

    [TestMethod]
    public void Vp9MvClassTree_Decode_RejectsNullReader()
    {
        Throws<ArgumentNullException>(() =>
            Vp9MvClassTree.Decode(null!, BuildProbs()));
    }

    private static byte[] BuildProbs()
    {
        var p = new byte[Vp9MvClassTree.Classes - 1];
        for (int i = 0; i < p.Length; i++) p[i] = 128;
        return p;
    }
}
