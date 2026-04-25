// Tests for Vp9MvFpTree (slice 237).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9MvFpTree_Constants_MatchLibvpx()
    {
        Equal(4, Vp9MvFpTree.FpSize);
        Equal(6, Vp9MvFpTree.Tree.Length);
    }

    [TestMethod]
    public void Vp9MvFpTree_Decode_AllZero_PicksFp0()
    {
        Equal(Vp9MvFpType.Fp0,
            Vp9MvFpTree.Decode(p => 0, new byte[] { 128, 128, 128 }));
    }

    [TestMethod]
    public void Vp9MvFpTree_Decode_OneZero_PicksFp1()
    {
        int callIdx = 0;
        int[] bits = { 1, 0 };
        Equal(Vp9MvFpType.Fp1,
            Vp9MvFpTree.Decode(p => bits[callIdx++], new byte[] { 128, 128, 128 }));
    }

    [TestMethod]
    public void Vp9MvFpTree_Decode_TwoZero_PicksFp2()
    {
        int callIdx = 0;
        int[] bits = { 1, 1, 0 };
        Equal(Vp9MvFpType.Fp2,
            Vp9MvFpTree.Decode(p => bits[callIdx++], new byte[] { 128, 128, 128 }));
    }

    [TestMethod]
    public void Vp9MvFpTree_Decode_AllOnes_PicksFp3()
    {
        int callIdx = 0;
        int[] bits = { 1, 1, 1 };
        Equal(Vp9MvFpType.Fp3,
            Vp9MvFpTree.Decode(p => bits[callIdx++], new byte[] { 128, 128, 128 }));
    }

    [TestMethod]
    public void Vp9MvFpTree_Decode_RejectsShortProbs()
    {
        Throws<ArgumentException>(() =>
            Vp9MvFpTree.Decode(p => 0, new byte[] { 128, 128 }));
    }

    [TestMethod]
    public void Vp9MvFpTree_Decode_RejectsNullReader()
    {
        Throws<ArgumentNullException>(() =>
            Vp9MvFpTree.Decode(null!, new byte[] { 128, 128, 128 }));
    }
}
