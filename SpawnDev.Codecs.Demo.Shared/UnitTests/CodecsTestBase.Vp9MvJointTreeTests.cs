// Tests for Vp9MvJointTree (slice 235).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9MvJointTree_Constants_MatchLibvpx()
    {
        Equal(4, Vp9MvJointTree.Joints);
        Equal(6, Vp9MvJointTree.Tree.Length);
    }

    [TestMethod]
    public void Vp9MvJointTree_Tree_Layout()
    {
        // Internal nodes:
        //   ROOT  : -Zero, 2
        //   i=2   : -Hnzvz, 4
        //   i=4   : -Hzvnz, -Hnzvnz
        Equal(0, (int)Vp9MvJointTree.Tree[0]);  // -Zero = 0
        Equal(2, (int)Vp9MvJointTree.Tree[1]);
        Equal(-1, (int)Vp9MvJointTree.Tree[2]); // -Hnzvz
        Equal(4, (int)Vp9MvJointTree.Tree[3]);
        Equal(-2, (int)Vp9MvJointTree.Tree[4]); // -Hzvnz
        Equal(-3, (int)Vp9MvJointTree.Tree[5]); // -Hnzvnz
    }

    [TestMethod]
    public void Vp9MvJointTree_Decode_AllZero_PicksZeroJoint()
    {
        // readBit returns 0 -> picks ROOT.left = -Zero = leaf 0.
        Equal(Vp9MvJointType.Zero,
            Vp9MvJointTree.Decode(p => 0, new byte[] { 128, 128, 128 }));
    }

    [TestMethod]
    public void Vp9MvJointTree_Decode_OneZeroOne_PicksHnzvz()
    {
        // bit 0 = 1 -> ROOT.right = 2.
        // bit 1 = 0 -> i2.left = -Hnzvz = leaf 1.
        int callIdx = 0;
        int[] bits = { 1, 0 };
        Equal(Vp9MvJointType.Hnzvz,
            Vp9MvJointTree.Decode(p => bits[callIdx++], new byte[] { 128, 128, 128 }));
    }

    [TestMethod]
    public void Vp9MvJointTree_Decode_BitsTwo_PicksHzvnz()
    {
        // 1 -> ROOT.right = 2; 1 -> i2.right = 4; 0 -> i4.left = -Hzvnz.
        int callIdx = 0;
        int[] bits = { 1, 1, 0 };
        Equal(Vp9MvJointType.Hzvnz,
            Vp9MvJointTree.Decode(p => bits[callIdx++], new byte[] { 128, 128, 128 }));
    }

    [TestMethod]
    public void Vp9MvJointTree_Decode_BitsThree_PicksHnzvnz()
    {
        // 1, 1, 1 -> all rights -> -Hnzvnz.
        int callIdx = 0;
        int[] bits = { 1, 1, 1 };
        Equal(Vp9MvJointType.Hnzvnz,
            Vp9MvJointTree.Decode(p => bits[callIdx++], new byte[] { 128, 128, 128 }));
    }

    [TestMethod]
    public void Vp9MvJointTree_Decode_RejectsShortProbs()
    {
        Throws<ArgumentException>(() =>
            Vp9MvJointTree.Decode(p => 0, new byte[] { 128, 128 }));
    }

    [TestMethod]
    public void Vp9MvJointTree_Decode_RejectsNullReader()
    {
        Throws<ArgumentNullException>(() =>
            Vp9MvJointTree.Decode(null!, new byte[] { 128, 128, 128 }));
    }

    [TestMethod]
    public void Vp9MvJointTree_Decode_UsesIndexedProbs()
    {
        // ROOT uses probs[0]; i2 uses probs[1]; i4 uses probs[2].
        // Verify by capturing prob arg per call.
        int callIdx = 0;
        byte[] capturedProbs = new byte[3];
        int[] bits = { 1, 1, 1 };
        Vp9MvJointTree.Decode(p =>
        {
            capturedProbs[callIdx] = p;
            return bits[callIdx++];
        }, new byte[] { 10, 20, 30 });
        Equal((byte)10, capturedProbs[0]);
        Equal((byte)20, capturedProbs[1]);
        Equal((byte)30, capturedProbs[2]);
    }
}
