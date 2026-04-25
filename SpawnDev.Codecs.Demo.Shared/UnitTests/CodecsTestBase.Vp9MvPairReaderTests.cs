// Tests for Vp9MvPairReader (slice 239).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9MvPairReader_JointHasVertical_HzvnzAndHnzvnzOnly()
    {
        Equal(false, Vp9MvPairReader.JointHasVertical(Vp9MvJointType.Zero));
        Equal(false, Vp9MvPairReader.JointHasVertical(Vp9MvJointType.Hnzvz));
        Equal(true, Vp9MvPairReader.JointHasVertical(Vp9MvJointType.Hzvnz));
        Equal(true, Vp9MvPairReader.JointHasVertical(Vp9MvJointType.Hnzvnz));
    }

    [TestMethod]
    public void Vp9MvPairReader_JointHasHorizontal_HnzvzAndHnzvnzOnly()
    {
        Equal(false, Vp9MvPairReader.JointHasHorizontal(Vp9MvJointType.Zero));
        Equal(true, Vp9MvPairReader.JointHasHorizontal(Vp9MvJointType.Hnzvz));
        Equal(false, Vp9MvPairReader.JointHasHorizontal(Vp9MvJointType.Hzvnz));
        Equal(true, Vp9MvPairReader.JointHasHorizontal(Vp9MvJointType.Hnzvnz));
    }

    [TestMethod]
    public void Vp9MvPairReader_ZeroJoint_ReturnsZeroPair()
    {
        // Joint tree: bit 0 = 0 -> Zero leaf. No component reads needed.
        var probs = NewMvProbsForPair();
        var (v, h) = Vp9MvPairReader.ReadDiff(p => 0, probs, useHp: false);
        Equal(0, v);
        Equal(0, h);
    }

    [TestMethod]
    public void Vp9MvPairReader_HnzvzJoint_ReadsOnlyHorizontal()
    {
        // 1, 0 -> Hnzvz (horizontal only).
        // Then horizontal component reads sign + class tree (zeros to Class0)
        // + d=0 + fp tree zeros + hp implicit 1 -> mag = 2.
        var probs = NewMvProbsForPair();
        int callIdx = 0;
        int[] bits = new int[20];
        bits[0] = 1; // joint bit 0 = 1
        bits[1] = 0; // joint bit 1 = 0 -> Hnzvz
        // remaining bits all 0
        var (v, h) = Vp9MvPairReader.ReadDiff(p => bits[callIdx++], probs, useHp: false);
        Equal(0, v);
        Equal(2, h);
    }

    [TestMethod]
    public void Vp9MvPairReader_HzvnzJoint_ReadsOnlyVertical()
    {
        // 1, 1, 0 -> Hzvnz (vertical only).
        var probs = NewMvProbsForPair();
        int callIdx = 0;
        int[] bits = new int[20];
        bits[0] = 1; // joint bit 0 = 1
        bits[1] = 1; // joint bit 1 = 1
        bits[2] = 0; // joint bit 2 = 0 -> Hzvnz
        // remaining bits all 0
        var (v, h) = Vp9MvPairReader.ReadDiff(p => bits[callIdx++], probs, useHp: false);
        Equal(2, v);
        Equal(0, h);
    }

    [TestMethod]
    public void Vp9MvPairReader_HnzvnzJoint_ReadsBothInOrder()
    {
        // 1, 1, 1 -> Hnzvnz.
        // Then vertical component (Class0 with d=0, fp=Fp0, no HP) -> 2.
        // Then horizontal component (same path) -> 2.
        // Joint reads 3 bits, each component reads sign+class(1)+d(1)+fp(3) = 6 bits without HP.
        // Total: 3 + 6 + 6 = 15 bits.
        var probs = NewMvProbsForPair();
        int callIdx = 0;
        int[] bits = new int[20];
        bits[0] = 1; bits[1] = 1; bits[2] = 1;  // joint = Hnzvnz
        var (v, h) = Vp9MvPairReader.ReadDiff(p => bits[callIdx++], probs, useHp: false);
        Equal(2, v);
        Equal(2, h);
    }

    [TestMethod]
    public void Vp9MvPairReader_RejectsNullReader()
    {
        var probs = NewMvProbsForPair();
        Throws<ArgumentNullException>(() =>
            Vp9MvPairReader.ReadDiff((Vp9BoolDecoder)null!, probs, false));
    }

    [TestMethod]
    public void Vp9MvPairReader_RejectsNullProbs()
    {
        Throws<ArgumentNullException>(() =>
            Vp9MvPairReader.ReadDiff(p => 0, null!, false));
    }

    private static Vp9MvProbs NewMvProbsForPair()
    {
        var p = new Vp9MvProbs();
        for (int i = 0; i < p.Joints.Length; i++) p.Joints[i] = 128;
        for (int c = 0; c < 2; c++)
        {
            var comp = p.Components[c];
            comp.Sign = 128;
            for (int i = 0; i < comp.Classes.Length; i++) comp.Classes[i] = 128;
            comp.Class0 = 128;
            for (int i = 0; i < comp.Bits.Length; i++) comp.Bits[i] = 128;
            for (int i = 0; i < comp.Class0Fp.GetLength(0); i++)
                for (int j = 0; j < comp.Class0Fp.GetLength(1); j++)
                    comp.Class0Fp[i, j] = 128;
            for (int i = 0; i < comp.Fp.Length; i++) comp.Fp[i] = 128;
            comp.Class0Hp = 128;
            comp.Hp = 128;
        }
        return p;
    }
}
