// Tests for Vp9MvProbsParser (slice 221).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9MvProbs_Constants_MatchLibvpx()
    {
        Equal(4, Vp9MvProbs.MvJoints);
        Equal(252, Vp9MvProbs.MvUpdateProb);
        Equal(11, Vp9MvComponentProbs.MvClasses);
        Equal(2, Vp9MvComponentProbs.Class0Size);
        Equal(10, Vp9MvComponentProbs.MvOffsetBits);
        Equal(4, Vp9MvComponentProbs.MvFpSize);
    }

    [TestMethod]
    public void Vp9MvProbs_TableShapes()
    {
        var probs = new Vp9MvProbs();
        Equal(3, probs.Joints.Length);
        Equal(2, probs.Components.Length);
        var c = probs.Components[0];
        Equal(10, c.Classes.Length);
        Equal(10, c.Bits.Length);
        Equal(2, c.Class0Fp.GetLength(0));
        Equal(3, c.Class0Fp.GetLength(1));
        Equal(3, c.Fp.Length);
    }

    [TestMethod]
    public void Vp9MvProbsParser_NoUpdates_LeavesTableUntouched_NoHp()
    {
        var data = new byte[64];
        var reader = new Vp9BoolDecoder(data, 0, data.Length);

        var probs = new Vp9MvProbs();
        for (int i = 0; i < 3; i++) probs.Joints[i] = (byte)(20 + i);
        for (int comp = 0; comp < 2; comp++)
        {
            var c = probs.Components[comp];
            c.Sign = (byte)(50 + comp);
            for (int i = 0; i < 10; i++) c.Classes[i] = (byte)(60 + comp * 10 + i);
            c.Class0 = (byte)(80 + comp);
            for (int i = 0; i < 10; i++) c.Bits[i] = (byte)(90 + comp * 10 + i);
        }

        Vp9MvProbsParser.Read(probs, allowHighPrecision: false, reader);

        for (int i = 0; i < 3; i++) Equal((byte)(20 + i), probs.Joints[i]);
        for (int comp = 0; comp < 2; comp++)
        {
            var c = probs.Components[comp];
            Equal((byte)(50 + comp), c.Sign);
            for (int i = 0; i < 10; i++) Equal((byte)(60 + comp * 10 + i), c.Classes[i]);
            Equal((byte)(80 + comp), c.Class0);
            for (int i = 0; i < 10; i++) Equal((byte)(90 + comp * 10 + i), c.Bits[i]);
        }
    }

    [TestMethod]
    public void Vp9MvProbsParser_NoUpdates_AllowHp_HpProbsAlsoUntouched()
    {
        var data = new byte[64];
        var reader = new Vp9BoolDecoder(data, 0, data.Length);

        var probs = new Vp9MvProbs();
        for (int comp = 0; comp < 2; comp++)
        {
            probs.Components[comp].Class0Hp = (byte)(150 + comp);
            probs.Components[comp].Hp = (byte)(160 + comp);
        }

        Vp9MvProbsParser.Read(probs, allowHighPrecision: true, reader);

        for (int comp = 0; comp < 2; comp++)
        {
            Equal((byte)(150 + comp), probs.Components[comp].Class0Hp);
            Equal((byte)(160 + comp), probs.Components[comp].Hp);
        }
    }

    [TestMethod]
    public void Vp9MvProbsParser_UpdateMvProb_ReturnsCurrent_WhenFlagZero()
    {
        // Zero buffer -> read flag = 0 -> return current.
        var data = new byte[8];
        var reader = new Vp9BoolDecoder(data, 0, data.Length);
        var newProb = Vp9MvProbsParser.UpdateMvProb(reader, current: 100);
        Equal((byte)100, newProb);
    }

    [TestMethod]
    public void Vp9MvProbsParser_RejectsNullArgs()
    {
        var data = new byte[8];
        var reader = new Vp9BoolDecoder(data, 0, data.Length);
        Throws<ArgumentNullException>(() =>
            Vp9MvProbsParser.Read(null!, false, reader));
        Throws<ArgumentNullException>(() =>
            Vp9MvProbsParser.Read(new Vp9MvProbs(), false, null!));
        Throws<ArgumentNullException>(() =>
            Vp9MvProbsParser.UpdateMvProb(null!, 100));
    }
}
