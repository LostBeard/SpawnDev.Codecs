// Tests for Vp9ReferenceModeProbsParser (slice 219).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9ReferenceModeProbs_Constants_MatchLibvpx()
    {
        Equal(5, Vp9ReferenceModeProbs.CompInterContexts);
        Equal(5, Vp9ReferenceModeProbs.RefContexts);
    }

    [TestMethod]
    public void Vp9ReferenceModeProbs_TableShapes()
    {
        var probs = new Vp9ReferenceModeProbs();
        Equal(5, probs.CompInterProb.Length);
        Equal(5, probs.SingleRefProb.GetLength(0));
        Equal(2, probs.SingleRefProb.GetLength(1));
        Equal(5, probs.CompRefProb.Length);
    }

    [TestMethod]
    public void Vp9ReferenceModeProbsParser_SingleReference_OnlySingleRefRead()
    {
        // SINGLE_REFERENCE: skip comp_inter (only on SELECT) and skip comp_ref.
        // Only single_ref is read.
        var data = new byte[16];
        var reader = new Vp9BoolDecoder(data, 0, data.Length);
        var probs = new Vp9ReferenceModeProbs();
        for (int i = 0; i < 5; i++)
        {
            probs.CompInterProb[i] = (byte)(50 + i);
            probs.SingleRefProb[i, 0] = (byte)(60 + i);
            probs.SingleRefProb[i, 1] = (byte)(70 + i);
            probs.CompRefProb[i] = (byte)(80 + i);
        }

        Vp9ReferenceModeProbsParser.Read(probs, Vp9ReferenceMode.SingleReference, reader);

        // No updates were signalled (zero buffer); verify nothing changed.
        for (int i = 0; i < 5; i++)
        {
            Equal((byte)(50 + i), probs.CompInterProb[i]);
            Equal((byte)(60 + i), probs.SingleRefProb[i, 0]);
            Equal((byte)(70 + i), probs.SingleRefProb[i, 1]);
            Equal((byte)(80 + i), probs.CompRefProb[i]);
        }
    }

    [TestMethod]
    public void Vp9ReferenceModeProbsParser_CompoundReference_OnlyCompRefRead()
    {
        var data = new byte[16];
        var reader = new Vp9BoolDecoder(data, 0, data.Length);
        var probs = new Vp9ReferenceModeProbs();
        for (int i = 0; i < 5; i++)
        {
            probs.CompInterProb[i] = (byte)(10 + i);
            probs.CompRefProb[i] = (byte)(20 + i);
        }

        Vp9ReferenceModeProbsParser.Read(probs, Vp9ReferenceMode.CompoundReference, reader);

        for (int i = 0; i < 5; i++)
        {
            Equal((byte)(10 + i), probs.CompInterProb[i]);
            Equal((byte)(20 + i), probs.CompRefProb[i]);
        }
    }

    [TestMethod]
    public void Vp9ReferenceModeProbsParser_ReferenceModeSelect_AllThreeRead()
    {
        // All three sub-tables read; with zero buffer, all probs unchanged.
        var data = new byte[32];
        var reader = new Vp9BoolDecoder(data, 0, data.Length);
        var probs = new Vp9ReferenceModeProbs();
        for (int i = 0; i < 5; i++)
        {
            probs.CompInterProb[i] = (byte)(100 + i);
            probs.SingleRefProb[i, 0] = (byte)(110 + i);
            probs.SingleRefProb[i, 1] = (byte)(120 + i);
            probs.CompRefProb[i] = (byte)(130 + i);
        }

        Vp9ReferenceModeProbsParser.Read(probs, Vp9ReferenceMode.ReferenceModeSelect, reader);

        for (int i = 0; i < 5; i++)
        {
            Equal((byte)(100 + i), probs.CompInterProb[i]);
            Equal((byte)(110 + i), probs.SingleRefProb[i, 0]);
            Equal((byte)(120 + i), probs.SingleRefProb[i, 1]);
            Equal((byte)(130 + i), probs.CompRefProb[i]);
        }
    }

    [TestMethod]
    public void Vp9ReferenceModeProbsParser_RejectsNullArgs()
    {
        var data = new byte[16];
        var reader = new Vp9BoolDecoder(data, 0, data.Length);
        Throws<ArgumentNullException>(() =>
            Vp9ReferenceModeProbsParser.Read(null!, Vp9ReferenceMode.SingleReference, reader));
        Throws<ArgumentNullException>(() =>
            Vp9ReferenceModeProbsParser.Read(new Vp9ReferenceModeProbs(), Vp9ReferenceMode.SingleReference, null!));
    }
}
