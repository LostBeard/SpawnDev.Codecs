// Tests for Vp9IntraInterProbsParser (slice 215).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9IntraInterProbs_Constants_MatchLibvpx()
    {
        Equal(4, Vp9IntraInterProbs.IntraInterContexts);
        Equal(4, new Vp9IntraInterProbs().Probs.Length);
    }

    [TestMethod]
    public void Vp9IntraInterProbs_Parse_NoUpdates_LeavesTableUntouched()
    {
        var data = new byte[8];
        var reader = new Vp9BoolDecoder(data, 0, data.Length);

        var probs = new Vp9IntraInterProbs();
        probs.Probs[0] = 9;
        probs.Probs[1] = 102;
        probs.Probs[2] = 187;
        probs.Probs[3] = 225;

        Vp9IntraInterProbsParser.Read(probs, reader);

        Equal((byte)9, probs.Probs[0]);
        Equal((byte)102, probs.Probs[1]);
        Equal((byte)187, probs.Probs[2]);
        Equal((byte)225, probs.Probs[3]);
    }

    [TestMethod]
    public void Vp9IntraInterProbs_Parse_RejectsNullArgs()
    {
        var data = new byte[8];
        var reader = new Vp9BoolDecoder(data, 0, data.Length);
        Throws<ArgumentNullException>(() =>
            Vp9IntraInterProbsParser.Read(null!, reader));
        Throws<ArgumentNullException>(() =>
            Vp9IntraInterProbsParser.Read(new Vp9IntraInterProbs(), null!));
    }
}
