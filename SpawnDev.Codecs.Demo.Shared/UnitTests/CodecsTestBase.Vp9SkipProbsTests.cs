// Tests for Vp9SkipProbsParser (slice 213).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9SkipProbs_Constants_MatchLibvpx()
    {
        Equal(3, Vp9SkipProbs.SkipContexts);
        Equal(3, new Vp9SkipProbs().Probs.Length);
    }

    [TestMethod]
    public void Vp9SkipProbs_Parse_NoUpdates_LeavesTableUntouched()
    {
        var data = new byte[8];
        var reader = new Vp9BoolDecoder(data, 0, data.Length);

        var probs = new Vp9SkipProbs();
        probs.Probs[0] = 50;
        probs.Probs[1] = 100;
        probs.Probs[2] = 200;

        Vp9SkipProbsParser.Read(probs, reader);

        Equal((byte)50, probs.Probs[0]);
        Equal((byte)100, probs.Probs[1]);
        Equal((byte)200, probs.Probs[2]);
    }

    [TestMethod]
    public void Vp9SkipProbs_Parse_RejectsNullArgs()
    {
        var data = new byte[8];
        var reader = new Vp9BoolDecoder(data, 0, data.Length);
        Throws<ArgumentNullException>(() =>
            Vp9SkipProbsParser.Read(null!, reader));
        Throws<ArgumentNullException>(() =>
            Vp9SkipProbsParser.Read(new Vp9SkipProbs(), null!));
    }
}
