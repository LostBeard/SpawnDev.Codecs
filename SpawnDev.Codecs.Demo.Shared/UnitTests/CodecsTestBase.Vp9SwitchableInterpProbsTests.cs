// Tests for Vp9SwitchableInterpProbsParser (slice 218).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9SwitchableInterpProbs_Constants_MatchLibvpx()
    {
        Equal(4, Vp9SwitchableInterpProbs.SwitchableFilterContexts);
        Equal(3, Vp9SwitchableInterpProbs.SwitchableFilters);
    }

    [TestMethod]
    public void Vp9SwitchableInterpProbs_TableShape()
    {
        var t = new Vp9SwitchableInterpProbs();
        Equal(4, t.Probs.GetLength(0));
        Equal(2, t.Probs.GetLength(1));
    }

    [TestMethod]
    public void Vp9SwitchableInterpProbsParser_NoUpdates_LeavesTableUntouched()
    {
        var data = new byte[8];
        var reader = new Vp9BoolDecoder(data, 0, data.Length);

        var probs = new Vp9SwitchableInterpProbs();
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 2; j++)
                probs.Probs[i, j] = (byte)(80 + i * 5 + j);

        Vp9SwitchableInterpProbsParser.Read(probs, reader);

        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 2; j++)
                Equal((byte)(80 + i * 5 + j), probs.Probs[i, j]);
    }

    [TestMethod]
    public void Vp9SwitchableInterpProbsParser_RejectsNullArgs()
    {
        var data = new byte[8];
        var reader = new Vp9BoolDecoder(data, 0, data.Length);
        Throws<ArgumentNullException>(() =>
            Vp9SwitchableInterpProbsParser.Read(null!, reader));
        Throws<ArgumentNullException>(() =>
            Vp9SwitchableInterpProbsParser.Read(new Vp9SwitchableInterpProbs(), null!));
    }
}
