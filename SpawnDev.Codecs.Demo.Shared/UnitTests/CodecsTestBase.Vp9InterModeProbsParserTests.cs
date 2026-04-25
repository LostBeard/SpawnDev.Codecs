// Tests for Vp9InterModeProbsParser (slice 214).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9InterModeProbsTable_Constants_MatchLibvpx()
    {
        Equal(7, Vp9InterModeProbsTable.InterModeContexts);
        Equal(4, Vp9InterModeProbsTable.InterModes);
    }

    [TestMethod]
    public void Vp9InterModeProbsTable_TableShape()
    {
        var t = new Vp9InterModeProbsTable();
        Equal(7, t.Probs.GetLength(0));
        Equal(3, t.Probs.GetLength(1));
    }

    [TestMethod]
    public void Vp9InterModeProbsParser_Parse_NoUpdates_LeavesTableUntouched()
    {
        var data = new byte[16];
        var reader = new Vp9BoolDecoder(data, 0, data.Length);

        var probs = new Vp9InterModeProbsTable();
        for (int i = 0; i < 7; i++)
            for (int j = 0; j < 3; j++)
                probs.Probs[i, j] = (byte)(50 + i * 3 + j);

        Vp9InterModeProbsParser.Read(probs, reader);

        for (int i = 0; i < 7; i++)
            for (int j = 0; j < 3; j++)
                Equal((byte)(50 + i * 3 + j), probs.Probs[i, j]);
    }

    [TestMethod]
    public void Vp9InterModeProbsParser_Parse_RejectsNullArgs()
    {
        var data = new byte[16];
        var reader = new Vp9BoolDecoder(data, 0, data.Length);
        Throws<ArgumentNullException>(() =>
            Vp9InterModeProbsParser.Read(null!, reader));
        Throws<ArgumentNullException>(() =>
            Vp9InterModeProbsParser.Read(new Vp9InterModeProbsTable(), null!));
    }
}
