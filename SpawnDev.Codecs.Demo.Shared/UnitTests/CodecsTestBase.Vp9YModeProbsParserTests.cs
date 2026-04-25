// Tests for Vp9YModeProbsParser (slice 217).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9YModeProbsParser_NoUpdates_LeavesTableUntouched()
    {
        var data = new byte[16];
        var reader = new Vp9BoolDecoder(data, 0, data.Length);

        var probs = (byte[])Vp9IntraModeProbs.DefaultIfYProbs.Clone();
        var snapshot = (byte[])probs.Clone();

        Vp9YModeProbsParser.Read(probs, reader);

        for (int i = 0; i < probs.Length; i++) Equal(snapshot[i], probs[i]);
    }

    [TestMethod]
    public void Vp9YModeProbsParser_RejectsUndersizedBuffer()
    {
        var data = new byte[16];
        var reader = new Vp9BoolDecoder(data, 0, data.Length);
        Throws<ArgumentException>(() =>
            Vp9YModeProbsParser.Read(new byte[35], reader));
    }

    [TestMethod]
    public void Vp9YModeProbsParser_RejectsNullArgs()
    {
        var data = new byte[16];
        var reader = new Vp9BoolDecoder(data, 0, data.Length);
        Throws<ArgumentNullException>(() =>
            Vp9YModeProbsParser.Read(null!, reader));
        Throws<ArgumentNullException>(() =>
            Vp9YModeProbsParser.Read(new byte[36], null!));
    }
}
