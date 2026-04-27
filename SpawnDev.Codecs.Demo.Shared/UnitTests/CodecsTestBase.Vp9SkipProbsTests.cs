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
    public void Vp9SkipProbs_DefaultProbs_MatchLibvpx()
    {
        // libvpx vp9_entropymode.c default_skip_probs[3] = { 192, 128, 64 }.
        // The compressed header applies diff_update_prob deltas FROM these
        // defaults; zero-init would corrupt every downstream skip read.
        Equal(3, Vp9SkipProbs.DefaultProbs.Length);
        Equal((byte)192, Vp9SkipProbs.DefaultProbs[0]);
        Equal((byte)128, Vp9SkipProbs.DefaultProbs[1]);
        Equal((byte)64,  Vp9SkipProbs.DefaultProbs[2]);

        // New instance is seeded from DefaultProbs.
        var probs = new Vp9SkipProbs();
        Equal((byte)192, probs.Probs[0]);
        Equal((byte)128, probs.Probs[1]);
        Equal((byte)64,  probs.Probs[2]);
    }

    [TestMethod]
    public void Vp9SkipProbs_DefaultProbs_IsDefensivelyCloned()
    {
        // Mutating one instance must not poison the static defaults or
        // other instances - the constructor uses Clone(), this pins it.
        var a = new Vp9SkipProbs();
        a.Probs[0] = 99;
        var b = new Vp9SkipProbs();
        Equal((byte)192, b.Probs[0]);
        Equal((byte)192, Vp9SkipProbs.DefaultProbs[0]);
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
