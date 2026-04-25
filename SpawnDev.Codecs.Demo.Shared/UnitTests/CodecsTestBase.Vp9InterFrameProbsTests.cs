// Tests for Vp9InterFrameProbs (slice 159).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9InterFrameProbs_DefaultInterModeProbs_HasCorrectShape()
    {
        Equal(21, Vp9InterFrameProbs.DefaultInterModeProbs.Length); // 7 * 3
        Equal(7, Vp9InterFrameProbs.InterModeContexts);
    }

    [TestMethod]
    public void Vp9InterFrameProbs_DefaultInterModeProbs_FirstAndLastContextsMatchLibvpx()
    {
        var t = Vp9InterFrameProbs.DefaultInterModeProbs;
        // ctx 0: { 2, 173, 34 }
        Equal((byte)2,   t[0]);
        Equal((byte)173, t[1]);
        Equal((byte)34,  t[2]);
        // ctx 6: { 25, 29, 30 }
        Equal((byte)25, t[18]);
        Equal((byte)29, t[19]);
        Equal((byte)30, t[20]);
    }

    [TestMethod]
    public void Vp9InterFrameProbs_DefaultSkipProbs_MatchesLibvpx()
    {
        Equal(3, Vp9InterFrameProbs.DefaultSkipProbs.Length);
        Equal((byte)192, Vp9InterFrameProbs.DefaultSkipProbs[0]);
        Equal((byte)128, Vp9InterFrameProbs.DefaultSkipProbs[1]);
        Equal((byte)64,  Vp9InterFrameProbs.DefaultSkipProbs[2]);
    }

    [TestMethod]
    public void Vp9InterFrameProbs_DefaultIntraInterProb_MatchesLibvpx()
    {
        Equal(4, Vp9InterFrameProbs.DefaultIntraInterProb.Length);
        Equal((byte)9,   Vp9InterFrameProbs.DefaultIntraInterProb[0]);
        Equal((byte)102, Vp9InterFrameProbs.DefaultIntraInterProb[1]);
        Equal((byte)187, Vp9InterFrameProbs.DefaultIntraInterProb[2]);
        Equal((byte)225, Vp9InterFrameProbs.DefaultIntraInterProb[3]);
    }

    [TestMethod]
    public void Vp9InterFrameProbs_InterModeProbsHelper_ReturnsExpectedSlices()
    {
        // Each context returns a 3-byte slice; spot-check at ctx 3
        // ({ 7, 94, 66 }) and ctx 4 ({ 8, 64, 46 }).
        var c3 = Vp9InterFrameProbs.InterModeProbs(3);
        Equal(3, c3.Length);
        Equal((byte)7,  c3[0]);
        Equal((byte)94, c3[1]);
        Equal((byte)66, c3[2]);

        var c4 = Vp9InterFrameProbs.InterModeProbs(4);
        Equal((byte)8,  c4[0]);
        Equal((byte)64, c4[1]);
        Equal((byte)46, c4[2]);
    }

    [TestMethod]
    public void Vp9InterFrameProbs_InterModeProbsHelper_RejectsOutOfRange()
    {
        Throws<ArgumentOutOfRangeException>(() =>
            Vp9InterFrameProbs.InterModeProbs(7));
        Throws<ArgumentOutOfRangeException>(() =>
            Vp9InterFrameProbs.InterModeProbs(-1));
    }

    [TestMethod]
    public void Vp9InterFrameProbs_DriveInterModeProbsIntoTree_DecodesZeroMvOnFirstZeroBit()
    {
        // End-to-end: pull a real probability slice and decode through
        // slice 158's Vp9InterModeTree.Decode. First bit = 0 -> ZeroMv.
        var probs = Vp9InterFrameProbs.InterModeProbs(0);
        var bits = new int[] { 0 };
        int idx = 0;
        var m = Vp9InterModeTree.Decode(_ => bits[idx++], probs);
        Equal(Vp9InterMode.ZeroMv, m);
    }
}
