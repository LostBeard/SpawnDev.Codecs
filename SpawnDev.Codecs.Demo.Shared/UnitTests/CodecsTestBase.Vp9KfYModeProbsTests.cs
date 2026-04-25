// Tests for Vp9IntraModeProbs.KfYModeProbs (slice 157). Length, pinned
// (above, left) contexts, and the KeyframeYProbs helper across the
// full 10x10 input domain.

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9KfYModeProbs_HasCorrect900EntryLength()
    {
        Equal(900, Vp9IntraModeProbs.KfYModeProbs.Length);
    }

    [TestMethod]
    public void Vp9KfYModeProbs_FirstContextMatchesLibvpx()
    {
        // (above = DcPred, left = DcPred): { 137, 30, 42, 148, 151, 207, 70, 52, 91 }
        var t = Vp9IntraModeProbs.KfYModeProbs;
        Equal((byte)137, t[0]);
        Equal((byte)30,  t[1]);
        Equal((byte)42,  t[2]);
        Equal((byte)91,  t[8]);
    }

    [TestMethod]
    public void Vp9KfYModeProbs_LastContextMatchesLibvpx()
    {
        // (above = TmPred, left = TmPred): { 43, 81, 53, 140, 169, 204, 68, 84, 72 }
        var t = Vp9IntraModeProbs.KfYModeProbs;
        Equal((byte)43, t[891]);
        Equal((byte)81, t[892]);
        Equal((byte)72, t[899]);
    }

    [TestMethod]
    public void Vp9KfYModeProbs_MidContextMatchesLibvpx()
    {
        // (above = D45Pred, left = HPred): { 62, 30, 23, 158, 200, 207, 59, 57, 50 }
        // Index = (3 * 10 + 2) * 9 = 32 * 9 = 288
        var t = Vp9IntraModeProbs.KfYModeProbs;
        Equal((byte)62,  t[288]);
        Equal((byte)30,  t[289]);
        Equal((byte)23,  t[290]);
        Equal((byte)158, t[291]);
        Equal((byte)200, t[292]);
        Equal((byte)207, t[293]);
        Equal((byte)59,  t[294]);
        Equal((byte)57,  t[295]);
        Equal((byte)50,  t[296]);
    }

    [TestMethod]
    public void Vp9KfYModeProbs_KeyframeYProbsHelper_ReturnsExpectedSlice()
    {
        // Spot-check (above = HPred, left = DcPred) against libvpx.
        // Row 2 (above = h) starts at index 2 * 10 = 20 contexts in,
        // = 20 * 9 = 180 bytes. Sub-index 0 (left = dc).
        var probs = Vp9IntraModeProbs.KeyframeYProbs(Vp9IntraMode.HPred, Vp9IntraMode.DcPred);
        Equal(9, probs.Length);
        // Libvpx: { 82, 26, 26, 171, 208, 204, 44, 32, 105 }
        Equal((byte)82,  probs[0]);
        Equal((byte)26,  probs[1]);
        Equal((byte)171, probs[3]);
        Equal((byte)105, probs[8]);
    }

    [TestMethod]
    public void Vp9KfYModeProbs_KeyframeYProbsHelper_CoversFull10x10ContextDomain()
    {
        // Every (above, left) pair returns a valid 9-byte slice and
        // the slice content matches the manually computed flat index.
        for (int a = 0; a < 10; a++)
        for (int l = 0; l < 10; l++)
        {
            var probs = Vp9IntraModeProbs.KeyframeYProbs((Vp9IntraMode)a, (Vp9IntraMode)l);
            Equal(9, probs.Length);
            int expectedStart = (a * 10 + l) * 9;
            Equal(Vp9IntraModeProbs.KfYModeProbs[expectedStart],     probs[0]);
            Equal(Vp9IntraModeProbs.KfYModeProbs[expectedStart + 8], probs[8]);
        }
    }

    [TestMethod]
    public void Vp9KfYModeProbs_KeyframeYProbsHelper_RejectsOutOfRangeArgs()
    {
        Throws<ArgumentOutOfRangeException>(() =>
            Vp9IntraModeProbs.KeyframeYProbs((Vp9IntraMode)10, Vp9IntraMode.DcPred));
        Throws<ArgumentOutOfRangeException>(() =>
            Vp9IntraModeProbs.KeyframeYProbs(Vp9IntraMode.DcPred, (Vp9IntraMode)10));
    }

    [TestMethod]
    public void Vp9KfYModeProbs_DriveProbsIntoIntraModeTree_DecodesDcPredOnFirstZeroBit()
    {
        // End-to-end: pull a real probability slice and decode through
        // slice 153's Vp9IntraModeTree.Decode. First bit = 0 -> DcPred.
        var probs = Vp9IntraModeProbs.KeyframeYProbs(Vp9IntraMode.DcPred, Vp9IntraMode.DcPred);
        var bits = new int[] { 0 };
        int idx = 0;
        var m = Vp9IntraModeTree.Decode(_ => bits[idx++], probs);
        Equal(Vp9IntraMode.DcPred, m);
    }
}
