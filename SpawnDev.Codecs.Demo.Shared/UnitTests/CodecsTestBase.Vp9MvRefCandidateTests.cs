// Tests for Vp9MvRefCandidate (slice 274).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9MvRefCandidate_None_HasIntraRef_ZeroMv()
    {
        Equal(Vp9MvReferenceFrame.Intra, Vp9MvRefCandidate.None.ReferenceFrame);
        Equal(Vp9Mv.Zero, Vp9MvRefCandidate.None.Mv);
        Equal(true, Vp9MvRefCandidate.None.IsIntra);
        Equal(false, Vp9MvRefCandidate.None.HasMotion);
    }

    [TestMethod]
    public void Vp9MvRefCandidate_IsIntra_OnlyForIntraRef()
    {
        var intra = new Vp9MvRefCandidate(Vp9Mv.Zero, Vp9MvReferenceFrame.Intra);
        var last = new Vp9MvRefCandidate(new Vp9Mv(8, -8), Vp9MvReferenceFrame.Last);
        Equal(true, intra.IsIntra);
        Equal(false, last.IsIntra);
    }

    [TestMethod]
    public void Vp9MvRefCandidate_HasMotion_FalseForZeroMv()
    {
        var c = new Vp9MvRefCandidate(Vp9Mv.Zero, Vp9MvReferenceFrame.Last);
        Equal(false, c.HasMotion);
    }

    [TestMethod]
    public void Vp9MvRefCandidate_HasMotion_TrueForNonzero()
    {
        Equal(true, new Vp9MvRefCandidate(new Vp9Mv(1, 0), Vp9MvReferenceFrame.Last).HasMotion);
        Equal(true, new Vp9MvRefCandidate(new Vp9Mv(0, 1), Vp9MvReferenceFrame.Last).HasMotion);
        Equal(true, new Vp9MvRefCandidate(new Vp9Mv(-5, 8), Vp9MvReferenceFrame.AltRef).HasMotion);
    }

    [TestMethod]
    public void Vp9MvRefCandidate_RecordEquality()
    {
        var a = new Vp9MvRefCandidate(new Vp9Mv(7, 11), Vp9MvReferenceFrame.Golden);
        var b = new Vp9MvRefCandidate(new Vp9Mv(7, 11), Vp9MvReferenceFrame.Golden);
        var c = new Vp9MvRefCandidate(new Vp9Mv(7, 11), Vp9MvReferenceFrame.Last);
        Equal(true, a == b);
        Equal(false, a == c);
        Equal(a.GetHashCode(), b.GetHashCode());
    }
}
