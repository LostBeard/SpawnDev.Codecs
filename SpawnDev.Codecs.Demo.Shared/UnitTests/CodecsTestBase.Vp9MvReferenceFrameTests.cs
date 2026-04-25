// Tests for Vp9MvReferenceFrame (slice 248).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9MvReferenceFrame_NumericValues_MatchLibvpx()
    {
        Equal(0, (int)Vp9MvReferenceFrame.Intra);
        Equal(1, (int)Vp9MvReferenceFrame.Last);
        Equal(2, (int)Vp9MvReferenceFrame.Golden);
        Equal(3, (int)Vp9MvReferenceFrame.AltRef);
        Equal(4, Vp9MvReferenceFrames.MaxRefFrames);
    }

    [TestMethod]
    public void Vp9MvReferenceFrame_IsInter_ExcludesIntra()
    {
        Equal(false, Vp9MvReferenceFrames.IsInter(Vp9MvReferenceFrame.Intra));
        Equal(true, Vp9MvReferenceFrames.IsInter(Vp9MvReferenceFrame.Last));
        Equal(true, Vp9MvReferenceFrames.IsInter(Vp9MvReferenceFrame.Golden));
        Equal(true, Vp9MvReferenceFrames.IsInter(Vp9MvReferenceFrame.AltRef));
    }

    [TestMethod]
    public void Vp9MvReferenceFrame_ToReferenceSlot_Mapping()
    {
        Equal(Vp9ReferenceSlot.Last, Vp9MvReferenceFrames.ToReferenceSlot(Vp9MvReferenceFrame.Last));
        Equal(Vp9ReferenceSlot.Golden, Vp9MvReferenceFrames.ToReferenceSlot(Vp9MvReferenceFrame.Golden));
        Equal(Vp9ReferenceSlot.AltRef, Vp9MvReferenceFrames.ToReferenceSlot(Vp9MvReferenceFrame.AltRef));
    }

    [TestMethod]
    public void Vp9MvReferenceFrame_ToReferenceSlot_RejectsIntra()
    {
        Throws<ArgumentOutOfRangeException>(() =>
            Vp9MvReferenceFrames.ToReferenceSlot(Vp9MvReferenceFrame.Intra));
    }
}
