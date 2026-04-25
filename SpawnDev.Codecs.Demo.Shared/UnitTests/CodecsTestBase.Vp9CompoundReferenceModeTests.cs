// Tests for Vp9CompoundReferenceMode (slice 263).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9CompoundReferenceMode_LastGoldenAgree_AltRefIsFixed()
    {
        // Last and Golden both biased = false; AltRef = true.
        var mode = Vp9CompoundReferenceMode.Compute(
            lastBias: false, goldenBias: false, altRefBias: true);
        Equal(Vp9MvReferenceFrame.AltRef, mode.FixedRef);
        Equal(Vp9MvReferenceFrame.Last, mode.VarRef0);
        Equal(Vp9MvReferenceFrame.Golden, mode.VarRef1);
    }

    [TestMethod]
    public void Vp9CompoundReferenceMode_LastGoldenAgree_BothBiased_AltRefIsFixed()
    {
        var mode = Vp9CompoundReferenceMode.Compute(
            lastBias: true, goldenBias: true, altRefBias: false);
        Equal(Vp9MvReferenceFrame.AltRef, mode.FixedRef);
        Equal(Vp9MvReferenceFrame.Last, mode.VarRef0);
        Equal(Vp9MvReferenceFrame.Golden, mode.VarRef1);
    }

    [TestMethod]
    public void Vp9CompoundReferenceMode_LastAltRefAgree_GoldenIsFixed()
    {
        var mode = Vp9CompoundReferenceMode.Compute(
            lastBias: false, goldenBias: true, altRefBias: false);
        Equal(Vp9MvReferenceFrame.Golden, mode.FixedRef);
        Equal(Vp9MvReferenceFrame.Last, mode.VarRef0);
        Equal(Vp9MvReferenceFrame.AltRef, mode.VarRef1);
    }

    [TestMethod]
    public void Vp9CompoundReferenceMode_GoldenAltRefAgree_LastIsFixed()
    {
        var mode = Vp9CompoundReferenceMode.Compute(
            lastBias: true, goldenBias: false, altRefBias: false);
        Equal(Vp9MvReferenceFrame.Last, mode.FixedRef);
        Equal(Vp9MvReferenceFrame.Golden, mode.VarRef0);
        Equal(Vp9MvReferenceFrame.AltRef, mode.VarRef1);
    }

    [TestMethod]
    public void Vp9CompoundReferenceMode_AllSame_FallsThroughToLastGoldenAgree()
    {
        // All three biased the same: lastBias == goldenBias is true,
        // so we hit the first branch -> AltRef fixed.
        var allFalse = Vp9CompoundReferenceMode.Compute(false, false, false);
        Equal(Vp9MvReferenceFrame.AltRef, allFalse.FixedRef);

        var allTrue = Vp9CompoundReferenceMode.Compute(true, true, true);
        Equal(Vp9MvReferenceFrame.AltRef, allTrue.FixedRef);
    }

    [TestMethod]
    public void Vp9CompoundReferenceMode_RecordEquality()
    {
        var a = Vp9CompoundReferenceMode.Compute(false, false, true);
        var b = Vp9CompoundReferenceMode.Compute(false, false, true);
        Equal(a, b);
        Equal(a.GetHashCode(), b.GetHashCode());
    }
}
