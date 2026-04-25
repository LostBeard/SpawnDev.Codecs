// Tests for Vp9CompoundMv (slice 279).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9CompoundMv_Zero_BothComponentsZero()
    {
        Equal(Vp9Mv.Zero, Vp9CompoundMv.Zero.Mv0);
        Equal(Vp9Mv.Zero, Vp9CompoundMv.Zero.Mv1);
        Equal(true, Vp9CompoundMv.Zero.IsZero);
    }

    [TestMethod]
    public void Vp9CompoundMv_IsZero_DetectsAnyNonzeroComponent()
    {
        Equal(false, new Vp9CompoundMv(new Vp9Mv(1, 0), Vp9Mv.Zero).IsZero);
        Equal(false, new Vp9CompoundMv(Vp9Mv.Zero, new Vp9Mv(0, 1)).IsZero);
        Equal(true, new Vp9CompoundMv(Vp9Mv.Zero, Vp9Mv.Zero).IsZero);
    }

    [TestMethod]
    public void Vp9CompoundMv_Sum_AddsComponents()
    {
        var c = new Vp9CompoundMv(new Vp9Mv(3, 5), new Vp9Mv(7, -2));
        Equal(new Vp9Mv(10, 3), c.Sum);
    }

    [TestMethod]
    public void Vp9CompoundMv_Clamp_AppliesToBoth()
    {
        var c = new Vp9CompoundMv(new Vp9Mv(50_000, 0), new Vp9Mv(-50_000, 100));
        var clamped = c.Clamp();
        Equal(16383, clamped.Mv0.Row);
        Equal(0, clamped.Mv0.Col);
        Equal(-16384, clamped.Mv1.Row);
        Equal(100, clamped.Mv1.Col);
    }

    [TestMethod]
    public void Vp9CompoundMv_RecordEquality()
    {
        var a = new Vp9CompoundMv(new Vp9Mv(1, 2), new Vp9Mv(3, 4));
        var b = new Vp9CompoundMv(new Vp9Mv(1, 2), new Vp9Mv(3, 4));
        Equal(a, b);
        Equal(a.GetHashCode(), b.GetHashCode());
    }
}
