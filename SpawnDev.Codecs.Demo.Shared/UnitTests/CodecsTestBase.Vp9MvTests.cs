// Tests for Vp9Mv (slice 250).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9Mv_Constants_MatchLibvpx()
    {
        Equal(14, Vp9Mv.InUseBits);
        Equal(-16384, Vp9Mv.Low);
        Equal(16384, Vp9Mv.Upp);
    }

    [TestMethod]
    public void Vp9Mv_Zero_HasZeroComponents()
    {
        Equal(0, Vp9Mv.Zero.Row);
        Equal(0, Vp9Mv.Zero.Col);
        Equal(true, Vp9Mv.Zero.IsZero);
    }

    [TestMethod]
    public void Vp9Mv_Construction_StoresComponents()
    {
        var mv = new Vp9Mv(Row: 5, Col: -7);
        Equal(5, mv.Row);
        Equal(-7, mv.Col);
        Equal(false, mv.IsZero);
    }

    [TestMethod]
    public void Vp9Mv_Add_ComponentWise()
    {
        var sum = new Vp9Mv(1, 2) + new Vp9Mv(3, 4);
        Equal(4, sum.Row);
        Equal(6, sum.Col);
    }

    [TestMethod]
    public void Vp9Mv_Subtract_ComponentWise()
    {
        var diff = new Vp9Mv(10, 20) - new Vp9Mv(3, 7);
        Equal(7, diff.Row);
        Equal(13, diff.Col);
    }

    [TestMethod]
    public void Vp9Mv_Negate_ComponentWise()
    {
        var neg = -new Vp9Mv(5, -8);
        Equal(-5, neg.Row);
        Equal(8, neg.Col);
    }

    [TestMethod]
    public void Vp9Mv_Equality_StructEquality()
    {
        var a = new Vp9Mv(7, 11);
        var b = new Vp9Mv(7, 11);
        var c = new Vp9Mv(7, 12);
        Equal(true, a == b);
        Equal(false, a == c);
        Equal(a.GetHashCode(), b.GetHashCode());
    }

    [TestMethod]
    public void Vp9Mv_Clamp_InRange_Unchanged()
    {
        var mv = new Vp9Mv(100, -200);
        Equal(mv, mv.Clamp());
    }

    [TestMethod]
    public void Vp9Mv_Clamp_OutOfRange_Clamped()
    {
        var mv = new Vp9Mv(50_000, -50_000);
        var clamped = mv.Clamp();
        Equal(16383, clamped.Row);
        Equal(-16384, clamped.Col);
    }
}
