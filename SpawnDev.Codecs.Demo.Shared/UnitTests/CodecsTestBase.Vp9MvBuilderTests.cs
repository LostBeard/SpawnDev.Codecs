// Tests for Vp9MvBuilder (slice 254).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9MvBuilder_LowerPrecision_AllowedHp_NoChange()
    {
        Equal(7, Vp9MvBuilder.LowerMvPrecisionComponent(7, allowHighPrecision: true));
        Equal(-3, Vp9MvBuilder.LowerMvPrecisionComponent(-3, allowHighPrecision: true));
    }

    [TestMethod]
    public void Vp9MvBuilder_LowerPrecision_EvenComponent_NoChange()
    {
        Equal(8, Vp9MvBuilder.LowerMvPrecisionComponent(8, allowHighPrecision: false));
        Equal(0, Vp9MvBuilder.LowerMvPrecisionComponent(0, allowHighPrecision: false));
        Equal(-12, Vp9MvBuilder.LowerMvPrecisionComponent(-12, allowHighPrecision: false));
    }

    [TestMethod]
    public void Vp9MvBuilder_LowerPrecision_PositiveOdd_SubtractsOne()
    {
        Equal(6, Vp9MvBuilder.LowerMvPrecisionComponent(7, allowHighPrecision: false));
        Equal(0, Vp9MvBuilder.LowerMvPrecisionComponent(1, allowHighPrecision: false));
    }

    [TestMethod]
    public void Vp9MvBuilder_LowerPrecision_NegativeOdd_AddsOne()
    {
        Equal(-6, Vp9MvBuilder.LowerMvPrecisionComponent(-7, allowHighPrecision: false));
        Equal(0, Vp9MvBuilder.LowerMvPrecisionComponent(-1, allowHighPrecision: false));
    }

    [TestMethod]
    public void Vp9MvBuilder_LowerPrecision_StructForm()
    {
        // (7, -3) with !allowHp -> (6, -2). Both LSB-cleared toward zero.
        var mv = new Vp9Mv(7, -3);
        var lowered = Vp9MvBuilder.LowerMvPrecision(mv, allowHighPrecision: false);
        Equal(6, lowered.Row);
        Equal(-2, lowered.Col);
    }

    [TestMethod]
    public void Vp9MvBuilder_ApplyDiff_AddsAndClamps()
    {
        // ref = (10, 20); diff = (5, -10); allowHp = true.
        // Result: (15, 10).
        var result = Vp9MvBuilder.ApplyDiff(
            new Vp9Mv(10, 20), vertDiff: 5, horizDiff: -10, allowHighPrecision: true);
        Equal(15, result.Row);
        Equal(10, result.Col);
    }

    [TestMethod]
    public void Vp9MvBuilder_ApplyDiff_NoHp_LowersPrecision()
    {
        // ref = (4, 4); diff = (3, 5); not allow hp.
        // sum = (7, 9) -> lower (6, 8). Both odd -> rounded toward zero.
        var result = Vp9MvBuilder.ApplyDiff(
            new Vp9Mv(4, 4), vertDiff: 3, horizDiff: 5, allowHighPrecision: false);
        Equal(6, result.Row);
        Equal(8, result.Col);
    }

    [TestMethod]
    public void Vp9MvBuilder_ApplyDiff_ClampsExtreme()
    {
        // ref near upper bound + huge positive diff -> clamps to Upp - 1.
        var result = Vp9MvBuilder.ApplyDiff(
            new Vp9Mv(16000, -16000), vertDiff: 5000, horizDiff: -5000, allowHighPrecision: true);
        Equal(16383, result.Row);
        Equal(-16384, result.Col);
    }
}
