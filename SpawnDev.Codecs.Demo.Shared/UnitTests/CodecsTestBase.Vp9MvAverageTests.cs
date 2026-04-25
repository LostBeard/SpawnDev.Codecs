// Tests for Vp9MvAverage (slice 264).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9MvAverage_Average2_RoundsHalfAwayFromZero()
    {
        // Average of (1, 0) and (1, 0): sum = 2, result = (2 + 1) / 2 = 1.
        Equal(new Vp9Mv(1, 0), Vp9MvAverage.Average2(new Vp9Mv(1, 0), new Vp9Mv(1, 0)));

        // Average of (1, 0) and (2, 0): sum = 3, result = (3 + 1) / 2 = 2.
        Equal(new Vp9Mv(2, 0), Vp9MvAverage.Average2(new Vp9Mv(1, 0), new Vp9Mv(2, 0)));

        // Negative: average of (-1, 0) and (-2, 0): sum = -3, result = (-3 - 1) / 2 = -2.
        Equal(new Vp9Mv(-2, 0), Vp9MvAverage.Average2(new Vp9Mv(-1, 0), new Vp9Mv(-2, 0)));
    }

    [TestMethod]
    public void Vp9MvAverage_Average2_BothComponents()
    {
        // (4, -8) and (6, -10) -> ((4+6+1)/2, (-18-1)/2) = (5, -9).
        Equal(new Vp9Mv(5, -9), Vp9MvAverage.Average2(new Vp9Mv(4, -8), new Vp9Mv(6, -10)));
    }

    [TestMethod]
    public void Vp9MvAverage_Average4_PositiveSum()
    {
        // 4 components sum to 7: (7 + 2) / 4 = 9 / 4 = 2.
        Equal(2, Vp9MvAverage.RoundComp4(7));
        // sum 8: (8 + 2) / 4 = 10/4 = 2 (truncation).
        Equal(2, Vp9MvAverage.RoundComp4(8));
        // sum 9: (9 + 2) / 4 = 11/4 = 2.
        Equal(2, Vp9MvAverage.RoundComp4(9));
        // sum 10: (10 + 2) / 4 = 12/4 = 3.
        Equal(3, Vp9MvAverage.RoundComp4(10));
    }

    [TestMethod]
    public void Vp9MvAverage_Average4_NegativeSum()
    {
        // sum -10: (-10 - 2) / 4 = -12/4 = -3.
        Equal(-3, Vp9MvAverage.RoundComp4(-10));
        // sum -9: (-9 - 2) / 4 = -11/4 = -2 (truncate toward zero).
        Equal(-2, Vp9MvAverage.RoundComp4(-9));
        // sum 0: 0+2/4 = 0.
        Equal(0, Vp9MvAverage.RoundComp4(0));
    }

    [TestMethod]
    public void Vp9MvAverage_Average4_FullStruct()
    {
        // 4 MVs: (1, -2), (2, -3), (3, -4), (4, -5).
        // Sum: (10, -14).
        // Row: (10 + 2) / 4 = 12 / 4 = 3.
        // Col: (-14 - 2) / 4 = -16 / 4 = -4.
        var avg = Vp9MvAverage.Average4(
            new Vp9Mv(1, -2), new Vp9Mv(2, -3),
            new Vp9Mv(3, -4), new Vp9Mv(4, -5));
        Equal(new Vp9Mv(3, -4), avg);
    }

    [TestMethod]
    public void Vp9MvAverage_Average4_AllZeros_IsZero()
    {
        Equal(Vp9Mv.Zero, Vp9MvAverage.Average4(
            Vp9Mv.Zero, Vp9Mv.Zero, Vp9Mv.Zero, Vp9Mv.Zero));
    }

    [TestMethod]
    public void Vp9MvAverage_Average4_AllSame_PreservesValue()
    {
        var v = new Vp9Mv(7, -11);
        Equal(v, Vp9MvAverage.Average4(v, v, v, v));
    }
}
