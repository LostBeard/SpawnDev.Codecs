// Tests for Vp9MvSubPel (slice 245).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9MvSubPel_OneEighthToQ4_DoublesValue()
    {
        Equal(0, Vp9MvSubPel.OneEighthPelToQ4(0));
        Equal(2, Vp9MvSubPel.OneEighthPelToQ4(1));
        Equal(8, Vp9MvSubPel.OneEighthPelToQ4(4));   // 1/2 pel = 8/16
        Equal(16, Vp9MvSubPel.OneEighthPelToQ4(8));  // 1 full pel
        Equal(-2, Vp9MvSubPel.OneEighthPelToQ4(-1));
    }

    [TestMethod]
    public void Vp9MvSubPel_Split_ZeroIsZeroZero()
    {
        var (pel, sub) = Vp9MvSubPel.Split(0);
        Equal(0, pel);
        Equal(0, sub);
    }

    [TestMethod]
    public void Vp9MvSubPel_Split_OnePixel_PelOneSubZero()
    {
        var (pel, sub) = Vp9MvSubPel.Split(16);
        Equal(1, pel);
        Equal(0, sub);
    }

    [TestMethod]
    public void Vp9MvSubPel_Split_HalfPel_PelZeroSub8()
    {
        var (pel, sub) = Vp9MvSubPel.Split(8);
        Equal(0, pel);
        Equal(8, sub);
    }

    [TestMethod]
    public void Vp9MvSubPel_Split_NegativeOne_PelMinusOneSub15()
    {
        // Arithmetic shift: (-1) >> 4 = -1; (-1) & 15 = 15.
        var (pel, sub) = Vp9MvSubPel.Split(-1);
        Equal(-1, pel);
        Equal(15, sub);
    }

    [TestMethod]
    public void Vp9MvSubPel_Split_NegativePixel()
    {
        // -16 = -1 pel exactly.
        var (pel, sub) = Vp9MvSubPel.Split(-16);
        Equal(-1, pel);
        Equal(0, sub);
    }

    [TestMethod]
    public void Vp9MvSubPel_Combine_RoundtripsPositive()
    {
        for (int q4 = 0; q4 < 100; q4++)
        {
            var (pel, sub) = Vp9MvSubPel.Split(q4);
            Equal(q4, Vp9MvSubPel.Combine(pel, sub));
        }
    }

    [TestMethod]
    public void Vp9MvSubPel_Combine_RoundtripsNegative()
    {
        for (int q4 = -64; q4 < 0; q4++)
        {
            var (pel, sub) = Vp9MvSubPel.Split(q4);
            Equal(q4, Vp9MvSubPel.Combine(pel, sub));
        }
    }

    [TestMethod]
    public void Vp9MvSubPel_Combine_RejectsOutOfRangeSubPel()
    {
        Throws<ArgumentOutOfRangeException>(() =>
            Vp9MvSubPel.Combine(0, 16));
        Throws<ArgumentOutOfRangeException>(() =>
            Vp9MvSubPel.Combine(0, -1));
    }
}
