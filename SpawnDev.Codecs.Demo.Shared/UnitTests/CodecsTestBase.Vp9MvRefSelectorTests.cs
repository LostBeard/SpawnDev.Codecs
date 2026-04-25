// Tests for Vp9MvRefSelector (slice 281).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9MvRefSelector_Nearest_EmptyList_IsZero()
    {
        Equal(Vp9Mv.Zero, Vp9MvRefSelector.Nearest(new Vp9MvCandidatesList()));
    }

    [TestMethod]
    public void Vp9MvRefSelector_Nearest_OneElement_ReturnsIt()
    {
        var list = new Vp9MvCandidatesList();
        list.TryAdd(new Vp9Mv(7, -3));
        Equal(new Vp9Mv(7, -3), Vp9MvRefSelector.Nearest(list));
    }

    [TestMethod]
    public void Vp9MvRefSelector_Near_LessThanTwoElements_IsZero()
    {
        var list = new Vp9MvCandidatesList();
        Equal(Vp9Mv.Zero, Vp9MvRefSelector.Near(list));
        list.TryAdd(new Vp9Mv(1, 2));
        Equal(Vp9Mv.Zero, Vp9MvRefSelector.Near(list));
    }

    [TestMethod]
    public void Vp9MvRefSelector_Near_TwoElements_ReturnsSecond()
    {
        var list = new Vp9MvCandidatesList();
        list.TryAdd(new Vp9Mv(1, 2));
        list.TryAdd(new Vp9Mv(3, 4));
        Equal(new Vp9Mv(3, 4), Vp9MvRefSelector.Near(list));
    }

    [TestMethod]
    public void Vp9MvRefSelector_ForInterMode_NearestMv_PicksFirst()
    {
        var list = new Vp9MvCandidatesList();
        list.TryAdd(new Vp9Mv(11, 22));
        list.TryAdd(new Vp9Mv(33, 44));
        Equal(new Vp9Mv(11, 22),
            Vp9MvRefSelector.ForInterMode(list, Vp9InterMode.NearestMv));
    }

    [TestMethod]
    public void Vp9MvRefSelector_ForInterMode_NearMv_PicksSecond()
    {
        var list = new Vp9MvCandidatesList();
        list.TryAdd(new Vp9Mv(11, 22));
        list.TryAdd(new Vp9Mv(33, 44));
        Equal(new Vp9Mv(33, 44),
            Vp9MvRefSelector.ForInterMode(list, Vp9InterMode.NearMv));
    }

    [TestMethod]
    public void Vp9MvRefSelector_ForInterMode_ZeroMv_AlwaysZero()
    {
        var list = new Vp9MvCandidatesList();
        list.TryAdd(new Vp9Mv(11, 22));
        list.TryAdd(new Vp9Mv(33, 44));
        Equal(Vp9Mv.Zero, Vp9MvRefSelector.ForInterMode(list, Vp9InterMode.ZeroMv));
    }

    [TestMethod]
    public void Vp9MvRefSelector_ForInterMode_NewMv_UsesNearest()
    {
        // NewMv reference for the diff is the nearest candidate.
        var list = new Vp9MvCandidatesList();
        list.TryAdd(new Vp9Mv(99, 100));
        Equal(new Vp9Mv(99, 100),
            Vp9MvRefSelector.ForInterMode(list, Vp9InterMode.NewMv));
    }

    [TestMethod]
    public void Vp9MvRefSelector_ForInterMode_RejectsInvalidMode()
    {
        var list = new Vp9MvCandidatesList();
        Throws<ArgumentOutOfRangeException>(() =>
            Vp9MvRefSelector.ForInterMode(list, (Vp9InterMode)99));
    }
}
