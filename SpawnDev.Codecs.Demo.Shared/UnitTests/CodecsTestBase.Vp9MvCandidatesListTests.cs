// Tests for Vp9MvCandidatesList (slice 275).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9MvCandidatesList_Capacity_Is2()
    {
        Equal(2, Vp9MvCandidatesList.MaxCandidates);
    }

    [TestMethod]
    public void Vp9MvCandidatesList_Empty_AfterConstruction()
    {
        var list = new Vp9MvCandidatesList();
        Equal(0, list.Count);
        Equal(false, list.IsFull);
        Equal(0, list.AsSpan().Length);
    }

    [TestMethod]
    public void Vp9MvCandidatesList_TryAdd_Distinct_Adds()
    {
        var list = new Vp9MvCandidatesList();
        Equal(true, list.TryAdd(new Vp9Mv(1, 2)));
        Equal(1, list.Count);
        Equal(true, list.TryAdd(new Vp9Mv(3, 4)));
        Equal(2, list.Count);
        Equal(true, list.IsFull);
    }

    [TestMethod]
    public void Vp9MvCandidatesList_TryAdd_Duplicate_Rejected()
    {
        var list = new Vp9MvCandidatesList();
        list.TryAdd(new Vp9Mv(1, 2));
        Equal(false, list.TryAdd(new Vp9Mv(1, 2))); // same as first
        Equal(1, list.Count);
    }

    [TestMethod]
    public void Vp9MvCandidatesList_TryAdd_Full_Rejected()
    {
        var list = new Vp9MvCandidatesList();
        list.TryAdd(new Vp9Mv(1, 2));
        list.TryAdd(new Vp9Mv(3, 4));
        Equal(false, list.TryAdd(new Vp9Mv(5, 6)));
        Equal(2, list.Count);
    }

    [TestMethod]
    public void Vp9MvCandidatesList_Indexer_AccessesAddedCandidates()
    {
        var list = new Vp9MvCandidatesList();
        list.TryAdd(new Vp9Mv(7, 11));
        list.TryAdd(new Vp9Mv(-3, 5));
        Equal(new Vp9Mv(7, 11), list[0]);
        Equal(new Vp9Mv(-3, 5), list[1]);
    }

    [TestMethod]
    public void Vp9MvCandidatesList_Indexer_OutOfRange_Throws()
    {
        var list = new Vp9MvCandidatesList();
        list.TryAdd(new Vp9Mv(1, 2));
        Throws<ArgumentOutOfRangeException>(() => _ = list[1]);  // only [0] valid
        Throws<ArgumentOutOfRangeException>(() => _ = list[-1]);
    }

    [TestMethod]
    public void Vp9MvCandidatesList_Clear_ResetsToEmpty()
    {
        var list = new Vp9MvCandidatesList();
        list.TryAdd(new Vp9Mv(1, 2));
        list.TryAdd(new Vp9Mv(3, 4));
        list.Clear();
        Equal(0, list.Count);
        Equal(false, list.IsFull);
    }

    [TestMethod]
    public void Vp9MvCandidatesList_AsSpan_ReflectsCount()
    {
        var list = new Vp9MvCandidatesList();
        Equal(0, list.AsSpan().Length);
        list.TryAdd(new Vp9Mv(1, 2));
        Equal(1, list.AsSpan().Length);
        Equal(new Vp9Mv(1, 2), list.AsSpan()[0]);
    }
}
