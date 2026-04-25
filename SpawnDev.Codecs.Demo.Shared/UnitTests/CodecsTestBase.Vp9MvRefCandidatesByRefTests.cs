// Tests for Vp9MvRefCandidatesByRef (slice 277).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9MvRefCandidatesByRef_AllRefs_StartEmpty()
    {
        var store = new Vp9MvRefCandidatesByRef();
        Equal(0, store.ForRef(Vp9MvReferenceFrame.Intra).Count);
        Equal(0, store.ForRef(Vp9MvReferenceFrame.Last).Count);
        Equal(0, store.ForRef(Vp9MvReferenceFrame.Golden).Count);
        Equal(0, store.ForRef(Vp9MvReferenceFrame.AltRef).Count);
    }

    [TestMethod]
    public void Vp9MvRefCandidatesByRef_PerRefIsolation()
    {
        var store = new Vp9MvRefCandidatesByRef();
        store.ForRef(Vp9MvReferenceFrame.Last).TryAdd(new Vp9Mv(1, 2));
        Equal(1, store.ForRef(Vp9MvReferenceFrame.Last).Count);
        // Other refs unaffected.
        Equal(0, store.ForRef(Vp9MvReferenceFrame.Golden).Count);
        Equal(0, store.ForRef(Vp9MvReferenceFrame.AltRef).Count);
    }

    [TestMethod]
    public void Vp9MvRefCandidatesByRef_Clear_ResetsAll()
    {
        var store = new Vp9MvRefCandidatesByRef();
        store.ForRef(Vp9MvReferenceFrame.Last).TryAdd(new Vp9Mv(1, 2));
        store.ForRef(Vp9MvReferenceFrame.Golden).TryAdd(new Vp9Mv(3, 4));
        store.ForRef(Vp9MvReferenceFrame.AltRef).TryAdd(new Vp9Mv(5, 6));
        store.Clear();
        Equal(0, store.ForRef(Vp9MvReferenceFrame.Last).Count);
        Equal(0, store.ForRef(Vp9MvReferenceFrame.Golden).Count);
        Equal(0, store.ForRef(Vp9MvReferenceFrame.AltRef).Count);
    }

    [TestMethod]
    public void Vp9MvRefCandidatesByRef_FillToCapacity()
    {
        var store = new Vp9MvRefCandidatesByRef();
        var list = store.ForRef(Vp9MvReferenceFrame.Last);
        Equal(true, list.TryAdd(new Vp9Mv(1, 2)));
        Equal(true, list.TryAdd(new Vp9Mv(3, 4)));
        Equal(true, list.IsFull);
        Equal(false, list.TryAdd(new Vp9Mv(5, 6)));
    }
}
