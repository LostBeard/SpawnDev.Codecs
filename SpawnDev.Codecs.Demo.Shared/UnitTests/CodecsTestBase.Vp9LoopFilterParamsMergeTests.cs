// Tests for Vp9LoopFilterParamsMerge (slice 273).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9LoopFilterParamsMerge_RefDeltas_NullParsed_PreservesPersistent()
    {
        var persistent = new int[] { 1, 2, 3, 4 };
        var merged = Vp9LoopFilterParamsMerge.MergeRefDeltas(null, persistent);
        Equal(4, merged.Length);
        Equal(1, merged[0]);
        Equal(2, merged[1]);
        Equal(3, merged[2]);
        Equal(4, merged[3]);
    }

    [TestMethod]
    public void Vp9LoopFilterParamsMerge_RefDeltas_AllUpdated_OverridesPersistent()
    {
        var persistent = new int[] { 1, 2, 3, 4 };
        var parsed = new int?[] { 10, 20, 30, 40 };
        var merged = Vp9LoopFilterParamsMerge.MergeRefDeltas(parsed, persistent);
        Equal(10, merged[0]);
        Equal(20, merged[1]);
        Equal(30, merged[2]);
        Equal(40, merged[3]);
    }

    [TestMethod]
    public void Vp9LoopFilterParamsMerge_RefDeltas_PartialUpdate_MixesValues()
    {
        var persistent = new int[] { 1, 2, 3, 4 };
        var parsed = new int?[] { 10, null, 30, null };
        var merged = Vp9LoopFilterParamsMerge.MergeRefDeltas(parsed, persistent);
        Equal(10, merged[0]); // updated
        Equal(2, merged[1]);  // persistent
        Equal(30, merged[2]); // updated
        Equal(4, merged[3]);  // persistent
    }

    [TestMethod]
    public void Vp9LoopFilterParamsMerge_RefDeltas_EmptyParsed_PreservesPersistent()
    {
        // Length-zero parsed array (e.g. when ModeRefDeltaUpdate=false).
        var persistent = new int[] { 5, 6, 7, 8 };
        var parsed = Array.Empty<int?>();
        var merged = Vp9LoopFilterParamsMerge.MergeRefDeltas(parsed, persistent);
        Equal(5, merged[0]);
        Equal(6, merged[1]);
        Equal(7, merged[2]);
        Equal(8, merged[3]);
    }

    [TestMethod]
    public void Vp9LoopFilterParamsMerge_ModeDeltas_NullParsed_PreservesPersistent()
    {
        var persistent = new int[] { -3, 7 };
        var merged = Vp9LoopFilterParamsMerge.MergeModeDeltas(null, persistent);
        Equal(-3, merged[0]);
        Equal(7, merged[1]);
    }

    [TestMethod]
    public void Vp9LoopFilterParamsMerge_ModeDeltas_PartialUpdate()
    {
        var persistent = new int[] { -3, 7 };
        var parsed = new int?[] { null, 11 };
        var merged = Vp9LoopFilterParamsMerge.MergeModeDeltas(parsed, persistent);
        Equal(-3, merged[0]); // persistent
        Equal(11, merged[1]); // updated
    }

    [TestMethod]
    public void Vp9LoopFilterParamsMerge_RejectsNullPersistent()
    {
        Throws<ArgumentNullException>(() =>
            Vp9LoopFilterParamsMerge.MergeRefDeltas(null, null!));
        Throws<ArgumentNullException>(() =>
            Vp9LoopFilterParamsMerge.MergeModeDeltas(null, null!));
    }

    [TestMethod]
    public void Vp9LoopFilterParamsMerge_RejectsWrongLengthPersistent()
    {
        Throws<ArgumentException>(() =>
            Vp9LoopFilterParamsMerge.MergeRefDeltas(null, new int[] { 1, 2 }));
        Throws<ArgumentException>(() =>
            Vp9LoopFilterParamsMerge.MergeModeDeltas(null, new int[] { 1, 2, 3 }));
    }
}
