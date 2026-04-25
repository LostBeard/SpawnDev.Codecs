// Tests for Vp9LoopFilterLookup (slice 247).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9LoopFilterLookup_MaxLoopFilter_Is63()
    {
        Equal(63, Vp9LoopFilterLookup.MaxLoopFilter);
    }

    [TestMethod]
    public void Vp9LoopFilterLookup_NoSegmentation_PassesThrough()
    {
        var seg = MakeLfSeg(enabled: false);
        Equal(20, Vp9LoopFilterLookup.ResolveSegmentLevel(seg, 0, 20));
    }

    [TestMethod]
    public void Vp9LoopFilterLookup_AltLfInactive_PassesThrough()
    {
        var seg = MakeLfSeg(enabled: true);
        // ALT_LF feature not enabled for any segment -> base level returned.
        Equal(35, Vp9LoopFilterLookup.ResolveSegmentLevel(seg, 0, 35));
    }

    [TestMethod]
    public void Vp9LoopFilterLookup_AltLfDeltaMode_AddsToBase()
    {
        var seg = MakeLfSeg(enabled: true, absDelta: false);
        seg.FeatureEnabled[2, (int)Vp9SegFeature.AltLf] = true;
        seg.FeatureData[2, (int)Vp9SegFeature.AltLf] = 5;
        Equal(45, Vp9LoopFilterLookup.ResolveSegmentLevel(seg, 2, 40));
    }

    [TestMethod]
    public void Vp9LoopFilterLookup_AltLfDeltaMode_NegativeDelta()
    {
        var seg = MakeLfSeg(enabled: true, absDelta: false);
        seg.FeatureEnabled[0, (int)Vp9SegFeature.AltLf] = true;
        seg.FeatureData[0, (int)Vp9SegFeature.AltLf] = -10;
        Equal(20, Vp9LoopFilterLookup.ResolveSegmentLevel(seg, 0, 30));
    }

    [TestMethod]
    public void Vp9LoopFilterLookup_AltLfAbsoluteMode_IgnoresBase()
    {
        var seg = MakeLfSeg(enabled: true, absDelta: true);
        seg.FeatureEnabled[3, (int)Vp9SegFeature.AltLf] = true;
        seg.FeatureData[3, (int)Vp9SegFeature.AltLf] = 17;
        Equal(17, Vp9LoopFilterLookup.ResolveSegmentLevel(seg, 3, 50));
    }

    [TestMethod]
    public void Vp9LoopFilterLookup_ClampsToZero()
    {
        var seg = MakeLfSeg(enabled: true, absDelta: false);
        seg.FeatureEnabled[0, (int)Vp9SegFeature.AltLf] = true;
        seg.FeatureData[0, (int)Vp9SegFeature.AltLf] = -100;
        // Base 30 + (-100) = -70 -> clamped to 0.
        Equal(0, Vp9LoopFilterLookup.ResolveSegmentLevel(seg, 0, 30));
    }

    [TestMethod]
    public void Vp9LoopFilterLookup_ClampsToMaxLoopFilter()
    {
        var seg = MakeLfSeg(enabled: true, absDelta: true);
        seg.FeatureEnabled[0, (int)Vp9SegFeature.AltLf] = true;
        seg.FeatureData[0, (int)Vp9SegFeature.AltLf] = 1000;
        // Absolute 1000 -> clamped to 63.
        Equal(63, Vp9LoopFilterLookup.ResolveSegmentLevel(seg, 0, 30));
    }

    [TestMethod]
    public void Vp9LoopFilterLookup_RejectsNullSegmentation()
    {
        Throws<ArgumentNullException>(() =>
            Vp9LoopFilterLookup.ResolveSegmentLevel(null!, 0, 30));
    }

    [TestMethod]
    public void Vp9LoopFilterLookup_BlockLevel_NoModeRefDelta_PassesSegmentLevel()
    {
        var seg = MakeLfSeg(enabled: false);
        Equal(30, Vp9LoopFilterLookup.ResolveBlockLevel(
            frameFilterLevel: 30, seg, segmentId: 0,
            modeRefDeltaEnabled: false,
            refDeltas: ReadOnlySpan<int>.Empty,
            modeDeltas: ReadOnlySpan<int>.Empty,
            Vp9MvReferenceFrame.Last,
            interMode: Vp9InterMode.ZeroMv));
    }

    [TestMethod]
    public void Vp9LoopFilterLookup_BlockLevel_IntraOnlyAddsRefDelta()
    {
        // Frame level 32, intra block. Ref deltas {1, 2, 3, 4}.
        // ref_deltas[Intra=0] = 1; scale = 1 << (32 >> 5) = 1 << 1 = 2.
        // level = 32 + 1*2 = 34.
        var seg = MakeLfSeg(enabled: false);
        Equal(34, Vp9LoopFilterLookup.ResolveBlockLevel(
            frameFilterLevel: 32, seg, segmentId: 0,
            modeRefDeltaEnabled: true,
            refDeltas: new int[] { 1, 2, 3, 4 },
            modeDeltas: new int[] { -1, 1 },
            Vp9MvReferenceFrame.Intra,
            interMode: null));
    }

    [TestMethod]
    public void Vp9LoopFilterLookup_BlockLevel_InterAddsBothRefAndModeDelta()
    {
        // Frame level 32, ref Last (idx 1), interMode = ZeroMv.
        // ref_deltas[Last] = 2; mode_deltas[0] = -3 (Zero index).
        // scale = 1 << (32 >> 5) = 2.
        // level = 32 + 2*2 + (-3)*2 = 32 + 4 - 6 = 30.
        var seg = MakeLfSeg(enabled: false);
        Equal(30, Vp9LoopFilterLookup.ResolveBlockLevel(
            frameFilterLevel: 32, seg, segmentId: 0,
            modeRefDeltaEnabled: true,
            refDeltas: new int[] { 1, 2, 3, 4 },
            modeDeltas: new int[] { -3, 1 },
            Vp9MvReferenceFrame.Last,
            interMode: Vp9InterMode.ZeroMv));
    }

    [TestMethod]
    public void Vp9LoopFilterLookup_BlockLevel_NewMvUsesIndex1OfModeDeltas()
    {
        // Frame level 0, scale = 1 << 0 = 1. ref_deltas[Last] = 0, mode_deltas[1] = 7.
        // level = 0 + 0*1 + 7*1 = 7.
        var seg = MakeLfSeg(enabled: false);
        Equal(7, Vp9LoopFilterLookup.ResolveBlockLevel(
            frameFilterLevel: 0, seg, segmentId: 0,
            modeRefDeltaEnabled: true,
            refDeltas: new int[] { 0, 0, 0, 0 },
            modeDeltas: new int[] { 0, 7 },
            Vp9MvReferenceFrame.Last,
            interMode: Vp9InterMode.NewMv));
    }

    [TestMethod]
    public void Vp9LoopFilterLookup_BlockLevel_ScaleAtLevel32_Is2()
    {
        // level >> 5 = 1 -> scale = 2 once we cross 32.
        var seg = MakeLfSeg(enabled: false);
        Equal(36, Vp9LoopFilterLookup.ResolveBlockLevel(
            frameFilterLevel: 32, seg, segmentId: 0,
            modeRefDeltaEnabled: true,
            refDeltas: new int[] { 0, 2, 0, 0 },
            modeDeltas: new int[] { 0, 0 },
            Vp9MvReferenceFrame.Last,
            interMode: Vp9InterMode.NewMv));
    }

    [TestMethod]
    public void Vp9LoopFilterLookup_BlockLevel_RejectsNullInterModeForInterRef()
    {
        var seg = MakeLfSeg(enabled: false);
        Throws<ArgumentNullException>(() =>
            Vp9LoopFilterLookup.ResolveBlockLevel(
                frameFilterLevel: 30, seg, segmentId: 0,
                modeRefDeltaEnabled: true,
                refDeltas: new int[] { 0, 0, 0, 0 },
                modeDeltas: new int[] { 0, 0 },
                Vp9MvReferenceFrame.Last,
                interMode: null));
    }

    [TestMethod]
    public void Vp9LoopFilterLookup_BlockLevel_RejectsShortRefDeltas()
    {
        var seg = MakeLfSeg(enabled: false);
        Throws<ArgumentException>(() =>
            Vp9LoopFilterLookup.ResolveBlockLevel(
                frameFilterLevel: 30, seg, segmentId: 0,
                modeRefDeltaEnabled: true,
                refDeltas: new int[] { 0, 0, 0 }, // only 3, need 4
                modeDeltas: new int[] { 0, 0 },
                Vp9MvReferenceFrame.Last,
                interMode: Vp9InterMode.ZeroMv));
    }

    private static Vp9SegmentationParams MakeLfSeg(bool enabled = true, bool absDelta = false)
    {
        return new Vp9SegmentationParams
        {
            Enabled = enabled,
            UpdateMap = false,
            TreeProbsArray = new byte[Vp9SegmentationParams.TreeProbs],
            TemporalUpdate = false,
            PredProbs = new byte[Vp9SegmentationParams.PredictionProbs],
            UpdateData = false,
            AbsDelta = absDelta,
            FeatureEnabled = new bool[Vp9SegmentationParams.MaxSegments, Vp9SegmentationParams.FeaturesPerSegment],
            FeatureData = new int[Vp9SegmentationParams.MaxSegments, Vp9SegmentationParams.FeaturesPerSegment],
        };
    }
}
