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
