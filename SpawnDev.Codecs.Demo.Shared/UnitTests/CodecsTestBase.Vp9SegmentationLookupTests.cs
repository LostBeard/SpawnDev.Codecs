// Tests for Vp9SegmentationLookup (slice 246).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9SegmentationLookup_MaxQuantizerIndex_Is255()
    {
        // VP9 Profile 0 (8-bit) MAXQ = 255.
        Equal(255, Vp9SegmentationLookup.MaxQuantizerIndex);
    }

    [TestMethod]
    public void Vp9SegmentationLookup_IsFeatureActive_DisabledSegmentation_AlwaysFalse()
    {
        var seg = MakeSegmentation(enabled: false);
        seg.FeatureEnabled[0, (int)Vp9SegFeature.AltQ] = true;
        Equal(false, Vp9SegmentationLookup.IsFeatureActive(seg, 0, Vp9SegFeature.AltQ));
    }

    [TestMethod]
    public void Vp9SegmentationLookup_IsFeatureActive_EnabledSegmentation_RespectsFlag()
    {
        var seg = MakeSegmentation(enabled: true);
        seg.FeatureEnabled[0, (int)Vp9SegFeature.AltQ] = true;
        seg.FeatureEnabled[0, (int)Vp9SegFeature.AltLf] = false;
        Equal(true, Vp9SegmentationLookup.IsFeatureActive(seg, 0, Vp9SegFeature.AltQ));
        Equal(false, Vp9SegmentationLookup.IsFeatureActive(seg, 0, Vp9SegFeature.AltLf));
    }

    [TestMethod]
    public void Vp9SegmentationLookup_GetFeatureData_ReturnsRawValue()
    {
        var seg = MakeSegmentation(enabled: true);
        seg.FeatureData[3, (int)Vp9SegFeature.AltQ] = 42;
        Equal(42, Vp9SegmentationLookup.GetFeatureData(seg, 3, Vp9SegFeature.AltQ));
    }

    [TestMethod]
    public void Vp9SegmentationLookup_ResolveQIndex_NoSegment_ReturnsBase()
    {
        var seg = MakeSegmentation(enabled: false);
        Equal(64, Vp9SegmentationLookup.ResolveQIndex(seg, 0, 64));
    }

    [TestMethod]
    public void Vp9SegmentationLookup_ResolveQIndex_AltQInactive_ReturnsBase()
    {
        var seg = MakeSegmentation(enabled: true);
        // AltQ feature is disabled for this segment.
        Equal(64, Vp9SegmentationLookup.ResolveQIndex(seg, 0, 64));
    }

    [TestMethod]
    public void Vp9SegmentationLookup_ResolveQIndex_DeltaMode_AddsToBase()
    {
        var seg = MakeSegmentation(enabled: true, absDelta: false);
        seg.FeatureEnabled[0, (int)Vp9SegFeature.AltQ] = true;
        seg.FeatureData[0, (int)Vp9SegFeature.AltQ] = -10;
        // base 64 + delta -10 = 54
        Equal(54, Vp9SegmentationLookup.ResolveQIndex(seg, 0, 64));
    }

    [TestMethod]
    public void Vp9SegmentationLookup_ResolveQIndex_AbsoluteMode_IgnoresBase()
    {
        var seg = MakeSegmentation(enabled: true, absDelta: true);
        seg.FeatureEnabled[1, (int)Vp9SegFeature.AltQ] = true;
        seg.FeatureData[1, (int)Vp9SegFeature.AltQ] = 100;
        // Absolute 100 wins regardless of base 64.
        Equal(100, Vp9SegmentationLookup.ResolveQIndex(seg, 1, 64));
    }

    [TestMethod]
    public void Vp9SegmentationLookup_ResolveQIndex_ClampsToZero()
    {
        var seg = MakeSegmentation(enabled: true, absDelta: false);
        seg.FeatureEnabled[0, (int)Vp9SegFeature.AltQ] = true;
        seg.FeatureData[0, (int)Vp9SegFeature.AltQ] = -1000;
        // base 64 + (-1000) = -936 -> clamped to 0.
        Equal(0, Vp9SegmentationLookup.ResolveQIndex(seg, 0, 64));
    }

    [TestMethod]
    public void Vp9SegmentationLookup_ResolveQIndex_ClampsToMaxQ()
    {
        var seg = MakeSegmentation(enabled: true, absDelta: true);
        seg.FeatureEnabled[0, (int)Vp9SegFeature.AltQ] = true;
        seg.FeatureData[0, (int)Vp9SegFeature.AltQ] = 9999;
        // Absolute 9999 -> clamped to 255.
        Equal(255, Vp9SegmentationLookup.ResolveQIndex(seg, 0, 64));
    }

    [TestMethod]
    public void Vp9SegmentationLookup_RejectsNullSegmentation()
    {
        Throws<ArgumentNullException>(() =>
            Vp9SegmentationLookup.IsFeatureActive(null!, 0, Vp9SegFeature.AltQ));
        Throws<ArgumentNullException>(() =>
            Vp9SegmentationLookup.GetFeatureData(null!, 0, Vp9SegFeature.AltQ));
        Throws<ArgumentNullException>(() =>
            Vp9SegmentationLookup.ResolveQIndex(null!, 0, 64));
    }

    private static Vp9SegmentationParams MakeSegmentation(bool enabled = true, bool absDelta = false)
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
