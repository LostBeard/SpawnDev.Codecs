// Av1StreamAnalyzer tests against the real BBB AV1 fixture.

using SpawnDev.Codecs.Video.Av1;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Av1StreamAnalyzer_BbbFixture_ReturnsCompleteSummary()
    {
        var bytes = LoadAv1Fixture();
        var summary = Av1StreamAnalyzer.Analyze(bytes);

        // IVF level
        Equal("AV01", summary.IvfHeader.FourCc);
        Equal(320, summary.IvfHeader.Width);
        Equal(180, summary.IvfHeader.Height);
        Equal(60, summary.TotalTemporalUnits);

        // Sequence header
        True(summary.SequenceHeader is not null, "expected SH from BBB");
        Equal(0, summary.SequenceHeader!.SeqProfile);
        Equal(8, summary.SequenceHeader.BitDepth);

        // OBU counts (matches earlier observations)
        Equal(60, summary.ObuCounts[Av1ObuType.TemporalDelimiter]);
        Equal(1, summary.ObuCounts[Av1ObuType.SequenceHeader]);
        Equal(62, summary.ObuCounts[Av1ObuType.Frame]);
        Equal(25, summary.ObuCounts[Av1ObuType.FrameHeader]);

        // Frame timeline split: coded vs show-existing
        Equal(62, summary.CodedFrames.Count);
        Equal(25, summary.ShowExistingFrames.Count);

        // Frame type breakdown across coded frames
        Equal(1, summary.FrameTypeCounts[Av1FrameType.KeyFrame]);
        Equal(61, summary.FrameTypeCounts[Av1FrameType.InterFrame]);
    }

    [TestMethod]
    public void Av1StreamAnalyzer_BbbFixture_FirstFrameIsVisibleKeyframe()
    {
        var bytes = LoadAv1Fixture();
        var summary = Av1StreamAnalyzer.Analyze(bytes);
        var first = summary.CodedFrames[0];
        Equal(1, first.TemporalUnit);
        Equal(1, first.IndexInTu);
        Equal(Av1FrameType.KeyFrame, first.Header.FrameType);
        Equal(true, first.Header.ShowFrame);
    }

    [TestMethod]
    public void Av1StreamAnalyzer_BbbFixture_PreservesFramePts()
    {
        var bytes = LoadAv1Fixture();
        var summary = Av1StreamAnalyzer.Analyze(bytes);
        // Each coded frame's Pts is the IVF Pts of its containing TU.
        // PTS values should be monotonically non-decreasing across the stream.
        long prevPts = -1;
        foreach (var f in summary.CodedFrames)
        {
            True(f.Pts >= prevPts,
                $"expected monotonic PTS; saw {f.Pts} after {prevPts} at TU {f.TemporalUnit}");
            prevPts = f.Pts;
        }
    }

    [TestMethod]
    public void Av1StreamAnalyzer_BbbFixture_ShowExistingMapsToValidSlot()
    {
        var bytes = LoadAv1Fixture();
        var summary = Av1StreamAnalyzer.Analyze(bytes);
        foreach (var se in summary.ShowExistingFrames)
        {
            True(se.Header.ShowExistingFrame, "expected ShowExistingFrame=true");
            True(se.Header.FrameToShowMapIdx >= 0 && se.Header.FrameToShowMapIdx < 8,
                $"map idx {se.Header.FrameToShowMapIdx} out of range");
        }
    }
}
