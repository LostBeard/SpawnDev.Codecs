// Vp9StreamAnalyzer tests against the BBB.webm fixture.

using SpawnDev.Codecs.Container.Matroska;
using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9StreamAnalyzer_BbbFixture_ReturnsCompleteSummary()
    {
        using var stream = LoadBigBuckBunnyWebM();
        var container = new MatroskaContainer(stream);
        var video = container.Tracks.First(t => t.IsVideo);

        var packets = container.Frames
            .Where(f => f.TrackNumber == video.TrackNumber)
            .Select(f => (ReadOnlyMemory<byte>)f.Data);

        var summary = Vp9StreamAnalyzer.Analyze(packets);

        // 300 video packets in 10s @ 30fps; 0 superframes here so total
        // slices == total packets.
        Equal(300, summary.TotalPackets);
        Equal(300, summary.TotalSlices);
        Equal(300, summary.CodedFrames.Count);
        Equal(0, summary.ShowExistingFrames.Count);

        // 3 keyframes + 297 inter (matches what Vp9Decoder.CumulativeFrameTypeCounts reports).
        Equal(3, summary.FrameTypeCounts[Vp9FrameType.Key]);
        Equal(297, summary.FrameTypeCounts[Vp9FrameType.NonKey]);

        // Size stays at 320x180 throughout (only one entry in the change list).
        Equal(1, summary.SizeChanges.Count);
        Equal(320, summary.SizeChanges[0].Width);
        Equal(180, summary.SizeChanges[0].Height);
    }

    [TestMethod]
    public void Vp9StreamAnalyzer_BbbFixture_FirstFrameIsKeyframe()
    {
        using var stream = LoadBigBuckBunnyWebM();
        var container = new MatroskaContainer(stream);
        var video = container.Tracks.First(t => t.IsVideo);

        var packets = container.Frames
            .Where(f => f.TrackNumber == video.TrackNumber)
            .Select(f => (ReadOnlyMemory<byte>)f.Data);

        var summary = Vp9StreamAnalyzer.Analyze(packets);
        var first = summary.CodedFrames[0];
        Equal(1, first.PacketIndex);
        Equal(1, first.SliceIndex);
        Equal(Vp9FrameType.Key, first.Header.FrameType);
        Equal(320, first.Header.FrameWidth);
        Equal(180, first.Header.FrameHeight);
        // Compressed header parsed for the first keyframe.
        True(first.CompressedResult is not null, "expected compressed header for keyframe");
    }
}
