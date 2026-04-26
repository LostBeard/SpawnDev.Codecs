// End-to-end Av1Decoder integration tests on real BBB AV1 stream.

using SpawnDev.Codecs.Container.Ivf;
using SpawnDev.Codecs.Video;
using SpawnDev.Codecs.Video.Av1;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private sealed class Av1RecordingSink : IVideoFrameSink
    {
        public List<(int Width, int Height, int YLen, int ULen, int VLen, long Pts)> Frames { get; } = new();
        public ValueTask OnFrameAsync(
            ReadOnlyMemory<byte> y, int ys,
            ReadOnlyMemory<byte> u, int us,
            ReadOnlyMemory<byte> v, int vs,
            long pts)
        {
            Frames.Add((ys, y.Length / ys, y.Length, u.Length, v.Length, pts));
            return ValueTask.CompletedTask;
        }
    }

    [TestMethod]
    public async Task Av1Decoder_BbbStream_ParsesEverySequenceHeader_LearnsDimensions()
    {
        var bytes = LoadAv1Fixture();
        await using var decoder = new Av1Decoder();
        var sink = new Av1RecordingSink();

        int packetsFed = 0;
        int frameDataPackets = 0;
        foreach (var ivfFrame in IvfReader.EnumerateFrames(bytes))
        {
            int emitted = await decoder.DecodeFrameAsync(ivfFrame.Data, sink);
            packetsFed++;
            if (emitted > 0) frameDataPackets++;
        }

        True(packetsFed > 0, "expected at least one IVF frame");
        True(frameDataPackets > 0, "expected at least one packet with coded frame data");

        // After the first SH lands, dimensions should be 320x180.
        Equal(320, decoder.Width);
        Equal(180, decoder.Height);

        // SequenceHeader exposed.
        True(decoder.LastSequenceHeader is not null, "LastSequenceHeader must be populated");
        Equal(0, decoder.LastSequenceHeader!.SeqProfile);
        Equal(8, decoder.LastSequenceHeader.BitDepth);
        Equal(false, decoder.LastSequenceHeader.Monochrome);
    }

    [TestMethod]
    public async Task Av1Decoder_BbbStream_EmitsFramesAtCorrectDimensions()
    {
        var bytes = LoadAv1Fixture();
        await using var decoder = new Av1Decoder();
        var sink = new Av1RecordingSink();

        foreach (var ivfFrame in IvfReader.EnumerateFrames(bytes))
        {
            await decoder.DecodeFrameAsync(ivfFrame.Data, sink);
        }

        // 60 frames in the fixture, all should emit a placeholder.
        True(sink.Frames.Count >= 1, "expected at least one emitted frame");
        foreach (var snap in sink.Frames)
        {
            Equal(320, snap.Width);
            Equal(180, snap.Height);
            Equal(57_600, snap.YLen);   // 320 * 180
            Equal(14_400, snap.ULen);   // 160 * 90 (4:2:0)
            Equal(14_400, snap.VLen);
        }
    }

    [TestMethod]
    public async Task Av1Decoder_BbbFirstFrame_LastFrameObuCounts_IncludesSequenceHeader()
    {
        var bytes = LoadAv1Fixture();
        await using var decoder = new Av1Decoder();
        var sink = new Av1RecordingSink();

        var first = IvfReader.EnumerateFrames(bytes).First();
        await decoder.DecodeFrameAsync(first.Data, sink);

        var counts = decoder.LastFrameObuCounts;
        True(counts.ContainsKey(Av1ObuType.SequenceHeader),
            $"first frame must contain SequenceHeader; saw {string.Join(',', counts.Keys)}");
    }

    [TestMethod]
    public async Task Av1Decoder_BbbStream_CumulativeStats_MatchObservedTotals()
    {
        // After driving every BBB AV1 frame through the decoder, cumulative
        // counts should reflect the OBU type distribution + frame type
        // breakdown observed by the per-frame metadata tests.
        var bytes = LoadAv1Fixture();
        await using var decoder = new Av1Decoder();
        var sink = new Av1RecordingSink();

        foreach (var ivfFrame in IvfReader.EnumerateFrames(bytes))
        {
            await decoder.DecodeFrameAsync(ivfFrame.Data, sink);
        }

        // 60 IVF frames = 60 Temporal Units.
        Equal(60, decoder.TotalTemporalUnits);

        // OBU type distribution observed in fixture:
        //   Frame: 62, TemporalDelimiter: 60, FrameHeader: 25, SequenceHeader: 1
        var obuCounts = decoder.CumulativeObuCounts;
        True(obuCounts.ContainsKey(Av1ObuType.SequenceHeader),
            $"expected SH OBUs; saw {string.Join(',', obuCounts.Keys)}");
        Equal(60, obuCounts[Av1ObuType.TemporalDelimiter]);
        Equal(1, obuCounts[Av1ObuType.SequenceHeader]);
        Equal(62, obuCounts[Av1ObuType.Frame]);
        Equal(25, obuCounts[Av1ObuType.FrameHeader]);

        // Frame type breakdown across all coded frame data OBUs.
        // 62 Frame + 25 FrameHeader = 87 frame-header parses total.
        // Some are show_existing_frame replays (no coded body) - those
        // are counted in ShowExistingFrameCount, not CumulativeFrameTypeCounts.
        var ftCounts = decoder.CumulativeFrameTypeCounts;
        int kfCount = ftCounts.GetValueOrDefault(Av1FrameType.KeyFrame, 0);
        int interCount = ftCounts.GetValueOrDefault(Av1FrameType.InterFrame, 0);
        int showExistCount = decoder.ShowExistingFrameCount;
        Equal(87, kfCount + interCount + showExistCount);
        // BBB has only 1 actual KeyFrame plus many show_existing replays.
        Equal(1, kfCount);
        // 25 show_existing replays (matches FrameHeader OBU count - those
        // are typically used for show_existing in BBB's hierarchical structure).
        Equal(25, showExistCount);
        Equal(61, interCount);
        // No IntraOnlyFrame or SwitchFrame in BBB.
    }

    [TestMethod]
    public async Task Av1Decoder_BbbStream_TracksShowExistingFrameSeparately()
    {
        // After driving every BBB AV1 frame, ShowExistingFrameCount + the
        // CumulativeFrameTypeCounts should add up to the total coded
        // frame data parses without overlap.
        var bytes = LoadAv1Fixture();
        await using var decoder = new Av1Decoder();
        var sink = new Av1RecordingSink();

        foreach (var ivfFrame in IvfReader.EnumerateFrames(bytes))
        {
            await decoder.DecodeFrameAsync(ivfFrame.Data, sink);
        }

        // BBB has libaom's hierarchical alt-ref structure. The 25
        // FrameHeader OBUs in the source are show_existing_frame replays
        // (one per visible "skip" frame in the GOP).
        Equal(25, decoder.ShowExistingFrameCount);

        // Show_existing entries are excluded from cumulative type counts.
        var ftCounts = decoder.CumulativeFrameTypeCounts;
        True(!ftCounts.ContainsKey(Av1FrameType.KeyFrame) || ftCounts[Av1FrameType.KeyFrame] < 25,
            "show_existing should not be counted as KeyFrame in cumulative type counts");
    }

    [TestMethod]
    public async Task Av1Decoder_BbbFirstFrame_PopulatesLastFrameHeader_AsKeyframe()
    {
        var bytes = LoadAv1Fixture();
        await using var decoder = new Av1Decoder();
        var sink = new Av1RecordingSink();

        var first = IvfReader.EnumerateFrames(bytes).First();
        await decoder.DecodeFrameAsync(first.Data, sink);

        True(decoder.LastFrameHeader is not null, "LastFrameHeader must be populated for the first frame");
        Equal(Av1FrameType.KeyFrame, decoder.LastFrameHeader!.FrameType);
        Equal(true, decoder.LastFrameHeader.ShowFrame);
        Equal(true, decoder.LastFrameHeader.FrameIsIntra);
        Equal(false, decoder.LastFrameHeader.ShowExistingFrame);
    }
}
