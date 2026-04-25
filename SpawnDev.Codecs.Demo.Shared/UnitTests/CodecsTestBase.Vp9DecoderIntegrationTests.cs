// End-to-end integration tests for Vp9Decoder. Drives the full
// Matroska -> Vp9SuperframeParser -> Vp9Decoder.DecodeFrameAsync ->
// IVideoFrameSink pipeline against the bundled Big Buck Bunny 10s
// VP9 fixture and verifies dimensions / frame counts.
//
// At this point the decoder emits a placeholder mid-gray frame for
// every visible packet; once block decode is wired up these same
// tests will pivot to pixel comparison against ffmpeg ground truth.

using SpawnDev.Codecs.Container.Matroska;
using SpawnDev.Codecs.Video;
using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    /// <summary>
    /// Recording <see cref="IVideoFrameSink"/>: copies each emitted
    /// frame into a list so tests can poke at dimensions and counts.
    /// </summary>
    private sealed class RecordingVp9FrameSink : IVideoFrameSink
    {
        public sealed record Snapshot(
            int Width, int Height,
            byte[] Y, int YStride,
            byte[] U, int UStride,
            byte[] V, int VStride,
            long Pts);

        public List<Snapshot> Frames { get; } = new();

        public ValueTask OnFrameAsync(
            ReadOnlyMemory<byte> yPlane, int yStride,
            ReadOnlyMemory<byte> uPlane, int uStride,
            ReadOnlyMemory<byte> vPlane, int vStride,
            long pts)
        {
            int yLen = yPlane.Length;
            int width = yStride;
            int height = yLen / yStride;
            Frames.Add(new Snapshot(
                width, height,
                yPlane.ToArray(), yStride,
                uPlane.ToArray(), uStride,
                vPlane.ToArray(), vStride,
                pts));
            return ValueTask.CompletedTask;
        }
    }

    [TestMethod]
    public async Task Vp9Decoder_DecodesBigBuckBunny_ReportsCorrectDimensions()
    {
        // Drive every packet from the BBB.webm fixture through the
        // VP9 decoder. Verify dimensions become 320x180 after the
        // first keyframe and stay there.
        using var stream = LoadBigBuckBunnyWebM();
        var container = new MatroskaContainer(stream);
        var video = container.Tracks.First(t => t.IsVideo);

        await using var decoder = new Vp9Decoder();
        var sink = new RecordingVp9FrameSink();
        int packetsFed = 0;

        foreach (var frame in container.Frames.Where(f => f.TrackNumber == video.TrackNumber))
        {
            await decoder.DecodeFrameAsync(frame.Data, sink);
            packetsFed++;
        }

        True(packetsFed > 0, "expected at least one VP9 packet in BBB fixture");
        Equal(320, decoder.Width);
        Equal(180, decoder.Height);
        True(sink.Frames.Count > 0, $"decoder emitted no frames after {packetsFed} packets");
    }

    [TestMethod]
    public async Task Vp9Decoder_DecodesBigBuckBunny_EmittedFramesHaveCorrectPlaneSizes()
    {
        // Every emitted frame must have:
        //   Y plane = 320 * 180 = 57_600 bytes
        //   U/V planes = 160 * 90 = 14_400 bytes (4:2:0)
        using var stream = LoadBigBuckBunnyWebM();
        var container = new MatroskaContainer(stream);
        var video = container.Tracks.First(t => t.IsVideo);

        await using var decoder = new Vp9Decoder();
        var sink = new RecordingVp9FrameSink();

        foreach (var frame in container.Frames.Where(f => f.TrackNumber == video.TrackNumber))
        {
            await decoder.DecodeFrameAsync(frame.Data, sink);
        }

        True(sink.Frames.Count >= 1, "expected at least one emitted frame");
        foreach (var snap in sink.Frames)
        {
            Equal(320, snap.YStride);
            Equal(57_600, snap.Y.Length);
            Equal(160, snap.UStride);
            Equal(14_400, snap.U.Length);
            Equal(160, snap.VStride);
            Equal(14_400, snap.V.Length);
        }
    }

    [TestMethod]
    public async Task Vp9Decoder_FirstFrameIsKeyframe_LearnsDimensions()
    {
        // The first packet in the BBB fixture must contain a keyframe.
        // After feeding only that packet, the decoder's Width / Height
        // should already be populated.
        using var stream = LoadBigBuckBunnyWebM();
        var container = new MatroskaContainer(stream);
        var video = container.Tracks.First(t => t.IsVideo);

        var first = container.Frames
            .First(f => f.TrackNumber == video.TrackNumber);

        await using var decoder = new Vp9Decoder();
        Equal(0, decoder.Width); // not yet learned
        Equal(0, decoder.Height);

        var sink = new RecordingVp9FrameSink();
        int frames = await decoder.DecodeFrameAsync(first.Data, sink);

        Equal(320, decoder.Width);
        Equal(180, decoder.Height);
        Equal(Vp9SubsamplingPair.Yuv420, decoder.Subsampling);
        Equal(Vp9BitDepth.Bits8, decoder.BitDepth);
        True(frames >= 1, "first packet carries a visible frame");

        // LastFrameHeader should be populated to a key frame.
        True(decoder.LastFrameHeader is not null, "LastFrameHeader populated");
        Equal(Vp9FrameType.Key, decoder.LastFrameHeader!.FrameType);
        Equal(320, decoder.LastFrameHeader.FrameWidth);
        Equal(180, decoder.LastFrameHeader.FrameHeight);
    }

    [TestMethod]
    public async Task Vp9Decoder_RejectsNullSink()
    {
        await using var decoder = new Vp9Decoder();
        var packet = new byte[] { 0x80, 0x49, 0x83, 0x42, 0, 0, 0 };
        bool threw = false;
        try { await decoder.DecodeFrameAsync(packet, null!); }
        catch (ArgumentNullException) { threw = true; }
        True(threw, "expected ArgumentNullException for null sink");
    }

    [TestMethod]
    public async Task Vp9Decoder_DecodesEntireStream_NoExceptions()
    {
        // Smoke test: feed every video packet, verify nothing throws.
        // This exercises every header variant (key, inter, hidden alt-ref)
        // present in the 10-second BBB fixture.
        using var stream = LoadBigBuckBunnyWebM();
        var container = new MatroskaContainer(stream);
        var video = container.Tracks.First(t => t.IsVideo);

        await using var decoder = new Vp9Decoder();
        var sink = new RecordingVp9FrameSink();

        int packets = 0;
        foreach (var frame in container.Frames.Where(f => f.TrackNumber == video.TrackNumber))
        {
            await decoder.DecodeFrameAsync(frame.Data, sink);
            packets++;
        }

        True(packets > 0, "fixture must contain VP9 packets");
        // Every visible frame should have produced a sink callback;
        // hidden alt-refs are silently consumed.
        True(sink.Frames.Count > 0, "decoder emitted at least one visible frame");
        True(sink.Frames.Count <= packets,
            $"emitted {sink.Frames.Count} frames from {packets} packets - alt-refs should NOT emit");
    }
}
