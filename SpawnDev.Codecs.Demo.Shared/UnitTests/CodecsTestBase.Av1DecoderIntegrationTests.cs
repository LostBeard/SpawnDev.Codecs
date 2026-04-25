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
