using SpawnDev.Codecs.Video;
using SpawnDev.Codecs.Video.Av1;
using SpawnDev.Codecs.Video.Vp8;
using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for the public IVideoDecoder surface. VP8 routes a real keyframe
/// through Vp8KeyframeWalker; VP9 + AV1 currently parse headers and emit
/// placeholder mid-gray frames while their walker integrations land.
/// Construction + dispose + codec identification are part of the contract
/// so callers can route by <see cref="VideoCodec"/> enum.
/// </summary>
public abstract partial class CodecsTestBase
{
    private sealed class CapturingFrameSink : IVideoFrameSink
    {
        public byte[]? Y { get; private set; }
        public int YStride { get; private set; }
        public int Width { get; private set; }
        public int Height { get; private set; }
        public int FrameCount { get; private set; }

        public ValueTask OnFrameAsync(
            ReadOnlyMemory<byte> y, int ys,
            ReadOnlyMemory<byte> u, int us,
            ReadOnlyMemory<byte> v, int vs,
            long pts)
        {
            Y = y.ToArray();
            YStride = ys;
            Width = ys; // sink contract: stride == width for the wired decoders
            Height = Y.Length / Math.Max(1, ys);
            FrameCount++;
            return ValueTask.CompletedTask;
        }
    }

    [TestMethod]
    public async Task Vp8Decoder_Codec_IsVp8()
    {
        await using var dec = new Vp8Decoder();
        Equal(VideoCodec.Vp8, dec.Codec);
    }

    [TestMethod]
    public async Task Vp9Decoder_Codec_IsVp9()
    {
        await using var dec = new Vp9Decoder();
        Equal(VideoCodec.Vp9, dec.Codec);
    }

    [TestMethod]
    public async Task Av1Decoder_Codec_IsAv1()
    {
        await using var dec = new Av1Decoder();
        Equal(VideoCodec.Av1, dec.Codec);
    }

    [TestMethod]
    public async Task Vp8Decoder_DecodesOwnEncoderKeyframe()
    {
        // Encode a 16x16 flat Y=128 keyframe with our encoder, then decode
        // it through the public IVideoDecoder surface and verify the sink
        // received a frame of the right dimensions with luma close to 128.
        const int W = 16, H = 16;
        var ySrc = new byte[W * H]; Array.Fill(ySrc, (byte)128);
        var uSrc = new byte[(W / 2) * (H / 2)]; Array.Fill(uSrc, (byte)128);
        var vSrc = new byte[(W / 2) * (H / 2)]; Array.Fill(vSrc, (byte)128);
        var frame = Vp8KeyframeEncoder.EncodeKeyFrame(
            ySrc, W, uSrc, W / 2, vSrc, W, H, baseQIndex: 30);

        var sink = new CapturingFrameSink();
        await using var dec = new Vp8Decoder();
        int n = await dec.DecodeFrameAsync(frame, sink);

        Equal(1, n);
        Equal(1, sink.FrameCount);
        Equal(W, dec.Width);
        Equal(H, dec.Height);
        True(sink.Y is not null);
        Equal(W * H, sink.Y!.Length);
        // Flat luma=128 should reconstruct close to 128 across the plane.
        long sum = 0;
        foreach (var b in sink.Y) sum += b;
        int mean = (int)(sum / sink.Y.Length);
        True(Math.Abs(mean - 128) <= 8);
    }

    [TestMethod]
    public async Task Vp8Decoder_InterFrame_ThrowsDescriptive()
    {
        // Synthesise a non-keyframe tag (low bit of byte 0 = 1 -> P-frame).
        // The decoder should reject it with a clear message rather than
        // crashing inside the walker.
        var fakeInter = new byte[] { 0x01, 0x00, 0x00 };
        await using var dec = new Vp8Decoder();
        bool threw = false;
        try
        {
            await dec.DecodeFrameAsync(fakeInter, new CapturingFrameSink());
        }
        catch (NotImplementedException ex)
        {
            threw = true;
            Contains("inter", ex.Message);
        }
        True(threw);
    }

    [TestMethod]
    public async Task Vp9Decoder_DecodesOwnEncoderKeyframe()
    {
        // Encode a 16x16 flat Y=128 keyframe with our VP9 encoder, then run
        // it through the public Vp9Decoder API which now drives the walker
        // (was a placeholder mid-gray emitter before commit 2814322).
        const int W = 16, H = 16;
        var ySrc = new byte[W * H]; Array.Fill(ySrc, (byte)128);
        var uSrc = new byte[(W / 2) * (H / 2)]; Array.Fill(uSrc, (byte)128);
        var vSrc = new byte[(W / 2) * (H / 2)]; Array.Fill(vSrc, (byte)128);
        var frame = SpawnDev.Codecs.Video.Vp9.Vp9KeyframeEncoder.EncodeKeyFrame(
            ySrc, W, uSrc, W / 2, vSrc, W, H, baseQIndex: 30);

        var sink = new CapturingFrameSink();
        await using var dec = new Vp9Decoder();
        int n = await dec.DecodeFrameAsync(frame, sink);

        Equal(1, n);
        Equal(1, sink.FrameCount);
        True(sink.Y is not null);
        Equal(W * H, sink.Y!.Length);
        long sum = 0;
        foreach (var b in sink.Y) sum += b;
        int mean = (int)(sum / sink.Y.Length);
        // Walker now drives real pixels; flat Y=128 should reconstruct exactly.
        True(Math.Abs(mean - 128) <= 4);
    }

    [TestMethod]
    public async Task Av1Decoder_DecodesOwnEncoderKeyframe()
    {
        // Encode a 16x16 flat Y=128 keyframe with our AV1 encoder, then run
        // it through the public Av1Decoder API which now drives the walker.
        // AV1 has a known small per-block drift so the tolerance is wider
        // than VP8/VP9.
        const int W = 16, H = 16;
        var ySrc = new byte[W * H]; Array.Fill(ySrc, (byte)128);
        var uSrc = new byte[(W / 2) * (H / 2)]; Array.Fill(uSrc, (byte)128);
        var vSrc = new byte[(W / 2) * (H / 2)]; Array.Fill(vSrc, (byte)128);
        var frame = SpawnDev.Codecs.Video.Av1.Av1KeyframeEncoder.EncodeKeyFrame(
            ySrc, W, uSrc, W / 2, vSrc, W, H, baseQIndex: 32);

        var sink = new CapturingFrameSink();
        await using var dec = new Av1Decoder();
        int n = await dec.DecodeFrameAsync(frame, sink);

        Equal(1, n);
        Equal(1, sink.FrameCount);
        True(sink.Y is not null);
        Equal(W * H, sink.Y!.Length);
        long sum = 0;
        foreach (var b in sink.Y) sum += b;
        int mean = (int)(sum / sink.Y.Length);
        True(Math.Abs(mean - 128) <= 16);
    }

}
