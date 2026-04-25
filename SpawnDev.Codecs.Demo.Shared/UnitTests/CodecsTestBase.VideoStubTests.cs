using SpawnDev.Codecs.Video;
using SpawnDev.Codecs.Video.Av1;
using SpawnDev.Codecs.Video.Vp8;
using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for the video codec scaffolding. All three decoders currently throw
/// a descriptive <see cref="NotImplementedException"/> from <c>DecodeFrameAsync</c>;
/// construction + dispose + codec identification are expected to work so
/// callers can route by <see cref="VideoCodec"/> enum.
/// </summary>
public abstract partial class CodecsTestBase
{
    private sealed class NoopFrameSink : IVideoFrameSink
    {
        public ValueTask OnFrameAsync(
            ReadOnlyMemory<byte> y, int ys,
            ReadOnlyMemory<byte> u, int us,
            ReadOnlyMemory<byte> v, int vs,
            long pts)
            => ValueTask.CompletedTask;
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
    public async Task Vp8Decoder_DecodeFrame_ThrowsDescriptive()
    {
        await using var dec = new Vp8Decoder();
        bool threw = false;
        try
        {
            await dec.DecodeFrameAsync(new byte[] { 0 }, new NoopFrameSink());
        }
        catch (NotImplementedException ex)
        {
            threw = true;
            Contains("VP8", ex.Message);
        }
        True(threw);
    }

    [TestMethod]
    public async Task Av1Decoder_DecodeFrame_ThrowsDescriptive()
    {
        await using var dec = new Av1Decoder();
        bool threw = false;
        try
        {
            await dec.DecodeFrameAsync(new byte[] { 0 }, new NoopFrameSink());
        }
        catch (NotImplementedException ex)
        {
            threw = true;
            Contains("AV1", ex.Message);
        }
        True(threw);
    }
}
