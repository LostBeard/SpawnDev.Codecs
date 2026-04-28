using SpawnDev.Codecs.Audio.Opus;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Configuration-shape tests for <see cref="OpusEncoder"/>. The "stub" name is
/// historical - as of the encoder-wiring slice the encoder is fully working
/// (see <see cref="CodecsTestBase"/>'s <c>CodecsTestBase.OpusEncoderTests.cs</c>
/// partial for round-trip coverage). These tests stay focused on the
/// public-API contracts that are independent of which backbone is wired
/// internally.
/// </summary>
public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void OpusEncoder_Construct_ExposesCodecAndConfig()
    {
        var enc = new OpusEncoder(new OpusEncoderConfig
        {
            SampleRateHz = 48000,
            ChannelCount = 1,
        });
        Equal(SpawnDev.Codecs.Audio.AudioCodec.Opus, enc.Codec);
        Equal(48000, enc.SampleRateHz);
        Equal(1, enc.ChannelCount);
    }

    [TestMethod]
    public void OpusEncoder_Construct_BadSampleRate_Throws()
    {
        Throws<ArgumentException>(() => new OpusEncoder(new OpusEncoderConfig
        {
            SampleRateHz = 11025,
            ChannelCount = 1,
        }));
    }

    [TestMethod]
    public void OpusEncoder_Construct_BadChannelCount_Throws()
    {
        Throws<ArgumentException>(() => new OpusEncoder(new OpusEncoderConfig
        {
            SampleRateHz = 48000,
            ChannelCount = 3,
        }));
    }

    [TestMethod]
    public void OpusEncoder_EncodeFrame_BadFrameSize_Throws()
    {
        var enc = new OpusEncoder(new OpusEncoderConfig
        {
            SampleRateHz = 48000,
            ChannelCount = 1,
        });
        var pcm = new float[960];
        var packet = new byte[1275];
        Throws<ArgumentOutOfRangeException>(
            () => enc.EncodeFrame(pcm, packet, frameSizeSamples: 0));
    }

    [TestMethod]
    public void OpusEncoder_EncodeFrame_PcmTooShort_Throws()
    {
        var enc = new OpusEncoder(new OpusEncoderConfig
        {
            SampleRateHz = 48000,
            ChannelCount = 2,
        });
        // Need 960 * 2 = 1920 samples; provide only 960 -> ArgumentException.
        var pcm = new float[960];
        var packet = new byte[1275];
        Throws<ArgumentException>(
            () => enc.EncodeFrame(pcm, packet, frameSizeSamples: 960));
    }

    [TestMethod]
    public void OpusEncoder_AfterDispose_Throws()
    {
        var enc = new OpusEncoder(new OpusEncoderConfig
        {
            SampleRateHz = 48000,
            ChannelCount = 1,
        });
        enc.Dispose();
        var pcm = new float[960];
        var packet = new byte[1275];
        Throws<ObjectDisposedException>(
            () => enc.EncodeFrame(pcm, packet, frameSizeSamples: 960));
    }
}
