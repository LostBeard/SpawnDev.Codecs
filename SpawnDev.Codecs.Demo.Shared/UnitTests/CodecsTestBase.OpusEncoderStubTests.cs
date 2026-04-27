using SpawnDev.Codecs.Audio.Opus;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for the top-level <see cref="OpusEncoder"/>. The encoder is
/// scaffolded - per-mode SILK and CELT encoders are not yet implemented -
/// so EncodeFrame throws NotImplementedException with a descriptive message.
/// These tests verify configuration validation, public API shape, and that
/// the failure mode is the documented one (rather than a cryptic crash).
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
    public void OpusEncoder_EncodeFrame_ThrowsNotImplemented()
    {
        var enc = new OpusEncoder(new OpusEncoderConfig
        {
            SampleRateHz = 48000,
            ChannelCount = 1,
        });
        var pcm = new float[960];
        var packet = new byte[1275];
        bool threw = false;
        try
        {
            enc.EncodeFrame(pcm, packet, frameSizeSamples: 960);
        }
        catch (NotImplementedException ex)
        {
            threw = true;
            True(ex.Message.Contains("not yet implemented"),
                $"Expected message to explain status; got: {ex.Message}");
        }
        True(threw, "EncodeFrame should throw NotImplementedException");
    }
}
