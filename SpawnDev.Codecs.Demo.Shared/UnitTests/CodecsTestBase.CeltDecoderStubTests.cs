using SpawnDev.Codecs.Audio.Opus;
using SpawnDev.Codecs.Audio.Opus.Celt;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for the <see cref="CeltDecoder"/> stub. Verifies the public API is
/// usable and that decode attempts throw <see cref="NotImplementedException"/>
/// with a descriptive message rather than a cryptic deeper failure.
/// </summary>
public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void CeltDecoder_Construct_ExposesModeFields()
    {
        var mode = CeltMode.Create(CeltConstants.FRAME_SIZE_20MS, CeltConstants.NB_BANDS_FULLBAND);
        var dec = new CeltDecoder(mode);
        Equal(48000, dec.SampleRateHz);
        Equal(960, dec.FrameSize);
        Equal(21, dec.EndBand);
    }

    [TestMethod]
    public void CeltDecoder_DecodeFrame_ThrowsNotImplementedWithContext()
    {
        var mode = CeltMode.Create(CeltConstants.FRAME_SIZE_10MS, CeltConstants.NB_BANDS_WB);
        var dec = new CeltDecoder(mode);

        bool threw = false;
        try
        {
            dec.DecodeFrame(new byte[32], new float[480], channels: 1);
        }
        catch (NotImplementedException ex)
        {
            threw = true;
            // Message should mention FrameSize and EndBand for diagnostic context.
            True(ex.Message.Contains("480"), $"Expected message to mention frame size 480; got: {ex.Message}");
            True(ex.Message.Contains("17"), $"Expected message to mention EndBand 17; got: {ex.Message}");
        }
        True(threw, "CeltDecoder.DecodeFrame should throw NotImplementedException");
    }

    [TestMethod]
    public void CeltDecoder_NullMode_Throws()
    {
        Throws<ArgumentNullException>(() => new CeltDecoder(null!));
    }

    [TestMethod]
    public void OpusDecoder_CeltPacket_ThrowsClearNotImplemented()
    {
        // Craft a minimal CELT TOC (config 29 = CELT WB 5 ms, mono).
        byte tocByte = (byte)(29 << 3);
        byte[] packet = new byte[] { tocByte, 0x00, 0x00 };

        var config = new OpusDecoderConfig { SampleRateHz = 48000, ChannelCount = 1 };
        var dec = new OpusDecoder(config);
        float[] pcm = new float[960];

        bool threw = false;
        try
        {
            _ = dec.DecodePacketAsync(packet.AsMemory(), pcm.AsMemory()).Result;
        }
        catch (NotImplementedException) { threw = true; }
        catch (AggregateException ae) when (ae.InnerException is NotImplementedException) { threw = true; }

        True(threw, "CELT-mode Opus packet should throw NotImplementedException");
    }
}
