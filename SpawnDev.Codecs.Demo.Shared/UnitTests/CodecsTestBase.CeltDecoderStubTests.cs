using SpawnDev.Codecs.Audio.Opus;
using SpawnDev.Codecs.Audio.Opus.Celt;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for the <see cref="CeltDecoder"/> public surface. The decoder now
/// produces real PCM via the BSD-3 Concentus backbone (see
/// <see cref="CeltDecoder"/> file header for full provenance and the
/// hand-port migration plan); these tests cover the construction surface
/// and basic decode-no-throw behavior. Bit-exact validation lives in
/// <c>CodecsTestBase.OpusDecoderConcentusCrossValidationTests</c> and the
/// new CELT cross-validation file.
/// </summary>
public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void CeltDecoder_Construct_ExposesModeFields()
    {
        var mode = CeltMode.Create(CeltConstants.FRAME_SIZE_20MS, CeltConstants.NB_BANDS_FULLBAND);
        using var dec = new CeltDecoder(mode);
        Equal(48000, dec.SampleRateHz);
        Equal(960, dec.FrameSize);
        Equal(21, dec.EndBand);
    }

    [TestMethod]
    public void CeltDecoder_Construct_DefaultsToMonoAt48k()
    {
        var mode = CeltMode.Create(CeltConstants.FRAME_SIZE_10MS, CeltConstants.NB_BANDS_WB);
        using var dec = new CeltDecoder(mode);
        Equal(1, dec.ChannelCount);
        Equal(48000, dec.SampleRateHz);
        // Mode geometry should be readable from the constructed decoder.
        Equal(480, dec.FrameSize);
        Equal(17, dec.EndBand);
    }

    [TestMethod]
    public void CeltDecoder_NullMode_Throws()
    {
        Throws<ArgumentNullException>(() => new CeltDecoder(null!));
    }

    [TestMethod]
    public void CeltDecoder_InvalidSampleRate_Throws()
    {
        var mode = CeltMode.Create(CeltConstants.FRAME_SIZE_20MS, CeltConstants.NB_BANDS_FULLBAND);
        Throws<ArgumentOutOfRangeException>(() => new CeltDecoder(mode, outputSampleRateHz: 44100));
    }

    [TestMethod]
    public void CeltDecoder_InvalidChannelCount_Throws()
    {
        var mode = CeltMode.Create(CeltConstants.FRAME_SIZE_20MS, CeltConstants.NB_BANDS_FULLBAND);
        Throws<ArgumentOutOfRangeException>(() => new CeltDecoder(mode, channelCount: 0));
        Throws<ArgumentOutOfRangeException>(() => new CeltDecoder(mode, channelCount: 3));
    }

    [TestMethod]
    public void CeltDecoder_DecodeFrame_ChannelMismatch_Throws()
    {
        var mode = CeltMode.Create(CeltConstants.FRAME_SIZE_20MS, CeltConstants.NB_BANDS_FULLBAND);
        using var dec = new CeltDecoder(mode, outputSampleRateHz: 48000, channelCount: 1);
        // Crafted CELT TOC, mono.
        byte tocByte = (byte)(31 << 3);
        byte[] packet = { tocByte, 0x00, 0x00 };
        // Invoking DecodeFrame with channels=2 but decoder is mono should reject.
        Throws<ArgumentException>(() => dec.DecodeFrame(packet, new float[960], channels: 2));
    }

    [TestMethod]
    public void CeltDecoder_DisposedThenUsed_Throws()
    {
        var mode = CeltMode.Create(CeltConstants.FRAME_SIZE_20MS, CeltConstants.NB_BANDS_FULLBAND);
        var dec = new CeltDecoder(mode);
        dec.Dispose();
        Throws<ObjectDisposedException>(() =>
            dec.DecodePacket(new byte[] { 0xF8, 0, 0 }, new float[960], 960));
    }

    [TestMethod]
    public void OpusDecoder_CeltPacket_ProducesValidPcm()
    {
        // Generate a CELT-mode Opus packet via Concentus by encoding music-like
        // audio with the AUDIO application hint at a high enough bitrate. We
        // verify the packet's mode then run our OpusDecoder and confirm it
        // produces sample-range-valid PCM (no NotImplementedException).
        var pcm = ReferenceOracle.GenerateSineWave(1000, 48000, 1, 960);
        byte[] packet = ReferenceOracle.EncodeFrame(
            pcm,
            sampleRateHz: 48000,
            channelCount: 1,
            frameSizeSamples: 960,
            application: Concentus.Enums.OpusApplication.OPUS_APPLICATION_RESTRICTED_LOWDELAY,
            bitrateBitsPerSecond: 64000);

        var toc = new OpusTocByte(packet[0]);
        // RESTRICTED_LOWDELAY forces CELT mode.
        if (toc.Mode != SpawnDev.Codecs.Audio.Opus.OpusMode.Celt)
        {
            throw new UnsupportedTestException(
                $"Concentus chose {toc.Mode} for RESTRICTED_LOWDELAY; CELT-routing test needs CELT mode.");
        }

        var config = new OpusDecoderConfig { SampleRateHz = 48000, ChannelCount = 1 };
        var dec = new OpusDecoder(config);
        float[] ourPcm = new float[960];
        int samples = dec.DecodePacketAsync(packet.AsMemory(), ourPcm.AsMemory()).Result;

        Equal(960, samples);
        for (int i = 0; i < samples; i++)
        {
            True(ourPcm[i] >= -1.0f && ourPcm[i] <= 1.0f, $"our pcm[{i}] = {ourPcm[i]} out of [-1, 1]");
        }
    }
}
