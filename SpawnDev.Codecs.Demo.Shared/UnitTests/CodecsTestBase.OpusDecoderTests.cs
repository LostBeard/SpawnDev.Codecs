using SpawnDev.Codecs.Audio;
using SpawnDev.Codecs.Audio.Opus;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Skeleton-level tests for <see cref="OpusDecoder"/>. Phase 1a state: packet parsing +
/// mode routing verified; SILK and CELT decode paths are expected to throw
/// <see cref="NotImplementedException"/>. When later slices wire the real decoders,
/// the NotImplementedException assertions become bit-exact RFC 6716 conformance checks.
/// </summary>
public abstract partial class CodecsTestBase
{
    // -------- Config validation --------

    [TestMethod]
    public void OpusDecoder_Config_InvalidSampleRate_Throws()
    {
        var config = new OpusDecoderConfig { SampleRateHz = 44100, ChannelCount = 1 };
        Throws<ArgumentOutOfRangeException>(() => new OpusDecoder(config));
    }

    [TestMethod]
    public void OpusDecoder_Config_InvalidChannelCount_Throws()
    {
        var config = new OpusDecoderConfig { SampleRateHz = 48000, ChannelCount = 3 };
        Throws<ArgumentOutOfRangeException>(() => new OpusDecoder(config));
    }

    [TestMethod]
    public void OpusDecoder_Config_NullThrows()
    {
        Throws<ArgumentNullException>(() => new OpusDecoder(null!));
    }

    [TestMethod]
    public void OpusDecoder_Config_ValidSampleRates_AllSupported()
    {
        int[] rates = { 8000, 12000, 16000, 24000, 48000 };
        foreach (int hz in rates)
        {
            var config = new OpusDecoderConfig { SampleRateHz = hz, ChannelCount = 2 };
            var dec = new OpusDecoder(config);
            Equal(hz, dec.SampleRateHz, $"SampleRate {hz}");
            Equal(2, dec.ChannelCount, $"ChannelCount for {hz}");
        }
    }

    [TestMethod]
    public void OpusDecoder_Codec_IsOpus()
    {
        var dec = new OpusDecoder(new OpusDecoderConfig { SampleRateHz = 48000, ChannelCount = 1 });
        Equal(AudioCodec.Opus, dec.Codec);
    }

    // -------- Factory --------

    [TestMethod]
    public void OpusCodec_CreateDecoder_ReturnsIAudioDecoder()
    {
        var decoder = OpusCodec.CreateDecoder(new OpusDecoderConfig { SampleRateHz = 48000, ChannelCount = 2 });
        NotNull(decoder);
        Equal(AudioCodec.Opus, decoder.Codec);
        Equal(48000, decoder.SampleRateHz);
        Equal(2, decoder.ChannelCount);
    }

    // -------- Packet handling --------

    [TestMethod]
    public async Task OpusDecoder_DecodePacket_InvalidPacket_Throws()
    {
        var dec = new OpusDecoder(new OpusDecoderConfig { SampleRateHz = 48000, ChannelCount = 1 });
        try
        {
            await dec.DecodePacketAsync(ReadOnlyMemory<byte>.Empty, new float[960]);
            throw new Exception("Expected ArgumentException for empty packet");
        }
        catch (ArgumentException) { }
    }

    [TestMethod]
    public async Task OpusDecoder_DecodePacket_BufferTooSmall_Throws()
    {
        var dec = new OpusDecoder(new OpusDecoderConfig { SampleRateHz = 48000, ChannelCount = 1 });
        // TOC config 0 = SILK NB 10ms mono = 480 samples @ 48kHz
        byte[] packet = { 0x00, 0x00, 0x00, 0x00 };
        try
        {
            // Buffer of 100 floats is way less than 480 required.
            await dec.DecodePacketAsync(packet, new float[100]);
            throw new Exception("Expected ArgumentException for too-small buffer");
        }
        catch (ArgumentException) { }
    }

    [TestMethod]
    public async Task OpusDecoder_DecodePacket_SilkPath_StubsWithNotImplementedException()
    {
        // TOC config 0: SILK NB 10ms mono = 480 samples @ 48kHz
        var dec = new OpusDecoder(new OpusDecoderConfig { SampleRateHz = 48000, ChannelCount = 1 });
        byte[] packet = { 0x00, 0x00, 0x00, 0x00 };
        try
        {
            await dec.DecodePacketAsync(packet, new float[960]);
            throw new Exception("Expected NotImplementedException from SILK stub");
        }
        catch (NotImplementedException ex)
        {
            Contains("SILK", ex.Message);
        }
    }

    [TestMethod]
    public async Task OpusDecoder_DecodePacket_HybridPath_StubsWithNotImplementedException()
    {
        // TOC config 12: Hybrid SWB 10ms mono = 480 samples @ 48kHz
        var dec = new OpusDecoder(new OpusDecoderConfig { SampleRateHz = 48000, ChannelCount = 1 });
        byte[] packet = { 0x60, 0x00, 0x00, 0x00 }; // config 12 << 3 = 0x60
        try
        {
            await dec.DecodePacketAsync(packet, new float[960]);
            throw new Exception("Expected NotImplementedException from Hybrid stub");
        }
        catch (NotImplementedException ex)
        {
            Contains("Hybrid", ex.Message);
        }
    }

    [TestMethod]
    public async Task OpusDecoder_DecodePacket_CeltPath_StubsWithNotImplementedException()
    {
        // TOC config 16: CELT NB 2.5ms mono = 120 samples @ 48kHz
        var dec = new OpusDecoder(new OpusDecoderConfig { SampleRateHz = 48000, ChannelCount = 1 });
        byte[] packet = { 0x80, 0x00, 0x00, 0x00 }; // config 16 << 3 = 0x80
        try
        {
            await dec.DecodePacketAsync(packet, new float[960]);
            throw new Exception("Expected NotImplementedException from CELT stub");
        }
        catch (NotImplementedException ex)
        {
            Contains("CELT", ex.Message);
        }
    }

    // -------- Disposal --------

    [TestMethod]
    public async Task OpusDecoder_DecodePacket_AfterDispose_Throws()
    {
        var dec = new OpusDecoder(new OpusDecoderConfig { SampleRateHz = 48000, ChannelCount = 1 });
        await dec.DisposeAsync();
        try
        {
            await dec.DecodePacketAsync(new byte[] { 0x00, 0x00 }, new float[480]);
            throw new Exception("Expected ObjectDisposedException");
        }
        catch (ObjectDisposedException) { }
    }

    // -------- Short overload path --------

    [TestMethod]
    public async Task OpusDecoder_DecodePacket_ShortOverload_InvalidPacket_Throws()
    {
        var dec = new OpusDecoder(new OpusDecoderConfig { SampleRateHz = 48000, ChannelCount = 1 });
        try
        {
            await dec.DecodePacketAsync(ReadOnlyMemory<byte>.Empty, new short[960]);
            throw new Exception("Expected ArgumentException from short overload");
        }
        catch (ArgumentException) { }
    }
}
