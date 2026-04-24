using SpawnDev.Codecs.Audio.Aiff;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for <see cref="AiffFileCodec"/>. Covers round-trip for all supported
/// bit depths and channel counts, the IEEE 80-bit extended-precision sample
/// rate field, and the end-to-end AIFF -> FLAC -> AIFF lossless roundtrip.
/// </summary>
public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Aiff_Write_Read_16Bit_Mono_Roundtrips()
    {
        var samples = new[] { 0, 100, -100, 32767, -32768, 1234, -1234 };
        byte[] aiff = AiffFileCodec.Write(samples, 44100, 1, 16);
        var parsed = AiffFileCodec.Read(aiff);
        Equal(44100, parsed.SampleRateHz);
        Equal(1, parsed.Channels);
        Equal(16, parsed.BitsPerSample);
        EqualInts(samples, parsed.InterleavedSamples);
    }

    [TestMethod]
    public void Aiff_Write_Read_24Bit_Stereo_Roundtrips()
    {
        // 24-bit signed: range [-8388608, 8388607].
        var samples = new[] { 0, 0, 8388607, -8388608, 1234567, -1234567 };
        byte[] aiff = AiffFileCodec.Write(samples, 48000, 2, 24);
        var parsed = AiffFileCodec.Read(aiff);
        Equal(48000, parsed.SampleRateHz);
        Equal(2, parsed.Channels);
        Equal(24, parsed.BitsPerSample);
        EqualInts(samples, parsed.InterleavedSamples);
    }

    [TestMethod]
    public void Aiff_Write_Read_8Bit_Roundtrips()
    {
        var samples = new[] { 0, 100, -100, 127, -128, 50 };
        byte[] aiff = AiffFileCodec.Write(samples, 22050, 1, 8);
        var parsed = AiffFileCodec.Read(aiff);
        Equal(22050, parsed.SampleRateHz);
        EqualInts(samples, parsed.InterleavedSamples);
    }

    [TestMethod]
    public void Aiff_Write_Read_32Bit_Roundtrips()
    {
        var samples = new[] { 0, int.MaxValue, int.MinValue, 1_000_000, -1_000_000 };
        byte[] aiff = AiffFileCodec.Write(samples, 96000, 1, 32);
        var parsed = AiffFileCodec.Read(aiff);
        Equal(32, parsed.BitsPerSample);
        EqualInts(samples, parsed.InterleavedSamples);
    }

    [TestMethod]
    public void Aiff_ExtendedFloat_SampleRate_RoundsCorrectly()
    {
        // Exactly integer rates like 44100 and 48000 must round-trip through the
        // 80-bit extended float without loss of integer value.
        int[] rates = { 8000, 11025, 16000, 22050, 32000, 44100, 48000, 88200, 96000, 176400, 192000 };
        foreach (var r in rates)
        {
            byte[] aiff = AiffFileCodec.Write(new int[] { 0 }, r, 1, 16);
            var parsed = AiffFileCodec.Read(aiff);
            Equal(r, parsed.SampleRateHz);
        }
    }

    [TestMethod]
    public void Aiff_BadForm_Throws()
    {
        var bad = new byte[60];
        bad[0] = (byte)'X'; bad[1] = (byte)'O'; bad[2] = (byte)'R'; bad[3] = (byte)'M';
        bool threw = false;
        try { _ = AiffFileCodec.Read(bad); } catch (InvalidDataException) { threw = true; }
        True(threw);
    }

    [TestMethod]
    public void Aiff_NonAlignedSize_Throws()
    {
        bool threw = false;
        try { _ = AiffFileCodec.Write(new int[5], 44100, 2, 16); }
        catch (ArgumentException) { threw = true; }
        True(threw);
    }

    [TestMethod]
    public void Aiff_ToFlacToAiff_RoundtripsLosslessly()
    {
        // AIFF -> FLAC -> AIFF should preserve samples exactly.
        var input = GenerateSineInt(samplesPerChannel: 1024, channels: 2, sampleRateHz: 44100, bps: 16);
        byte[] aiffBytes = AiffFileCodec.Write(input, 44100, 2, 16);
        var aiff = AiffFileCodec.Read(aiffBytes);
        byte[] flacBytes = SpawnDev.Codecs.Audio.Flac.FlacEncoder.EncodeStream(
            aiff.InterleavedSamples, aiff.SampleRateHz, aiff.Channels, aiff.BitsPerSample, blockSize: 1024);
        var flac = SpawnDev.Codecs.Audio.Flac.FlacDecoder.Decode(flacBytes);
        byte[] outAiff = AiffFileCodec.Write(flac.InterleavedSamples, flac.StreamInfo.SampleRateHz,
            flac.StreamInfo.Channels, flac.StreamInfo.BitsPerSample);
        var reparsed = AiffFileCodec.Read(outAiff);
        EqualInts(input, reparsed.InterleavedSamples);
    }

    [TestMethod]
    public void Aiff_5Channels_Roundtrips()
    {
        // AIFF doesn't limit channel count the way WAV does; 5-channel surround works.
        int samplesPerChannel = 32;
        int channels = 5;
        var samples = new int[samplesPerChannel * channels];
        for (int i = 0; i < samples.Length; i++) samples[i] = i * 31 - 100;
        byte[] aiff = AiffFileCodec.Write(samples, 48000, channels, 16);
        var parsed = AiffFileCodec.Read(aiff);
        Equal(5, parsed.Channels);
        EqualInts(samples, parsed.InterleavedSamples);
    }
}
