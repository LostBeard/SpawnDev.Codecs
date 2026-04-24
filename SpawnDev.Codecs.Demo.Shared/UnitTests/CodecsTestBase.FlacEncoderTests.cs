using SpawnDev.Codecs.Audio.Flac;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Roundtrip tests for the minimal <see cref="FlacEncoder"/>. Each test
/// generates PCM samples, encodes to a FLAC byte stream, decodes via the
/// public <see cref="FlacDecoder"/>, and asserts the decoded samples match
/// the originals exactly (lossless contract). This also proves the encoder's
/// CRC-8 / CRC-16 / metadata / frame-layout are spec-compliant because the
/// decoder validates all of them.
/// </summary>
public abstract partial class CodecsTestBase
{
    private static int[] GenerateSineInt(int samplesPerChannel, int channels, int sampleRateHz, int bps,
        double frequencyHz = 440.0, double amplitudeFraction = 0.5)
    {
        int maxValue = (1 << (bps - 1)) - 1;
        int amplitude = (int)(maxValue * amplitudeFraction);
        var samples = new int[samplesPerChannel * channels];
        for (int n = 0; n < samplesPerChannel; n++)
        {
            double phase = 2.0 * Math.PI * frequencyHz * n / sampleRateHz;
            int v = (int)(Math.Sin(phase) * amplitude);
            for (int ch = 0; ch < channels; ch++)
                samples[n * channels + ch] = v;
        }
        return samples;
    }

    [TestMethod]
    public void FlacEncoder_Mono_Silence_Roundtrips()
    {
        var input = new int[512]; // all zero
        byte[] encoded = FlacEncoder.EncodeStream(input, sampleRateHz: 44100, channels: 1, bitsPerSample: 16, blockSize: 256);
        var decoded = FlacDecoder.Decode(encoded);
        Equal(1, decoded.StreamInfo.Channels);
        Equal(44100, decoded.StreamInfo.SampleRateHz);
        Equal(16, decoded.StreamInfo.BitsPerSample);
        Equal(512, decoded.TotalSamplesPerChannel);
        EqualInts(input, decoded.InterleavedSamples);
    }

    [TestMethod]
    public void FlacEncoder_Mono_Sine_Roundtrips_ExactlyLossless()
    {
        var input = GenerateSineInt(samplesPerChannel: 1024, channels: 1, sampleRateHz: 44100, bps: 16);
        byte[] encoded = FlacEncoder.EncodeStream(input, 44100, 1, 16, blockSize: 256);
        var decoded = FlacDecoder.Decode(encoded);
        Equal(1024, decoded.TotalSamplesPerChannel);
        EqualInts(input, decoded.InterleavedSamples);
    }

    [TestMethod]
    public void FlacEncoder_Stereo_Sine_Roundtrips()
    {
        var input = GenerateSineInt(samplesPerChannel: 512, channels: 2, sampleRateHz: 48000, bps: 16);
        byte[] encoded = FlacEncoder.EncodeStream(input, 48000, 2, 16, blockSize: 128);
        var decoded = FlacDecoder.Decode(encoded);
        Equal(2, decoded.StreamInfo.Channels);
        Equal(48000, decoded.StreamInfo.SampleRateHz);
        Equal(512, decoded.TotalSamplesPerChannel);
        EqualInts(input, decoded.InterleavedSamples);
    }

    [TestMethod]
    public void FlacEncoder_24Bit_Roundtrips()
    {
        var input = GenerateSineInt(samplesPerChannel: 300, channels: 1, sampleRateHz: 96000, bps: 24);
        byte[] encoded = FlacEncoder.EncodeStream(input, 96000, 1, 24, blockSize: 128);
        var decoded = FlacDecoder.Decode(encoded);
        Equal(24, decoded.StreamInfo.BitsPerSample);
        Equal(96000, decoded.StreamInfo.SampleRateHz);
        EqualInts(input, decoded.InterleavedSamples);
    }

    [TestMethod]
    public void FlacEncoder_8Bit_Roundtrips()
    {
        var input = GenerateSineInt(samplesPerChannel: 200, channels: 1, sampleRateHz: 8000, bps: 8);
        byte[] encoded = FlacEncoder.EncodeStream(input, 8000, 1, 8, blockSize: 64);
        var decoded = FlacDecoder.Decode(encoded);
        Equal(8, decoded.StreamInfo.BitsPerSample);
        Equal(8000, decoded.StreamInfo.SampleRateHz);
        EqualInts(input, decoded.InterleavedSamples);
    }

    [TestMethod]
    public void FlacEncoder_MultipleFrames_Concatenate()
    {
        // 10 frames of 64 samples each = 640 samples total.
        var input = GenerateSineInt(samplesPerChannel: 640, channels: 1, sampleRateHz: 44100, bps: 16);
        byte[] encoded = FlacEncoder.EncodeStream(input, 44100, 1, 16, blockSize: 64);
        var decoded = FlacDecoder.Decode(encoded);
        Equal(640, decoded.TotalSamplesPerChannel);
        EqualInts(input, decoded.InterleavedSamples);
    }

    [TestMethod]
    public void FlacEncoder_PartialLastBlock_Roundtrips()
    {
        // 500 samples with blockSize 128 → 3 full blocks + 1 partial block of 116 samples.
        var input = GenerateSineInt(samplesPerChannel: 500, channels: 1, sampleRateHz: 44100, bps: 16);
        byte[] encoded = FlacEncoder.EncodeStream(input, 44100, 1, 16, blockSize: 128);
        var decoded = FlacDecoder.Decode(encoded);
        Equal(500, decoded.TotalSamplesPerChannel);
        EqualInts(input, decoded.InterleavedSamples);
    }

    [TestMethod]
    public void FlacEncoder_FourChannel_Roundtrips()
    {
        // 4-channel surround: generate 256 samples with unique data per channel.
        int samplesPerChannel = 256;
        int channels = 4;
        var input = new int[samplesPerChannel * channels];
        for (int n = 0; n < samplesPerChannel; n++)
        {
            for (int ch = 0; ch < channels; ch++)
                input[n * channels + ch] = (ch + 1) * 100 + (n % 31) - 15;
        }
        byte[] encoded = FlacEncoder.EncodeStream(input, 44100, channels, 16, blockSize: 128);
        var decoded = FlacDecoder.Decode(encoded);
        Equal(4, decoded.StreamInfo.Channels);
        EqualInts(input, decoded.InterleavedSamples);
    }

    [TestMethod]
    public void FlacEncoder_CustomSampleRate_WithSideBytes_Roundtrips()
    {
        // 37500 Hz does not match any fixed code; must use 16-bit side byte via code 0b1101.
        var input = GenerateSineInt(samplesPerChannel: 256, channels: 1, sampleRateHz: 37500, bps: 16);
        byte[] encoded = FlacEncoder.EncodeStream(input, 37500, 1, 16, blockSize: 128);
        var decoded = FlacDecoder.Decode(encoded);
        Equal(37500, decoded.StreamInfo.SampleRateHz);
        EqualInts(input, decoded.InterleavedSamples);
    }

    [TestMethod]
    public void FlacEncoder_CustomBlockSize_WithSideBytes_Roundtrips()
    {
        // 100-sample block size: not in fixed table, use 8-bit side via code 0b0110.
        var input = GenerateSineInt(samplesPerChannel: 400, channels: 1, sampleRateHz: 44100, bps: 16);
        byte[] encoded = FlacEncoder.EncodeStream(input, 44100, 1, 16, blockSize: 100);
        var decoded = FlacDecoder.Decode(encoded);
        Equal(400, decoded.TotalSamplesPerChannel);
        EqualInts(input, decoded.InterleavedSamples);
    }

    [TestMethod]
    public void FlacEncoder_ValidatesBadChannels_Throws()
    {
        bool threw = false;
        try { _ = FlacEncoder.EncodeStream(new int[100], 44100, 0, 16); }
        catch (ArgumentException) { threw = true; }
        True(threw, "0 channels should throw.");
    }

    [TestMethod]
    public void FlacEncoder_ValidatesBadBps_Throws()
    {
        bool threw = false;
        try { _ = FlacEncoder.EncodeStream(new int[100], 44100, 1, 11); }
        catch (ArgumentException) { threw = true; }
        True(threw, "Unsupported bps 11 should throw.");
    }

    [TestMethod]
    public void FlacEncoder_StreamInfo_RecordsTotalSamples()
    {
        var input = GenerateSineInt(samplesPerChannel: 777, channels: 1, sampleRateHz: 44100, bps: 16);
        byte[] encoded = FlacEncoder.EncodeStream(input, 44100, 1, 16, blockSize: 128);
        var decoded = FlacDecoder.Decode(encoded);
        Equal(777UL, decoded.StreamInfo.TotalSamples);
    }
}
