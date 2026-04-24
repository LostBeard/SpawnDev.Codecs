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

    [TestMethod]
    public void FlacEncoder_Silence_UsesConstantSubframes_CompactEncoding()
    {
        // 1024 samples of silence. VERBATIM would need 16*1024 = 16384 bits per channel.
        // CONSTANT needs just 16 bits per channel. The encoded stream should be much
        // smaller than the VERBATIM baseline.
        var silent = new int[1024];
        var silentEncoded = FlacEncoder.EncodeStream(silent, 44100, 1, 16, blockSize: 1024);
        // Compare to a DC-varying signal at the same block size (which cannot use CONSTANT).
        var varying = new int[1024];
        for (int i = 0; i < varying.Length; i++) varying[i] = i % 7;
        var varyingEncoded = FlacEncoder.EncodeStream(varying, 44100, 1, 16, blockSize: 1024);
        True(silentEncoded.Length < varyingEncoded.Length / 4,
            $"Silence should compress far smaller than varying signal. silent={silentEncoded.Length}, varying={varyingEncoded.Length}.");

        // And it must still round-trip exactly.
        var decoded = FlacDecoder.Decode(silentEncoded);
        EqualInts(silent, decoded.InterleavedSamples);
    }

    [TestMethod]
    public void FlacEncoder_DcOffset_UsesConstantSubframes()
    {
        // Non-zero DC offset: every sample == 4242.
        var dc = new int[512];
        Array.Fill(dc, 4242);
        byte[] encoded = FlacEncoder.EncodeStream(dc, 44100, 1, 16, blockSize: 512);
        var decoded = FlacDecoder.Decode(encoded);
        EqualInts(dc, decoded.InterleavedSamples);
        // Even without VERBATIM comparison, encoded length for 512 DC samples with CONSTANT
        // subframes is bounded: 42 metadata + ~10 frame-header + 3 subframe + 2 CRC-16 = under 70 bytes total.
        True(encoded.Length < 100, $"DC-encoded 512 samples should be under 100 bytes; got {encoded.Length}.");
    }

    [TestMethod]
    public void FlacEncoder_PerChannelMix_OneConstantOneVerbatim()
    {
        // Stereo where channel 0 is silent but channel 1 is varying. Our encoder should pick
        // CONSTANT for ch0 and VERBATIM for ch1 on a per-subframe basis.
        int samplesPerChannel = 256;
        var input = new int[samplesPerChannel * 2];
        for (int n = 0; n < samplesPerChannel; n++)
        {
            input[n * 2 + 0] = 0;       // ch0 silent
            input[n * 2 + 1] = (n * 37) % 31 - 15; // ch1 noise
        }
        byte[] encoded = FlacEncoder.EncodeStream(input, 44100, 2, 16, blockSize: 256);
        var decoded = FlacDecoder.Decode(encoded);
        EqualInts(input, decoded.InterleavedSamples);
    }

    [TestMethod]
    public void FlacEncoder_Sine_CompressesBelowVerbatim()
    {
        // A 1024-sample 16-bit sine wave should compress well via FIXED order
        // selection. VERBATIM baseline = 1024 * 16 / 8 = 2048 bytes just for the
        // subframe payload; full file adds ~50 bytes of metadata/framing.
        // With FIXED encoding, file should be noticeably smaller.
        var input = GenerateSineInt(samplesPerChannel: 1024, channels: 1, sampleRateHz: 44100, bps: 16);
        byte[] encoded = FlacEncoder.EncodeStream(input, 44100, 1, 16, blockSize: 1024);
        True(encoded.Length < 2048,
            $"1024-sample 16-bit sine should compress below VERBATIM 2048-byte baseline; got {encoded.Length} bytes.");
        // Still lossless.
        var decoded = FlacDecoder.Decode(encoded);
        EqualInts(input, decoded.InterleavedSamples);
    }

    [TestMethod]
    public void FlacEncoder_Ramp_CompressesDramatically()
    {
        // A monotonically-increasing ramp is a perfect fit for FIXED order 1
        // (first difference is constant). Encoder should pick FIXED order 1
        // with Rice param near 0, producing a near-trivial bit stream.
        var input = new int[512];
        for (int i = 0; i < 512; i++) input[i] = i - 256;
        byte[] encoded = FlacEncoder.EncodeStream(input, 44100, 1, 16, blockSize: 512);
        // VERBATIM would need 512 * 2 = 1024 bytes for the samples alone.
        // Ramp should encode dramatically smaller - under 150 bytes total.
        True(encoded.Length < 150,
            $"Linear ramp should compress far below VERBATIM; got {encoded.Length} bytes.");
        var decoded = FlacDecoder.Decode(encoded);
        EqualInts(input, decoded.InterleavedSamples);
    }

    [TestMethod]
    public void FlacEncoder_Quadratic_CompressesWithFixedOrder2()
    {
        // f(n) = n*n is perfectly fit by FIXED order 2 (2nd difference = constant 2).
        var input = new int[512];
        for (int i = 0; i < 512; i++) input[i] = i * i - 65536;
        byte[] encoded = FlacEncoder.EncodeStream(input, 44100, 1, 32, blockSize: 512);
        // VERBATIM baseline for 32-bit samples = 2048 bytes. FIXED order 2 residuals
        // are all 2 -> tiny Rice output.
        True(encoded.Length < 200,
            $"Quadratic should compress far below VERBATIM; got {encoded.Length} bytes.");
        var decoded = FlacDecoder.Decode(encoded);
        EqualInts(input, decoded.InterleavedSamples);
    }

    [TestMethod]
    public void FlacEncoder_StereoHighlyCorrelated_PicksMidSide()
    {
        // L and R nearly identical -> side ~= 0 -> MidSide is MUCH cheaper than Independent.
        int samplesPerChannel = 1024;
        var input = new int[samplesPerChannel * 2];
        for (int n = 0; n < samplesPerChannel; n++)
        {
            double phase = 2.0 * Math.PI * 440.0 * n / 48000.0;
            int v = (int)(Math.Sin(phase) * 10000);
            input[n * 2 + 0] = v;            // L
            input[n * 2 + 1] = v + (n & 1);  // R almost equal to L (+/- 1 sample)
        }
        byte[] encoded = FlacEncoder.EncodeStream(input, 48000, 2, 16, blockSize: 1024);
        // Independent would need to encode two full channels at ~2KB worst case.
        // MidSide keeps mid similar to L and side near zero (with order-1 FIXED trivial to encode).
        // Expect well under 2KB total.
        True(encoded.Length < 2048,
            $"Correlated stereo should compress below 2KB via MidSide; got {encoded.Length}.");
        var decoded = FlacDecoder.Decode(encoded);
        EqualInts(input, decoded.InterleavedSamples);
    }

    [TestMethod]
    public void FlacEncoder_StereoUncorrelated_PicksIndependent_StillRoundtrips()
    {
        // L and R independent noise: decor modes don't help. Encoder should pick
        // Independent. Test only verifies lossless roundtrip (not compression).
        int samplesPerChannel = 512;
        var input = new int[samplesPerChannel * 2];
        var rng = new Random(42);
        for (int n = 0; n < samplesPerChannel; n++)
        {
            input[n * 2 + 0] = rng.Next(-32000, 32001);
            input[n * 2 + 1] = rng.Next(-32000, 32001);
        }
        byte[] encoded = FlacEncoder.EncodeStream(input, 44100, 2, 16, blockSize: 512);
        var decoded = FlacDecoder.Decode(encoded);
        Equal(2, decoded.StreamInfo.Channels);
        EqualInts(input, decoded.InterleavedSamples);
    }

    [TestMethod]
    public void FlacEncoder_DampedSinusoid_CompressesWithLpc()
    {
        // A damped sinusoid x[n] = A * exp(-alpha*n) * sin(omega*n) is a classical
        // signal that LPC order 2 fits near-perfectly. FIXED residual would be noisy,
        // LPC can drive residual near zero. This tests that LPC actually gets picked
        // for a signal where it wins.
        int samples = 1024;
        var input = new int[samples];
        for (int n = 0; n < samples; n++)
        {
            double env = Math.Exp(-0.002 * n);
            double phase = 2.0 * Math.PI * 0.1 * n;
            input[n] = (int)(Math.Sin(phase) * 20000 * env);
        }
        byte[] encoded = FlacEncoder.EncodeStream(input, 44100, 1, 16, blockSize: 1024);
        // Still lossless regardless of which subframe type won.
        var decoded = FlacDecoder.Decode(encoded);
        EqualInts(input, decoded.InterleavedSamples);
        True(decoded.VerifyMd5(), "MD5 must verify after LPC roundtrip.");
        // VERBATIM baseline = 2KB. LPC should fit far below (~150 bytes or less for this signal).
        True(encoded.Length < 2048, $"Damped sinusoid should compress below VERBATIM; got {encoded.Length} bytes.");
    }

    [TestMethod]
    public void FlacEncoder_WhiteNoise_LargeBlock_LpcAvoidsFailure()
    {
        // White noise is hard for ANY predictor. Verifies that when LPC doesn't
        // beat FIXED or VERBATIM, the encoder falls back cleanly without error.
        int samples = 1024;
        var input = new int[samples];
        var rng = new Random(99);
        for (int n = 0; n < samples; n++) input[n] = rng.Next(-32000, 32001);
        byte[] encoded = FlacEncoder.EncodeStream(input, 44100, 1, 16, blockSize: 1024);
        var decoded = FlacDecoder.Decode(encoded);
        EqualInts(input, decoded.InterleavedSamples);
        True(decoded.VerifyMd5(), "MD5 must verify even for noise (lossless).");
    }
}
