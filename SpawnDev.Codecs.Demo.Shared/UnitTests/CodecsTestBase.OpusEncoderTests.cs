using SpawnDev.Codecs.Audio.Opus;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// End-to-end tests for <see cref="OpusEncoder"/>. Each test encodes a
/// synthetic signal (sine wave or silence) into an Opus packet and decodes it
/// back through our <see cref="OpusDecoder"/>. The contract verified is the
/// classical Opus round-trip: encode a known frame size, decode the produced
/// packet, get the same number of samples per channel back.
///
/// The encoder currently delegates per-frame work to the BSD-3 Concentus
/// pure-C# port of libopus (see <see cref="OpusEncoder"/> file header), so
/// these tests also implicitly exercise our decoder's bit-exact CELT/Hybrid
/// path against packets produced by Concentus's own encoder. SILK-mode
/// packets (low bitrate VOIP) round-trip through our SILK decoder.
///
/// We keep the suite parameter-rich (multiple frame sizes, sample rates,
/// channel counts, applications) so a regression in any one wiring path
/// surfaces as a focused failure rather than a single coarse smoke test.
/// </summary>
public abstract partial class CodecsTestBase
{
    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    private static float[] OpusEncoderTests_GenerateSine(
        double frequencyHz,
        int sampleRateHz,
        int channelCount,
        int samplesPerChannel,
        double amplitude = 0.4)
    {
        var result = new float[samplesPerChannel * channelCount];
        double k = 2.0 * Math.PI * frequencyHz / sampleRateHz;
        for (int i = 0; i < samplesPerChannel; i++)
        {
            float v = (float)(amplitude * Math.Sin(k * i));
            for (int c = 0; c < channelCount; c++)
                result[i * channelCount + c] = v;
        }
        return result;
    }

    private static int OpusEncoderTests_EncodeAndDecode(
        OpusEncoderApplication application,
        int sampleRateHz,
        int channelCount,
        int frameSizeSamples,
        double frequencyHz,
        out float[] decodedPcm)
    {
        var pcm = OpusEncoderTests_GenerateSine(
            frequencyHz, sampleRateHz, channelCount, frameSizeSamples);

        using var enc = new OpusEncoder(new OpusEncoderConfig
        {
            SampleRateHz = sampleRateHz,
            ChannelCount = channelCount,
            Application = application,
        });

        var packet = new byte[1275];
        int bytes = enc.EncodeFrame(pcm, packet, frameSizeSamples);
        True(bytes > 0 && bytes <= 1275,
            $"encoded packet length {bytes} out of (0, 1275]");

        var dec = new OpusDecoder(new OpusDecoderConfig
        {
            SampleRateHz = sampleRateHz,
            ChannelCount = channelCount,
        });
        decodedPcm = new float[frameSizeSamples * channelCount];
        int samples = dec.DecodePacketAsync(
            packet.AsMemory(0, bytes), decodedPcm.AsMemory()).Result;
        return samples;
    }

    // -----------------------------------------------------------------------
    // Application matrix
    // -----------------------------------------------------------------------

    [TestMethod]
    public void OpusEncoder_Voip_Mono_8kHz_20ms_Roundtrips()
    {
        int samples = OpusEncoderTests_EncodeAndDecode(
            OpusEncoderApplication.Voip,
            sampleRateHz: 8000, channelCount: 1, frameSizeSamples: 160,
            frequencyHz: 440, out var pcm);
        Equal(160, samples);
        for (int i = 0; i < pcm.Length; i++)
            True(pcm[i] >= -1.0f && pcm[i] <= 1.0f, $"pcm[{i}] out of [-1,1]: {pcm[i]}");
    }

    [TestMethod]
    public void OpusEncoder_Audio_Mono_48kHz_20ms_Roundtrips()
    {
        int samples = OpusEncoderTests_EncodeAndDecode(
            OpusEncoderApplication.Audio,
            sampleRateHz: 48000, channelCount: 1, frameSizeSamples: 960,
            frequencyHz: 1000, out var pcm);
        Equal(960, samples);
        for (int i = 0; i < pcm.Length; i++)
            True(pcm[i] >= -1.0f && pcm[i] <= 1.0f);
    }

    [TestMethod]
    public void OpusEncoder_RestrictedLowDelay_Mono_48kHz_20ms_Roundtrips()
    {
        int samples = OpusEncoderTests_EncodeAndDecode(
            OpusEncoderApplication.RestrictedLowDelay,
            sampleRateHz: 48000, channelCount: 1, frameSizeSamples: 960,
            frequencyHz: 880, out var pcm);
        Equal(960, samples);
        for (int i = 0; i < pcm.Length; i++)
            True(pcm[i] >= -1.0f && pcm[i] <= 1.0f);
    }

    // -----------------------------------------------------------------------
    // Frame-size matrix at 48 kHz mono (Audio application)
    // -----------------------------------------------------------------------

    [TestMethod]
    public void OpusEncoder_Audio_Mono_48kHz_2_5ms_Roundtrips()
    {
        // 2.5 ms at 48 kHz = 120 samples/channel.
        int samples = OpusEncoderTests_EncodeAndDecode(
            OpusEncoderApplication.RestrictedLowDelay,
            sampleRateHz: 48000, channelCount: 1, frameSizeSamples: 120,
            frequencyHz: 1000, out var pcm);
        Equal(120, samples);
        Equal(120, pcm.Length);
    }

    [TestMethod]
    public void OpusEncoder_Audio_Mono_48kHz_5ms_Roundtrips()
    {
        // 5 ms at 48 kHz = 240 samples/channel.
        int samples = OpusEncoderTests_EncodeAndDecode(
            OpusEncoderApplication.RestrictedLowDelay,
            sampleRateHz: 48000, channelCount: 1, frameSizeSamples: 240,
            frequencyHz: 1000, out var pcm);
        Equal(240, samples);
        Equal(240, pcm.Length);
    }

    [TestMethod]
    public void OpusEncoder_Audio_Mono_48kHz_10ms_Roundtrips()
    {
        // 10 ms at 48 kHz = 480 samples/channel.
        int samples = OpusEncoderTests_EncodeAndDecode(
            OpusEncoderApplication.Audio,
            sampleRateHz: 48000, channelCount: 1, frameSizeSamples: 480,
            frequencyHz: 1000, out var pcm);
        Equal(480, samples);
        Equal(480, pcm.Length);
    }

    [TestMethod]
    public void OpusEncoder_Audio_Mono_48kHz_20ms_Roundtrips_FrameSizes()
    {
        // 20 ms at 48 kHz = 960 samples/channel.
        int samples = OpusEncoderTests_EncodeAndDecode(
            OpusEncoderApplication.Audio,
            sampleRateHz: 48000, channelCount: 1, frameSizeSamples: 960,
            frequencyHz: 1000, out var pcm);
        Equal(960, samples);
        Equal(960, pcm.Length);
    }

    // -----------------------------------------------------------------------
    // Sample-rate matrix at mono 20 ms (Audio application)
    // -----------------------------------------------------------------------

    [TestMethod]
    public void OpusEncoder_Audio_Mono_8kHz_20ms_Roundtrips()
    {
        // 20 ms at 8 kHz = 160 samples/channel.
        int samples = OpusEncoderTests_EncodeAndDecode(
            OpusEncoderApplication.Audio,
            sampleRateHz: 8000, channelCount: 1, frameSizeSamples: 160,
            frequencyHz: 440, out var pcm);
        Equal(160, samples);
        Equal(160, pcm.Length);
    }

    [TestMethod]
    public void OpusEncoder_Audio_Mono_16kHz_20ms_Roundtrips()
    {
        // 20 ms at 16 kHz = 320 samples/channel.
        int samples = OpusEncoderTests_EncodeAndDecode(
            OpusEncoderApplication.Audio,
            sampleRateHz: 16000, channelCount: 1, frameSizeSamples: 320,
            frequencyHz: 440, out var pcm);
        Equal(320, samples);
        Equal(320, pcm.Length);
    }

    [TestMethod]
    public void OpusEncoder_Audio_Mono_24kHz_20ms_Roundtrips()
    {
        // 20 ms at 24 kHz = 480 samples/channel.
        int samples = OpusEncoderTests_EncodeAndDecode(
            OpusEncoderApplication.Audio,
            sampleRateHz: 24000, channelCount: 1, frameSizeSamples: 480,
            frequencyHz: 440, out var pcm);
        Equal(480, samples);
        Equal(480, pcm.Length);
    }

    // -----------------------------------------------------------------------
    // Stereo coverage
    // -----------------------------------------------------------------------

    [TestMethod]
    public void OpusEncoder_Audio_Stereo_48kHz_20ms_Roundtrips()
    {
        int samples = OpusEncoderTests_EncodeAndDecode(
            OpusEncoderApplication.Audio,
            sampleRateHz: 48000, channelCount: 2, frameSizeSamples: 960,
            frequencyHz: 1000, out var pcm);
        Equal(960, samples);
        Equal(960 * 2, pcm.Length);
    }

    [TestMethod]
    public void OpusEncoder_Voip_Stereo_16kHz_20ms_Roundtrips()
    {
        int samples = OpusEncoderTests_EncodeAndDecode(
            OpusEncoderApplication.Voip,
            sampleRateHz: 16000, channelCount: 2, frameSizeSamples: 320,
            frequencyHz: 440, out var pcm);
        Equal(320, samples);
        Equal(320 * 2, pcm.Length);
    }

    // -----------------------------------------------------------------------
    // Multi-frame stream continuity (encoder state must carry across frames)
    // -----------------------------------------------------------------------

    [TestMethod]
    public void OpusEncoder_Audio_Mono_48kHz_ThreeFrames_Stream_Roundtrips()
    {
        const int sampleRateHz = 48000;
        const int channelCount = 1;
        const int frameSizeSamples = 960; // 20 ms at 48 kHz
        const int frameCount = 3;

        var pcm = OpusEncoderTests_GenerateSine(
            1000, sampleRateHz, channelCount, frameSizeSamples * frameCount);

        using var enc = new OpusEncoder(new OpusEncoderConfig
        {
            SampleRateHz = sampleRateHz,
            ChannelCount = channelCount,
            Application = OpusEncoderApplication.Audio,
        });

        var dec = new OpusDecoder(new OpusDecoderConfig
        {
            SampleRateHz = sampleRateHz,
            ChannelCount = channelCount,
        });

        int totalDecoded = 0;
        var packetBuf = new byte[1275];
        var decodeBuf = new float[frameSizeSamples * channelCount];

        for (int f = 0; f < frameCount; f++)
        {
            int srcOffset = f * frameSizeSamples * channelCount;
            int bytes = enc.EncodeFrame(
                pcm.AsSpan(srcOffset, frameSizeSamples * channelCount),
                packetBuf,
                frameSizeSamples);
            True(bytes > 0, $"frame {f}: encoder returned {bytes} bytes");

            int decoded = dec.DecodePacketAsync(
                packetBuf.AsMemory(0, bytes),
                decodeBuf.AsMemory()).Result;
            Equal(frameSizeSamples, decoded);
            totalDecoded += decoded;
        }

        Equal(frameSizeSamples * frameCount, totalDecoded);
    }

    // -----------------------------------------------------------------------
    // Energy / silence sanity checks
    // -----------------------------------------------------------------------

    [TestMethod]
    public void OpusEncoder_Silence_Roundtrips_NearZero()
    {
        const int sampleRateHz = 48000;
        const int channelCount = 1;
        const int frameSizeSamples = 960;
        var silence = new float[frameSizeSamples * channelCount]; // all 0

        using var enc = new OpusEncoder(new OpusEncoderConfig
        {
            SampleRateHz = sampleRateHz,
            ChannelCount = channelCount,
            Application = OpusEncoderApplication.Audio,
        });

        var packet = new byte[1275];
        int bytes = enc.EncodeFrame(silence, packet, frameSizeSamples);
        True(bytes > 0, $"silence encode returned {bytes}");

        var dec = new OpusDecoder(new OpusDecoderConfig
        {
            SampleRateHz = sampleRateHz,
            ChannelCount = channelCount,
        });
        var decoded = new float[frameSizeSamples * channelCount];
        int samples = dec.DecodePacketAsync(
            packet.AsMemory(0, bytes), decoded.AsMemory()).Result;
        Equal(frameSizeSamples, samples);

        double sumSq = 0;
        for (int i = 0; i < decoded.Length; i++) sumSq += decoded[i] * decoded[i];
        double rms = Math.Sqrt(sumSq / Math.Max(1, decoded.Length));
        True(rms < 0.05, $"silence in -> low-energy out expected, got RMS {rms:F4}");
    }

    [TestMethod]
    public void OpusEncoder_Sine_Roundtrips_NonZeroEnergy()
    {
        // Encode a clearly-audible sine then decode and confirm the decoded
        // signal is not trivially zero. This catches the failure mode where
        // the encoder silently produces a DTX packet for a tone.
        int samples = OpusEncoderTests_EncodeAndDecode(
            OpusEncoderApplication.Audio,
            sampleRateHz: 48000, channelCount: 1, frameSizeSamples: 960,
            frequencyHz: 1000, out var pcm);
        Equal(960, samples);

        double sumSq = 0;
        for (int i = 0; i < pcm.Length; i++) sumSq += pcm[i] * pcm[i];
        double rms = Math.Sqrt(sumSq / Math.Max(1, pcm.Length));
        True(rms > 0.05, $"sine in -> non-trivial signal out expected, got RMS {rms:F4}");
    }
}
