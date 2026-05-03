// Tests that mirror the pipeline used by the AudioTranscode demo page
// (Pages/AudioTranscode.razor in SpawnDev.Codecs.Demo). Verifies the exact
// encode -> decode flow the page wires together so the demo can be trusted
// to produce a sane round-trip without manual browser testing every commit.

using SpawnDev.Codecs.Audio.Flac;
using SpawnDev.Codecs.Audio.Opus;
using SpawnDev.Codecs.Audio.Vorbis;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public Task AudioTranscodeDemo_FlacRoundTrip_LosslessSineRecovers()
    {
        const int sampleRate = 44100;
        const double durationSec = 1.0;
        const int toneHz = 440;
        const float amplitude = 0.5f;

        int sampleCount = (int)(durationSec * sampleRate);
        var srcInt16 = new int[sampleCount];
        var srcFloat = new float[sampleCount];
        for (int i = 0; i < sampleCount; i++)
        {
            double t = i / (double)sampleRate;
            float v = (float)(amplitude * Math.Sin(2.0 * Math.PI * toneHz * t));
            srcFloat[i] = v;
            srcInt16[i] = (int)Math.Round(v * 32767.0);
        }

        var bytes = FlacEncoder.EncodeStream(srcInt16, sampleRate, channels: 1, bitsPerSample: 16);
        True(bytes.Length > 0, "FLAC encoder must produce non-empty output");

        var dec = FlacDecoder.Decode(bytes);
        Equal(sampleCount, dec.InterleavedSamples.Length,
            "FLAC is lossless: decoded sample count must match source");

        // FLAC is lossless within the 16-bit quantization. Recovered samples
        // must equal the rounded source samples bit-for-bit.
        int firstMismatch = -1;
        for (int i = 0; i < sampleCount; i++)
        {
            if (dec.InterleavedSamples[i] != srcInt16[i])
            {
                firstMismatch = i;
                break;
            }
        }
        Equal(-1, firstMismatch,
            $"FLAC must be bit-exact; first mismatch at sample {firstMismatch}");

        return Task.CompletedTask;
    }

    [TestMethod]
    public Task AudioTranscodeDemo_VorbisRoundTrip_LossySineRecovers()
    {
        const int sampleRate = 44100;
        const double durationSec = 1.0;
        const int toneHz = 440;
        const float amplitude = 0.5f;

        int sampleCount = (int)(durationSec * sampleRate);
        var srcFloat = new float[sampleCount];
        for (int i = 0; i < sampleCount; i++)
        {
            double t = i / (double)sampleRate;
            srcFloat[i] = (float)(amplitude * Math.Sin(2.0 * Math.PI * toneHz * t));
        }

        var enc = new VorbisAudioEncoder(new VorbisAudioEncoderOptions
        {
            SampleRateHz = sampleRate,
            Channels = 1,
            BlockSize = 1024,
        });
        var bytes = enc.EncodeStream(srcFloat);
        True(bytes.Length > 0, "Vorbis encoder must produce non-empty output");

        var dec = VorbisOggDecoder.Decode(bytes);
        True(dec.InterleavedSamples.Length > 0,
            "Vorbis decoder must produce non-empty samples");

        // Lossy: enforce SNR floor rather than bit-exactness. A clean 440 Hz
        // sine at amplitude 0.5 should recover comfortably above 10 dB SNR
        // through the mono encoder; the production target is much higher,
        // but a low floor catches catastrophic regressions while staying
        // robust against minor output-amplitude tweaks.
        double snr = ComputeSnrDbForTest(srcFloat, dec.InterleavedSamples);
        True(snr > 10.0,
            $"Vorbis SNR floor 10 dB; got {snr:F1} dB (sineHz={toneHz})");

        return Task.CompletedTask;
    }

    [TestMethod]
    public async Task AudioTranscodeDemo_OpusRoundTrip_LossySineRecovers()
    {
        // Opus is locked to 48 kHz internally. Use 48 kHz directly to skip
        // resampling for this test - the demo resamples for cross-codec
        // alignment, but here we just want round-trip integrity.
        const int sampleRate = 48000;
        const double durationSec = 1.0;
        const int toneHz = 440;
        const float amplitude = 0.5f;

        int sampleCount = (int)(durationSec * sampleRate);
        var srcFloat = new float[sampleCount];
        for (int i = 0; i < sampleCount; i++)
        {
            double t = i / (double)sampleRate;
            srcFloat[i] = (float)(amplitude * Math.Sin(2.0 * Math.PI * toneHz * t));
        }

        // Same chunking flow the demo page uses: 20 ms frames at 48 kHz.
        const int frameSamples = 48000 / 50; // 960 = 20 ms
        var packets = new List<byte[]>();
        var packetBuf = new byte[1500];
        using (var enc = new OpusEncoder(new OpusEncoderConfig
        {
            SampleRateHz = sampleRate,
            ChannelCount = 1,
            Application = OpusEncoderApplication.Audio,
        }))
        {
            int idx = 0;
            while (idx + frameSamples <= sampleCount)
            {
                int n = enc.EncodeFrame(srcFloat.AsSpan(idx, frameSamples), packetBuf, frameSamples);
                packets.Add(packetBuf.AsSpan(0, n).ToArray());
                idx += frameSamples;
            }
            True(packets.Count > 0, "Opus encoder must produce at least one packet");
        }

        var oggBytes = OpusOggEncoder.Encode(packets, new OpusOggEncoderOptions
        {
            OutputChannels = 1,
            InputSampleRateHz = (uint)sampleRate,
            PreSkip = 312,
            Vendor = "AudioTranscodeDemo unit test",
        });
        True(oggBytes.Length > 0, "Opus-in-Ogg encoder must produce non-empty output");

        var dec = await OpusOggDecoder.DecodeAsync(oggBytes);
        True(dec.InterleavedSamples48kHz.Length > 0,
            "Opus decoder must produce non-empty samples");

        // SNR floor (lossy + Opus has codec-side latency / pre-skip).
        double snr = ComputeSnrDbForTest(srcFloat, dec.InterleavedSamples48kHz);
        True(snr > 5.0,
            $"Opus SNR floor 5 dB; got {snr:F1} dB (sineHz={toneHz})");
    }

    private static double ComputeSnrDbForTest(float[] reference, float[] decoded)
    {
        if (reference.Length == 0 || decoded.Length == 0) return 0;
        int n = Math.Min(reference.Length, decoded.Length);
        double sigPow = 0;
        double errPow = 0;
        for (int i = 0; i < n; i++)
        {
            sigPow += (double)reference[i] * reference[i];
            double e = reference[i] - decoded[i];
            errPow += e * e;
        }
        if (errPow <= 0) return double.PositiveInfinity;
        if (sigPow <= 0) return 0;
        return 10.0 * Math.Log10(sigPow / errPow);
    }
}
