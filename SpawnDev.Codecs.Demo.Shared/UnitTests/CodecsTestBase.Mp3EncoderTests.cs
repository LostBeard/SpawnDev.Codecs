// Tests for the MP3 encoder (Mp3EncoderGpu, Mp3EncodeKernels): FFmpeg must decode every file without an error, and the
// decoded audio (aligned for the codec delay) must be close to the source. Desktop lanes (FFmpeg on PATH).

using SpawnDev.Codecs.Audio.Mp3;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private static float[] Mp3TestTone(int samples, int channels, int rate, int seed)
    {
        var r = new Random(seed);
        var pcm = new float[samples * channels];
        for (int c = 0; c < channels; c++)
        {
            double f1 = 220 * (1 + 0.5 * c), f2 = 1500 + 300 * c, f3 = rate / 6.0;
            for (int i = 0; i < samples; i++)
            {
                double t = (double)i / rate;
                pcm[i * channels + c] = (float)(0.3 * Math.Sin(2 * Math.PI * f1 * t) + 0.15 * Math.Sin(2 * Math.PI * f2 * t)
                    + 0.05 * Math.Sin(2 * Math.PI * f3 * t) + 0.01 * (r.NextDouble() - 0.5));
            }
        }
        return pcm;
    }

    /// <summary>SNR (dB) of decoded vs source at the best delay in [0, maxDelay) samples, and that delay.</summary>
    private static (double Snr, int Delay) AlignedSnr(float[] src, float[] dec, int channels, int maxDelay)
    {
        int n = src.Length / channels;
        int bestDelay = 0;
        double bestCorr = double.MinValue;
        for (int d = 0; d < maxDelay; d++)
        {
            double corr = 0;
            for (int i = 0; i < n && (i + d) * channels < dec.Length; i += 7) corr += src[i * channels] * dec[(i + d) * channels];
            if (corr > bestCorr) { bestCorr = corr; bestDelay = d; }
        }
        double sig = 0, err = 0;
        for (int i = 0; i < n; i++)
            for (int c = 0; c < channels; c++)
            {
                long j = (long)(i + bestDelay) * channels + c;
                double y = j < dec.Length ? dec[j] : 0;
                double x = src[i * channels + c];
                sig += x * x;
                err += (x - y) * (x - y);
            }
        return (10 * Math.Log10(sig / Math.Max(err, 1e-30)), bestDelay);
    }

    /// <summary>The gain g minimising |dec - g * src| at a delay (diagnostics: 1.0 when the levels agree).</summary>
    private static double FitGain(float[] src, float[] dec, int channels, int delay)
    {
        double xy = 0, xx = 0;
        int n = src.Length / channels;
        for (int i = 0; i < n; i++)
            for (int c = 0; c < channels; c++)
            {
                long j = (long)(i + delay) * channels + c;
                if (j >= dec.Length) continue;
                xy += src[i * channels + c] * dec[j];
                xx += src[i * channels + c] * src[i * channels + c];
            }
        return xy / Math.Max(xx, 1e-30);
    }

    [TestMethod(Timeout = 900000, Category = "DesktopOnly,NotOnWebGL")]
    public async Task Mp3EncoderGpu_Cbr_FfmpegDecodes_CloseToTheSource()
    {
        FfmpegOracle.RequireAvailable();
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var enc = new Mp3EncoderGpu(acc);
            var report = new List<string>();
            var failures = new List<string>();
            foreach (var (rate, ch, kbps, minSnr) in new[]
                     {
                         // measured 2026-11-08 (flat-noise quantization, no psychoacoustic model): 38..64 dB; the floor is 5 dB below
                         (44100, 2, 320, 43.0), (44100, 1, 320, 59.0), (48000, 2, 320, 42.0), (44100, 2, 128, 33.0),
                         (48000, 1, 192, 46.0), (32000, 2, 256, 47.0), (22050, 2, 128, 40.0), (24000, 1, 64, 39.0),
                         (16000, 1, 48, 41.0), (11025, 1, 32, 40.0), (8000, 2, 32, 33.0),
                     })
            {
                var src = Mp3TestTone(rate + 1234, ch, rate, rate + ch);
                byte[] file;
                try { file = await enc.EncodeAsync(src, ch, rate, new Mp3EncodeOptions { BitrateKbps = kbps }); }
                catch (Exception e) { failures.Add($"{rate} Hz {ch} ch {kbps}k: encode threw {e.Message}"); continue; }
                // diagnostics: MP3_TEST_DUMP=<dir> keeps the files
                string? dump = Environment.GetEnvironmentVariable("MP3_TEST_DUMP");
                if (!string.IsNullOrEmpty(dump) && !OperatingSystem.IsBrowser()) File.WriteAllBytes(Path.Combine(dump, $"enc_{rate}_{ch}_{kbps}.mp3"), file);
                float[] dec;
                try { dec = FfmpegOracle.DecodeFloat(file, ".mp3"); }
                catch (Exception e) { failures.Add($"{rate} Hz {ch} ch {kbps}k: {e.Message}"); continue; }
                var (snr, delay) = AlignedSnr(src, dec, ch, 3000);
                double actualKbps = file.Length * 8.0 / ((double)src.Length / ch / rate) / 1000;
                report.Add($"{rate} Hz {ch} ch {kbps}k: {file.Length} bytes ({actualKbps:F1} kbps), SNR {snr:F1} dB at delay {delay}, gain {FitGain(src, dec, ch, delay):F4}");
                if (snr < minSnr) failures.Add($"{rate} Hz {ch} ch {kbps}k: SNR {snr:F1} dB < {minSnr}");
            }
            Console.WriteLine(string.Join("\n", report));
            True(failures.Count == 0, string.Join("\n", failures));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }
}
