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
    /// <summary>MPEG-1 stereo / mono frames of an MP3: how many granule-channels have scalefac_compress != 0.</summary>
    private static (int Units, int WithScalefactors) Mp3ScalefactorUse(byte[] f)
    {
        int p = 0, units = 0, used = 0;
        while (p + 4 <= f.Length && f[p] == 0xFF && (f[p + 1] & 0xE0) == 0xE0)
        {
            bool mpeg1 = (f[p + 1] & 0x18) == 0x18;
            int brIdx = f[p + 2] >> 4, srIdx = (f[p + 2] >> 2) & 3, pad = (f[p + 2] >> 1) & 1;
            bool mono = (f[p + 3] >> 6) == 3;
            int[] br1 = { 0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320 };
            int[] sr1 = { 44100, 48000, 32000 };
            if (!mpeg1) break;
            int size = 144000 * br1[brIdx] / sr1[srIdx] + pad;
            // side info bits after the 4-byte header
            long bit = (p + 4) * 8L;
            int Read(int n) { int v = 0; for (int k = 0; k < n; k++, bit++) v = (v << 1) | ((f[bit >> 3] >> (7 - (int)(bit & 7))) & 1); return v; }
            int ch = mono ? 1 : 2;
            Read(9); Read(mono ? 5 : 3); Read(4 * ch);
            for (int g = 0; g < 2; g++)
                for (int c = 0; c < ch; c++)
                {
                    Read(12); Read(9); Read(8);
                    int compress = Read(4);
                    int ws = Read(1);
                    if (ws != 0) { Read(2); Read(1); Read(10); Read(9); } else { Read(15); Read(4); Read(3); }
                    Read(3);
                    units++;
                    if (compress != 0) used++;
                }
            p += size;
        }
        return (units, used);
    }

    /// <summary>Local SNR (dB) of a tone at <paramref name="freq"/>: decoded vs source energy within +-60 Hz, channel 0.</summary>
    private static double ToneSnr(float[] src, float[] dec, int channels, int delay, int rate, double freq)
    {
        int n = 1 << 14, start = 4096;
        var a = new System.Numerics.Complex[n];
        var e = new System.Numerics.Complex[n];
        for (int i = 0; i < n; i++)
        {
            double w = 0.5 - 0.5 * Math.Cos(2 * Math.PI * i / n);
            double x = src[(start + i) * channels], y = dec[(start + i + delay) * channels];
            a[i] = x * w;
            e[i] = (y - x) * w;
        }
        static void Fft(System.Numerics.Complex[] v)
        {
            int n = v.Length;
            for (int i = 1, j = 0; i < n; i++)
            {
                int bit = n >> 1;
                for (; (j & bit) != 0; bit >>= 1) j ^= bit;
                j ^= bit;
                if (i < j) (v[i], v[j]) = (v[j], v[i]);
            }
            for (int len = 2; len <= n; len <<= 1)
            {
                var wl = System.Numerics.Complex.FromPolarCoordinates(1, -2 * Math.PI / len);
                for (int i = 0; i < n; i += len)
                {
                    var w = System.Numerics.Complex.One;
                    for (int k = 0; k < len / 2; k++, w *= wl)
                    {
                        var u = v[i + k];
                        var t = v[i + k + len / 2] * w;
                        v[i + k] = u + t;
                        v[i + k + len / 2] = u - t;
                    }
                }
            }
        }
        Fft(a);
        Fft(e);
        int lo = (int)((freq - 60) * n / rate), hi = (int)((freq + 60) * n / rate);
        double sig = 0, err = 0;
        for (int k = lo; k <= hi; k++) { sig += a[k].Magnitude * a[k].Magnitude; err += e[k].Magnitude * e[k].Magnitude; }
        return 10 * Math.Log10(sig / Math.Max(err, 1e-30));
    }

    /// <summary>
    /// Noise shaping at 128 kbps: a loud 200 Hz tone, a quiet 9 kHz tone 44 dB down and faint noise. The psychoacoustic
    /// model must keep the quiet tone above its own noise (local SNR), and the encoder must use scalefactors to do so.
    /// </summary>
    [TestMethod(Timeout = 900000, Category = "DesktopOnly,NotOnWebGL")]
    public async Task Mp3EncoderGpu_NoiseShaping_KeepsQuietToneAboveItsNoise()
    {
        FfmpegOracle.RequireAvailable();
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            const int rate = 44100, ch = 2;
            int n = rate * 2;
            var r = new Random(9);
            var src = new float[n * ch];
            for (int i = 0; i < n; i++)
                for (int c = 0; c < ch; c++)
                    src[i * ch + c] = (float)(0.5 * Math.Sin(2 * Math.PI * 200 * i / rate) + 0.003 * Math.Sin(2 * Math.PI * 9000 * i / rate)
                        + 0.002 * (r.NextDouble() - 0.5));
            using var enc = new Mp3EncoderGpu(acc);
            var file = await enc.EncodeAsync(src, ch, rate, new Mp3EncodeOptions { BitrateKbps = 128 });
            var dec = FfmpegOracle.DecodeFloat(file, ".mp3");
            var (units, used) = Mp3ScalefactorUse(file);
            double quiet = ToneSnr(src, dec, ch, Mp3EncoderGpu.CodecDelay, rate, 9000);
            double loud = ToneSnr(src, dec, ch, Mp3EncoderGpu.CodecDelay, rate, 200);
            Console.WriteLine($"scalefactors in {used} of {units} granule-channels; local SNR 9 kHz {quiet:F1} dB, 200 Hz {loud:F1} dB");
            True(used > 0, "no granule used scalefactors");
            True(quiet >= 12, $"quiet 9 kHz tone local SNR {quiet:F1} dB < 12");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }
}
