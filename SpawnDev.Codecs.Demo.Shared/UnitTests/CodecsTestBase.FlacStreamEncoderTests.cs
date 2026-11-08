// Tests for FlacEncoderGpu.EncodeAsync (Audio/Flac/FlacEncoderGpu.Stream.cs, FlacStreamKernels, FlacStreamWriteKernels):
// a port of FFmpeg 8.1's FLAC encoder + muxer, compression levels 0..8.
//
// Fixtures (scripts/audio/make_flac_encoder_fixtures.py, TestData/FlacEnc): source PCM (stereo / mono / 5 channels,
// 16 / 24 bits, 22.05 .. 96 kHz and a non-standard rate, lengths that are not block multiples, a 3-sample clip, silence
// with a burst, a constant pair, samples with wasted bits, correlated stereo) and FFmpeg's .flac of each at levels 0, 1,
// 2, 5, 8 (bit-exact mode).
//
// What must hold:
//   * levels 0..2 (FFmpeg's integer fixed predictors): our file is BYTE-IDENTICAL with FFmpeg's, on every backend;
//   * levels 5 and 8 (LPC, FFmpeg's double-precision analysis; fixtures made with -cpuflags 0, FFmpeg's C code):
//     BYTE-IDENTICAL on the backends with IEEE doubles (CPU, CUDA, OpenCL, Wasm). WebGPU / WebGL emulate doubles, so
//     there the coefficients may differ: the same stream header as FFmpeg's (STREAMINFO with the same MD5 of the samples,
//     Vorbis comment, padding) apart from the min / max frame sizes, and a file no more than 2 % larger;
//   * (desktop) libFLAC decodes every file of ours, every level, to exactly the source samples.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Audio.Flac;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private const string FlacEncPrefix = "SpawnDev.Codecs.Demo.Shared.TestData.FlacEnc.";

    private sealed record FlacEncCase(string Name, string Source, int Channels, int Rate, int Bits, int Samples, int Level);

    private static List<FlacEncCase> FlacEncCases()
    {
        var list = new List<FlacEncCase>();
        foreach (var line in System.Text.Encoding.UTF8.GetString(ReadEmbedded(FlacEncPrefix + "manifest.tsv")).Split('\n'))
        {
            var row = line.TrimEnd('\r');
            if (row.Length == 0) continue;
            var c = row.Split('\t');
            list.Add(new FlacEncCase(c[0], c[1], int.Parse(c[2]), int.Parse(c[3]), int.Parse(c[4]), int.Parse(c[5]), int.Parse(c[6])));
        }
        return list;
    }

    private static int[] FlacEncSource(FlacEncCase c, Dictionary<string, int[]> cache)
    {
        if (cache.TryGetValue(c.Source, out var s)) return s;
        var raw = Gunzip(ReadEmbedded(FlacEncPrefix + c.Source + ".s32.gz"));
        s = new int[raw.Length / 4];
        Buffer.BlockCopy(raw, 0, s, 0, raw.Length);
        True(s.Length == c.Samples * c.Channels, $"{c.Source}: {s.Length} samples, manifest says {c.Samples} x {c.Channels}");
        return cache[c.Source] = s;
    }

    private static Task<byte[]> FlacEncode(FlacEncoderGpu enc, FlacEncCase c, int[] samples) =>
        enc.EncodeAsync(samples, c.Channels, c.Rate, c.Bits, new FlacEncodeOptions
        {
            CompressionLevel = c.Level,
            ChannelMask = c.Channels == 5 ? 0x37UL : null,   // FFmpeg's default 5-channel layout (5.0 back), not native to FLAC
        });

    private static int FirstDifference(byte[] a, byte[] b)
    {
        int i = 0, n = Math.Min(a.Length, b.Length);
        while (i < n && a[i] == b[i]) i++;
        return i;
    }

    /// <summary>WebGPU and WebGL emulate double (no IEEE f64 in their shading languages).</summary>
    private static bool IsEmulatedDoubleBackend(Accelerator acc)
    {
        string type = acc.AcceleratorType.ToString();
        return type.Contains("WebGPU", StringComparison.OrdinalIgnoreCase) || type.Contains("WebGL", StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>The bytes before the first audio frame: "fLaC" + every metadata block (until the last-block flag).</summary>
    private static int FlacHeaderLength(byte[] f)
    {
        int p = 4;
        while (true)
        {
            bool last = (f[p] & 0x80) != 0;
            int len = (f[p + 1] << 16) | (f[p + 2] << 8) | f[p + 3];
            p += 4 + len;
            if (last) return p;
        }
    }

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task FlacEncoderGpu_FixedLevels_ByteIdenticalWithFfmpeg()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var enc = new FlacEncoderGpu(acc);
            var cache = new Dictionary<string, int[]>();
            var failures = new List<string>();
            var cases = FlacEncCases().Where(c => c.Level <= 2).ToList();
            foreach (var c in cases)
            {
                try
                {
                    var ours = await FlacEncode(enc, c, FlacEncSource(c, cache));
                    var ffmpeg = ReadEmbedded(FlacEncPrefix + c.Name + ".flac");
                    if (!ours.AsSpan().SequenceEqual(ffmpeg))
                        failures.Add($"{c.Name}: {ours.Length} bytes vs FFmpeg {ffmpeg.Length}, first difference at byte {FirstDifference(ours, ffmpeg)}");
                }
                catch (Exception ex)
                {
                    failures.Add($"{c.Name}: {ex.GetType().Name}: {ex.Message}");
                }
            }
            True(cases.Count >= 30, $"only {cases.Count} fixtures");
            True(failures.Count == 0, $"{failures.Count} of {cases.Count} differ:\n" + string.Join("\n", failures));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task FlacEncoderGpu_LpcLevels_ByteIdenticalWithFfmpeg_WhereDoublesAreIeee()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var enc = new FlacEncoderGpu(acc);
            var cache = new Dictionary<string, int[]>();
            var failures = new List<string>();
            var report = new List<string>();
            int identical = 0;
            var cases = FlacEncCases().Where(c => c.Level > 2).ToList();
            bool ieeeDoubles = !IsEmulatedDoubleBackend(acc);
            foreach (var c in cases)
            {
                try
                {
                    var ours = await FlacEncode(enc, c, FlacEncSource(c, cache));
                    var ffmpeg = ReadEmbedded(FlacEncPrefix + c.Name + ".flac");
                    int h = FlacHeaderLength(ffmpeg);
                    if (FlacHeaderLength(ours) != h) { failures.Add($"{c.Name}: header length {FlacHeaderLength(ours)} vs FFmpeg {h}"); continue; }
                    // STREAMINFO bytes 8..42: min / max frame size (12..18) depend on the frames; everything else must match,
                    // the MD5 of the samples (26..42) included
                    var a = ours.AsSpan(0, h).ToArray();
                    var b = ffmpeg.AsSpan(0, h).ToArray();
                    for (int i = 12; i < 18; i++) { a[i] = 0; b[i] = 0; }
                    if (!a.AsSpan().SequenceEqual(b)) { failures.Add($"{c.Name}: stream header differs at byte {FirstDifference(a, b)}"); continue; }
                    double ratio = (double)(ours.Length - h) / (ffmpeg.Length - h);
                    bool same = ours.AsSpan().SequenceEqual(ffmpeg);
                    if (same) identical++;
                    else if (ieeeDoubles) failures.Add($"{c.Name}: {ours.Length} bytes vs FFmpeg {ffmpeg.Length}, first difference at byte {FirstDifference(ours, ffmpeg)}");
                    report.Add($"{c.Name} {(same ? "identical" : ratio.ToString("F4") + " first difference at byte " + FirstDifference(ours, ffmpeg))}");
                    if (ratio > 1.02) failures.Add($"{c.Name}: frames {ours.Length - h} bytes vs FFmpeg {ffmpeg.Length - h} ({ratio:F4})");
                }
                catch (Exception ex)
                {
                    failures.Add($"{c.Name}: {ex.GetType().Name}: {ex.Message}");
                }
            }
            Console.WriteLine($"FLAC LPC levels: {identical} of {cases.Count} byte-identical with FFmpeg; " + string.Join(", ", report));
            True(cases.Count >= 20, $"only {cases.Count} fixtures");
            True(failures.Count == 0, $"{failures.Count} of {cases.Count}:\n" + string.Join("\n", failures));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod(Timeout = 600000, Category = "DesktopOnly,NotOnWebGL")]
    public async Task FlacEncoderGpu_EveryLevel_LibflacDecodesTheSourceExactly()
    {
        FlacOracle.RequireAvailable();
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var enc = new FlacEncoderGpu(acc);
            var cache = new Dictionary<string, int[]>();
            var failures = new List<string>();
            var cases = FlacEncCases();
            foreach (var c in cases)
            {
                var src = FlacEncSource(c, cache);
                var ours = await FlacEncode(enc, c, src);
                var (decoded, ch, rate) = FlacOracle.Decode(ours);
                if (ch != c.Channels || rate != c.Rate) { failures.Add($"{c.Name}: libFLAC sees {ch} ch {rate} Hz"); continue; }
                if (decoded.Length != src.Length) { failures.Add($"{c.Name}: {decoded.Length} samples decoded, {src.Length} encoded"); continue; }
                for (int i = 0; i < src.Length; i++)
                {
                    if (decoded[i] != src[i]) { failures.Add($"{c.Name}: sample {i} decodes to {decoded[i]}, source {src[i]}"); break; }
                }
            }
            True(failures.Count == 0, $"{failures.Count} of {cases.Count}:\n" + string.Join("\n", failures));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task FlacEncoderGpu_EncodeAsync_ArrayViewOverload_WritesOnlyTheFile()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var enc = new FlacEncoderGpu(acc);
            var cache = new Dictionary<string, int[]>();
            foreach (var c in FlacEncCases().Where(k => k.Level == 1 && (k.Source == "stereo_tone_16" || k.Source == "stereo_tone_24")))
            {
                var src = FlacEncSource(c, cache);
                var ffmpeg = ReadEmbedded(FlacEncPrefix + c.Name + ".flac");
                using var dIn = acc.Allocate1D<int>(src.Length);
                dIn.View.CopyFromCPU(src);
                using var dOut = DeviceVerify.Poisoned(acc, ffmpeg.Length + 64);
                int length = await enc.EncodeAsync(dIn.View, c.Channels, c.Rate, c.Bits, dOut.View.SubView(0, ffmpeg.Length),
                    new FlacEncodeOptions { CompressionLevel = c.Level });
                Equal(ffmpeg.Length, length, $"{c.Name}: length");
                Equal(null, await DeviceVerify.AllEqualAsync(acc, dOut.View.SubView(length, 64), 64, 0xAB), $"{c.Name}: bytes past the end");
                True((await DeviceTransfer.DownloadAsync(dOut.View, length)).AsSpan().SequenceEqual(ffmpeg), $"{c.Name}: file differs from FFmpeg's");
                using var small = DeviceVerify.Poisoned(acc, ffmpeg.Length - 1);
                string? message = null;
                try { await enc.EncodeAsync(dIn.View, c.Channels, c.Rate, c.Bits, small.View, new FlacEncodeOptions { CompressionLevel = c.Level }); }
                catch (ArgumentException ex) { message = ex.Message; }
                True(message != null && message.Contains(ffmpeg.Length.ToString()), $"{c.Name}: too-small output: {message ?? "no exception"}");
            }
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }
}
