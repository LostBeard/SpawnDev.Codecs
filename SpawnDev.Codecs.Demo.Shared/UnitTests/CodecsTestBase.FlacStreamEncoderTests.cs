// Tests for the FLAC stream encoder, FlacEncoderGpu.EncodeAsync (Audio/Flac/FlacEncoderGpu.Encode.cs, FlacEncodeKernels):
// compression levels 0..8, any rate, 1..8 channels, 4..24 bits.
//
// Fixtures (scripts/audio/make_flac_encoder_fixtures.py, TestData/FlacEnc): source PCM (stereo / mono / 5 channels,
// 16 / 24 bits, 22.05 .. 96 kHz and a non-standard rate, lengths that are not block multiples, a 3-sample clip, silence
// with a burst, a constant pair, samples with wasted bits, correlated stereo, full-scale white noise) and FFmpeg's .flac
// of each at levels 0, 1, 2, 5, 8 (bit-exact mode). Our frames need not equal FFmpeg's (a different encoder); what must
// hold for every fixture:
//   * lossless: our FlacDecoderGpu decodes our file to exactly the source samples, every frame passes its CRCs and the
//     decoded samples' MD5 equals our STREAMINFO's; with SPAWNDEV_IMG_PYTHON set, libFLAC decodes it to the source too;
//   * the metadata blocks (everything before the first frame) equal the fixture's byte for byte, except STREAMINFO's
//     minimum / maximum frame size (file bytes 12..17), which must instead equal the smallest / largest frame of OUR
//     file (measured by walking its frames). So block sizes, rate, channels, bits, sample count, the MD5 of the samples,
//     the VORBIS_COMMENT (vendor, channel mask tag) and the PADDING block are FFmpeg's exactly;
//   * size: our frames (file minus metadata) no more than 2 % larger than the fixture's at the same level (the ratios
//     are printed).

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

    /// <summary>Largest ratio of our frame bytes to the fixture's at the same level.</summary>
    private const double FlacEncMaxSizeRatio = 1.02;

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

    private static FlacEncodeOptions FlacEncOptions(FlacEncCase c) => new()
    {
        CompressionLevel = c.Level,
        ChannelMask = c.Channels == 5 ? 0x37 : null,   // FFmpeg's default 5-channel layout (5.0 back), not FLAC's default (5.0 side)
    };

    private static Task<byte[]> FlacEncode(FlacEncoderGpu enc, FlacEncCase c, int[] samples) =>
        enc.EncodeAsync(samples, c.Channels, c.Rate, c.Bits, FlacEncOptions(c));

    private static int FirstDifference(byte[] a, byte[] b)
    {
        int i = 0, n = Math.Min(a.Length, b.Length);
        while (i < n && a[i] == b[i]) i++;
        return i;
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

    /// <summary>The smallest and largest frame of a FLAC file, measured by walking its frames (sync code + CRC-16).</summary>
    private static (int Min, int Max) FlacMeasuredFrameSizes(byte[] f)
    {
        var starts = FlacFrameOffsets(f);
        int mn = int.MaxValue, mx = 0;
        for (int i = 0; i < starts.Count; i++)
        {
            int size = (i + 1 < starts.Count ? starts[i + 1] : f.Length) - starts[i];
            mn = Math.Min(mn, size);
            mx = Math.Max(mx, size);
        }
        return starts.Count == 0 ? (0, 0) : (mn, mx);
    }

    /// <summary>
    /// Null when <paramref name="ours"/> meets the fixture rules above (metadata, STREAMINFO frame sizes, lossless decode
    /// through <paramref name="dec"/>), else what failed. <paramref name="ratio"/>: our frame bytes / the fixture's.
    /// </summary>
    private static async Task<string?> FlacEncCheck(FlacDecoderGpu dec, FlacEncCase c, int[] src, byte[] ours, byte[] ffmpeg, Action<double> ratio)
    {
        int h = FlacHeaderLength(ffmpeg);
        if (FlacHeaderLength(ours) != h) return $"metadata is {FlacHeaderLength(ours)} bytes, the fixture's {h}";
        var a = ours.AsSpan(0, h).ToArray();
        var b = ffmpeg.AsSpan(0, h).ToArray();
        for (int i = 12; i < 18; i++) { a[i] = 0; b[i] = 0; }
        if (!a.AsSpan().SequenceEqual(b)) return $"metadata differs at byte {FirstDifference(a, b)}";
        int minFrame = ours[12] << 16 | ours[13] << 8 | ours[14];
        int maxFrame = ours[15] << 16 | ours[16] << 8 | ours[17];
        var (mn, mx) = FlacMeasuredFrameSizes(ours);
        if (minFrame != mn || maxFrame != mx) return $"STREAMINFO frame sizes {minFrame}..{maxFrame}, the frames measure {mn}..{mx}";
        var r = await dec.DecodeAsync(ours);
        if (r.Channels != c.Channels || r.SampleRateHz != c.Rate || r.BitsPerSample != c.Bits) return $"decoder sees {r.Channels} ch {r.SampleRateHz} Hz {r.BitsPerSample} bits";
        if (r.BadFrames != 0 || r.LostSamples != 0) return $"{r.BadFrames} bad frames, {r.LostSamples} lost samples";
        if (!r.Md5Checked || !r.Md5Matches) return "the decoded samples' MD5 differs from STREAMINFO's";
        int diff = FirstDifference(r.InterleavedSamples, src);
        if (diff >= 0) return $"decoded sample {diff} differs from the source";
        ratio((double)(ours.Length - h) / (ffmpeg.Length - h));
        return null;
    }

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task FlacEncoderGpu_EveryFixture_LosslessMetadataAsFfmpeg_SizeWithinTwoPercent()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var enc = new FlacEncoderGpu(acc);
            using var dec = new FlacDecoderGpu(acc);
            var cache = new Dictionary<string, int[]>();
            var failures = new List<string>();
            var cases = FlacEncCases();
            var byLevel = new SortedDictionary<int, List<double>>();
            var report = new List<string>();
            long oursTotal = 0, ffTotal = 0;
            foreach (var c in cases)
            {
                try
                {
                    var src = FlacEncSource(c, cache);
                    var ours = await FlacEncode(enc, c, src);
                    var ffmpeg = ReadEmbedded(FlacEncPrefix + c.Name + ".flac");
                    double ratio = 0;
                    string? err = await FlacEncCheck(dec, c, src, ours, ffmpeg, x => ratio = x);
                    if (err != null) { failures.Add($"{c.Name}: {err}"); continue; }
                    (byLevel.TryGetValue(c.Level, out var l) ? l : byLevel[c.Level] = new List<double>()).Add(ratio);
                    report.Add($"{c.Name} {ratio:F4} (file {(double)ours.Length / ffmpeg.Length:F4})");
                    oursTotal += ours.Length; ffTotal += ffmpeg.Length;
                    if (ratio > FlacEncMaxSizeRatio) failures.Add($"{c.Name}: frames {ratio:F4} x the fixture's");
                }
                catch (Exception ex)
                {
                    failures.Add($"{c.Name}: {ex.GetType().Name}: {ex.Message}");
                }
            }
            Console.WriteLine("FLAC encoder frame-size ratios vs FFmpeg: " + string.Join(", ", report));
            Console.WriteLine("FLAC encoder per level (min / mean / max frame ratio): " + string.Join("; ",
                byLevel.Select(kv => $"l{kv.Key} {kv.Value.Min():F4} / {kv.Value.Average():F4} / {kv.Value.Max():F4}"))
                + $"; all files {(double)oursTotal / ffTotal:F4}");
            True(cases.Count >= 60, $"only {cases.Count} fixtures");
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
            foreach (var c in FlacEncCases().Where(k => (k.Level == 1 || k.Level == 5) && (k.Source == "stereo_tone_16" || k.Source == "stereo_tone_24")))
            {
                var src = FlacEncSource(c, cache);
                var managed = await FlacEncode(enc, c, src);
                using var dIn = acc.Allocate1D<int>(src.Length);
                dIn.View.CopyFromCPU(src);
                // an output exactly the file's size (smaller than the encoder's worst case: coded aside, then copied) and one
                // with room for the worst case (coded in place)
                foreach (int room in new[] { 0, src.Length * 8 })
                {
                    using var dOut = DeviceVerify.Poisoned(acc, managed.Length + room + 64);
                    int length = await enc.EncodeAsync(dIn.View, c.Channels, c.Rate, c.Bits, dOut.View.SubView(0, managed.Length + room), FlacEncOptions(c));
                    Equal(managed.Length, length, $"{c.Name} room {room}: length");
                    Equal(null, await DeviceVerify.AllEqualAsync(acc, dOut.View.SubView(length, room + 64), room + 64, 0xAB), $"{c.Name} room {room}: bytes past the end");
                    True((await DeviceTransfer.DownloadAsync(dOut.View, length)).AsSpan().SequenceEqual(managed), $"{c.Name} room {room}: differs from the managed overload's file");
                }
                using var small = DeviceVerify.Poisoned(acc, managed.Length - 1);
                string? message = null;
                try { await enc.EncodeAsync(dIn.View, c.Channels, c.Rate, c.Bits, small.View, FlacEncOptions(c)); }
                catch (ArgumentException ex) { message = ex.Message; }
                True(message != null && message.Contains(managed.Length.ToString()), $"{c.Name}: too-small output: {message ?? "no exception"}");
                Equal(null, await DeviceVerify.AllEqualAsync(acc, small.View, managed.Length - 1, 0xAB), $"{c.Name}: a too-small output was written");
            }
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task FlacEncoderGpu_WarmUp_LoadsTheKernels_LaterEncodesAreUnchanged()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            var cache = new Dictionary<string, int[]>();
            var cases = FlacEncCases().Where(x => x.Name is "stereo_tone_16_l5" or "mono_silence_mix_16_l0").ToList();
            Equal(2, cases.Count, "cases");
            var cold = new List<byte[]>();
            using (var enc = new FlacEncoderGpu(acc))
                foreach (var c in cases) cold.Add(await FlacEncode(enc, c, FlacEncSource(c, cache)));
            using var warm = new FlacEncoderGpu(acc);
            await warm.WarmUpAsync();
            using var dec = new FlacDecoderGpu(acc);
            for (int i = 0; i < cases.Count; i++)
            {
                var c = cases[i];
                var src = FlacEncSource(c, cache);
                var after = await FlacEncode(warm, c, src);
                True(after.AsSpan().SequenceEqual(cold[i]), $"{c.Name}: the file after WarmUpAsync differs (first difference at byte {FirstDifference(after, cold[i])})");
                var r = await dec.DecodeAsync(after);
                True(FirstDifference(r.InterleavedSamples, src) < 0, $"{c.Name}: not lossless after WarmUpAsync");
            }
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    /// <summary>A deterministic test signal: tones plus integer noise per channel, in the signed <paramref name="bits"/>-bit range.</summary>
    private static int[] FlacEncSynth(int samples, int channels, int bits, int seed)
    {
        var pcm = new int[samples * channels];
        uint s = (uint)seed;
        double amp = (1 << (bits - 1)) * 0.6;
        int noise = Math.Max(1, (1 << (bits - 1)) / 512);
        for (int i = 0; i < samples; i++)
            for (int c = 0; c < channels; c++)
            {
                s = s * 1664525 + 1013904223;
                double v = amp * (0.7 * Math.Sin(i * (0.031 + 0.007 * c)) + 0.2 * Math.Sin(i * 0.173 + c));
                pcm[i * channels + c] = (int)Math.Clamp(Math.Round(v) + (int)(s % (uint)(2 * noise + 1)) - noise, -(1 << (bits - 1)), (1 << (bits - 1)) - 1);
            }
        return pcm;
    }

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task FlacEncoderGpu_Options_EveryLevelModeDepthAndBlockSize_Lossless()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var enc = new FlacEncoderGpu(acc);
            using var dec = new FlacDecoderGpu(acc);
            var failures = new List<string>();
            async Task Check(string name, int[] pcm, int channels, int rate, int bits, FlacEncodeOptions o, Action<byte[]>? more = null)
            {
                try
                {
                    var file = await enc.EncodeAsync(pcm, channels, rate, bits, o);
                    var r = await dec.DecodeAsync(file);
                    if (r.Channels != channels || r.SampleRateHz != rate || r.BitsPerSample != bits) failures.Add($"{name}: decoder sees {r.Channels} ch {r.SampleRateHz} Hz {r.BitsPerSample} bits");
                    else if (r.BadFrames != 0 || r.LostSamples != 0 || !r.Md5Matches) failures.Add($"{name}: {r.BadFrames} bad frames, {r.LostSamples} lost, MD5 match {r.Md5Matches}");
                    else if (FirstDifference(r.InterleavedSamples, pcm) is int d && d >= 0) failures.Add($"{name}: sample {d} differs");
                    else more?.Invoke(file);
                }
                catch (Exception ex) { failures.Add($"{name}: {ex.GetType().Name}: {ex.Message}"); }
            }
            var stereo = FlacEncSynth(30011, 2, 16, 1);
            for (int level = 0; level <= 8; level++)
                await Check($"level {level}", stereo, 2, 44100, 16, new FlacEncodeOptions { CompressionLevel = level });
            foreach (var mode in Enum.GetValues<FlacChannelMode>())
                await Check($"mode {mode}", stereo, 2, 48000, 16, new FlacEncodeOptions { ChannelMode = mode, CompressionLevel = 3 });
            foreach (int bits in new[] { 4, 8, 12, 20, 24 })
                await Check($"{bits} bits", FlacEncSynth(9001, 2, bits, bits), 2, 32000, bits, new FlacEncodeOptions { CompressionLevel = 7 });
            foreach (int channels in new[] { 1, 3, 4, 6, 7, 8 })
                await Check($"{channels} channels", FlacEncSynth(7000, channels, 16, channels), channels, 22050, 16, new FlacEncodeOptions { CompressionLevel = 4 });
            // odd block sizes (8-bit and 16-bit header codes), a rate in tens of Hz and one only STREAMINFO can carry
            await Check("block 100", stereo, 2, 44100, 16, new FlacEncodeOptions { BlockSize = 100, CompressionLevel = 6 });
            await Check("block 5000", stereo, 2, 44100, 16, new FlacEncodeOptions { BlockSize = 5000, CompressionLevel = 8 });
            await Check("rate 655350", stereo, 2, 655350, 16, new FlacEncodeOptions { CompressionLevel = 2 });
            await Check("rate 700001", stereo, 2, 700001, 16, new FlacEncodeOptions { CompressionLevel = 2 });
            // more frames than one dispatch chunk holds, with every chunk boundary inside the stream
            await Check("long", FlacEncSynth(44100 * 7 + 13, 2, 16, 99), 2, 44100, 16, new FlacEncodeOptions { CompressionLevel = 5 });
            // metadata: vendor, tags in order, FLAC-default channel mask (no tag), no padding
            await Check("metadata", stereo, 2, 44100, 16, new FlacEncodeOptions
            {
                Vendor = "SpawnDev", Padding = 0, ChannelMask = 0x3,
                Tags = new[] { new KeyValuePair<string, string>("TITLE", "x"), new("artist", "Y é") },
            }, file =>
            {
                int h = FlacHeaderLength(file);
                var vc = System.Text.Encoding.UTF8.GetBytes("\u0008\u0000\u0000\u0000SpawnDev\u0002\u0000\u0000\u0000\u0007\u0000\u0000\u0000TITLE=x\u000b\u0000\u0000\u0000artist=Y é");
                // fLaC, STREAMINFO (not last), VORBIS_COMMENT (last, nothing after it)
                if (file[4] != 0x00 || file[42] != 0x84 || h != 46 + vc.Length || !file.AsSpan(46, vc.Length).SequenceEqual(vc))
                    failures.Add($"metadata: blocks are not STREAMINFO + VORBIS_COMMENT(SpawnDev; TITLE=x, artist=Y é) without padding (metadata {h} bytes)");
            });
            // an empty stream
            await Check("empty", Array.Empty<int>(), 2, 44100, 16, new FlacEncodeOptions());
            // a sample outside the bit range is refused
            string? refused = null;
            try { await enc.EncodeAsync(new[] { 0, 1, 40000, 0 }, 2, 44100, 16); }
            catch (ArgumentException ex) { refused = ex.Message; }
            True(refused != null && refused.Contains("16-bit"), $"out-of-range sample: {refused ?? "accepted"}");
            True(failures.Count == 0, $"{failures.Count} failures:\n" + string.Join("\n", failures));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }
}
