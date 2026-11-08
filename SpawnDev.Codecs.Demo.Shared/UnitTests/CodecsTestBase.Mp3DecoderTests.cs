// Tests for the MP3 decoder (Mp3DecoderGpu, Mp3DecodeKernels; port of minimp3, CC0).
//
// Fixtures (scripts/audio/make_mp3_decoder_fixtures.py, TestData/Mp3Dec): files from FFmpeg's libmp3lame and Windows'
// MediaFoundation encoder (black boxes): CBR 128 / 320, VBR V0 / V4 / V6, mono / stereo, MPEG-1 / 2 / 2.5 at every rate
// group, short-block-heavy transients, joint (M/S) / plain stereo, intensity stereo, ID3v2 (TXXX) + Info frame + ID3v1,
// no tags, free format; each with minimp3's own decode (x64 MSVC build, float output, minimp3_ex's gapless trim).
//
// What must hold: every file decodes to minimp3's floats (bit for bit on the CPU accelerator; within 1e-5 elsewhere,
// where kernels compile to other float instruction sequences), to the source's length for gapless encodes (ours and
// LAME's); FFmpeg's decoder (a different implementation) is close (reported); a damaged frame comes out as silence and
// keeps its time while the frames after it decode as in the clean file; batches of frames and resumed walks join
// without a seam.

using ILGPU.Runtime;
using SpawnDev.Codecs.Audio.Mp3;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private const string Mp3DecPrefix = "SpawnDev.Codecs.Demo.Shared.TestData.Mp3Dec.";

    private sealed record Mp3DecCase(string Name, int Rate, int Channels, long SourceSamples, long RefSamples, long LoadBufSamples, string Note);

    private static List<Mp3DecCase> Mp3DecCases() =>
        System.Text.Encoding.UTF8.GetString(ReadEmbedded(Mp3DecPrefix + "manifest.tsv")).Split('\n')
            .Where(l => l.Trim().Length > 0)
            .Select(l => l.TrimEnd('\r').Split('\t'))
            .Select(c => new Mp3DecCase(c[0], int.Parse(c[1]), int.Parse(c[2]), long.Parse(c[3]), long.Parse(c[4]), long.Parse(c[5]), c[6]))
            .ToList();

    /// <summary>minimp3's decode of a fixture (interleaved float32).</summary>
    private static float[] Mp3DecReference(string name)
    {
        var raw = Gunzip(ReadEmbedded(Mp3DecPrefix + name + ".mm3.f32.gz"));
        var f = new float[raw.Length / 4];
        Buffer.BlockCopy(raw, 0, f, 0, raw.Length);
        return f;
    }

    /// <summary>Largest |a - b|, how many values differ in any bit, and the first such index (-1 when none); a length
    /// difference counts every missing value.</summary>
    private static (double MaxAbs, long BitDiffs, long First) Mp3Diff(float[] a, float[] b)
    {
        long n = Math.Min(a.Length, b.Length), diffs = Math.Abs(a.Length - b.Length), first = a.Length == b.Length ? -1 : n;
        double max = 0;
        for (long i = 0; i < n; i++)
        {
            if (BitConverter.SingleToInt32Bits(a[i]) == BitConverter.SingleToInt32Bits(b[i])) continue;
            diffs++;
            if (first < 0 || i < first) first = i;
            double d = Math.Abs((double)a[i] - b[i]);
            if (!(d <= max)) max = d;   // NaN counts as the largest difference
        }
        return (max, diffs, first);
    }

    /// <summary>Byte offsets and sizes of the MPEG-1 Layer III frames of a file (after an ID3v2 tag; the Info frame is
    /// the first), for tests that damage frames.</summary>
    private static List<(int Offset, int Size)> Mp3TestFrames(byte[] f)
    {
        int p = 0;
        if (f.Length > 10 && f[0] == 'I' && f[1] == 'D' && f[2] == '3')
            p = 10 + ((f[6] & 0x7F) << 21 | (f[7] & 0x7F) << 14 | (f[8] & 0x7F) << 7 | (f[9] & 0x7F));
        int[] br = { 0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320 };
        int[] sr = { 44100, 48000, 32000 };
        var list = new List<(int, int)>();
        while (p + 4 <= f.Length && f[p] == 0xFF && (f[p + 1] & 0xFE) == 0xFA)
        {
            int size = 144000 * br[f[p + 2] >> 4] / sr[(f[p + 2] >> 2) & 3] + ((f[p + 2] >> 1) & 1);
            if (size <= 0 || p + size > f.Length) break;
            list.Add((p, size));
            p += size;
        }
        return list;
    }

    [TestMethod(Timeout = 900000, Category = "NotOnWebGL")]
    public async Task Mp3DecoderGpu_Fixtures_MatchMinimp3()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            bool exact = acc.AcceleratorType == AcceleratorType.CPU;
            const double tolerance = 1e-5;
            using var dec = new Mp3DecoderGpu(acc);
            var failures = new List<string>();
            var report = new List<string>();
            double worst = 0;
            foreach (var c in Mp3DecCases())
            {
                byte[] file = ReadEmbedded(Mp3DecPrefix + c.Name + ".mp3");
                var want = Mp3DecReference(c.Name);
                Mp3DecodeResult r;
                try { r = await dec.DecodeAsync(file); }
                catch (Exception e) { failures.Add($"{c.Name}: {e.GetType().Name}: {e.Message}"); continue; }
                if (r.SampleRateHz != c.Rate || r.Channels != c.Channels)
                {
                    failures.Add($"{c.Name}: {r.SampleRateHz} Hz {r.Channels} ch, want {c.Rate} / {c.Channels}");
                    continue;
                }
                if (r.TotalSamplesPerChannel != c.RefSamples)
                    failures.Add($"{c.Name}: {r.TotalSamplesPerChannel} samples, minimp3 gives {c.RefSamples}");
                // gapless encodes (LAME's Info frame): the source's length
                if (c.SourceSamples >= 0 && r.TotalSamplesPerChannel != c.SourceSamples)
                    failures.Add($"{c.Name}: {r.TotalSamplesPerChannel} samples, the source has {c.SourceSamples}");
                if (r.LostFrames != 0) failures.Add($"{c.Name}: {r.LostFrames} frames lost in a clean file");
                var (max, diffs, first) = Mp3Diff(r.Interleaved, want);
                worst = Math.Max(worst, max);
                report.Add($"{c.Name}: {r.Frames} frames, {r.TotalSamplesPerChannel} samples, vs minimp3 max |diff| {max:G3}, {diffs} values differ ({c.Note})");
                if (exact ? diffs != 0 : max > tolerance)
                    failures.Add($"{c.Name}: {diffs} values differ from minimp3, max |diff| {max:G3}, first at {first} " +
                        $"(got {(first >= 0 && first < r.Interleaved.Length ? r.Interleaved[first] : float.NaN):R}, " +
                        $"want {(first >= 0 && first < want.Length ? want[first] : float.NaN):R})");
                // the planar output holds the same values
                var p = await dec.DecodeFloatAsync(file);
                var planar = p.PlanarFloat!;
                long n = r.TotalSamplesPerChannel;
                if (p.Interleaved.Length != 0 || planar.Length != r.Interleaved.Length || p.TotalSamplesPerChannel != n)
                    failures.Add($"{c.Name}: planar decode gives {planar.Length} floats, {p.Interleaved.Length} interleaved");
                else
                    for (long s = 0; s < n; s++)
                        for (int k = 0; k < c.Channels; k++)
                            if (BitConverter.SingleToInt32Bits(planar[k * n + s]) != BitConverter.SingleToInt32Bits(r.Interleaved[s * c.Channels + k]))
                            {
                                failures.Add($"{c.Name}: planar channel {k} sample {s} differs from the interleaved decode");
                                s = n;
                                break;
                            }
            }
            Console.WriteLine(string.Join("\n", report));
            Console.WriteLine($"{report.Count} files, worst max |diff| vs minimp3 {worst:G3} ({(exact ? "bit-exact required" : $"tolerance {tolerance}")})");
            True(failures.Count == 0, string.Join("\n", failures));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    /// <summary>FFmpeg 8.1's own MP3 decoder (mp3float, a different implementation) against ours: same length, max |diff| and
    /// SNR reported per file; the floor is far below what the files measure (a wrong decode is far below it).</summary>
    [TestMethod(Timeout = 900000, Category = "DesktopOnly,NotOnWebGL")]
    public async Task Mp3DecoderGpu_Fixtures_CloseToFfmpeg()
    {
        FfmpegOracle.RequireAvailable();
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            const double minSnr = 90;
            using var dec = new Mp3DecoderGpu(acc);
            var failures = new List<string>();
            var report = new List<string>();
            foreach (var c in Mp3DecCases())
            {
                byte[] file = ReadEmbedded(Mp3DecPrefix + c.Name + ".mp3");
                var r = await dec.DecodeAsync(file);
                // FFmpeg does not open free-format streams: free_st44 holds cbr128_st44's frames, so FFmpeg decodes that file.
                // Lenient: the MediaFoundation files overrun part2_3 in places, which FFmpeg's error detection turns into a
                // failure; decoded as a player decodes them, they are compared as the others.
                byte[] ffInput = c.Name == "free_st44" ? ReadEmbedded(Mp3DecPrefix + "cbr128_st44.mp3") : file;
                float[] ff;
                try { ff = FfmpegOracle.DecodeFloat(ffInput, ".mp3", strict: !c.Name.StartsWith("is_")); }
                catch (Exception e) { failures.Add($"{c.Name}: FFmpeg: {e.Message}"); continue; }
                long n = Math.Min(ff.Length, r.Interleaved.Length);
                double sig = 0, err = 0, max = 0;
                for (long i = 0; i < n; i++)
                {
                    double d = (double)r.Interleaved[i] - ff[i];
                    sig += (double)ff[i] * ff[i];
                    err += d * d;
                    max = Math.Max(max, Math.Abs(d));
                }
                double snr = 10 * Math.Log10(sig / Math.Max(err, 1e-30));
                report.Add($"{c.Name}: vs FFmpeg max |diff| {max:G3}, SNR {snr:F1} dB, {r.Interleaved.Length / c.Channels} vs {ff.Length / c.Channels} samples");
                if (ff.Length != r.Interleaved.Length) failures.Add($"{c.Name}: {r.Interleaved.Length / c.Channels} samples, FFmpeg gives {ff.Length / c.Channels}");
                if (snr < minSnr) failures.Add($"{c.Name}: SNR vs FFmpeg {snr:F1} dB < {minSnr}");
            }
            Console.WriteLine(string.Join("\n", report));
            True(failures.Count == 0, string.Join("\n", failures));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    /// <summary>Files from our own encoder (Info frame: delay 528, padding frames * N - samples - 528) decode to exactly the
    /// source's length, aligned with it (no delay).</summary>
    [TestMethod(Timeout = 900000, Category = "NotOnWebGL")]
    public async Task Mp3DecoderGpu_OwnEncoder_GaplessLength()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var enc = new Mp3EncoderGpu(acc);
            using var dec = new Mp3DecoderGpu(acc);
            var failures = new List<string>();
            var report = new List<string>();
            // Not here: CBR rates whose frames are under 169 bytes (16 kHz 32k, 8 kHz 16k: 144 bytes). Mp3EncoderGpu writes its
            // Info frame at the stream's rate, and a 144-byte frame cannot hold the Xing + LAME tag: the tag runs into the
            // first audio frame, minimp3 (and this decoder) bound-check the tag and decode the Info frame as audio.
            // Reported for the encoder (it should pick a larger rate for the Info frame, as its VBR path does).
            foreach (var (rate, ch, kbps, vbr) in new (int, int, int, int?)[]
                     {
                         (44100, 2, 128, null), (48000, 1, 192, null), (32000, 2, 96, null), (22050, 2, 64, null),
                         (16000, 1, 48, null), (8000, 1, 24, null), (11025, 2, 32, null), (44100, 2, 0, 0), (24000, 1, 0, 4),
                     })
            {
                int samples = rate / 3 + 1234;
                var src = Mp3TestTone(samples, ch, rate, rate + 7 * ch);
                var options = vbr is int q ? new Mp3EncodeOptions { VbrQuality = q } : new Mp3EncodeOptions { BitrateKbps = kbps };
                byte[] file = await enc.EncodeAsync(src, ch, rate, options);
                string name = $"{rate} Hz {ch} ch {(vbr is int v ? $"V{v}" : $"{kbps}k")}";
                // diagnostics: MP3_TEST_DUMP=<dir> keeps the files
                string? dump = Environment.GetEnvironmentVariable("MP3_TEST_DUMP");
                if (!string.IsNullOrEmpty(dump) && !OperatingSystem.IsBrowser())
                    File.WriteAllBytes(Path.Combine(dump, $"own_{rate}_{ch}_{(vbr is int v2 ? $"v{v2}" : $"{kbps}k")}.mp3"), file);
                var r = await dec.DecodeAsync(file);
                var (snr, delay) = AlignedSnr(src, r.Interleaved, ch, 64);
                report.Add($"{name}: {r.TotalSamplesPerChannel} samples (source {samples}), trims {r.StartTrim} / {r.EndTrim}, SNR {snr:F1} dB at delay {delay}");
                if (r.TotalSamplesPerChannel != samples) failures.Add($"{name}: {r.TotalSamplesPerChannel} samples, want {samples}");
                if (delay != 0) failures.Add($"{name}: decoded audio sits {delay} samples late");
                if (snr < 20) failures.Add($"{name}: SNR {snr:F1} dB");
            }
            Console.WriteLine(string.Join("\n", report));
            True(failures.Count == 0, string.Join("\n", failures));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod(Timeout = 900000, Category = "NotOnWebGL")]
    public async Task Mp3DecoderGpu_Id3v2Tags_Read()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var dec = new Mp3DecoderGpu(acc);
            var r = await dec.DecodeAsync(ReadEmbedded(Mp3DecPrefix + "tags_st44.mp3"));
            var tags = r.Tags.ToDictionary(t => t.Key, t => t.Value);
            Console.WriteLine(string.Join("\n", r.Tags.Select(t => $"{t.Key} = {t.Value}")));
            Equal("{\"3\": {\"class_type\": \"KSampler\"}}", tags.GetValueOrDefault("prompt"), "TXXX prompt");
            Equal("spawndev", tags.GetValueOrDefault("workflow"), "TXXX workflow");
            Equal("MP3 decoder fixture", tags.GetValueOrDefault("TIT2"), "TIT2");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    /// <summary>
    /// Damage: (a) a frame header wiped: the walk resyncs, the frame (and any frame whose bit reservoir reached into it)
    /// comes out as zero lines, the length and every later sample's time are kept, the lost frame ends in exact
    /// silence, and from three frames on the output is the clean file's bit for bit; (b) a frame's main data garbled:
    /// finite output, same length, clean again five frames on; (c) the file cut inside a frame: no throw, the samples
    /// before the cut are the clean file's.
    /// </summary>
    [TestMethod(Timeout = 900000, Category = "NotOnWebGL")]
    public async Task Mp3DecoderGpu_DamagedFrames_SilenceAndRecovery()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var dec = new Mp3DecoderGpu(acc);
            byte[] clean = ReadEmbedded(Mp3DecPrefix + "cbr128_st44.mp3");
            var frames = Mp3TestFrames(clean);
            True(frames.Count > 20, $"{frames.Count} frames");
            var r0 = await dec.DecodeAsync(clean);
            int ch = r0.Channels;
            long skip = r0.StartTrim;
            // sample index (per channel) of audio frame k's first sample in the trimmed output
            long At(int k) => (long)k * 1152 - skip;
            void SameFrom(Mp3DecodeResult r, long fromSample, long toSample, string what)
            {
                for (long i = Math.Max(0, fromSample) * ch; i < Math.Min(toSample, r.TotalSamplesPerChannel) * ch; i++)
                    if (BitConverter.SingleToInt32Bits(r.Interleaved[i]) != BitConverter.SingleToInt32Bits(r0.Interleaved[i]))
                        throw new Exception($"{what}: sample {i / ch} channel {i % ch} = {r.Interleaved[i]:R}, clean file {r0.Interleaved[i]:R}");
            }
            void Finite(Mp3DecodeResult r, string what)
            {
                foreach (var v in r.Interleaved)
                    if (!float.IsFinite(v)) throw new Exception($"{what}: non-finite sample");
            }

            // (a) wiped header of audio frame 10 (file frame 11: frame 0 is the Info frame)
            var a = (byte[])clean.Clone();
            for (int i = 0; i < 4; i++) a[frames[11].Offset + i] = 0;
            var ra = await dec.DecodeAsync(a);
            Console.WriteLine($"(a) wiped header: {ra.Frames} frames, {ra.LostFrames} lost, {ra.TotalSamplesPerChannel} samples (clean {r0.TotalSamplesPerChannel})");
            Equal(r0.TotalSamplesPerChannel, ra.TotalSamplesPerChannel, "(a) length");
            Equal(r0.Frames, ra.Frames, "(a) frames");
            True(ra.LostFrames >= 1, "(a) no frame reported lost");
            Finite(ra, "(a)");
            SameFrom(ra, 0, At(10), "(a) before the damage");
            SameFrom(ra, At(13), long.MaxValue, "(a) after the damage");
            // the lost frame's lines are zero: its last 3 slots (96 samples) see only its own second granule
            void Silent(Mp3DecodeResult r, int k0, string what)
            {
                for (long s = At(k0) + 1056; s < At(k0 + 1); s++)
                    for (int k = 0; k < ch; k++)
                        if (r.Interleaved[s * ch + k] != 0f) throw new Exception($"{what}: not silent at sample {s}: {r.Interleaved[s * ch + k]:R}");
            }
            Silent(ra, 10, "(a) lost frame");
            // audio frame 11 reaches main_data_begin bytes back, into frame 10's (lost) main data: minimp3's reservoir has
            // none of them, so it is not decoded either
            int mdb11 = (clean[frames[12].Offset + 4] << 1) | (clean[frames[12].Offset + 5] >> 7);
            Console.WriteLine($"(a) audio frame 11 main_data_begin {mdb11}");
            if (mdb11 > 0)
            {
                True(ra.LostFrames >= 2, $"(a) {ra.LostFrames} lost: frame 11 needed the lost frame's main data");
                Silent(ra, 11, "(a) frame 11 (its reservoir was lost)");
            }

            // (b) garbled main data of audio frame 15
            var b = (byte[])clean.Clone();
            var rng = new Random(1108);
            for (int i = frames[16].Offset + 4 + 32; i < frames[16].Offset + frames[16].Size; i++) b[i] = (byte)rng.Next(256);
            var rb = await dec.DecodeAsync(b);
            Console.WriteLine($"(b) garbled main data: {rb.LostFrames} lost, {rb.TotalSamplesPerChannel} samples");
            Equal(r0.TotalSamplesPerChannel, rb.TotalSamplesPerChannel, "(b) length");
            Finite(rb, "(b)");
            SameFrom(rb, 0, At(15), "(b) before the damage");
            SameFrom(rb, At(20), long.MaxValue, "(b) after the damage");

            // (c) cut inside audio frame 19
            var cut = clean.AsSpan(0, frames[20].Offset + frames[20].Size / 2).ToArray();
            var rc = await dec.DecodeAsync(cut);
            Console.WriteLine($"(c) cut: {rc.Frames} frames, {rc.TotalSamplesPerChannel} samples");
            True(rc.TotalSamplesPerChannel <= r0.TotalSamplesPerChannel && rc.TotalSamplesPerChannel >= At(18), $"(c) {rc.TotalSamplesPerChannel} samples");
            Finite(rc, "(c)");
            SameFrom(rc, 0, At(18), "(c) before the cut");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    /// <summary>
    /// A stream longer than one decode batch (1024 frames): a fixture's audio frames repeated 45 times (no Info frame:
    /// no trim). The decoder state at the start of every repetition after the first is the same, so repetitions 2..45
    /// decode to repetition 1's samples bit for bit, across the batch seam (the carried IMDCT overlap and synthesis
    /// history). Then 200000 random bytes before the frames: the walk scans them over several launches and more than one
    /// round of launches (resumed state, the host's loop until the walk is done) and the decode is the clean file's.
    /// </summary>
    [TestMethod(Timeout = 900000, Category = "NotOnWebGL")]
    public async Task Mp3DecoderGpu_LongStream_BatchSeamsAndResumedWalk()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var dec = new Mp3DecoderGpu(acc);
            byte[] clean = ReadEmbedded(Mp3DecPrefix + "cbr128_st44.mp3");
            var frames = Mp3TestFrames(clean);
            int audioStart = frames[1].Offset, audioEnd = frames[^1].Offset + frames[^1].Size;
            int perRep = frames.Count - 1, reps = 45;
            var unit = clean.AsSpan(audioStart, audioEnd - audioStart).ToArray();
            var longFile = new byte[unit.Length * reps];
            for (int r = 0; r < reps; r++) Buffer.BlockCopy(unit, 0, longFile, r * unit.Length, unit.Length);
            var rl = await dec.DecodeAsync(longFile);
            Console.WriteLine($"long: {rl.Frames} frames, {rl.TotalSamplesPerChannel} samples, {rl.LostFrames} lost");
            Equal(perRep * reps, rl.Frames, "frames");
            Equal((long)perRep * reps * 1152, rl.TotalSamplesPerChannel, "samples (no Info frame: no trim)");
            Equal(0, rl.LostFrames, "lost frames");
            long repFloats = (long)perRep * 1152 * rl.Channels;
            for (int r = 2; r < reps; r++)
                for (long i = 0; i < repFloats; i++)
                    if (BitConverter.SingleToInt32Bits(rl.Interleaved[r * repFloats + i]) != BitConverter.SingleToInt32Bits(rl.Interleaved[repFloats + i]))
                        throw new Exception($"repetition {r} (frame {r * perRep + i / rl.Channels / 1152}) sample {i / rl.Channels} differs from repetition 1");

            // random bytes between the ID3v2 tag and the Info frame
            var r0 = await dec.DecodeAsync(clean);
            var junk = new byte[200000];
            new Random(42).NextBytes(junk);
            var withJunk = new byte[clean.Length + junk.Length];
            Buffer.BlockCopy(clean, 0, withJunk, 0, frames[0].Offset);
            Buffer.BlockCopy(junk, 0, withJunk, frames[0].Offset, junk.Length);
            Buffer.BlockCopy(clean, frames[0].Offset, withJunk, frames[0].Offset + junk.Length, clean.Length - frames[0].Offset);
            var rj = await dec.DecodeAsync(withJunk);
            Equal(r0.TotalSamplesPerChannel, rj.TotalSamplesPerChannel, "junk prefix: samples");
            var (max, diffs, first) = Mp3Diff(rj.Interleaved, r0.Interleaved);
            True(diffs == 0, $"junk prefix: {diffs} values differ from the clean decode (first {first}, max {max:G3})");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }
}
