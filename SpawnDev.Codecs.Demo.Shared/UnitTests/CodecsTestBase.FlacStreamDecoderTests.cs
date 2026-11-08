// Tests for the full FLAC decoder (FlacDecoderGpu.DecodeAsync, FlacStreamDecodeKernels).
//
// Fixtures (scripts/audio/make_flac_decoder_fixtures.py, TestData/FlacDec): files from FFmpeg 8.1 (levels 0 / 5 / 8 / 12,
// fixed-only, every stereo mode, exact Rice parameters, block sizes 192 / 1000 / 16384, 16 / 24 / 32 bits, 1 / 2 / 6
// channels, noise, wasted bits, constant runs), libFLAC (8 / 16 / 24 bits, 6 channels), a variable-block-size stream
// and hand-made frames (escaped partitions of 0..16 raw bits, Rice2 with parameter 20, LPC order 32 at precision 15,
// negative coefficients, constant / verbatim subframes, wasted bits, STREAMINFO length 0). libFLAC (FFmpeg for 32-bit and
// unknown-length files) decoded every file to its source when the fixtures were made.
//
// What must hold: every file decodes to its source exactly, with STREAMINFO's rate / channels / bits; the MD5 matches
// where STREAMINFO has one; a damaged frame comes out as silence of its length while every other sample keeps its place.

using SpawnDev.Codecs.Audio.Flac;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private const string FlacDecPrefix = "SpawnDev.Codecs.Demo.Shared.TestData.FlacDec.";

    private sealed record FlacDecCase(string Name, string Source, int Channels, int Rate, int Bits, long Samples, string Note);

    private static List<FlacDecCase> FlacDecCases() =>
        System.Text.Encoding.UTF8.GetString(ReadEmbedded(FlacDecPrefix + "manifest.tsv")).Split('\n')
            .Where(l => l.Trim().Length > 0)
            .Select(l => l.TrimEnd('\r').Split('\t'))
            .Select(c => new FlacDecCase(c[0], c[1], int.Parse(c[2]), int.Parse(c[3]), int.Parse(c[4]), long.Parse(c[5]), c[6]))
            .ToList();

    private static int[] FlacDecSource(string source)
    {
        var raw = Gunzip(ReadEmbedded(FlacDecPrefix + source + ".s32.gz"));
        var s = new int[raw.Length / 4];
        Buffer.BlockCopy(raw, 0, s, 0, raw.Length);
        return s;
    }

    [TestMethod(Timeout = 900000, Category = "NotOnWebGL")]
    public async Task FlacDecoderGpu_Fixtures_DecodeToTheSource()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var dec = new FlacDecoderGpu(acc);
            var failures = new List<string>();
            int md5Checked = 0;
            var sources = new Dictionary<string, int[]>();
            foreach (var c in FlacDecCases())
            {
                if (!sources.TryGetValue(c.Source, out var src)) sources[c.Source] = src = FlacDecSource(c.Source);
                FlacDecodeResult r;
                try { r = await dec.DecodeAsync(ReadEmbedded(FlacDecPrefix + c.Name + ".flac")); }
                catch (Exception e) { failures.Add($"{c.Name}: {e.GetType().Name}: {e.Message}"); continue; }
                if (r.SampleRateHz != c.Rate || r.Channels != c.Channels || r.BitsPerSample != c.Bits || r.TotalSamplesPerChannel != c.Samples)
                {
                    failures.Add($"{c.Name}: {r.SampleRateHz} Hz {r.Channels} ch {r.BitsPerSample} bit {r.TotalSamplesPerChannel} samples, " +
                        $"want {c.Rate} / {c.Channels} / {c.Bits} / {c.Samples}");
                    continue;
                }
                int bad = FirstDifference(r.InterleavedSamples, src);
                if (bad >= 0)
                {
                    failures.Add($"{c.Name} ({c.Note}): sample {bad} = {(bad < r.InterleavedSamples.Length ? r.InterleavedSamples[bad] : 0)}, " +
                        $"want {(bad < src.Length ? src[bad] : 0)}; frames {r.Frames}, bad {r.BadFrames}, lost {r.LostSamples}");
                    continue;
                }
                if (r.BadFrames != 0 || r.LostSamples != 0) failures.Add($"{c.Name}: {r.BadFrames} bad frames, {r.LostSamples} lost samples");
                if (r.Md5Checked)
                {
                    md5Checked++;
                    if (!r.Md5Matches) failures.Add($"{c.Name}: MD5 differs from STREAMINFO's");
                }
            }
            Console.WriteLine($"{FlacDecCases().Count} FLAC files, {md5Checked} with an MD5 checked");
            True(md5Checked > 0, "no file had its MD5 checked");
            True(failures.Count == 0, string.Join("\n", failures));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    /// <summary>
    /// DecodeFloatAsync gives what ComfyUI's Load Audio gets through PyAV: FFmpeg's FLAC decoder shifts samples into
    /// int16 (bits &lt;= 16) or int32 (more), ComfyUI casts that to float32 and divides by 2^15 or 2^31; planar
    /// [channel][sample]. Every fixture (8 / 16 / 24 / 32 bits, 1 / 2 / 6 channels), bit for bit.
    /// </summary>
    [TestMethod(Timeout = 900000, Category = "NotOnWebGL")]
    public async Task FlacDecoderGpu_DecodeFloat_AsComfyUiLoadAudio()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var dec = new FlacDecoderGpu(acc);
            var failures = new List<string>();
            var sources = new Dictionary<string, int[]>();
            foreach (var c in FlacDecCases())
            {
                if (!sources.TryGetValue(c.Source, out var src)) sources[c.Source] = src = FlacDecSource(c.Source);
                var r = await dec.DecodeFloatAsync(ReadEmbedded(FlacDecPrefix + c.Name + ".flac"));
                var f = r.PlanarFloat!;
                long n = c.Samples;
                if (f.Length != n * c.Channels || r.InterleavedSamples.Length != 0) { failures.Add($"{c.Name}: {f.Length} floats"); continue; }
                for (long s = 0; s < n; s++)
                {
                    bool bad = false;
                    for (int k = 0; k < c.Channels; k++)
                    {
                        int v = src[s * c.Channels + k];
                        // PyAV's sample (FFmpeg flacdec: s16 for <= 16 bits, s32 above), then torch's int -> float32 / 2^15 or 2^31
                        float want = c.Bits <= 16 ? (float)(v << (16 - c.Bits)) / 32768f : (float)(v << (32 - c.Bits)) / 2147483648f;
                        float got = f[k * n + s];
                        if (BitConverter.SingleToInt32Bits(got) != BitConverter.SingleToInt32Bits(want))
                        {
                            failures.Add($"{c.Name}: channel {k} sample {s} = {got:R}, want {want:R}");
                            bad = true;
                            break;
                        }
                    }
                    if (bad) break;
                }
            }
            True(failures.Count == 0, string.Join("\n", failures));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    /// <summary>The first index where two sample arrays differ (or the shorter length), -1 when equal.</summary>
    private static int FirstDifference(int[] a, int[] b)
    {
        int n = Math.Min(a.Length, b.Length);
        for (int i = 0; i < n; i++) if (a[i] != b[i]) return i;
        return a.Length == b.Length ? -1 : n;
    }

    /// <summary>Byte offsets of the frames of a FLAC file (sync code + the next sync code's CRC-16), for damaging one.</summary>
    private static List<int> FlacFrameOffsets(byte[] f)
    {
        int p = 4;
        while (true)
        {
            bool last = (f[p] & 0x80) != 0;
            p += 4 + (f[p + 1] << 16 | f[p + 2] << 8 | f[p + 3]);
            if (last) break;
        }
        var starts = new List<int>();
        int i = p;
        while (i < f.Length)
        {
            starts.Add(i);
            int j = i + 2;
            while (true)
            {
                while (j + 1 < f.Length && !(f[j] == 0xFF && (f[j + 1] & 0xFE) == 0xF8)) j++;
                if (j + 1 >= f.Length) { j = f.Length; break; }
                if (Crc16(f, i, j - i) == 0) break;
                j++;
            }
            i = j;
        }
        return starts;
    }

    private static int Crc16(byte[] b, int at, int count)
    {
        int c = 0;
        for (int i = 0; i < count; i++)
        {
            c ^= b[at + i] << 8;
            for (int k = 0; k < 8; k++) c = (c & 0x8000) != 0 ? ((c << 1) ^ 0x8005) & 0xFFFF : (c << 1) & 0xFFFF;
        }
        return c;
    }

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task FlacDecoderGpu_DamagedFrames_ComeOutAsSilence_OthersKeepTheirPlace()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var dec = new FlacDecoderGpu(acc);
            var failures = new List<string>();
            // fixed blocking (FFmpeg level 5) and variable blocking (sample numbers in the headers)
            foreach (var (name, frameIndex) in new[] { ("ff_st16_l5", 3), ("var_blocksize", 9), ("var_blocksize", 12) })
            {
                var good = ReadEmbedded(FlacDecPrefix + name + ".flac");
                var c = FlacDecCases().First(x => x.Name == name);
                var src = FlacDecSource(c.Source);
                var frames = FlacFrameOffsets(good);
                int start = frames[frameIndex], end = frames[frameIndex + 1];
                var ok = await dec.DecodeAsync(good);
                // which samples the frame holds: decode the intact file's frames up to it
                foreach (var (what, at) in new[] { ("a subframe byte (CRC-16)", start + (end - start) / 2), ("a header byte (CRC-8)", start + 2) })
                {
                    var bad = (byte[])good.Clone();
                    bad[at] ^= 0x10;
                    var r = await dec.DecodeAsync(bad);
                    if (r.TotalSamplesPerChannel != c.Samples) { failures.Add($"{name} {what}: {r.TotalSamplesPerChannel} samples, want {c.Samples}"); continue; }
                    // the silent run must be one frame long, everything else intact
                    int ch = c.Channels;
                    long zeroFrom = -1, zeroTo = -1;
                    for (long s = 0; s < c.Samples; s++)
                    {
                        bool same = true;
                        for (int k = 0; k < ch; k++) same &= r.InterleavedSamples[s * ch + k] == src[s * ch + k];
                        if (same) continue;
                        bool silent = true;
                        for (int k = 0; k < ch; k++) silent &= r.InterleavedSamples[s * ch + k] == 0;
                        if (!silent) { failures.Add($"{name} {what}: sample {s} is neither the source nor silence"); zeroFrom = -2; break; }
                        if (zeroFrom < 0) zeroFrom = s;
                        zeroTo = s;
                    }
                    if (zeroFrom == -2) continue;
                    if (zeroFrom < 0) { failures.Add($"{name} {what}: nothing was lost"); continue; }
                    if (r.LostSamples < zeroTo - zeroFrom + 1 - 0 || r.LostSamples > 16384)
                        failures.Add($"{name} {what}: LostSamples {r.LostSamples} for a silent run {zeroFrom}..{zeroTo}");
                    if (r.Frames != ok.Frames - 1) failures.Add($"{name} {what}: {r.Frames} frames, want {ok.Frames - 1}");
                    if (r.Md5Checked && r.Md5Matches) failures.Add($"{name} {what}: MD5 still matches");
                }
            }

            // a frame whose CRCs pass but whose subframe type is reserved: a bad frame, silent, the rest intact
            {
                var good = ReadEmbedded(FlacDecPrefix + "ff_st16_l5.flac");
                var src = FlacDecSource("st16_44k");
                var frames = FlacFrameOffsets(good);
                int start = frames[2], end = frames[3];
                var bad = (byte[])good.Clone();
                // the first subframe header follows the frame header: FFmpeg level 5 frame 2 has a 1-byte frame number and
                // block size code 5 (4608, no extra bytes), so the header is 6 bytes with CRC-8; the subframe header byte
                // becomes "type 2" (reserved)
                int sub = start + 6;
                bad[sub] = 0x04;
                int crc = Crc16(bad, start, end - start - 2);
                bad[end - 2] = (byte)(crc >> 8);
                bad[end - 1] = (byte)crc;
                var r = await dec.DecodeAsync(bad);
                if (r.BadFrames != 1) failures.Add($"reserved subframe type: {r.BadFrames} bad frames, want 1");
                int diff = FirstDifference(r.InterleavedSamples, src);
                if (diff != 2 * 4608 * 2) failures.Add($"reserved subframe type: first difference at {diff}, want {2 * 4608 * 2}");
                for (int i = 2 * 4608 * 2; i < 3 * 4608 * 2; i++)
                    if (r.InterleavedSamples[i] != 0) { failures.Add($"reserved subframe type: sample {i} not silent"); break; }
                if (FirstDifference(r.InterleavedSamples[(3 * 4608 * 2)..], src[(3 * 4608 * 2)..]) >= 0) failures.Add("reserved subframe type: later samples differ");
            }
            True(failures.Count == 0, string.Join("\n", failures));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }
}
