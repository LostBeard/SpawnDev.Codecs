// Tests for SnappyCompressorGpu and SnappyDecompressorGpu (Compression/Snappy).
//
// Compressor: raw and framed streams BYTE-IDENTICAL with the Rust snap crate (cramjam, python-snappy's backend). The
// truth is snap's own output, never our code. Fixtures (TestData/Snappy, scripts/compression/make_snappy_fixtures.py):
// procedural text / noise / structured / mixed inputs (the generators below rebuild the same bytes) at 0, 1, 15 .. 18
// (the 17-byte minimum for a non-literal block), 64 KB +- 1, 200 KB .. 40 MB (two waves at the default wave size),
// incompressible blocks, all-zero data (long overlapping copies), mixed_65536 (its raw stream lands between 6/7 and 7/8
// of the block: decides the framed "stored uncompressed" rule), and three real media files.
// What can fail: the table size / hash shift, the skip schedule, match extension, literal and copy forms (snap's
// 68 / 64 / 60 copy split), the resumable state across launches, the wave layout and copy, the framed chunk headers and
// CRC-32C, the stored-chunk rule, writes past the stream (output poisoned).
//
// Decompressor: snap's streams, streams snap never writes (TestData/SnappyDec: a whole-stream greedy encoder with
// copies 400 KB back, 1- to 4-byte literal length forms, copy-4 elements; framed streams with chunks of any size,
// uncompressed / padding / skippable chunks, a repeated stream identifier) decode to their content; damaged streams
// are refused with the reason; every compressor fixture decodes back to its input.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Compression.Snappy;
using SpawnDev.ILGPU;
using SpawnDev.SpawnJS.JSObjects;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private const string SnappyFixturePrefix = "SpawnDev.Codecs.Demo.Shared.TestData.Snappy.";
    private const string SnappyDecFixturePrefix = "SpawnDev.Codecs.Demo.Shared.TestData.SnappyDec.";

    /// <summary>Noise, except every period-th 32-byte run repeats the run before it (make_snappy_fixtures.gen_mixed).</summary>
    private static byte[] SnappyGenMixed(int n, int seed, int period = 4)
    {
        var outp = new byte[n];
        uint x = (uint)(seed | 1);
        for (int i = 0; i < n; i++)
        {
            if ((i / 32) % period == period - 1 && i >= 32)
            {
                outp[i] = outp[i - 32];
                continue;
            }
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            outp[i] = (byte)x;
        }
        return outp;
    }

    /// <summary>Text, noise, then the same text (make_snappy_fixtures.gen_far).</summary>
    private static byte[] SnappyGenFar(int n, int seed)
    {
        int a = n * 3 / 7;
        int b = n - 2 * a;
        var t = Lz4GenText(a, seed);
        return t.Concat(Lz4GenNoise(b, seed + 1)).Concat(t).ToArray();
    }

    private static byte[] SnappyContent(string kind, int n, int seed) => kind switch
    {
        "text" => Lz4GenText(n, seed),
        "noise" => Lz4GenNoise(n, seed),
        "struct" => Lz4GenStruct(n, seed),
        "mixed" => SnappyGenMixed(n, seed),
        "far" => SnappyGenFar(n, seed),
        _ when kind.StartsWith("file:", StringComparison.Ordinal) => ReadEmbedded("SpawnDev.Codecs.Demo.Shared.TestData." + kind.Substring(5)),
        _ => throw new InvalidOperationException($"unknown kind {kind}"),
    };

    private sealed record SnappyCase(string Name, string Kind, int Length, int Seed, int RawLength, string RawSha, bool RawStored,
        int FramedLength, string FramedSha, bool FramedStored)
    {
        public int StreamLength(SnappyFormat f) => f == SnappyFormat.Raw ? RawLength : FramedLength;
        public string Sha(SnappyFormat f) => f == SnappyFormat.Raw ? RawSha : FramedSha;
        public bool Stored(SnappyFormat f) => f == SnappyFormat.Raw ? RawStored : FramedStored;
        public string File(SnappyFormat f) => SnappyFixturePrefix + Name + (f == SnappyFormat.Raw ? ".raw.sz" : ".framed.sz");
    }

    private static List<SnappyCase> SnappyCases()
    {
        var list = new List<SnappyCase>();
        foreach (var raw in ReadEmbeddedText(SnappyFixturePrefix + "manifest.tsv").Split('\n'))
        {
            var line = raw.Trim('\r', ' ');
            if (line.Length == 0 || line.StartsWith('#')) continue;
            var f = line.Split('\t');
            list.Add(new SnappyCase(f[0], f[1], int.Parse(f[2]), int.Parse(f[3]), int.Parse(f[4]), f[5], f[6] == "1",
                int.Parse(f[7]), f[8], f[9] == "1"));
        }
        return list;
    }

    private static byte[] SnappyInput(SnappyCase c) => SnappyContent(c.Kind, c.Length, c.Seed);

    private static readonly SnappyFormat[] SnappyFormats = { SnappyFormat.Raw, SnappyFormat.Framed };

    /// <summary>Compresses one case through the ArrayView1D overload into a poisoned buffer; null when it matches snap.</summary>
    private static async Task<string?> RunSnappyCaseAsync(Accelerator acc, SnappyCompressorGpu comp, SnappyCase c, byte[] input, SnappyFormat format)
    {
        string name = $"{c.Name} {format}";
        if (input.Length != c.Length) return $"{name}: input is {input.Length} bytes, manifest says {c.Length}";
        long cap = SnappyCompressorGpu.GetMaxCompressedSize(input.Length, format) + 41;
        using var dIn = DeviceVerify.Upload(acc, input, 7);
        using var dOut = DeviceVerify.Poisoned(acc, cap);
        int n = await comp.CompressAsync(dIn.View, input.Length, dOut.View, format);
        if (n != c.StreamLength(format)) return $"{name}: stream is {n} bytes, snap wrote {c.StreamLength(format)}";
        if (c.Stored(format))
        {
            byte[] truth = ReadEmbedded(c.File(format));
            var withPoison = new byte[truth.Length + 32];
            truth.CopyTo(withPoison, 0);
            System.Array.Fill(withPoison, (byte)0xAB, truth.Length, 32);
            string? diff = await DeviceVerify.CompareAsync(acc, dOut.View, withPoison);
            return diff == null ? null : $"{name}: {diff}";
        }
        string sha = Sha256Hex(await DeviceTransfer.DownloadAsync(dOut.View, n));
        return sha == c.Sha(format) ? null : $"{name}: SHA-256 {sha} != snap {c.Sha(format)}";
    }

    private async Task AssertSnappyCasesAsync(Func<SnappyCase, bool> which, int waveBlocks, int minCases)
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, Lz4Requirements);
        try
        {
            using var comp = new SnappyCompressorGpu(acc, waveBlocks);
            var cases = SnappyCases().Where(which).ToList();
            True(cases.Count >= minCases, $"only {cases.Count} Snappy fixtures selected");
            var failures = new List<string>();
            foreach (var c in cases)
            {
                byte[] input = SnappyInput(c);
                foreach (var format in SnappyFormats)
                {
                    string? r = await RunSnappyCaseAsync(acc, comp, c, input, format);
                    if (r != null) failures.Add(r);
                }
            }
            True(failures.Count == 0, $"{failures.Count} of {cases.Count * 2} streams differ from snap:\n" + string.Join("\n", failures));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task SnappyCompressorGpu_AllSmallFixtures_ByteIdenticalWithSnap() =>
        await AssertSnappyCasesAsync(c => c.Length <= 1_100_000, SnappyCompressorGpu.DefaultWaveBlocks, 27);

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task SnappyCompressorGpu_ManySmallWaves_ByteIdenticalWithSnap() =>
        await AssertSnappyCasesAsync(c => c.Name is "struct_1000003" or "mixed_200000" or "noise_300000", 3, 3);

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task SnappyCompressorGpu_40MB_TwoWaves_ByteIdenticalWithSnap() =>
        await AssertSnappyCasesAsync(c => c.Length > 1_100_000, SnappyCompressorGpu.DefaultWaveBlocks, 1);

    [TestMethod(Category = "NotOnWebGL")]
    public async Task SnappyCompressorGpu_ManagedOverload_MatchesSnap()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, Lz4Requirements);
        try
        {
            using var comp = new SnappyCompressorGpu(acc, 8);
            foreach (var c in SnappyCases().Where(c => c.Length is 0 or 17 or 65537 or 1000 or 200000))
                foreach (var format in SnappyFormats)
                {
                    byte[] stream = await comp.CompressAsync(SnappyInput(c), format);
                    EqualBytes(ReadEmbedded(c.File(format)), stream, $"{c.Name} {format}: managed overload");
                }
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod(Category = "BrowserOnly,NotOnWebGL")]
    public async Task SnappyCompressorGpu_TypedArrayOverload_MatchesSnap()
    {
        if (!OperatingSystem.IsBrowser())
            throw new UnsupportedTestException("TypedArray overloads need a browser accelerator.");
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, Lz4Requirements);
        try
        {
            using var comp = new SnappyCompressorGpu(acc, 8);
            await RunSnappyTypedArrayCases(comp);
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [System.Runtime.Versioning.SupportedOSPlatform("browser")]
    private static async Task RunSnappyTypedArrayCases(SnappyCompressorGpu comp)
    {
        foreach (var c in SnappyCases().Where(c => c.Length is 17 or 65537 or 200000))
            foreach (var format in SnappyFormats)
            {
                byte[] input = SnappyInput(c);
                using var js = new Uint8Array(input.Length);
                js.Set(input);
                using var stream = await comp.CompressAsync(js, format);
                EqualBytes(ReadEmbedded(c.File(format)), stream.ReadBytes(), $"{c.Name} {format}: TypedArray overload");
            }
    }

    [TestMethod(Category = "NotOnWebGL")]
    public async Task SnappyCompressorGpu_RejectsBadArguments()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, Lz4Requirements);
        try
        {
            using var comp = new SnappyCompressorGpu(acc, 2);
            using var a = acc.Allocate1D<byte>(1000);
            using var small = acc.Allocate1D<byte>(100);
            await ThrowsAsync<ArgumentException>(async () => await comp.CompressAsync(a.View, 1000, small.View), "output below GetMaxCompressedSize");
            await ThrowsAsync<ArgumentOutOfRangeException>(async () => await comp.CompressAsync(small.View, 1000, a.View), "length past the input");
            using var big = acc.Allocate1D<byte>(SnappyCompressorGpu.GetMaxCompressedSize(1000, SnappyFormat.Framed));
            await ThrowsAsync<ArgumentOutOfRangeException>(async () => await comp.CompressAsync(a.View, 1000, big.View, (SnappyFormat)7), "unknown format");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    // ------------------------------------------------------------------ decompressor

    private sealed record SnappyDecCase(string Name, SnappyFormat Format, string Kind, int Length, int Seed, string Expected, string Description);

    private static List<SnappyDecCase> SnappyDecCases()
    {
        var list = new List<SnappyDecCase>();
        foreach (var raw in ReadEmbeddedText(SnappyDecFixturePrefix + "manifest.tsv").Split('\n'))
        {
            var line = raw.Trim('\r', ' ');
            if (line.Length == 0 || line.StartsWith('#')) continue;
            var f = line.Split('\t');
            list.Add(new SnappyDecCase(f[0], f[1] == "raw" ? SnappyFormat.Raw : SnappyFormat.Framed, f[2], int.Parse(f[3]), int.Parse(f[4]),
                f[5], f.Length > 6 ? f[6] : ""));
        }
        return list;
    }

    private static byte[] SnappyDecContent(SnappyDecCase c) =>
        c.Kind == "file" ? ReadEmbedded(SnappyDecFixturePrefix + c.Name + ".content") : SnappyContent(c.Kind, c.Length, c.Seed);

    /// <summary>Decodes a stream through the ArrayView1D overload into a poisoned buffer and compares it with the content on the accelerator.</summary>
    private static async Task<string?> DecodeSnappyAndCompareAsync(Accelerator acc, SnappyDecompressorGpu dec, string name, byte[] stream, SnappyFormat format, byte[] content)
    {
        using var dIn = DeviceVerify.Upload(acc, stream.Length == 0 ? new byte[1] : stream, 11);
        using var dOut = DeviceVerify.Poisoned(acc, content.Length + 64);
        int n;
        try { n = await dec.DecompressAsync(dIn.View, stream.Length, dOut.View, format); }
        catch (Exception ex) { return $"{name}: threw {ex.GetType().Name}: {ex.Message}"; }
        if (n != content.Length) return $"{name}: decoded {n} bytes, content is {content.Length}";
        var expected = new byte[content.Length + 64];
        content.CopyTo(expected, 0);
        System.Array.Fill(expected, (byte)0xAB, content.Length, 64);
        string? diff = await DeviceVerify.CompareAsync(acc, dOut.View, expected);
        return diff == null ? null : $"{name}: {diff}";
    }

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task SnappyDecompressorGpu_GoodStreams_DecodeToTheContent()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, Lz4Requirements);
        try
        {
            using var dec = new SnappyDecompressorGpu(acc);
            var cases = SnappyDecCases().Where(c => c.Expected == "ok").ToList();
            True(cases.Count >= 11, $"only {cases.Count} good streams embedded");
            var failures = new List<string>();
            foreach (var c in cases)
            {
                string? r = await DecodeSnappyAndCompareAsync(acc, dec, c.Name, ReadEmbedded(SnappyDecFixturePrefix + c.Name + ".sz"), c.Format, SnappyDecContent(c));
                if (r != null) failures.Add($"{r} ({c.Description})");
            }
            True(failures.Count == 0, $"{failures.Count} of {cases.Count} streams decoded wrong:\n" + string.Join("\n", failures));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task SnappyDecompressorGpu_DamagedStreams_RefusedWithTheReason()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, Lz4Requirements);
        try
        {
            using var dec = new SnappyDecompressorGpu(acc);
            var cases = SnappyDecCases().Where(c => c.Expected != "ok").ToList();
            True(cases.Count >= 15, $"only {cases.Count} damaged streams embedded");
            var failures = new List<string>();
            foreach (var c in cases)
            {
                byte[] stream = ReadEmbedded(SnappyDecFixturePrefix + c.Name + ".sz");
                using var dIn = DeviceVerify.Upload(acc, stream.Length == 0 ? new byte[1] : stream);
                using var dOut = DeviceVerify.Poisoned(acc, c.Length + 64);
                try
                {
                    int n = await dec.DecompressAsync(dIn.View, stream.Length, dOut.View, c.Format);
                    failures.Add($"{c.Name}: decoded {n} bytes, expected {c.Expected}");
                }
                catch (SnappyDecompressorGpu.SnappyFormatException ex)
                {
                    if (ex.Reason.ToString() != c.Expected) failures.Add($"{c.Name}: refused with {ex.Reason} at {ex.Offset}, expected {c.Expected}");
                }
                catch (Exception ex) { failures.Add($"{c.Name}: threw {ex.GetType().Name}: {ex.Message}"); }
            }
            True(failures.Count == 0, string.Join("\n", failures));

            // The decoder still works after refusing (no state left behind).
            var good = SnappyDecCases().First(c => c.Name == "text_200000_framed_mixed_chunks");
            string? after = await DecodeSnappyAndCompareAsync(acc, dec, good.Name, ReadEmbedded(SnappyDecFixturePrefix + good.Name + ".sz"), good.Format, SnappyDecContent(good));
            True(after == null, $"after the damaged streams: {after}");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    /// <summary>Every snap stream of the compressor's fixtures (raw and framed, real files included) decodes to its input.</summary>
    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task SnappyDecompressorGpu_CompressorFixtures_DecodeToTheirInputs()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, Lz4Requirements);
        try
        {
            using var dec = new SnappyDecompressorGpu(acc);
            var failures = new List<string>();
            int count = 0;
            foreach (var c in SnappyCases())
            {
                byte[] input = SnappyInput(c);
                foreach (var format in SnappyFormats)
                {
                    if (!c.Stored(format)) continue;
                    count++;
                    string? r = await DecodeSnappyAndCompareAsync(acc, dec, $"{c.Name} {format}", ReadEmbedded(c.File(format)), format, input);
                    if (r != null) failures.Add(r);
                }
            }
            True(count >= 50, $"only {count} streams");
            True(failures.Count == 0, $"{failures.Count} of {count} streams decoded wrong:\n" + string.Join("\n", failures));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    /// <summary>40 MB round trips on the accelerator in both formats: our compressor, then our decompressor, compared with the input on the device.</summary>
    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task Snappy_RoundTrip40MB_OnTheAccelerator()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, Lz4Requirements);
        try
        {
            using var comp = new SnappyCompressorGpu(acc);
            using var dec = new SnappyDecompressorGpu(acc);
            byte[] input = Lz4GenText(40_000_000, 31);
            using var dIn = DeviceVerify.Upload(acc, input);
            foreach (var format in SnappyFormats)
            {
                using var dStream = acc.Allocate1D<byte>(SnappyCompressorGpu.GetMaxCompressedSize(input.Length, format));
                int n = await comp.CompressAsync(dIn.View, input.Length, dStream.View, format);
                using var dBack = DeviceVerify.Poisoned(acc, input.Length + 64);
                int m = await dec.DecompressAsync(dStream.View, n, dBack.View, format);
                Equal(input.Length, m, $"{format}: decoded length");
                string? diff = await DeviceVerify.CompareAsync(acc, dBack.View, dIn.View, input.Length);
                True(diff == null, $"{format} round trip: {diff}");
                True(await DeviceVerify.AllEqualAsync(acc, dBack.View.SubView(input.Length, 64), 64, 0xAB) == null, $"{format}: wrote past the output");
            }
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod(Category = "NotOnWebGL")]
    public async Task SnappyDecompressorGpu_ManagedOverload_AndOutputTooSmall()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, Lz4Requirements);
        try
        {
            using var dec = new SnappyDecompressorGpu(acc);
            var c = SnappyDecCases().First(c => c.Name == "far_700000_greedy_raw");
            byte[] stream = ReadEmbedded(SnappyDecFixturePrefix + c.Name + ".sz");
            EqualBytes(SnappyDecContent(c), await dec.DecompressAsync(stream, SnappyFormat.Raw), "managed overload");
            Equal(0, (await dec.DecompressAsync(System.Array.Empty<byte>(), SnappyFormat.Framed)).Length, "empty framed stream");
            Equal(0, (await dec.DecompressAsync(new byte[] { 0 }, SnappyFormat.Raw)).Length, "empty raw stream");
            using var dIn = DeviceVerify.Upload(acc, stream);
            using var small = acc.Allocate1D<byte>(1000);
            try
            {
                await dec.DecompressAsync(dIn.View, stream.Length, small.View, SnappyFormat.Raw);
                throw new Exception($"output of 1000 bytes accepted for {c.Length}");
            }
            catch (SnappyDecompressorGpu.SnappyFormatException ex) { Equal(SnappyDecompressorGpu.SnappyError.OutputTooSmall, ex.Reason, "reason"); }
            Equal(c.Length, await dec.GetDecompressedLengthAsync(dIn.View, stream.Length, SnappyFormat.Raw), "GetDecompressedLengthAsync");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod(Category = "BrowserOnly,NotOnWebGL")]
    public async Task SnappyDecompressorGpu_TypedArrayOverload()
    {
        if (!OperatingSystem.IsBrowser())
            throw new UnsupportedTestException("TypedArray overloads need a browser accelerator.");
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, Lz4Requirements);
        try
        {
            using var dec = new SnappyDecompressorGpu(acc);
            await RunSnappyDecTypedArray(dec);
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [System.Runtime.Versioning.SupportedOSPlatform("browser")]
    private static async Task RunSnappyDecTypedArray(SnappyDecompressorGpu dec)
    {
        var c = SnappyDecCases().First(c => c.Name == "text_200000_framed_mixed_chunks");
        byte[] stream = ReadEmbedded(SnappyDecFixturePrefix + c.Name + ".sz");
        using var js = new Uint8Array(stream.Length);
        js.Set(stream);
        using var outp = await dec.DecompressAsync(js, c.Format);
        EqualBytes(SnappyDecContent(c), outp.ReadBytes(), "TypedArray overload");
    }
}
