// Tests for GDeflateDecompressorGpu (Compression/GDeflate): GDeflate tile streams written by Microsoft's DirectStorage
// reference implementation (NVIDIA's libdeflate fork), decoded on the accelerator and compared with their content on
// the accelerator. The truth is the content the reference compressed (and decoded back, checked when the fixtures were
// made), never our code.
//
// Fixtures (TestData/GDeflate, scripts/compression/make_gdeflate_fixtures.py): text of 1 B .. 300 KB and 8 MB (123
// tiles), incompressible noise (stored blocks), structured data, all-zero data (long overlapping matches), mixed and
// far-repeat data, three real media files, at compression levels 1, 6, 9 and 12; damaged streams (magic, codec id,
// reserved block type) refused by the reference and by us with the reason.
//
// What can fail: the tile header and table, the 32 sub-stream bit buffers and their refill order, the deferred copies
// (length in one slot, distance in the same sub-stream's next), the end-of-block flush, stored blocks (a byte per slot),
// dynamic headers (precode lengths one per slot, code lengths), static blocks, the first-level tables and the canonical
// walk for long codes, the last tile's size, writes past the output (poisoned).

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Compression.GDeflate;
using SpawnDev.ILGPU;
using SpawnDev.SpawnJS.JSObjects;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private const string GDeflateFixturePrefix = "SpawnDev.Codecs.Demo.Shared.TestData.GDeflate.";

    private sealed record GDeflateCase(string Name, string Kind, int Length, int Seed, int Level, int StreamLength, string Expected, string Description);

    private static List<GDeflateCase> GDeflateCases()
    {
        var list = new List<GDeflateCase>();
        foreach (var raw in ReadEmbeddedText(GDeflateFixturePrefix + "manifest.tsv").Split('\n'))
        {
            var line = raw.Trim('\r', ' ');
            if (line.Length == 0 || line.StartsWith('#')) continue;
            var f = line.Split('\t');
            list.Add(new GDeflateCase(f[0], f[1], int.Parse(f[2]), int.Parse(f[3]), int.Parse(f[4]), int.Parse(f[5]), f[6], f.Length > 7 ? f[7] : ""));
        }
        return list;
    }

    private static byte[] GDeflateContent(GDeflateCase c) => SnappyContent(c.Kind, c.Length, c.Seed);

    /// <summary>Decodes a stream through the ArrayView1D overload into a poisoned buffer and compares it with the content on the accelerator.</summary>
    private static async Task<string?> DecodeGDeflateAndCompareAsync(Accelerator acc, GDeflateDecompressorGpu dec, string name, byte[] stream, byte[] content)
    {
        using var dIn = DeviceVerify.Upload(acc, stream, 9);
        using var dOut = DeviceVerify.Poisoned(acc, content.Length + 64);
        int n;
        try { n = await dec.DecompressAsync(dIn.View, stream.Length, dOut.View); }
        catch (Exception ex) { return $"{name}: threw {ex.GetType().Name}: {ex.Message}"; }
        if (n != content.Length) return $"{name}: decoded {n} bytes, content is {content.Length}";
        var expected = new byte[content.Length + 64];
        content.CopyTo(expected, 0);
        System.Array.Fill(expected, (byte)0xAB, content.Length, 64);
        string? diff = await DeviceVerify.CompareAsync(acc, dOut.View, expected);
        return diff == null ? null : $"{name}: {diff}";
    }

    private async Task AssertGDeflateCasesAsync(Func<GDeflateCase, bool> which, int wave, int minCases, GDeflateDecodeMode mode = GDeflateDecodeMode.LanesPerTile)
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, Lz4Requirements);
        try
        {
            using var dec = new GDeflateDecompressorGpu(acc, wave, mode);
            var cases = GDeflateCases().Where(c => c.Expected == "ok").Where(which).ToList();
            True(cases.Count >= minCases, $"only {cases.Count} GDeflate streams selected");
            var failures = new List<string>();
            foreach (var c in cases)
            {
                string? r = await DecodeGDeflateAndCompareAsync(acc, dec, c.Name, ReadEmbedded(GDeflateFixturePrefix + c.Name + ".gdf"), GDeflateContent(c));
                if (r != null) failures.Add($"{r} ({c.Description})");
            }
            True(failures.Count == 0, $"{failures.Count} of {cases.Count} reference streams decoded wrong:\n" + string.Join("\n", failures));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod(Timeout = 600000)]
    public async Task GDeflateDecompressorGpu_ReferenceStreams_DecodeToTheContent() =>
        await AssertGDeflateCasesAsync(_ => true, GDeflateDecompressorGpu.DefaultWave, 30);

    /// <summary>The one-thread-per-tile mode (the reference decoder's order) on every reference stream.</summary>
    [TestMethod(Timeout = 600000)]
    public async Task GDeflateDecompressorGpu_ThreadPerTile_DecodeToTheContent() =>
        await AssertGDeflateCasesAsync(_ => true, GDeflateDecompressorGpu.DefaultWave, 30, GDeflateDecodeMode.ThreadPerTile);

    [TestMethod(Timeout = 600000)]
    public async Task GDeflateDecompressorGpu_SmallWaves_DecodeToTheContent() =>
        await AssertGDeflateCasesAsync(c => c.Name is "text_8000000_l6" or "struct_1000003_l12" or "far_700000_l12", 7, 3);

    [TestMethod(Timeout = 600000)]
    public async Task GDeflateDecompressorGpu_DamagedStreams_RefusedWithTheReason()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, Lz4Requirements);
        try
        {
            using var dec = new GDeflateDecompressorGpu(acc);
            var cases = GDeflateCases().Where(c => c.Expected != "ok").ToList();
            // Truncation the reference cannot see (it reads past its buffer): cut a good stream's last tile short.
            var good = GDeflateCases().First(c => c.Name == "text_300000_l6");
            byte[] goodStream = ReadEmbedded(GDeflateFixturePrefix + good.Name + ".gdf");
            var extra = new List<(string name, byte[] stream, string expected)>
            {
                ("cut last tile", goodStream.AsSpan(0, goodStream.Length - 8).ToArray(), "BadTileTable"),
                ("header only", goodStream.AsSpan(0, 8).ToArray(), "Truncated"),
                ("4 bytes", goodStream.AsSpan(0, 4).ToArray(), "Truncated"),
            };
            True(cases.Count >= 3, $"only {cases.Count} damaged streams embedded");
            var all = cases.Select(c => (c.Name, ReadEmbedded(GDeflateFixturePrefix + c.Name + ".gdf"), c.Expected)).Concat(extra).ToList();
            var failures = new List<string>();
            foreach (var (name, stream, expected) in all)
            {
                using var dIn = DeviceVerify.Upload(acc, stream);
                using var dOut = DeviceVerify.Poisoned(acc, 400_000);
                try
                {
                    int n = await dec.DecompressAsync(dIn.View, stream.Length, dOut.View);
                    failures.Add($"{name}: decoded {n} bytes, expected {expected}");
                }
                catch (GDeflateDecompressorGpu.GDeflateFormatException ex)
                {
                    if (ex.Reason.ToString() != expected) failures.Add($"{name}: refused with {ex.Reason} at {ex.Offset}, expected {expected}");
                }
                catch (Exception ex) { failures.Add($"{name}: threw {ex.GetType().Name}: {ex.Message}"); }
            }
            True(failures.Count == 0, string.Join("\n", failures));

            // The decoder still works after refusing (no state left behind).
            string? after = await DecodeGDeflateAndCompareAsync(acc, dec, good.Name, goodStream, GDeflateContent(good));
            True(after == null, $"after the damaged streams: {after}");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod]
    public async Task GDeflateDecompressorGpu_ManagedOverload_AndOutputTooSmall()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, Lz4Requirements);
        try
        {
            using var dec = new GDeflateDecompressorGpu(acc);
            var c = GDeflateCases().First(c => c.Name == "far_700000_l12");
            byte[] stream = ReadEmbedded(GDeflateFixturePrefix + c.Name + ".gdf");
            EqualBytes(GDeflateContent(c), await dec.DecompressAsync(stream), "managed overload");
            using var dIn = DeviceVerify.Upload(acc, stream);
            using var small = acc.Allocate1D<byte>(1000);
            try
            {
                await dec.DecompressAsync(dIn.View, stream.Length, small.View);
                throw new Exception($"output of 1000 bytes accepted for {c.Length}");
            }
            catch (GDeflateDecompressorGpu.GDeflateFormatException ex) { Equal(GDeflateDecompressorGpu.GDeflateError.OutputTooSmall, ex.Reason, "reason"); }
            Equal(c.Length, await dec.GetDecompressedLengthAsync(dIn.View, stream.Length), "GetDecompressedLengthAsync");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    // ------------------------------------------------------------------ compressor

    /// <summary>
    /// GDeflateCompressorGpu at every level on the fixture inputs: each stream decodes with our decoder in BOTH modes
    /// (32 lanes and thread per tile) to its input, and at level 3 it is at most 10 % larger than the reference library's
    /// level-6 stream of the same input (a different matcher; the reference itself is checked offline by
    /// scripts/compression/check_gdeflate_encoder.py).
    /// </summary>
    [TestMethod(Timeout = 600000)]
    public async Task GDeflateCompressorGpu_AllLevels_RoundTripThroughBothDecoders()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, Lz4Requirements);
        try
        {
            using var comp = new GDeflateCompressorGpu(acc, 8);
            using var lanes = new GDeflateDecompressorGpu(acc);
            using var serial = new GDeflateDecompressorGpu(acc, 64, GDeflateDecodeMode.ThreadPerTile);
            var cases = GDeflateCases().Where(c => c.Expected == "ok" && c.Level == 6 && c.Length <= 2_000_000).ToList();
            True(cases.Count >= 5, $"only {cases.Count} inputs");
            var failures = new List<string>();
            foreach (var c in cases)
            {
                byte[] input = GDeflateContent(c);
                using var dIn = DeviceVerify.Upload(acc, input, 5);
                for (int level = 0; level <= 4; level++)
                {
                    using var dStream = DeviceVerify.Poisoned(acc, GDeflateCompressorGpu.GetMaxCompressedSize(input.Length) + 32);
                    int n = await comp.CompressAsync(dIn.View, input.Length, dStream.View, level);
                    if (level == 3 && n > c.StreamLength * 1.10)
                        failures.Add($"{c.Name} level 3: {n} bytes, the reference's level 6 is {c.StreamLength}");
                    byte[] stream = await DeviceTransfer.DownloadAsync(dStream.View, n);
                    string? a = await DecodeGDeflateAndCompareAsync(acc, lanes, $"{c.Name} level {level} (lanes)", stream, input);
                    if (a != null) failures.Add(a);
                    string? b = await DecodeGDeflateAndCompareAsync(acc, serial, $"{c.Name} level {level} (thread per tile)", stream, input);
                    if (b != null) failures.Add(b);
                }
            }
            True(failures.Count == 0, $"{failures.Count} failures:\n" + string.Join("\n", failures));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod]
    public async Task GDeflateCompressorGpu_ManagedOverload_AndEmptyInputRefused()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, Lz4Requirements);
        try
        {
            using var comp = new GDeflateCompressorGpu(acc, 4);
            using var dec = new GDeflateDecompressorGpu(acc);
            byte[] input = Lz4GenText(300_000, 77);
            byte[] stream = await comp.CompressAsync(input, 2);
            EqualBytes(input, await dec.DecompressAsync(stream), "managed round trip");
            await ThrowsAsync<ArgumentOutOfRangeException>(async () => await comp.CompressAsync(System.Array.Empty<byte>()), "empty input");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod]
    public async Task GDeflateDecompressorGpu_TypedArrayOverload()
    {
        if (!OperatingSystem.IsBrowser())
            throw new UnsupportedTestException("TypedArray overloads need a browser accelerator.");
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, Lz4Requirements);
        try
        {
            using var dec = new GDeflateDecompressorGpu(acc);
            await RunGDeflateTypedArray(dec);
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [System.Runtime.Versioning.SupportedOSPlatform("browser")]
    private static async Task RunGDeflateTypedArray(GDeflateDecompressorGpu dec)
    {
        var c = GDeflateCases().First(c => c.Name == "struct_1000003_l6");
        byte[] stream = ReadEmbedded(GDeflateFixturePrefix + c.Name + ".gdf");
        using var js = new Uint8Array(stream.Length);
        js.Set(stream);
        using var outp = await dec.DecompressAsync(js);
        EqualBytes(GDeflateContent(c), outp.ReadBytes(), "TypedArray overload");
    }
}
