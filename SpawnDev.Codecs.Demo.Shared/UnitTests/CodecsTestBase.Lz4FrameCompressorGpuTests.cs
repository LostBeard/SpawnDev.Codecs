// Tests for Lz4FrameCompressorGpu (Compression/Lz4): LZ4 frames of independent 64 KB blocks, BYTE-IDENTICAL with liblz4
// 1.9.4 (python-lz4's LZ4F_compressFrame, block_linked=False). The truth is liblz4's own output, never our code.
//
// Fixtures (TestData/Lz4, scripts/compression/make_lz4_fixtures.py): procedural text / noise / structured inputs (the
// generators below rebuild the same bytes) at 0, 1, 12, 13 (the minimum compressible length), 64 KB +- 1, 200 KB ..
// 40 MB (two waves at the default wave size), incompressible blocks (stored raw), all-zero data (long overlapping
// matches), every option (content size, block checksums, content checksum, acceleration 3 and 8), and three real media
// files embedded in the test assembly. Small frames are compared whole on the accelerator; large ones by SHA-256.
//
// What can fail: the hash, the skip/step schedule, catch-up, literal and match length coding, the conservative
// limitedOutput bounds that decide raw blocks, the last-literals rule, the resumable state across launches, the wave
// layout and copy, the header checksum, block / content XXH32, writes past the frame (output poisoned).
// .NET has no LZ4, so the round trip through a decoder comes with Lz4FrameDecompressorGpu; byte identity with liblz4
// already implies liblz4 reads every frame.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Compression.Lz4;
using SpawnDev.ILGPU;
using SpawnDev.SpawnJS.JSObjects;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    /// <summary>
    /// The LZ4 kernels scatter: one thread writes a whole block (compressor slots, decoded blocks, the header walk). WebGL
    /// stores only at a thread's own slot, so these tests do not run there.
    /// </summary>
    private static readonly AcceleratorRequirements Lz4Requirements = new() { RequiresScatterStores = true };

    private const string Lz4FixturePrefix = "SpawnDev.Codecs.Demo.Shared.TestData.Lz4.";

    private static readonly string[] Lz4Words = { "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "pack", "my", "box",
        "with", "five", "dozen", "liquor", "jugs", "sphinx", "of", "black", "quartz" };

    private static byte[] Lz4GenText(int n, int seed)
    {
        var outp = new byte[n];
        int len = 0;
        long x = seed & 0x7FFFFFFF;
        while (len < n)
        {
            x = (x * 1103515245 + 12345) & 0x7FFFFFFF;
            var w = Lz4Words[(int)(x % 20)];
            for (int i = 0; i < w.Length && len < n; i++) outp[len++] = (byte)w[i];
            if ((x >> 8) % 9 == 0)
            {
                if (len < n) outp[len++] = (byte)'.';
                if (len < n) outp[len++] = (byte)'\n';
            }
            else if (len < n) outp[len++] = (byte)' ';
        }
        return outp;
    }

    private static byte[] Lz4GenNoise(int n, int seed)
    {
        var outp = new byte[n];
        uint x = (uint)(seed | 1);
        for (int i = 0; i < n; i++)
        {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            outp[i] = (byte)x;
        }
        return outp;
    }

    private static byte[] Lz4GenStruct(int n, int seed)
    {
        var outp = new byte[n];
        if (seed < 0) return outp; // zeros
        for (int i = 0; i < n; i++)
        {
            int r = (i / 37) % 11;
            int v;
            if (r < 3) v = 0;
            else if (r < 6) v = (i * 3 + seed) & 0xFF;
            else if (r < 9) v = ((i % 23) * 7 + (i / 4096)) & 0xFF;
            else v = (int)((((ulong)i * 2654435761UL + (ulong)seed) >> 11) & 0xFF);
            outp[i] = (byte)v;
        }
        return outp;
    }

    private sealed record Lz4Case(string Name, string Kind, int Length, int Seed, bool StoreSize, bool BlockChecksum, bool ContentChecksum,
        int Acceleration, int FrameLength, string Sha256, bool Stored)
    {
        public Lz4FrameOptions Options => new() { StoreContentSize = StoreSize, BlockChecksum = BlockChecksum, ContentChecksum = ContentChecksum, Acceleration = Acceleration };
    }

    private static List<Lz4Case> Lz4Cases()
    {
        var list = new List<Lz4Case>();
        foreach (var raw in ReadEmbeddedText(Lz4FixturePrefix + "manifest.tsv").Split('\n'))
        {
            var line = raw.Trim('\r', ' ');
            if (line.Length == 0 || line.StartsWith('#')) continue;
            var f = line.Split('\t');
            list.Add(new Lz4Case(f[0], f[1], int.Parse(f[2]), int.Parse(f[3]), f[4] == "1", f[5] == "1", f[6] == "1", int.Parse(f[7]),
                int.Parse(f[8]), f[9], f[10] == "1"));
        }
        return list;
    }

    private static byte[] Lz4Input(Lz4Case c)
    {
        if (c.Kind.StartsWith("file:", StringComparison.Ordinal))
            return ReadEmbedded("SpawnDev.Codecs.Demo.Shared.TestData." + c.Kind.Substring(5));
        return c.Kind switch
        {
            "text" => Lz4GenText(c.Length, c.Seed),
            "noise" => Lz4GenNoise(c.Length, c.Seed),
            "struct" => Lz4GenStruct(c.Length, c.Seed),
            _ => throw new InvalidOperationException($"unknown kind {c.Kind}"),
        };
    }

    /// <summary>Compresses one case through the ArrayView1D overload into a poisoned buffer; null when it matches liblz4.</summary>
    private static async Task<string?> RunLz4CaseAsync(Accelerator acc, Lz4FrameCompressorGpu comp, Lz4Case c)
    {
        byte[] input = Lz4Input(c);
        if (input.Length != c.Length) return $"{c.Name}: input is {input.Length} bytes, manifest says {c.Length}";
        long cap = Lz4FrameCompressorGpu.GetMaxCompressedSize(input.Length) + 41;
        using var dIn = DeviceVerify.Upload(acc, input, 7);
        using var dOut = DeviceVerify.Poisoned(acc, cap);
        int n = await comp.CompressAsync(dIn.View, input.Length, dOut.View, c.Options);
        if (n != c.FrameLength) return $"{c.Name}: frame is {n} bytes, liblz4 wrote {c.FrameLength}";
        if (c.Stored)
        {
            byte[] truth = ReadEmbedded(Lz4FixturePrefix + c.Name + ".lz4");
            var withPoison = new byte[truth.Length + 32];
            truth.CopyTo(withPoison, 0);
            System.Array.Fill(withPoison, (byte)0xAB, truth.Length, 32);
            string? diff = await DeviceVerify.CompareAsync(acc, dOut.View, withPoison);
            return diff == null ? null : $"{c.Name}: {diff}";
        }
        string sha = Sha256Hex(await DeviceTransfer.DownloadAsync(dOut.View, n));
        return sha == c.Sha256 ? null : $"{c.Name}: SHA-256 {sha} != liblz4 {c.Sha256}";
    }

    private async Task AssertLz4CasesAsync(Func<Lz4Case, bool> which, int waveBlocks, int minCases)
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, Lz4Requirements);
        try
        {
            using var comp = new Lz4FrameCompressorGpu(acc, waveBlocks);
            var cases = Lz4Cases().Where(which).ToList();
            True(cases.Count >= minCases, $"only {cases.Count} LZ4 fixtures selected");
            var failures = new List<string>();
            foreach (var c in cases)
            {
                string? r = await RunLz4CaseAsync(acc, comp, c);
                if (r != null) failures.Add(r);
            }
            True(failures.Count == 0, $"{failures.Count} of {cases.Count} frames differ from liblz4:\n" + string.Join("\n", failures));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod(Timeout = 600000)]
    public async Task Lz4FrameCompressorGpu_AllSmallFixtures_ByteIdenticalWithLiblz4() =>
        await AssertLz4CasesAsync(c => c.Length <= 1_100_000, Lz4FrameCompressorGpu.DefaultWaveBlocks, 25);

    [TestMethod(Timeout = 600000)]
    public async Task Lz4FrameCompressorGpu_ManySmallWaves_ByteIdenticalWithLiblz4() =>
        await AssertLz4CasesAsync(c => c.Name is "struct_1000003" or "text_300000_accel3_cc" or "text_150001_bc_cc_nosize", 3, 3);

    [TestMethod(Timeout = 600000)]
    public async Task Lz4FrameCompressorGpu_40MB_TwoWaves_ByteIdenticalWithLiblz4() =>
        await AssertLz4CasesAsync(c => c.Length > 1_100_000, Lz4FrameCompressorGpu.DefaultWaveBlocks, 1);

    [TestMethod]
    public async Task Lz4FrameCompressorGpu_ManagedOverload_MatchesLiblz4()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, Lz4Requirements);
        try
        {
            using var comp = new Lz4FrameCompressorGpu(acc, 8);
            foreach (var c in Lz4Cases().Where(c => c.Stored && c.Length is 0 or 13 or 65537 or 1000 or 200000))
            {
                byte[] frame = await comp.CompressAsync(Lz4Input(c), c.Options);
                EqualBytes(ReadEmbedded(Lz4FixturePrefix + c.Name + ".lz4"), frame, $"{c.Name}: managed overload");
            }
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod]
    public async Task Lz4FrameCompressorGpu_TypedArrayOverload_MatchesLiblz4()
    {
        if (!OperatingSystem.IsBrowser())
            throw new UnsupportedTestException("TypedArray overloads need a browser accelerator.");
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, Lz4Requirements);
        try
        {
            using var comp = new Lz4FrameCompressorGpu(acc, 8);
            await RunLz4TypedArrayCases(comp);
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [System.Runtime.Versioning.SupportedOSPlatform("browser")]
    private static async Task RunLz4TypedArrayCases(Lz4FrameCompressorGpu comp)
    {
        foreach (var c in Lz4Cases().Where(c => c.Stored && c.Length is 13 or 65537 or 200000))
        {
            byte[] input = Lz4Input(c);
            using var js = new Uint8Array(input.Length);
            js.Set(input);
            using var frame = await comp.CompressAsync(js, c.Options);
            EqualBytes(ReadEmbedded(Lz4FixturePrefix + c.Name + ".lz4"), frame.ReadBytes(), $"{c.Name}: TypedArray overload");
        }
    }

    [TestMethod]
    public async Task Lz4FrameCompressorGpu_RejectsBadArguments()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, Lz4Requirements);
        try
        {
            using var comp = new Lz4FrameCompressorGpu(acc, 2);
            using var a = acc.Allocate1D<byte>(1000);
            using var small = acc.Allocate1D<byte>(100);
            await ThrowsAsync<ArgumentException>(async () => await comp.CompressAsync(a.View, 1000, small.View), "output below GetMaxCompressedSize");
            await ThrowsAsync<ArgumentOutOfRangeException>(async () => await comp.CompressAsync(small.View, 1000, a.View), "length past the input");
            using var big = acc.Allocate1D<byte>(Lz4FrameCompressorGpu.GetMaxCompressedSize(1000));
            await ThrowsAsync<ArgumentOutOfRangeException>(async () => await comp.CompressAsync(a.View, 1000, big.View, new Lz4FrameOptions { Acceleration = 0 }), "acceleration 0");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }
}
