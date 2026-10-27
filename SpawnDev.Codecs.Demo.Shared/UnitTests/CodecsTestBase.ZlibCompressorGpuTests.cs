// Tests for the GPU zlib compressor (Compression/DeflateGpu + ZlibCompressorGpu).
//
// Independent oracle: System.IO.Compression.ZLibStream (the .NET zlib) DECOMPRESSES what the accelerator
// compressed, and must return the exact input; our own InflateGpu agrees. Coverage: every level, empty /
// one-byte / sub-block / multi-block inputs (small block size to force many blocks and the sync-flush
// concatenation), text, structured data with long matches and overlapping distance-1 runs, incompressible
// random data (must fall back to stored blocks, not expand), and the size sanity bound that proves
// compression actually happens (a test that only round-trips would pass with stored blocks everywhere).

using System.IO.Compression;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Compression;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private static byte[] DotNetZlibDecompress(byte[] zlib)
    {
        using var input = new MemoryStream(zlib);
        using var z = new ZLibStream(input, CompressionMode.Decompress);
        using var output = new MemoryStream();
        z.CopyTo(output);
        return output.ToArray();
    }

    private static byte[] TextData(int length, int seed)
    {
        var sb = new System.Text.StringBuilder();
        var rng = new Random(seed);
        string[] words = { "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "pack", "my", "box", "with", "five", "dozen", "liquor", "jugs", "sphinx", "of", "black", "quartz" };
        while (sb.Length < length) sb.Append(words[rng.Next(words.Length)]).Append(rng.Next(9) == 0 ? ".\n" : " ");
        return System.Text.Encoding.ASCII.GetBytes(sb.ToString(0, length));
    }

    private async Task AssertZlibRoundTripAsync(byte[] raw, int level, int blockSize, string what, double? maxRatio = null)
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var comp = new ZlibCompressorGpu(acc, blockSize, threadsPerWave: 4);
            byte[] zlib = await comp.CompressAsync(raw, level);

            byte[] viaDotNet;
            try { viaDotNet = DotNetZlibDecompress(zlib); }
            catch (Exception ex) { throw new Exception($"{what}: .NET zlib rejected our stream ({ex.GetType().Name}: {ex.Message}), {zlib.Length} bytes", ex); }
            Equal(raw.Length, viaDotNet.Length, $"{what}: .NET inflated length");
            EqualBytes(raw, viaDotNet, $"{what}: .NET inflated bytes");

            // Our own GPU inflate must agree too.
            using var inflate = new InflateKernel(acc);
            using var dZ = acc.Allocate1D<byte>(zlib.Length);
            using var dOut = acc.Allocate1D<byte>(Math.Max(1, raw.Length));
            using var dScratch = acc.Allocate1D<int>(InflateGpu.ScratchInts);
            using var dResult = acc.Allocate1D<int>(1);
            DeviceTransfer.Upload(dZ.View, zlib);
            await inflate.RunAsync(dZ.View, zlib.Length, dOut.View, raw.Length, dScratch.View, dResult.View, verifyAdler: true);
            await acc.SynchronizeAsync();
            Equal(raw.Length, (await dResult.View.CopyToHostAsync())[0], $"{what}: InflateGpu result");

            if (maxRatio.HasValue)
                True(zlib.Length <= raw.Length * maxRatio.Value,
                    $"{what}: compressed {zlib.Length} of {raw.Length} bytes (ratio {(double)zlib.Length / raw.Length:F3}) exceeds {maxRatio.Value}");
            if (level == 0 || raw.Length == 0) return;
            // Never worse than stored blocks + framing.
            True(zlib.Length <= raw.Length + 5 * (raw.Length / 65535 + 2) + 6 + 10 * (raw.Length / blockSize + 1),
                $"{what}: expanded to {zlib.Length} bytes from {raw.Length}");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod]
    public async Task ZlibCompressorGpu_Empty() => await AssertZlibRoundTripAsync(System.Array.Empty<byte>(), 2, 65536, "empty");

    [TestMethod]
    public async Task ZlibCompressorGpu_OneByte() => await AssertZlibRoundTripAsync(new byte[] { 0x42 }, 2, 65536, "one byte");

    [TestMethod]
    public async Task ZlibCompressorGpu_SmallText_Level1() => await AssertZlibRoundTripAsync(TextData(500, 1), 1, 65536, "small text L1");

    [TestMethod]
    public async Task ZlibCompressorGpu_Text_Level1() => await AssertZlibRoundTripAsync(TextData(150_000, 2), 1, 65536, "text L1", 0.45);

    [TestMethod]
    public async Task ZlibCompressorGpu_Text_Level2() => await AssertZlibRoundTripAsync(TextData(150_000, 3), 2, 65536, "text L2", 0.40);

    [TestMethod]
    public async Task ZlibCompressorGpu_Text_Level3_Lazy() => await AssertZlibRoundTripAsync(TextData(150_000, 4), 3, 65536, "text L3", 0.40);

    [TestMethod]
    public async Task ZlibCompressorGpu_Stored_Level0() => await AssertZlibRoundTripAsync(TextData(70_000, 5), 0, 65536, "stored L0");

    [TestMethod]
    public async Task ZlibCompressorGpu_Structured_ManySmallBlocks() =>
        await AssertZlibRoundTripAsync(StructuredData(120_000, 0x2B01), 2, 4096, "structured, 4 KiB blocks (sync-flush concatenation)");

    [TestMethod]
    public async Task ZlibCompressorGpu_Structured_Level3() =>
        await AssertZlibRoundTripAsync(StructuredData(250_000, 0x2B02), 3, 65536, "structured L3");

    [TestMethod]
    public async Task ZlibCompressorGpu_RepeatedByte_OverlappingMatches() =>
        await AssertZlibRoundTripAsync(Enumerable.Repeat((byte)0x7A, 200_000).ToArray(), 2, 65536, "repeated byte", 0.01);

    /// <summary>
    /// Level 4 = Huffman only (zlib Z_HUFFMAN_ONLY): bytes with a skewed distribution and no repeats worth matching (like the
    /// exponent plane of BF16 weights). .NET's zlib must inflate it exactly, and the size must be within 1 % (+ block headers)
    /// of an optimal Huffman code of every 64 KiB block, built in the test from the data (the order-0 entropy is NOT a
    /// bound a Huffman coder can reach on skewed data): stored blocks or a broken code-length builder fail it.
    /// </summary>
    [TestMethod(Timeout = 600000)]
    public async Task ZlibCompressorGpu_HuffmanOnly_Level4_NearOptimalHuffman()
    {
        var rng = new Random(0x4F11);
        var raw = new byte[1_000_000];
        for (int i = 0; i < raw.Length; i++)
        {
            // Two-sided geometric around 120 (exponent-like): ~3 bits of entropy per byte.
            int k = 0;
            while (rng.Next(3) == 0 && k < 40) k++;
            raw[i] = (byte)(rng.Next(2) == 0 ? 120 + k : 119 - k);
        }
        int blocks = (raw.Length + 65535) / 65536;
        double optimal = PlaneChunkHuffmanBytes(raw, 1); // 64 KiB chunks = the 65536-byte blocks below
        await AssertZlibRoundTripAsync(raw, DeflateGpu.LevelHuffmanOnly, 65536, "Huffman-only, skewed bytes",
            maxRatio: (optimal * 1.01 + blocks * 120 + 6) / raw.Length);
    }

    [TestMethod]
    public async Task ZlibCompressorGpu_IncompressibleRandom_FallsBackToStored()
    {
        var rng = new Random(0x2B03);
        var raw = new byte[100_000];
        rng.NextBytes(raw);
        await AssertZlibRoundTripAsync(raw, 2, 65536, "random");
    }

    [TestMethod]
    public async Task ZlibCompressorGpu_ArrayViewOverload_WritesCallerBuffer()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var comp = new ZlibCompressorGpu(acc, 8192, 4);
            byte[] raw = StructuredData(60_000, 0x2B04);
            using var dIn = acc.Allocate1D<byte>(raw.Length + 9);
            DeviceTransfer.Upload(dIn.View, raw);
            long cap = ZlibCompressorGpu.GetMaxCompressedSize(raw.Length, 8192);
            using var dOut = acc.Allocate1D<byte>(cap + 13);
            var fill = new byte[dOut.Length];
            System.Array.Fill(fill, (byte)0xAB);
            DeviceTransfer.Upload(dOut.View, fill);
            int n = await comp.CompressAsync(dIn.View, raw.Length, dOut.View, 2);
            byte[] zlib = await DeviceTransfer.DownloadAsync(dOut.View, n);
            EqualBytes(raw, DotNetZlibDecompress(zlib), "ArrayView1D overload round trip");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    // ---- bounded launches: inputs that need many launches of every resumable stage ----

    [TestMethod(Timeout = 600000)]
    public async Task ZlibCompressorGpu_LargeInput_ManyWaves() =>
        await AssertZlibRoundTripAsync(TextData(3_000_000, 11), 2, 65536, "3 MB text, 46 blocks in waves of 4", 0.40);

    [TestMethod(Timeout = 600000)]
    public async Task ZlibCompressorGpu_LargeBlocks_Level3_ManyParseAndEmitLaunches() =>
        await AssertZlibRoundTripAsync(StructuredData(2_500_000, 0x2B05), 3, 1 << 20, "2.5 MB structured, 1 MiB blocks, lazy");

    /// <summary>
    /// The parallel parse (DeflateGpu.ParallelParse) writes the bytes of the serial DeflateGpu.ParseStep, its oracle, at
    /// levels 1-3. The serial lazy parse runs with a compare budget far below what the blocks need, so it finishes only
    /// through the host's unfinished-parse loop.
    /// </summary>
    [TestMethod(Timeout = 600000)]
    public async Task ZlibCompressorGpu_ParallelParse_SameBytesAsSerialParse()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            foreach (var (name, raw) in new[] { ("text", TextData(400_000, 12)), ("structured", StructuredData(300_000, 0x2B06)) })
            {
                using var parallel = new ZlibCompressorGpu(acc, 65536, threadsPerWave: 4);
                using var serial = new ZlibCompressorGpu(acc, 65536, threadsPerWave: 4) { SerialParse = true, ParseCompareBudget = 4096 };
                for (int level = 1; level <= 3; level++)
                {
                    byte[] a = await parallel.CompressAsync(raw, level);
                    byte[] b = await serial.CompressAsync(raw, level);
                    EqualBytes(b, a, $"{name}: level {level} parallel parse vs the serial ParseStep (4096-compare budget)");
                    EqualBytes(raw, DotNetZlibDecompress(a), $"{name}: .NET inflates the level {level} stream");
                }
            }
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }
}
