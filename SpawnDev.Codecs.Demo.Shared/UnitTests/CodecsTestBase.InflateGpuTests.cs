// Tests for the GPU zlib/DEFLATE decoder (Compression/InflateGpu + InflateKernel).
//
// Oracle: System.IO.Compression.ZLibStream (the .NET zlib) compresses the inputs; the kernel must
// reproduce the exact bytes. Inputs cover every deflate block type and the matcher corner cases:
//  - stored blocks  (CompressionLevel.NoCompression, incompressible random data)
//  - fixed Huffman  (a hand-held zlib vector: zlib.compress(b"hello"))
//  - dynamic Huffman (text, structured data, long runs; Optimal / SmallestSize / Fastest)
//  - overlapping back-references (a single repeated byte: distance 1, length 258)
//  - the error paths: bad Adler trailer, truncated input, bad header, output capacity too small
// A test that cannot fail is not a test: the error tests assert the specific InflateError.

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
    private static byte[] ZlibCompress(byte[] raw, CompressionLevel level)
    {
        using var ms = new MemoryStream();
        using (var z = new ZLibStream(ms, level, leaveOpen: true))
            z.Write(raw, 0, raw.Length);
        return ms.ToArray();
    }

    /// <summary>Runs the kernel on <paramref name="zlib"/>; returns the raw result slot and the output bytes.</summary>
    private static async Task<(int result, byte[] output)> InflateOnGpuAsync(
        Accelerator acc, byte[] zlib, int outputCapacity, bool verifyAdler = true)
    {
        using var kernel = new InflateKernel(acc);
        using var dZ = acc.Allocate1D<byte>(System.Math.Max(zlib.Length, 1));
        using var dOut = acc.Allocate1D<byte>(System.Math.Max(outputCapacity, 1));
        using var dScratch = acc.Allocate1D<int>(InflateGpu.ScratchInts);
        using var dResult = acc.Allocate1D<int>(1);
        DeviceTransfer.Upload(dZ.View, zlib);
        dOut.View.MemSetToZero();
        await kernel.RunAsync(dZ.View, zlib.Length, dOut.View, outputCapacity, dScratch.View, dResult.View, verifyAdler);
        await acc.SynchronizeAsync();
        int result = (await dResult.View.CopyToHostAsync())[0];
        byte[] output = result > 0
            ? await DeviceTransfer.DownloadAsync(dOut.View, result)
            : System.Array.Empty<byte>();
        return (result, output);
    }

    private async Task AssertInflateMatchesAsync(byte[] raw, CompressionLevel level, string what)
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            byte[] zlib = ZlibCompress(raw, level);
            var (result, output) = await InflateOnGpuAsync(acc, zlib, raw.Length);
            Equal(raw.Length, result, $"{what}: inflated length (negative = -InflateError)");
            EqualBytes(raw, output, $"{what}: inflated bytes");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    private static byte[] StructuredData(int length, int seed)
    {
        // Mixed content: runs, repeated phrases with small edits, ramps and noise, so the encoder
        // emits literals, near and far matches, and several dynamic Huffman tables.
        var rng = new Random(seed);
        var data = new byte[length];
        string[] words = { "alpha ", "beta ", "gamma ", "delta ", "epsilon ", "zeta ", "eta ", "theta " };
        int i = 0;
        while (i < length)
        {
            int kind = rng.Next(0, 5);
            int n = rng.Next(1, 200);
            for (int k = 0; k < n && i < length; k++, i++)
            {
                switch (kind)
                {
                    case 0: data[i] = 0x41; break;                                   // run
                    case 1: data[i] = (byte)words[rng.Next(words.Length)][k % 4]; break;
                    case 2: data[i] = (byte)(i & 0xFF); break;                       // ramp
                    case 3: data[i] = (byte)rng.Next(256); break;                    // noise
                    default: data[i] = (byte)(i > 300 ? data[i - 300] : 7); break;   // far copy
                }
            }
        }
        return data;
    }

    [TestMethod]
    public async Task InflateGpu_FixedHuffmanVector_Hello()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // zlib.compress(b"hello"): a fixed-Huffman block.
            byte[] zlib = { 0x78, 0x9C, 0xCB, 0x48, 0xCD, 0xC9, 0xC9, 0x07, 0x00, 0x06, 0x2C, 0x02, 0x15 };
            var (result, output) = await InflateOnGpuAsync(acc, zlib, 16);
            Equal(5, result, "fixed-Huffman inflated length");
            EqualBytes(System.Text.Encoding.ASCII.GetBytes("hello"), output, "fixed-Huffman inflated bytes");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod]
    public async Task InflateGpu_EmptyStream()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // zlib.compress(b""): a fixed block holding only end-of-block, Adler-32 of nothing = 1.
            // (.NET's ZLibStream writes NO bytes at all for empty input, so it cannot make this vector.)
            byte[] zlib = { 0x78, 0x9C, 0x03, 0x00, 0x00, 0x00, 0x00, 0x01 };
            var (result, _) = await InflateOnGpuAsync(acc, zlib, 1);
            Equal(0, result, "empty stream inflated length");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod]
    public async Task InflateGpu_SingleByte() =>
        await AssertInflateMatchesAsync(new byte[] { 0x5A }, CompressionLevel.Optimal, "single byte");

    [TestMethod]
    public async Task InflateGpu_StoredBlocks_NoCompression() =>
        await AssertInflateMatchesAsync(StructuredData(200_000, 0x1F01), CompressionLevel.NoCompression, "stored blocks");

    [TestMethod]
    public async Task InflateGpu_IncompressibleRandom_Optimal()
    {
        var rng = new Random(0x1F02);
        var raw = new byte[70_000];
        rng.NextBytes(raw);
        await AssertInflateMatchesAsync(raw, CompressionLevel.Optimal, "incompressible random");
    }

    [TestMethod]
    public async Task InflateGpu_RepeatedByte_OverlappingBackReferences() =>
        await AssertInflateMatchesAsync(Enumerable.Repeat((byte)0x61, 100_000).ToArray(), CompressionLevel.Optimal, "repeated byte (dist 1, len 258)");

    [TestMethod]
    public async Task InflateGpu_Text_Optimal()
    {
        var sb = new System.Text.StringBuilder();
        var rng = new Random(0x1F03);
        string[] words = { "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "pack", "my", "box", "with", "five", "dozen", "liquor", "jugs" };
        while (sb.Length < 150_000) sb.Append(words[rng.Next(words.Length)]).Append(rng.Next(8) == 0 ? ".\n" : " ");
        await AssertInflateMatchesAsync(System.Text.Encoding.ASCII.GetBytes(sb.ToString()), CompressionLevel.Optimal, "text optimal");
    }

    [TestMethod]
    public async Task InflateGpu_Structured_Fastest() =>
        await AssertInflateMatchesAsync(StructuredData(300_000, 0x1F04), CompressionLevel.Fastest, "structured fastest");

    [TestMethod]
    public async Task InflateGpu_Structured_Optimal() =>
        await AssertInflateMatchesAsync(StructuredData(300_000, 0x1F05), CompressionLevel.Optimal, "structured optimal");

    [TestMethod]
    public async Task InflateGpu_Structured_SmallestSize() =>
        await AssertInflateMatchesAsync(StructuredData(300_000, 0x1F06), CompressionLevel.SmallestSize, "structured smallest");

    [TestMethod]
    public async Task InflateGpu_BadAdlerTrailer_ReportsAdlerMismatch()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            byte[] raw = StructuredData(5_000, 0x1F07);
            byte[] zlib = ZlibCompress(raw, CompressionLevel.Optimal);
            zlib[^1] ^= 0xFF;
            var (result, _) = await InflateOnGpuAsync(acc, zlib, raw.Length);
            Equal(-(int)InflateError.AdlerMismatch, result, "corrupted trailer");
            // The same stream with verification off still inflates: proves the data was fine and the
            // error above came from the trailer check, not from the data.
            var (ok, output) = await InflateOnGpuAsync(acc, zlib, raw.Length, verifyAdler: false);
            Equal(raw.Length, ok, "verifyAdler=false inflated length");
            EqualBytes(raw, output, "verifyAdler=false bytes");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod]
    public async Task InflateGpu_TruncatedInput_ReportsInputTruncated()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            byte[] raw = StructuredData(20_000, 0x1F08);
            byte[] zlib = ZlibCompress(raw, CompressionLevel.Optimal);
            byte[] cut = zlib.AsSpan(0, zlib.Length / 2).ToArray();
            var (result, _) = await InflateOnGpuAsync(acc, cut, raw.Length);
            Equal(-(int)InflateError.InputTruncated, result, "half a stream");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod]
    public async Task InflateGpu_BadHeader_ReportsBadZlibHeader()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            byte[] zlib = ZlibCompress(StructuredData(1_000, 0x1F09), CompressionLevel.Optimal);
            zlib[0] = 0x79; // CM != 8
            var (result, _) = await InflateOnGpuAsync(acc, zlib, 1_000);
            Equal(-(int)InflateError.BadZlibHeader, result, "bad CM");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod]
    public async Task InflateGpu_OutputCapacityTooSmall_ReportsOutputOverflow()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            byte[] raw = StructuredData(10_000, 0x1F0A);
            byte[] zlib = ZlibCompress(raw, CompressionLevel.Optimal);
            var (result, _) = await InflateOnGpuAsync(acc, zlib, raw.Length - 1);
            Equal(-(int)InflateError.OutputOverflow, result, "capacity one byte short");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    // ---- bounded launches: streams that need many resumable Step launches ----

    [TestMethod(Timeout = 600000)]
    public async Task InflateGpu_LargeStoredBlocks_SplitAcrossLaunches() =>
        await AssertInflateMatchesAsync(StructuredData(3_000_000, 0x1F0B), CompressionLevel.NoCompression, "3 MB stored blocks");

    [TestMethod(Timeout = 600000)]
    public async Task InflateGpu_LargeDynamic_ManyLaunches() =>
        await AssertInflateMatchesAsync(StructuredData(4_000_000, 0x1F0C), CompressionLevel.Optimal, "4 MB dynamic blocks");

    /// <summary>The Adler-32 is folded launch by launch: a corrupted trailer on a many-launch stream must still be caught.</summary>
    [TestMethod(Timeout = 600000)]
    public async Task InflateGpu_LargeStream_BadAdlerTrailer_ReportsAdlerMismatch()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            byte[] raw = StructuredData(1_500_000, 0x1F0D);
            byte[] zlib = ZlibCompress(raw, CompressionLevel.Optimal);
            zlib[^2] ^= 0x01;
            var (result, _) = await InflateOnGpuAsync(acc, zlib, raw.Length);
            Equal(-(int)InflateError.AdlerMismatch, result, "corrupted trailer of a 1.5 MB stream");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }
}
