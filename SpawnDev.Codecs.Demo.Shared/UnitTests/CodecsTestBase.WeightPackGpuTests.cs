// Tests for WeightPackGpu (Compression): lossless tensor packing = byte planes + chunked Huffman-only DEFLATE.
//
// Oracles, none of them our code: (1) the unpacked bytes equal the input (compared on the accelerator); (2) .NET's own
// DeflateStream inflates every chunk of a packed blob (the format is standard DEFLATE behind a small table), and a host
// un-shuffle of those planes gives the input back; (3) the packed size is within 1 % (+ per-chunk header bytes) of an
// optimal Huffman code of every plane chunk, built here from the data (the bound a per-chunk Huffman coder cannot beat,
// unlike the order-0 entropy, which a skewed chunk undercuts by up to 1 bit per byte): stored chunks or a broken code
// builder fail it.
//
// Inputs: normally distributed weights (sigma 0.02, like transformer layers) as BF16, FP16 and FP32, element sizes 1 / 3 /
// 8, empty input, a 64 MiB BF16 tensor and an 80 MB FP32 tensor (16+ compression waves), damaged and truncated blobs.

using System.IO.Compression;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Compression;
using SpawnDev.ILGPU;
using SpawnDev.SpawnJS.JSObjects;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    /// <summary>Normally distributed weights (mean 0, sigma 0.02) as little-endian elements: 2 = BF16, 3 = FP16, 4 = FP32.</summary>
    private static byte[] Weights(int count, int kind, int seed)
    {
        var rng = new Random(seed);
        int size = kind == 4 ? 4 : 2;
        var b = new byte[count * size];
        for (int i = 0; i < count; i++)
        {
            double u1 = 1.0 - rng.NextDouble(), u2 = rng.NextDouble();
            float f = (float)(0.02 * Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2));
            if (kind == 4) BitConverter.TryWriteBytes(b.AsSpan(i * 4), f);
            else if (kind == 2) BitConverter.TryWriteBytes(b.AsSpan(i * 2), (ushort)(BitConverter.SingleToUInt32Bits(f) >> 16));
            else BitConverter.TryWriteBytes(b.AsSpan(i * 2), BitConverter.HalfToUInt16Bits((System.Half)f));
        }
        return b;
    }

    /// <summary>
    /// Sum over plane chunks of an OPTIMAL (unlimited-length) Huffman code of the chunk's bytes plus one end-of-block symbol,
    /// in bytes, or the chunk's raw size when that is smaller. Total code bits = the sum of the weights of the merged nodes.
    /// </summary>
    private static double PlaneChunkHuffmanBytes(byte[] data, int elementSize)
    {
        int n = data.Length / elementSize;
        double total = 0;
        var counts = new long[256];
        var queue = new PriorityQueue<long, long>();
        for (int p = 0; p < elementSize; p++)
            for (int c0 = 0; c0 < n; c0 += WeightPackGpu.ChunkBytes)
            {
                int c1 = Math.Min(n, c0 + WeightPackGpu.ChunkBytes);
                System.Array.Clear(counts);
                for (int i = c0; i < c1; i++) counts[data[i * elementSize + p]]++;
                queue.Clear();
                queue.Enqueue(1, 1); // end of block
                foreach (var k in counts)
                    if (k > 0) queue.Enqueue(k, k);
                long bits = 0;
                while (queue.Count > 1)
                {
                    long a = queue.Dequeue(), b = queue.Dequeue();
                    bits += a + b;
                    queue.Enqueue(a + b, a + b);
                }
                total += Math.Min(bits / 8.0, c1 - c0);
            }
        return total;
    }

    /// <summary>Packs on the accelerator, checks the size bound, unpacks into a poisoned buffer and compares on the accelerator.</summary>
    /// <summary>
    /// Packs with <paramref name="pack"/> and unpacks with <paramref name="unpack"/>, a DIFFERENT instance (as in real use:
    /// packed once, unpacked at load): one instance's plane buffer still holds the shuffled input after packing, so a chunk
    /// that never decoded would pass a same-instance round trip.
    /// </summary>
    private static async Task<int> AssertWeightPackRoundTripAsync(Accelerator acc, WeightPackGpu pack, WeightPackGpu unpack, byte[] data, int elementSize, string what, bool sizeBound = true)
    {
        using var dIn = DeviceVerify.Upload(acc, data, 13);
        using var dPacked = DeviceVerify.Poisoned(acc, WeightPackGpu.GetMaxPackedSize(data.Length, elementSize) + 17);
        int n = await pack.PackAsync(dIn.View, data.Length, elementSize, dPacked.View);
        if (sizeBound && data.Length > 0)
        {
            int chunks = elementSize * WeightPackGpu.ChunksPerPlane(data.Length / elementSize);
            double bound = PlaneChunkHuffmanBytes(data, elementSize) * 1.01 + chunks * 140 + WeightPackGpu.HeaderBytes;
            True(n <= bound, $"{what}: packed {n} bytes, the per-chunk optimal Huffman bound is {bound:F0} ({(double)n / data.Length:F4} of {data.Length})");
        }
        using var dOut = DeviceVerify.Poisoned(acc, data.Length + 29);
        int m = await unpack.UnpackAsync(dPacked.View, n, dOut.View);
        Equal(data.Length, m, $"{what}: unpacked length");
        var expected = new byte[data.Length + 29];
        data.CopyTo(expected, 0);
        System.Array.Fill(expected, (byte)0xAB, data.Length, 29);
        string? diff = await DeviceVerify.CompareAsync(acc, dOut.View, expected);
        True(diff == null, $"{what}: round trip: {diff}");
        return n;
    }

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task WeightPackGpu_Bf16Fp16Fp32_RoundTrip_NearPerChunkEntropy()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var pack = new WeightPackGpu(acc);
            using var unpack = new WeightPackGpu(acc);
            int bf = await AssertWeightPackRoundTripAsync(acc, pack, unpack, Weights(3_000_001, 2, 1), 2, "BF16 3M");
            int hf = await AssertWeightPackRoundTripAsync(acc, pack, unpack, Weights(1_000_000, 3, 2), 2, "FP16 1M");
            int f32 = await AssertWeightPackRoundTripAsync(acc, pack, unpack, Weights(1_000_000, 4, 3), 4, "FP32 1M");
            // A coder that falls back to stored everywhere passes the round trips but not this.
            True(bf < 3_000_001 * 2 * 0.75, $"BF16 packed to {(double)bf / (3_000_001 * 2):F3} of its size (exponent plane not coded?)");
            True(f32 < 4_000_000 * 0.90, $"FP32 packed to {f32 / 4_000_000.0:F3} of its size");
            True(hf < 2_000_000, $"FP16 packed to {hf / 2_000_000.0:F3} of its size");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task WeightPackGpu_OddElementSizes_Empty_AndTiny()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var pack = new WeightPackGpu(acc);
            using var unpack = new WeightPackGpu(acc);
            await AssertWeightPackRoundTripAsync(acc, pack, unpack, System.Array.Empty<byte>(), 2, "empty");
            await AssertWeightPackRoundTripAsync(acc, pack, unpack, new byte[] { 7, 9 }, 2, "one BF16");
            var rng = new Random(5);
            var b1 = new byte[200_003];
            for (int i = 0; i < b1.Length; i++) b1[i] = (byte)(rng.Next(4) == 0 ? rng.Next(256) : 17);
            await AssertWeightPackRoundTripAsync(acc, pack, unpack, b1, 1, "element size 1");
            var b3 = new byte[3 * 70_001];
            rng.NextBytes(b3);
            await AssertWeightPackRoundTripAsync(acc, pack, unpack, b3, 3, "element size 3, random (stored chunks)", sizeBound: true);
            var b8 = new byte[8 * 40_000];
            for (int i = 0; i < 40_000; i++) BitConverter.TryWriteBytes(b8.AsSpan(i * 8), (double)(i % 977) * 0.125);
            await AssertWeightPackRoundTripAsync(acc, pack, unpack, b8, 8, "FP64 element size 8");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    /// <summary>The format is standard DEFLATE behind a table: .NET's DeflateStream reads every chunk, and a host un-shuffle gives the input.</summary>
    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task WeightPackGpu_EveryChunk_InflatesWithDotNetDeflate()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var pack = new WeightPackGpu(acc);
            byte[] data = Weights(150_000, 2, 9);
            byte[] packed = await pack.PackAsync(data, 2);
            Equal((byte)'S', packed[0], "magic");
            Equal(2, (int)packed[5], "element size");
            int elements = BitConverter.ToInt32(packed, 8);
            int perPlane = BitConverter.ToInt32(packed, 20);
            Equal(150_000, elements, "element count");
            Equal(WeightPackGpu.ChunksPerPlane(150_000), perPlane, "chunks per plane");
            int pos = WeightPackGpu.HeaderBytes + 4 * 2 * perPlane;
            var back = new byte[data.Length];
            for (int g = 0; g < 2 * perPlane; g++)
            {
                int n = BitConverter.ToInt32(packed, WeightPackGpu.HeaderBytes + 4 * g);
                int plane = g / perPlane, c = g % perPlane;
                int len = Math.Min(WeightPackGpu.ChunkBytes, elements - c * WeightPackGpu.ChunkBytes);
                using var z = new DeflateStream(new MemoryStream(packed, pos, n), CompressionMode.Decompress);
                var chunk = new byte[len + 1];
                int got = 0, r;
                while ((r = z.Read(chunk, got, chunk.Length - got)) > 0) got += r;
                Equal(len, got, $"chunk {g} inflated length");
                for (int i = 0; i < len; i++) back[(c * WeightPackGpu.ChunkBytes + i) * 2 + plane] = chunk[i];
                pos += n;
            }
            Equal(packed.Length, pos, "chunks fill the blob exactly");
            EqualBytes(data, back, ".NET inflate + host un-shuffle");
            using var unpack = new WeightPackGpu(acc);
            EqualBytes(data, await unpack.UnpackAsync(packed), "managed unpack (another instance)");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task WeightPackGpu_DamagedBlobs_AreRefused()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var pack = new WeightPackGpu(acc);
            byte[] good = await pack.PackAsync(Weights(100_000, 2, 11), 2);
            async Task Refused(byte[] blob, string what)
            {
                try
                {
                    await pack.UnpackAsync(blob);
                    throw new Exception($"{what}: accepted");
                }
                catch (InvalidDataException) { }
            }
            var badMagic = (byte[])good.Clone(); badMagic[1] ^= 1;
            await Refused(badMagic, "bad magic");
            await Refused(good.AsSpan(0, 20).ToArray(), "header cut");
            await Refused(good.AsSpan(0, WeightPackGpu.HeaderBytes + 3).ToArray(), "table cut");
            await Refused(good.AsSpan(0, good.Length - 5).ToArray(), "data cut");
            var badTable = (byte[])good.Clone(); badTable[WeightPackGpu.HeaderBytes + 1] ^= 0x40;
            await Refused(badTable, "chunk length changed");
            var badChunk = (byte[])good.Clone();
            int firstChunk = WeightPackGpu.HeaderBytes + 4 * 2 * WeightPackGpu.ChunksPerPlane(100_000);
            badChunk[firstChunk] ^= 0x06; // the first block header's type bits: 3 = reserved
            await Refused(badChunk, "reserved block type");
            EqualBytes(Weights(100_000, 2, 11), await pack.UnpackAsync(good), "still works after the refusals");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    /// <summary>64 MiB of BF16 (1024 chunks = 16 compress waves) and an FP32 tensor of 20M elements (1220 chunks).</summary>
    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task WeightPackGpu_Large_ManyWaves_RoundTrip()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var pack = new WeightPackGpu(acc);
            using var unpack = new WeightPackGpu(acc);
            await AssertWeightPackRoundTripAsync(acc, pack, unpack, Weights(32 * 1024 * 1024, 2, 21), 2, "BF16 64 MiB", sizeBound: false);
            await AssertWeightPackRoundTripAsync(acc, pack, unpack, Weights(20_000_000, 4, 22), 4, "FP32 80 MB", sizeBound: false);
            // Decode waves of 300 chunks: the 64 MiB BF16 tensor (1024 chunks) crosses three wave boundaries.
            using var smallWaves = new WeightPackGpu(acc, decompressWave: 300);
            await AssertWeightPackRoundTripAsync(acc, pack, smallWaves, Weights(32 * 1024 * 1024, 2, 21), 2, "BF16 64 MiB, decode waves of 300", sizeBound: false);
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod(Category = "BrowserOnly,NotOnWebGL")]
    public async Task WeightPackGpu_TypedArrayOverloads()
    {
        if (!OperatingSystem.IsBrowser())
            throw new UnsupportedTestException("TypedArray overloads need a browser accelerator.");
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var pack = new WeightPackGpu(acc);
            using var unpack = new WeightPackGpu(acc);
            await RunWeightPackTypedArray(pack, unpack);
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [System.Runtime.Versioning.SupportedOSPlatform("browser")]
    private static async Task RunWeightPackTypedArray(WeightPackGpu pack, WeightPackGpu unpack)
    {
        byte[] data = Weights(120_000, 2, 31);
        using var js = new Uint8Array(data.Length);
        js.Set(data);
        using var packed = await pack.PackAsync(js, 2);
        byte[] managed = await pack.PackAsync(data, 2);
        EqualBytes(managed, packed.ReadBytes(), "TypedArray pack = managed pack");
        using var back = await unpack.UnpackAsync(packed);
        EqualBytes(data, back.ReadBytes(), "TypedArray unpack");
    }

    [TestMethod(Category = "NotOnWebGL")]
    public async Task WeightPackGpu_RejectsBadArguments()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var pack = new WeightPackGpu(acc);
            using var a = acc.Allocate1D<byte>(1000);
            using var big = acc.Allocate1D<byte>(WeightPackGpu.GetMaxPackedSize(1000, 4));
            await ThrowsAsync<ArgumentException>(async () => await pack.PackAsync(a.View, 999, 2, big.View), "not whole elements");
            await ThrowsAsync<ArgumentOutOfRangeException>(async () => await pack.PackAsync(a.View, 1000, 0, big.View), "element size 0");
            await ThrowsAsync<ArgumentException>(async () => await pack.PackAsync(a.View, 1000, 4, a.View.SubView(0, 10)), "output too small");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }
}
