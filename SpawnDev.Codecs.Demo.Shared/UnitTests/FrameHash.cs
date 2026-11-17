// Frame hash for tests: a 64-bit hash of a frame's bytes that the accelerator computes in parallel, so a test reads back
// 8 bytes per frame instead of the frame. The same function as libvpx_ref.c's fh_* (scripts/video/libvpx_ref), which
// writes the references (libvpx_modeinfo.tsv.gz, make_modeinfo_tsv.py): the stream is cut into 4096-byte chunks; chunk
// c gets two MurmurHash3 x86_32 values, seeds c * 0x9E3779B9 + 0x243F6A88 and c * 0x9E3779B9 + 0x85A308D3; the A values
// (little-endian words) are then hashed with seed = stream length and the B values with seed = length ^ 0x13198A2E.
// Text form "%08x%08x" (A, B). Not a cryptographic hash: it detects a wrong frame, nothing adversarial.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

internal static class FrameHash
{
    public const int Chunk = 4096;
    private const uint SeedStep = 0x9E3779B9u, SeedA = 0x243F6A88u, SeedB = 0x85A308D3u, CombineB = 0x13198A2Eu;

    private static uint Rotl(uint x, int r) => (x << r) | (x >> (32 - r));
    private static uint Mix(uint k) { k *= 0xcc9e2d51u; k = Rotl(k, 15); return k * 0x1b873593u; }
    private static uint Round(uint h, uint k) { h ^= k; h = Rotl(h, 13); return h * 5 + 0xe6546b64u; }
    private static uint Fmix(uint h)
    {
        h ^= h >> 16; h *= 0x85ebca6bu; h ^= h >> 13; h *= 0xc2b2ae35u; h ^= h >> 16;
        return h;
    }

    /// <summary>The hash of <paramref name="bytes"/> on the host (the reference for the kernels).</summary>
    public static string Compute(ReadOnlySpan<byte> bytes)
    {
        int n = bytes.Length, chunks = (n + Chunk - 1) / Chunk;
        uint ca = (uint)n, cb = (uint)n ^ CombineB;
        for (int c = 0; c < chunks; c++)
        {
            int start = c * Chunk, len = Math.Min(Chunk, n - start);
            uint s = (uint)c * SeedStep, ha = s + SeedA, hb = s + SeedB;
            int i = 0;
            for (; i + 4 <= len; i += 4)
            {
                uint k = Mix(System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(bytes.Slice(start + i, 4)));
                ha = Round(ha, k); hb = Round(hb, k);
            }
            uint t = 0;
            int rem = len & 3;
            if (rem == 3) t ^= (uint)bytes[start + i + 2] << 16;
            if (rem >= 2) t ^= (uint)bytes[start + i + 1] << 8;
            if (rem >= 1) { t ^= bytes[start + i]; t = Mix(t); ha ^= t; hb ^= t; }
            ca = Round(ca, Mix(Fmix(ha ^ (uint)len)));
            cb = Round(cb, Mix(Fmix(hb ^ (uint)len)));
        }
        uint words = (uint)(chunks * 4);
        return $"{Fmix(ca ^ words):x8}{Fmix(cb ^ words):x8}";
    }

    // one thread per chunk: both lanes of the chunk's hash
    private static void ChunkKernel(Index1D c, ArrayView<byte> data, ArrayView<uint> a, ArrayView<uint> b, int n)
    {
        int start = c * Chunk;
        int len = n - start;
        if (len > Chunk) len = Chunk;
        uint s = (uint)(int)c * SeedStep;
        uint ha = s + SeedA, hb = s + SeedB;
        int i = 0;
        for (; i + 4 <= len; i += 4)
        {
            uint k = Mix((uint)data[start + i] | ((uint)data[start + i + 1] << 8) | ((uint)data[start + i + 2] << 16)
                | ((uint)data[start + i + 3] << 24));
            ha = Round(ha, k);
            hb = Round(hb, k);
        }
        uint t = 0;
        int rem = len & 3;
        if (rem == 3) t ^= (uint)data[start + i + 2] << 16;
        if (rem >= 2) t ^= (uint)data[start + i + 1] << 8;
        if (rem >= 1)
        {
            t ^= data[start + i];
            t = Mix(t);
            ha ^= t;
            hb ^= t;
        }
        a[c] = Fmix(ha ^ (uint)len);
        b[c] = Fmix(hb ^ (uint)len);
    }

    // one thread: the chunk hashes in order, into the two halves of the result
    private static void CombineKernel(Index1D _, ArrayView<uint> a, ArrayView<uint> b, ArrayView<uint> ra, ArrayView<uint> rb,
        int chunks, int n)
    {
        uint ca = (uint)n, cb = (uint)n ^ CombineB;
        for (int c = 0; c < chunks; c++)
        {
            ca = Round(ca, Mix(a[c]));
            cb = Round(cb, Mix(b[c]));
        }
        uint words = (uint)(chunks * 4);
        ra[0] = Fmix(ca ^ words);
        rb[0] = Fmix(cb ^ words);
    }

    private sealed class Kernels
    {
        public required Action<Index1D, ArrayView<byte>, ArrayView<uint>, ArrayView<uint>, int> Chunks;
        public required Action<Index1D, ArrayView<uint>, ArrayView<uint>, ArrayView<uint>, ArrayView<uint>, int, int> Combine;
    }

    private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<Accelerator, Kernels> Cache = new();

    private static Kernels For(Accelerator acc) => Cache.GetValue(acc, a => new Kernels
    {
        Chunks = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<uint>, ArrayView<uint>, int>(ChunkKernel),
        Combine = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<uint>, ArrayView<uint>, ArrayView<uint>, ArrayView<uint>, int, int>(CombineKernel),
    });

    /// <summary>The hash of the first <paramref name="length"/> bytes of <paramref name="data"/>, computed on the accelerator.</summary>
    public static async Task<string> ComputeAsync(Accelerator acc, ArrayView1D<byte, Stride1D.Dense> data, long length)
    {
        if (length < 0 || length > data.Length || length > int.MaxValue - Chunk) throw new ArgumentOutOfRangeException(nameof(length));
        var k = For(acc);
        int n = (int)length, chunks = (n + Chunk - 1) / Chunk;
        using var a = acc.Allocate1D<uint>(Math.Max(1, chunks));
        using var b = acc.Allocate1D<uint>(Math.Max(1, chunks));
        using var ra = acc.Allocate1D<uint>(1);
        using var rb = acc.Allocate1D<uint>(1);
        if (chunks > 0) k.Chunks(chunks, data.SubView(0, Math.Max(1, n)), a.View, b.View, n);
        k.Combine(1, a.View, b.View, ra.View, rb.View, chunks, n);
        uint ha = (await ra.View.CopyToHostAsync())[0];
        uint hb = (await rb.View.CopyToHostAsync())[0];
        return $"{ha:x8}{hb:x8}";
    }
}
