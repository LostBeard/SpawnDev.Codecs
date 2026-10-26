// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// XXH32 (xxHash, 32-bit; Yann Collet, BSD-2) as GPU-callable functions: the LZ4 frame header checksum, the optional
// per-block checksums and the optional content checksum. One-shot Hash() over a range for short inputs, and a resumable
// form (four lane accumulators + the byte count, kept in device memory) for a long range hashed by a chain of bounded
// launches: Begin, then Stripes over whole 16-byte stripes, then Finish over the tail.

using ILGPU;

namespace SpawnDev.Codecs.Compression.Lz4;

/// <summary>GPU-callable XXH32.</summary>
public static class Xxh32Gpu
{
    public const uint Prime1 = 0x9E3779B1u, Prime2 = 0x85EBCA77u, Prime3 = 0xC2B2AE3Du, Prime4 = 0x27D4EB2Fu, Prime5 = 0x165667B1u;

    /// <summary>Ints of a resumable state: v1..v4 and the total length (bytes, low 31 bits enough per call).</summary>
    public const int StateInts = 5;

    private static uint Rotl(uint x, int r) => (x << r) | (x >> (32 - r));

    /// <summary>Little-endian 32-bit read of 4 bytes at <paramref name="at"/>.</summary>
    public static uint Read32(ArrayView<byte> data, int at) =>
        (uint)data[at] | ((uint)data[at + 1] << 8) | ((uint)data[at + 2] << 16) | ((uint)data[at + 3] << 24);

    private static uint Round(uint acc, uint input)
    {
        acc += input * Prime2;
        acc = Rotl(acc, 13);
        return acc * Prime1;
    }

    /// <summary>XXH32 of <paramref name="length"/> bytes at <paramref name="offset"/> with <paramref name="seed"/> (one thread, whole range).</summary>
    public static uint Hash(ArrayView<byte> data, int offset, int length, uint seed)
    {
        uint h;
        int p = offset;
        int end = offset + length;
        if (length >= 16)
        {
            uint v1 = seed + Prime1 + Prime2, v2 = seed + Prime2, v3 = seed, v4 = seed - Prime1;
            int limit = end - 16;
            while (p <= limit)
            {
                v1 = Round(v1, Read32(data, p));
                v2 = Round(v2, Read32(data, p + 4));
                v3 = Round(v3, Read32(data, p + 8));
                v4 = Round(v4, Read32(data, p + 12));
                p += 16;
            }
            h = Rotl(v1, 1) + Rotl(v2, 7) + Rotl(v3, 12) + Rotl(v4, 18);
        }
        else h = seed + Prime5;
        h += (uint)length;
        return Tail(h, data, p, end);
    }

    private static uint Tail(uint h, ArrayView<byte> data, int p, int end)
    {
        while (p + 4 <= end)
        {
            h += Read32(data, p) * Prime3;
            h = Rotl(h, 17) * Prime4;
            p += 4;
        }
        while (p < end)
        {
            h += data[p] * Prime5;
            h = Rotl(h, 11) * Prime1;
            p++;
        }
        h ^= h >> 15;
        h *= Prime2;
        h ^= h >> 13;
        h *= Prime3;
        h ^= h >> 16;
        return h;
    }

    /// <summary>Resumable form, step 1: the lane accumulators of <paramref name="seed"/> into state[at ..].</summary>
    public static void Begin(ArrayView<int> state, int at, uint seed)
    {
        state[at] = (int)(seed + Prime1 + Prime2);
        state[at + 1] = (int)(seed + Prime2);
        state[at + 2] = (int)seed;
        state[at + 3] = (int)(seed - Prime1);
        state[at + 4] = 0;
    }

    /// <summary>
    /// Resumable form, step 2: folds <paramref name="stripes"/> whole 16-byte stripes at <paramref name="offset"/> (the next
    /// bytes of the range) into the state and adds their length.
    /// </summary>
    public static void Stripes(ArrayView<int> state, int at, ArrayView<byte> data, int offset, int stripes)
    {
        uint v1 = (uint)state[at], v2 = (uint)state[at + 1], v3 = (uint)state[at + 2], v4 = (uint)state[at + 3];
        int p = offset;
        for (int s = 0; s < stripes; s++)
        {
            v1 = Round(v1, Read32(data, p));
            v2 = Round(v2, Read32(data, p + 4));
            v3 = Round(v3, Read32(data, p + 8));
            v4 = Round(v4, Read32(data, p + 12));
            p += 16;
        }
        state[at] = (int)v1;
        state[at + 1] = (int)v2;
        state[at + 2] = (int)v3;
        state[at + 3] = (int)v4;
        state[at + 4] += stripes * 16;
    }

    /// <summary>
    /// Resumable form, step 3: the hash of the whole range, given the state after every whole stripe and the
    /// <paramref name="tailLength"/> (&lt; 16) remaining bytes at <paramref name="tailOffset"/>.
    /// </summary>
    public static uint Finish(ArrayView<int> state, int at, ArrayView<byte> data, int tailOffset, int tailLength, uint seed)
    {
        int folded = state[at + 4];
        uint h;
        if (folded > 0)
            h = Rotl((uint)state[at], 1) + Rotl((uint)state[at + 1], 7) + Rotl((uint)state[at + 2], 12) + Rotl((uint)state[at + 3], 18);
        else h = seed + Prime5;
        h += (uint)(folded + tailLength);
        return Tail(h, data, tailOffset, tailOffset + tailLength);
    }
}
