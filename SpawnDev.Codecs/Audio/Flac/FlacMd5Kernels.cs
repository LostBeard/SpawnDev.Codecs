// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// MD5 (RFC 1321) of FLAC samples on the accelerator: the digest STREAMINFO carries, over every sample's little-endian
// bytes (ceil(bits / 8) per sample, interleaved). Bounded launches: Md5BlocksKernel runs a fixed number of 64-byte blocks
// per call (one thread, the state in md5[0..4)), Md5FinalKernel the last partial block, padding and length, then writes
// the 16 digest bytes at md5[Md5Digest..].

using ILGPU;

namespace SpawnDev.Codecs.Audio.Flac;

/// <summary>MD5 of FLAC sample bytes (see the file header).</summary>
public static class FlacMd5Kernels
{
    /// <summary>The md5 view: state words at [0..4), digest bytes at [Md5Digest..Md5Digest + 16).</summary>
    public const int Md5A = 0, Md5Digest = 4, Md5Size = 20;

    /// <summary>Blocks per Md5BlocksKernel launch (bounded work per launch).</summary>
    public const int Md5BlocksPerLaunch = 4096;

    /// <summary>The MD5 sine table (64 words) and shift amounts (64) as the kernels read them.</summary>
    public static uint[] BuildTables()
    {
        var t = new uint[128];
        for (int i = 0; i < 64; i++) t[i] = (uint)(long)Math.Floor(Math.Abs(Math.Sin(i + 1)) * 4294967296.0);
        int[] s = { 7, 12, 17, 22, 5, 9, 14, 20, 4, 11, 16, 23, 6, 10, 15, 21 };
        for (int i = 0; i < 64; i++) t[64 + i] = (uint)s[(i / 16) * 4 + (i & 3)];
        return t;
    }

    private static uint Rotl(uint x, int c) => (x << c) | (x >> (32 - c));

    /// <summary>Byte i of the sample stream as STREAMINFO's MD5 covers it: little-endian, bytesPerSample bytes.</summary>
    private static uint SampleByte(ArrayView<int> input, long i, int bytesPerSample)
    {
        long s = i / bytesPerSample;
        int k = (int)(i - s * bytesPerSample);
        return (uint)(input[s] >> (8 * k)) & 0xFF;
    }

    /// <summary>
    /// One MD5 block at <paramref name="baseByte"/> of the padded message: sample bytes below <paramref name="total"/>, the
    /// 0x80 pad at <paramref name="total"/> when <paramref name="pad"/>, the bit length in the last 8 bytes when
    /// <paramref name="withLength"/>, zeros elsewhere.
    /// </summary>
    private static void Md5Block(ArrayView<int> input, long baseByte, int bytesPerSample, long total, ArrayView<uint> tables,
        ref uint a0, ref uint b0, ref uint c0, ref uint d0, bool pad, bool withLength, long lengthBits)
    {
        uint a = a0, b = b0, c = c0, d = d0;
        for (int i = 0; i < 64; i++)
        {
            uint fn;
            int g;
            if (i < 16) { fn = (b & c) | (~b & d); g = i; }
            else if (i < 32) { fn = (d & b) | (~d & c); g = (5 * i + 1) & 15; }
            else if (i < 48) { fn = b ^ c ^ d; g = (3 * i + 5) & 15; }
            else { fn = c ^ (b | ~d); g = (7 * i) & 15; }
            // message word g: 4 bytes of the (padded) stream at baseByte + 4g
            uint m = 0;
            for (int k = 0; k < 4; k++)
            {
                long at = baseByte + 4 * g + k;
                uint by = 0;
                long lenAt = at - (baseByte + 56);
                if (at < total) by = SampleByte(input, at, bytesPerSample);
                else if (at == total & pad) by = 0x80;
                else if (withLength & lenAt >= 0) by = (uint)((lengthBits >> (int)(8 * lenAt)) & 0xFF);
                m |= by << (8 * k);
            }
            uint tmp = d;
            d = c;
            c = b;
            b = b + Rotl(a + fn + tables[i] + m, (int)tables[64 + i]);
            a = tmp;
        }
        a0 += a; b0 += b; c0 += c; d0 += d;
    }

    /// <summary>One thread: MD5 blocks [first, first + count) of the sample bytes (full blocks only).</summary>
    public static void Md5BlocksKernel(Index1D _, ArrayView<int> input, ArrayView<uint> md5, ArrayView<uint> tables,
        int bytesPerSample, long totalBytes, long firstBlock, int count)
    {
        uint a = md5[0], b = md5[1], c = md5[2], d = md5[3];
        for (int k = 0; k < count; k++)
            Md5Block(input, (firstBlock + k) * 64, bytesPerSample, totalBytes, tables, ref a, ref b, ref c, ref d, false, false, 0);
        md5[0] = a; md5[1] = b; md5[2] = c; md5[3] = d;
    }

    /// <summary>One thread: the trailing partial block, the 0x80 pad and the length (one or two blocks), the digest bytes.</summary>
    public static void Md5FinalKernel(Index1D _, ArrayView<int> input, ArrayView<uint> md5, ArrayView<uint> tables,
        int bytesPerSample, long totalBytes)
    {
        uint a = md5[0], b = md5[1], c = md5[2], d = md5[3];
        long fullBlocks = totalBytes / 64;
        long rest = totalBytes - fullBlocks * 64;
        long baseByte = fullBlocks * 64;
        long bits = totalBytes * 8;
        if (rest < 56)
        {
            Md5Block(input, baseByte, bytesPerSample, totalBytes, tables, ref a, ref b, ref c, ref d, true, true, bits);
        }
        else
        {
            // the 0x80 pad still fits this block; the length needs the next one
            Md5Block(input, baseByte, bytesPerSample, totalBytes, tables, ref a, ref b, ref c, ref d, true, false, 0);
            Md5Block(input, baseByte + 64, bytesPerSample, totalBytes, tables, ref a, ref b, ref c, ref d, false, true, bits);
        }
        md5[0] = a; md5[1] = b; md5[2] = c; md5[3] = d;
        for (int i = 0; i < 16; i++)
        {
            uint w = i < 4 ? a : i < 8 ? b : i < 12 ? c : d;
            md5[Md5Digest + i] = (w >> (8 * (i & 3))) & 0xFF;
        }
    }
}
