// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Compute the MD5 signature of FLAC PCM samples per RFC 9639 Section 8.1:
// "The MD5 signature of the unencoded audio data. [...] Samples shall be
// packed tightly ... in the exact bit depth used in the stream, little-endian
// signed." Matches libFLAC's digest pipeline.
//
// Uses a self-contained pure-C# MD5 implementation (RFC 1321) to keep the
// library browser-compatible - System.Security.Cryptography.MD5 is not
// supported on the 'browser' target for Blazor WASM.

namespace SpawnDev.Codecs.Audio.Flac;

internal static class FlacMd5
{
    /// <summary>
    /// Compute MD5 over interleaved signed integer PCM samples, packed as
    /// little-endian two's-complement at <paramref name="bitsPerSample"/> bits
    /// (rounded up to a whole number of bytes - 12-bit packs into 16-bit LE,
    /// 20-bit packs into 24-bit LE, following libFLAC's convention).
    /// </summary>
    internal static byte[] Compute(ReadOnlySpan<int> interleavedSamples, int bitsPerSample)
    {
        int bytesPerSample = (bitsPerSample + 7) / 8;
        int totalBytes = interleavedSamples.Length * bytesPerSample;
        byte[] buffer = new byte[totalBytes];
        int pos = 0;
        for (int i = 0; i < interleavedSamples.Length; i++)
        {
            int v = interleavedSamples[i];
            for (int b = 0; b < bytesPerSample; b++)
                buffer[pos + b] = (byte)(v >> (8 * b));
            pos += bytesPerSample;
        }
        return Md5(buffer);
    }

    // RFC 1321 per-round constants K[i] = floor(|sin(i+1)| * 2^32), as 32-bit unsigned integers.
    private static readonly uint[] K = new uint[64]
    {
        0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee,
        0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
        0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be,
        0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
        0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa,
        0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
        0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
        0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
        0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c,
        0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
        0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05,
        0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
        0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039,
        0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
        0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1,
        0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391,
    };

    // RFC 1321 per-round left-rotation amounts.
    private static readonly int[] S = new int[64]
    {
         7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
         5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20,
         4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
         6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
    };

    private static byte[] Md5(ReadOnlySpan<byte> data)
    {
        // Padding: append 0x80, then zeros until length ≡ 56 (mod 64), then 8-byte LE bit count.
        long bitLength = (long)data.Length * 8;
        int padLength = (56 - (data.Length + 1) % 64 + 64) % 64;
        int totalLength = data.Length + 1 + padLength + 8;
        byte[] padded = new byte[totalLength];
        data.CopyTo(padded);
        padded[data.Length] = 0x80;
        for (int i = 0; i < 8; i++)
            padded[totalLength - 8 + i] = (byte)(bitLength >> (8 * i));

        uint A = 0x67452301, B = 0xefcdab89, C = 0x98badcfe, D = 0x10325476;
        Span<uint> M = stackalloc uint[16];
        for (int offset = 0; offset < totalLength; offset += 64)
        {
            for (int i = 0; i < 16; i++)
            {
                M[i] = (uint)padded[offset + i * 4]
                     | ((uint)padded[offset + i * 4 + 1] << 8)
                     | ((uint)padded[offset + i * 4 + 2] << 16)
                     | ((uint)padded[offset + i * 4 + 3] << 24);
            }
            uint a = A, b = B, c = C, d = D;
            for (int i = 0; i < 64; i++)
            {
                uint f;
                int g;
                if (i < 16) { f = (b & c) | (~b & d); g = i; }
                else if (i < 32) { f = (d & b) | (~d & c); g = (5 * i + 1) % 16; }
                else if (i < 48) { f = b ^ c ^ d; g = (3 * i + 5) % 16; }
                else { f = c ^ (b | ~d); g = (7 * i) % 16; }
                uint sum = a + f + K[i] + M[g];
                uint rotated = (sum << S[i]) | (sum >> (32 - S[i]));
                a = d;
                d = c;
                c = b;
                b = b + rotated;
            }
            A += a; B += b; C += c; D += d;
        }

        byte[] result = new byte[16];
        for (int i = 0; i < 4; i++)
        {
            result[i] = (byte)(A >> (8 * i));
            result[i + 4] = (byte)(B >> (8 * i));
            result[i + 8] = (byte)(C >> (8 * i));
            result[i + 12] = (byte)(D >> (8 * i));
        }
        return result;
    }
}
