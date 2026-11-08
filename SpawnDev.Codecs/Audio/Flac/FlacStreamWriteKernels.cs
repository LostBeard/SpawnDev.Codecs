// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Frame sizes, the bit stream, the MD5 signature and the stream header of FlacEncoderGpu.EncodeAsync (FFmpeg 8.1
// libavcodec/flacenc.c + libavformat/flacenc.c):
//   FrameBitsKernel     one thread per frame: count_frame_header + the subframes' exact bits + alignment + CRC-16; marks
//                       a frame larger than verbatim coding for the verbatim fallback (flac_get_max_frame_size)
//   OffsetsKernel       one thread: each frame's byte offset in the file, total size, min / max frame size
//   WriteFrameKernel    one thread per frame: frame header (CRC-8), subframes (constant / verbatim / fixed / LPC with
//                       partitioned rice residual), byte alignment, CRC-16
//   Md5BlocksKernel     one thread, a bounded run of 64-byte MD5 blocks per launch over the samples' little-endian bytes
//   Md5FinalKernel      one thread: the last partial block, padding and length
//   HeaderKernel        one thread per header byte: "fLaC", STREAMINFO (with the MD5), VORBIS_COMMENT (payload from the
//                       caller's metadata blob), PADDING

using ILGPU;
using ILGPU.Runtime;
using static SpawnDev.Codecs.Audio.Flac.FlacStreamKernels;

namespace SpawnDev.Codecs.Audio.Flac;

internal static class FlacStreamWriteKernels
{
    /// <summary>MD5 state words in the md5 buffer: A, B, C, D, then the digest bytes (16) as ints.</summary>
    public const int Md5A = 0, Md5Digest = 4, Md5Size = 20;

    /// <summary>Blocks per Md5BlocksKernel launch (bounded work per launch).</summary>
    public const int Md5BlocksPerLaunch = 4096;

    private static int Utf8Length(long v)
    {
        if (v < 0x80) return 1;
        if (v < 0x800) return 2;
        if (v < 0x10000) return 3;
        if (v < 0x200000) return 4;
        if (v < 0x4000000) return 5;
        return 6;
    }

    /// <summary>ff_flac_blocksize_table index of n (0 when not in the table: explicit 6 / 7).</summary>
    private static int BlockSizeCode(int n)
    {
        int code = 0;
        if (n == 192) code = 1;
        else if (n == 576) code = 2;
        else if (n == 1152) code = 3;
        else if (n == 2304) code = 4;
        else if (n == 4608) code = 5;
        else if (n == 256) code = 8;
        else if (n == 512) code = 9;
        else if (n == 1024) code = 10;
        else if (n == 2048) code = 11;
        else if (n == 4096) code = 12;
        else if (n == 8192) code = 13;
        else if (n == 16384) code = 14;
        else if (n == 32768) code = 15;
        if (code == 0) code = n <= 256 ? 6 : 7;
        return code;
    }

    private static int HeaderBits(int f, int n, ArrayView<int> meta)
    {
        int count = 32 + 8 * Utf8Length(f);
        int bs = BlockSizeCode(n);
        if (bs == 6) count += 8;
        else if (bs == 7) count += 16;
        int sr = meta[MSrCode0];
        if (sr == 12) count += 8;
        else if (sr > 12) count += 16;
        return count + 8;
    }

    /// <summary>flac_get_max_frame_size: the verbatim size bound of an n-sample frame.</summary>
    private static int MaxFrameSize(int n, int channels, int bps)
    {
        int count = 16 + channels * ((7 + bps + 7) / 8);
        if (channels == 2) count += ((2 * bps + 1) * n + 7) / 8;
        else count += (channels * bps * n + 7) / 8;
        return count + 2;
    }

    /// <summary>
    /// One thread per frame: its size in bytes; when larger than the verbatim bound (and not yet verbatim) the frame is
    /// marked for the verbatim re-encode (FVerbatim = 1) and its size left for the second pass.
    /// </summary>
    public static void FrameBitsKernel(Index1D index, ArrayView<int> frames, ArrayView<int> subs, ArrayView<int> meta)
    {
        int f = index.X, channels = meta[MChannels];
        int fr = f * FrameStride;
        int n = frames[fr + FN];
        long count = HeaderBits(f, n, meta);
        for (int c = 0; c < channels; c++) count += subs[(f * channels + c) * SubStride + SBits];
        count += (8 - (count & 7)) & 7;
        count += 16;
        int bytes = (int)(count >> 3);
        int limit = f == meta[MFrames] - 1 && n < meta[MBlockSize] ? MaxFrameSize(n, channels, meta[MBps]) : meta[MMaxFrameInit];
        if (bytes > limit & frames[fr + FVerbatim] == 0) frames[fr + FVerbatim] = 1;   // FFmpeg: encode again, verbatim only
        else if (frames[fr + FVerbatim] == 1) frames[fr + FVerbatim] = 2;              // the verbatim pass is done
        frames[fr + FBytes] = bytes;
    }

    /// <summary>One thread: frame offsets after the stream header, the file size, STREAMINFO's min / max frame size.</summary>
    public static void OffsetsKernel(Index1D _, ArrayView<int> frames, ArrayView<int> meta)
    {
        int frameCount = meta[MFrames];
        long at = meta[MFramesStart];
        int min = meta[MMaxFrameInit], max = 0;
        for (int f = 0; f < frameCount; f++)
        {
            int fr = f * FrameStride;
            frames[fr + FOffset] = (int)at;
            int bytes = frames[fr + FBytes];
            at += bytes;
            if (bytes < min) min = bytes;
            if (bytes > max) max = bytes;
        }
        meta[MTotalBytes] = (int)at;
        meta[MMinFrame] = min;
        meta[MMaxFrame] = max;
    }

    // ------------------------------------------------------------------ bit writer (MSB first)

    private static void Put(ArrayView<byte> o, ref long pos, ref ulong acc, ref int nbits, int count, ulong value)
    {
        // count <= 32 per call
        if (count == 0) return;
        acc = (acc << count) | (value & ((1UL << count) - 1));
        nbits += count;
        while (nbits >= 8)
        {
            nbits -= 8;
            o[pos] = (byte)(acc >> nbits);
            pos++;
        }
        acc &= (1UL << nbits) - 1;
    }

    private static void PutSigned(ArrayView<byte> o, ref long pos, ref ulong acc, ref int nbits, int count, int value) =>
        Put(o, ref pos, ref acc, ref nbits, count, (ulong)(uint)value);

    private static void Flush(ArrayView<byte> o, ref long pos, ref ulong acc, ref int nbits)
    {
        if (nbits > 0)
        {
            o[pos] = (byte)(acc << (8 - nbits));
            pos++;
            nbits = 0;
            acc = 0;
        }
    }

    private static int Crc8(ArrayView<byte> o, long start, long end)
    {
        int crc = 0;
        for (long i = start; i < end; i++)
        {
            crc ^= o[i];
            for (int b = 0; b < 8; b++) crc = (crc & 0x80) != 0 ? ((crc << 1) ^ 0x07) & 0xFF : (crc << 1) & 0xFF;
        }
        return crc;
    }

    private static int Crc16(ArrayView<byte> o, long start, long end)
    {
        int crc = 0;
        for (long i = start; i < end; i++)
        {
            crc ^= o[i] << 8;
            for (int b = 0; b < 8; b++) crc = (crc & 0x8000) != 0 ? ((crc << 1) ^ 0x8005) & 0xFFFF : (crc << 1) & 0xFFFF;
        }
        return crc;
    }

    /// <summary>One thread per frame: write_frame (header + CRC-8, subframes, alignment, CRC-16) at its offset.</summary>
    public static void WriteFrameKernel(Index1D index, ArrayView<int> r, ArrayView<int> frames, ArrayView<int> subs,
        ArrayView<int> parmsAll, ArrayView<byte> o, ArrayView<int> meta)
    {
        int f = index.X, channels = meta[MChannels], b = meta[MBlockSize];
        int fr = f * FrameStride;
        int n = frames[fr + FN], mode = frames[fr + FChMode];
        long start = frames[fr + FOffset];
        long pos = start;
        ulong acc = 0;
        int nb = 0;

        // frame header
        Put(o, ref pos, ref acc, ref nb, 16, 0xFFF8);
        int bs = BlockSizeCode(n);
        Put(o, ref pos, ref acc, ref nb, 4, (ulong)bs);
        Put(o, ref pos, ref acc, ref nb, 4, (ulong)meta[MSrCode0]);
        Put(o, ref pos, ref acc, ref nb, 4, (ulong)(mode == 0 ? channels - 1 : mode + 8 - 1));
        Put(o, ref pos, ref acc, ref nb, 3, (ulong)meta[MBpsCode]);
        Put(o, ref pos, ref acc, ref nb, 1, 0);
        // UTF-8 coded frame number
        long v = f;
        int len = Utf8Length(v);
        if (len == 1)
        {
            Put(o, ref pos, ref acc, ref nb, 8, (ulong)v);
        }
        else
        {
            int lead = (0xFF00 >> len) & 0xFF;
            Put(o, ref pos, ref acc, ref nb, 8, (ulong)(lead | (int)(v >> (6 * (len - 1)))));
            for (int i = len - 2; i >= 0; i--) Put(o, ref pos, ref acc, ref nb, 8, (ulong)(0x80 | (int)((v >> (6 * i)) & 0x3F)));
        }
        if (bs == 6) Put(o, ref pos, ref acc, ref nb, 8, (ulong)(n - 1));
        else if (bs == 7) Put(o, ref pos, ref acc, ref nb, 16, (ulong)(n - 1));
        int src = meta[MSrCode0];
        if (src == 12) Put(o, ref pos, ref acc, ref nb, 8, (ulong)meta[MSrCode1]);
        else if (src > 12) Put(o, ref pos, ref acc, ref nb, 16, (ulong)meta[MSrCode1]);
        Flush(o, ref pos, ref acc, ref nb);
        Put(o, ref pos, ref acc, ref nb, 8, (ulong)Crc8(o, start, pos));

        int precision = meta[MPrecision];
        for (int c = 0; c < channels; c++)
        {
            int sr = (f * channels + c) * SubStride;
            int slot = (f * channels + c) * b;
            int type = subs[sr + SType], obits = subs[sr + SObits], wasted = subs[sr + SWasted], order = subs[sr + SOrder];
            int typeCode = type == TypeConstant ? 0 : type == TypeVerbatim ? 1 : type == TypeFixed ? 8 | order : 32 | (order - 1);
            Put(o, ref pos, ref acc, ref nb, 1, 0);
            Put(o, ref pos, ref acc, ref nb, 6, (ulong)typeCode);
            Put(o, ref pos, ref acc, ref nb, 1, wasted != 0 ? 1UL : 0UL);
            if (wasted != 0) Put(o, ref pos, ref acc, ref nb, wasted, 1);
            if (type == TypeConstant)
            {
                PutSigned(o, ref pos, ref acc, ref nb, obits, r[slot]);
            }
            else if (type == TypeVerbatim)
            {
                for (int i = 0; i < n; i++) PutSigned(o, ref pos, ref acc, ref nb, obits, r[slot + i]);
            }
            else
            {
                for (int i = 0; i < order; i++) PutSigned(o, ref pos, ref acc, ref nb, obits, r[slot + i]);
                if (type == TypeLpc)
                {
                    Put(o, ref pos, ref acc, ref nb, 4, (ulong)(precision - 1));
                    PutSigned(o, ref pos, ref acc, ref nb, 5, subs[sr + SShift]);
                    for (int i = 0; i < order; i++) PutSigned(o, ref pos, ref acc, ref nb, precision, subs[sr + SCoef + i]);
                }
                int cm = subs[sr + SCodingMode];
                Put(o, ref pos, ref acc, ref nb, 2, (ulong)(cm - 4));
                int porder = subs[sr + SPorder];
                Put(o, ref pos, ref acc, ref nb, 4, (ulong)porder);
                int psize = n >> porder;
                int at = order, end = psize;
                for (int p = 0; p < (1 << porder); p++)
                {
                    int k = parmsAll[(f * channels + c) * MaxPartitions + p];
                    Put(o, ref pos, ref acc, ref nb, cm, (ulong)k);
                    for (int i = at; i < end; i++)
                    {
                        int x = r[slot + i];
                        uint u = (uint)(x << 1) ^ (uint)(x >> 31);
                        uint e = (u >> k) + 1;
                        while (e > 31)
                        {
                            Put(o, ref pos, ref acc, ref nb, 31, 0);
                            e -= 31;
                        }
                        Put(o, ref pos, ref acc, ref nb, (int)e, 1);
                        if (k > 0) Put(o, ref pos, ref acc, ref nb, k, u & (uint.MaxValue >> (32 - k)));
                    }
                    at = end;
                    end = end + psize < n ? end + psize : n;
                }
            }
        }
        Flush(o, ref pos, ref acc, ref nb);
        int crc = Crc16(o, start, pos);
        o[pos] = (byte)(crc >> 8);
        o[pos + 1] = (byte)crc;
    }

    // ------------------------------------------------------------------ MD5 of the samples (little-endian, bps bytes each)

    private static uint Rotl(uint x, int c) => (x << c) | (x >> (32 - c));

    /// <summary>Byte i of the sample stream as FFmpeg hashes it (16-bit: 2 bytes LE; 24-bit: 3 bytes LE).</summary>
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

    // ------------------------------------------------------------------ stream header

    /// <summary>
    /// One thread per byte of the stream header: "fLaC", STREAMINFO block (34 bytes, MD5 included), VORBIS_COMMENT block
    /// (payload = the blob), PADDING block of <paramref name="padding"/> zero bytes (none when 0: the comment is last).
    /// </summary>
    public static void HeaderKernel(Index1D index, ArrayView<byte> o, ArrayView<byte> comment, int commentLength, int padding,
        ArrayView<int> meta, ArrayView<uint> md5)
    {
        int i = index.X;
        byte v = 0;
        if (i < 4) v = (byte)(i == 0 ? 0x66 : i == 1 ? 0x4C : i == 2 ? 0x61 : 0x43);   // "fLaC"
        else if (i < 8) v = i == 4 ? (byte)0x00 : i == 7 ? (byte)0x22 : (byte)0;
        else if (i < 42) v = StreamInfoByte(i - 8, meta, md5);
        else if (i < 46)
        {
            int k = i - 42;
            v = k == 0 ? (byte)(padding > 0 ? 0x04 : 0x84) : (byte)((commentLength >> (8 * (3 - k))) & 0xFF);
        }
        else if (i < 46 + commentLength) v = comment[i - 46];
        else if (i < 46 + commentLength + 4)
        {
            int k = i - 46 - commentLength;
            v = k == 0 ? (byte)0x81 : (byte)((padding >> (8 * (3 - k))) & 0xFF);
        }
        o[i] = v;
    }

    /// <summary>Byte k of write_streaminfo's 34 bytes.</summary>
    private static byte StreamInfoByte(int k, ArrayView<int> meta, ArrayView<uint> md5)
    {
        if (k >= 18) return (byte)md5[Md5Digest + k - 18];
        int bsz = meta[MBlockSize], minF = meta[MMinFrame], maxF = meta[MMaxFrame];
        long total = ((long)meta[MTotalHi] << 32) | (uint)meta[MTotalLo];
        int rate = meta[MRate];
        int ch = meta[MChannels], bps = meta[MBps];
        // 16 min block, 16 max block, 24 min frame, 24 max frame, 20 rate, 3 channels-1, 5 bps-1, 36 total samples
        if (k < 2) return (byte)(bsz >> (8 * (1 - k)));
        if (k < 4) return (byte)(bsz >> (8 * (3 - k)));
        if (k < 7) return (byte)(minF >> (8 * (6 - k)));
        if (k < 10) return (byte)(maxF >> (8 * (9 - k)));
        // bits 80..: rate(20) ch(3) bps(5) total(36) = 64 bits in bytes 10..17
        ulong packed = ((ulong)(uint)rate << 44) | ((ulong)(ch - 1) << 41) | ((ulong)(bps - 1) << 36) | ((ulong)total & 0xFFFFFFFFFUL);
        return (byte)(packed >> (8 * (17 - k)));
    }
}
