// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// The FLAC decoder's kernels (RFC 9639): any conforming FLAC stream - fixed or variable block size, 4..32 bits per
// sample, 1..8 channels, every subframe type (constant, verbatim, fixed 0..4, LPC 1..32), Rice / Rice2 residuals with
// escaped partitions, wasted bits, every stereo decorrelation. The whole file sits on the accelerator; the host parses
// only the metadata blocks and dispatches:
//   1. FindSync: one thread per byte flags a frame header - sync code, valid fields that agree with STREAMINFO, CRC-8.
//   2. CountSync / Offsets / Compact: the flagged positions in file order (block counts, one prefix, ordered writes).
//   3. FrameEnd: one thread per candidate scans forward, CRC-16 over the bytes, to the first later sync code (or the
//      end of the file) where the CRC comes out 0: that is where the frame ends - also when the next frame's header is
//      damaged past its sync code. A candidate inside frame data (the sync pattern by chance, with a matching CRC-8)
//      finds no such end within a frame's largest size.
//   4. Walk: one thread follows the frames from the first verified one, end to start, and gives each its first sample.
//   5. DecodeFrame: one thread per frame decodes its subframes into a channel-major region (the side channel of a 32-bit
//      stream, 33 bits wide, into a separate int64 view).
//   6. Interleave: one thread per sample undoes the stereo decorrelation and writes interleaved int32.
// libFLAC's decoder checks frames the same way (header CRC-8, frame CRC-16); a frame that fails comes out as silence.

using ILGPU;
using ILGPU.Algorithms;

namespace SpawnDev.Codecs.Audio.Flac;

/// <summary>Kernels of the FLAC decoder (see the file header).</summary>
public static class FlacStreamDecodeKernels
{
    /// <summary>Bytes per block of the sync-flag compaction.</summary>
    public const int SyncBlock = 4096;

    /// <summary>The meta view (ints): STREAMINFO fields in, counts out.</summary>
    public const int MChannels = 0, MBps = 1, MRate = 2, MMaxFrame = 3, MCandidates = 4, MFrames = 5, MSkippedBytes = 6,
        MBadFrames = 7, MSyncBlocks = 8, MBlockSize = 9, MGapSamples = 10, MetaSize = 11;

    /// <summary>Per-frame ints: block size, channel assignment, status (0 = decoded).</summary>
    public const int FBlockSize = 0, FAssign = 1, FStatus = 2, FrameInts = 3;

    // ------------------------------------------------------------------ bit reading (MSB first)

    private static int ByteAt(ArrayView<byte> d, long at) => at < d.Length ? d[at] : 0;

    /// <summary>n (0..24) bits at bit position pos.</summary>
    private static uint Read24(ArrayView<byte> d, ref long pos, int n)
    {
        if (n == 0) return 0;
        long at = pos >> 3;
        int off = (int)(pos & 7);
        uint w = ((uint)ByteAt(d, at) << 24) | ((uint)ByteAt(d, at + 1) << 16) | ((uint)ByteAt(d, at + 2) << 8) | (uint)ByteAt(d, at + 3);
        pos += n;
        return (w << off) >> (32 - n);
    }

    /// <summary>n (0..32) bits.</summary>
    private static uint Read(ArrayView<byte> d, ref long pos, int n)
    {
        if (n <= 24) return Read24(d, ref pos, n);
        uint hi = Read24(d, ref pos, n - 16);
        uint lo = Read24(d, ref pos, 16);
        return (hi << 16) | lo;
    }

    /// <summary>A signed n-bit (1..32) value.</summary>
    private static int ReadSigned(ArrayView<byte> d, ref long pos, int n)
    {
        uint u = Read(d, ref pos, n);
        if (n < 32 && (u & (1u << (n - 1))) != 0) u |= ~((1u << n) - 1);
        return (int)u;
    }

    /// <summary>A signed n-bit (1..33) value as int64.</summary>
    private static long ReadSignedLong(ArrayView<byte> d, ref long pos, int n)
    {
        if (n <= 32) return ReadSigned(d, ref pos, n);
        uint top = Read24(d, ref pos, 1);
        uint low = Read(d, ref pos, 32);
        return top != 0 ? (long)low - (1L << 32) : low;
    }

    /// <summary>A unary count (zeros before a one), or -1 past <paramref name="endBit"/>.</summary>
    private static int Unary(ArrayView<byte> d, ref long pos, long endBit)
    {
        int count = 0;
        while (pos < endBit)
        {
            int off = (int)(pos & 7);
            int b = (ByteAt(d, pos >> 3) << off) & 0xFF;
            if (b == 0)
            {
                count += 8 - off;
                pos += 8 - off;
                continue;
            }
            int lz = 0;
            while ((b & 0x80) == 0) { b <<= 1; lz++; }
            count += lz;
            pos += lz + 1;
            return count;
        }
        return -1;
    }

    // ------------------------------------------------------------------ frame header

    private static int Crc8(ArrayView<byte> d, long at, int count)
    {
        int crc = 0;
        for (int i = 0; i < count; i++)
        {
            crc ^= d[at + i];
            for (int k = 0; k < 8; k++) crc = (crc & 0x80) != 0 ? ((crc << 1) ^ 0x07) & 0xFF : (crc << 1) & 0xFF;
        }
        return crc;
    }

    /// <summary>
    /// The frame header at <paramref name="at"/>: its length in bytes (CRC-8 included), or 0 when there is none (no sync,
    /// a reserved value, disagreement with STREAMINFO's channels / bits / rate, a wrong CRC-8).
    /// </summary>
    public static int ParseHeader(ArrayView<byte> d, long at, long length, int siChannels, int siBps, int siRate,
        out int blockSize, out int assign, out int bps, out long number, out bool variable)
    {
        blockSize = 0; assign = 0; bps = 0; number = 0; variable = false;
        if (at + 5 > length) return 0;
        if (d[at] != 0xFF || (d[at + 1] & 0xFE) != 0xF8) return 0;
        variable = (d[at + 1] & 1) != 0;
        int b2 = d[at + 2], b3 = d[at + 3];
        int bsCode = b2 >> 4, rateCode = b2 & 15;
        assign = b3 >> 4;
        int bpsCode = (b3 >> 1) & 7;
        if (bsCode == 0 || rateCode == 15 || assign >= 11 || bpsCode == 3 || (b3 & 1) != 0) return 0;
        // UTF-8 coded frame / sample number
        long p = at + 4;
        int first = d[p++];
        int extra;
        if (first < 0x80) extra = 0;
        else if ((first & 0xE0) == 0xC0) extra = 1;
        else if ((first & 0xF0) == 0xE0) extra = 2;
        else if ((first & 0xF8) == 0xF0) extra = 3;
        else if ((first & 0xFC) == 0xF8) extra = 4;
        else if ((first & 0xFE) == 0xFC) extra = 5;
        else if (first == 0xFE) extra = 6;
        else return 0;
        if (p + extra + 4 > length) return 0;
        number = extra == 0 ? first : first & (0x3F >> extra);
        for (int i = 0; i < extra; i++)
        {
            if ((d[p + i] & 0xC0) != 0x80) return 0;
            number = (number << 6) | (long)(d[p + i] & 0x3F);
        }
        p += extra;
        // block size
        if (bsCode == 1) blockSize = 192;
        else if (bsCode <= 5) blockSize = 576 << (bsCode - 2);
        else if (bsCode == 6) blockSize = d[p++] + 1;
        else if (bsCode == 7) { blockSize = ((d[p] << 8) | d[p + 1]) + 1; p += 2; }
        else blockSize = 256 << (bsCode - 8);
        // sample rate
        int rate;
        if (rateCode == 0) rate = siRate;
        else if (rateCode == 1) rate = 88200;
        else if (rateCode == 2) rate = 176400;
        else if (rateCode == 3) rate = 192000;
        else if (rateCode == 4) rate = 8000;
        else if (rateCode == 5) rate = 16000;
        else if (rateCode == 6) rate = 22050;
        else if (rateCode == 7) rate = 24000;
        else if (rateCode == 8) rate = 32000;
        else if (rateCode == 9) rate = 44100;
        else if (rateCode == 10) rate = 48000;
        else if (rateCode == 11) rate = 96000;
        else if (rateCode == 12) rate = d[p++] * 1000;
        else if (rateCode == 13) { rate = (d[p] << 8) | d[p + 1]; p += 2; }
        else { rate = ((d[p] << 8) | d[p + 1]) * 10; p += 2; }
        // bits per sample
        if (bpsCode == 0) bps = siBps;
        else if (bpsCode == 1) bps = 8;
        else if (bpsCode == 2) bps = 12;
        else if (bpsCode == 4) bps = 16;
        else if (bpsCode == 5) bps = 20;
        else if (bpsCode == 6) bps = 24;
        else bps = 32;
        int channels = assign < 8 ? assign + 1 : 2;
        if (channels != siChannels || (siBps > 0 && bps != siBps) || (siRate > 0 && rate != siRate) || bps < 4) return 0;
        if (p >= length) return 0;
        int headerLength = (int)(p - at);
        if (Crc8(d, at, headerLength) != d[p]) return 0;
        return headerLength + 1;
    }

    // ------------------------------------------------------------------ 1-2. sync candidates, in file order

    /// <summary>One thread per byte: 1 where a frame header starts.</summary>
    public static void FindSyncKernel(Index1D i, ArrayView<byte> data, ArrayView<byte> flags, ArrayView<int> meta, long first)
    {
        long at = first + i;
        flags[at] = (byte)(ParseHeader(data, at, data.Length, meta[MChannels], meta[MBps], meta[MRate], out _, out _, out _, out _, out _) > 0 ? 1 : 0);
    }

    /// <summary>One thread per block of <see cref="SyncBlock"/> bytes: how many headers it holds.</summary>
    public static void CountSyncKernel(Index1D b, ArrayView<byte> flags, ArrayView<int> blockCounts)
    {
        long start = (long)b.X * SyncBlock, end = XMath.Min(start + SyncBlock, flags.Length);
        int n = 0;
        for (long i = start; i < end; i++) n += flags[i];
        blockCounts[b] = n;
    }

    /// <summary>One thread: exclusive prefix of the block counts; the total into MCandidates.</summary>
    public static void SyncOffsetsKernel(Index1D _, ArrayView<int> blockCounts, ArrayView<int> meta)
    {
        int blocks = meta[MSyncBlocks];
        int sum = 0;
        for (int b = 0; b < blocks; b++)
        {
            int c = blockCounts[b];
            blockCounts[b] = sum;
            sum += c;
        }
        meta[MCandidates] = sum;
    }

    /// <summary>One thread per block: its header positions, in order, at the block's offset.</summary>
    public static void CompactSyncKernel(Index1D b, ArrayView<byte> flags, ArrayView<int> blockOffsets, ArrayView<int> candidates)
    {
        long start = (long)b.X * SyncBlock, end = XMath.Min(start + SyncBlock, flags.Length);
        int at = blockOffsets[b];
        for (long i = start; i < end; i++)
            if (flags[i] != 0) candidates[at++] = (int)i;
    }

    // ------------------------------------------------------------------ 3. frame ends (CRC-16)

    /// <summary>
    /// One thread per candidate: the byte where its frame ends - the first sync code (or the end of the file) the CRC-16
    /// of the bytes between comes out 0 at - or -1 when none does within a frame's largest size.
    /// </summary>
    public static void FrameEndKernel(Index1D index, ArrayView<byte> data, ArrayView<int> candidates, ArrayView<int> crcTable,
        ArrayView<int> ends, ArrayView<int> meta, int firstCandidate)
    {
        int k = firstCandidate + index;
        int count = meta[MCandidates];
        if (k >= count) return;
        long start = candidates[k];
        long length = data.Length;
        int hl = ParseHeader(data, start, length, meta[MChannels], meta[MBps], meta[MRate], out int blockSize, out _, out int bps, out _, out _);
        // the largest frame: every subframe verbatim at bps + 1 bits, generous headers, and Rice's partition overhead;
        // or STREAMINFO's maximum when it states a larger one
        long limit = 64 + 2L * meta[MChannels] * (8 + ((long)blockSize * (bps + 1) + 7) / 8);
        if (meta[MMaxFrame] > limit) limit = meta[MMaxFrame];
        long stop = XMath.Min(length, start + limit);
        int crc = 0;
        int result = -1;
        long minEnd = start + hl + 2;
        for (long p = start; p < stop; p++)
        {
            crc = ((crc << 8) ^ crcTable[((crc >> 8) ^ data[p]) & 0xFF]) & 0xFFFF;
            long end = p + 1;
            if (crc != 0 || end <= minEnd) continue;
            if (end == length || (end + 1 < length && data[end] == 0xFF && (data[end + 1] & 0xFE) == 0xF8))
            {
                result = (int)end;
                break;
            }
        }
        ends[k] = result;
    }

    // ------------------------------------------------------------------ 4. the frame chain

    /// <summary>
    /// One thread: from the first candidate with a verified end, follow the frames; each gets its byte offset and first
    /// sample. The first sample is the one the header codes (the sample number, or the frame number times STREAMINFO's
    /// block size) when that lies past the previous frame: frames lost to damage leave a gap of silence, as libFLAC
    /// does, and the samples after it keep their time. Candidates that do not end a frame are skipped.
    /// </summary>
    public static void WalkKernel(Index1D index, ArrayView<byte> data, ArrayView<int> candidates, ArrayView<int> ends,
        ArrayView<int> frameOffsets, ArrayView<int> frameEnds, ArrayView<long> frameSamples, ArrayView<int> meta)
    {
        int count = meta[MCandidates];
        int siBlock = meta[MBlockSize];
        int frames = 0;
        long sample = 0, gap = 0, skipped = 0;
        int k = 0;
        long expect = 0;   // where the next frame should start
        while (k < count)
        {
            if (ends[k] < 0)
            {
                k++;
                continue;
            }
            int at = candidates[k];
            if (at > expect) skipped += at - expect;
            ParseHeader(data, at, data.Length, meta[MChannels], meta[MBps], meta[MRate], out int blockSize, out _, out _,
                out long number, out bool variable);
            long coded = variable ? number : number * (siBlock > 0 ? siBlock : blockSize);
            if (coded > sample)
            {
                gap += coded - sample;
                sample = coded;
            }
            frameOffsets[frames] = at;
            frameEnds[frames] = ends[k];
            frameSamples[frames] = sample;
            frames++;
            sample += blockSize;
            expect = ends[k];
            // the next frame starts at the first candidate at or past this one's end
            k++;
            while (k < count && candidates[k] < expect) k++;
        }
        if (data.Length > expect) skipped += data.Length - expect;
        frameSamples[frames] = sample;
        meta[MFrames] = frames;
        meta[MSkippedBytes] = (int)XMath.Min(skipped, int.MaxValue);
        meta[MGapSamples] = (int)XMath.Min(gap, int.MaxValue);
    }

    // ------------------------------------------------------------------ 5. frame decode

    /// <summary>The channel of an assignment that carries the side signal (one bit wider), or -1.</summary>
    private static int SideChannel(int assign) => assign == 8 ? 1 : assign == 9 ? 0 : assign == 10 ? 1 : -1;

    private static long Load(ArrayView<int> buf, ArrayView<long> wide, long chBase, long wideBase, bool isWide, int i)
        => isWide ? wide[wideBase + i] : buf[chBase + i];

    private static void Store(ArrayView<int> buf, ArrayView<long> wide, long chBase, long wideBase, bool isWide, int i, long v)
    {
        if (isWide) wide[wideBase + i] = v;
        else buf[chBase + i] = (int)v;
    }

    /// <summary>
    /// Decode the Rice / Rice2 residual of a predicted subframe into samples order..n-1 (residuals, before prediction).
    /// False on an invalid partition layout or a residual outside int32.
    /// </summary>
    private static bool Residual(ArrayView<byte> d, ref long pos, long endBit, int n, int order,
        ArrayView<int> buf, ArrayView<long> wide, long chBase, long wideBase, bool isWide)
    {
        int method = (int)Read24(d, ref pos, 2);
        if (method > 1) return false;
        int paramBits = method == 0 ? 4 : 5;
        int escape = method == 0 ? 15 : 31;
        int porder = (int)Read24(d, ref pos, 4);
        int parts = 1 << porder;
        if ((n & (parts - 1)) != 0) return false;
        int psize = n >> porder;
        if (psize < order) return false;
        int i = order;
        for (int p = 0; p < parts; p++)
        {
            int cnt = p == 0 ? psize - order : psize;
            int param = (int)Read24(d, ref pos, paramBits);
            if (param == escape)
            {
                int raw = (int)Read24(d, ref pos, 5);
                for (int j = 0; j < cnt; j++)
                    Store(buf, wide, chBase, wideBase, isWide, i++, raw == 0 ? 0 : ReadSigned(d, ref pos, raw));
            }
            else
            {
                for (int j = 0; j < cnt; j++)
                {
                    int q = Unary(d, ref pos, endBit);
                    if (q < 0) return false;
                    long u = ((long)q << param) | Read(d, ref pos, param);
                    if (u > 0xFFFFFFFFL) return false;
                    long v = (u & 1) != 0 ? -(u >> 1) - 1 : u >> 1;
                    Store(buf, wide, chBase, wideBase, isWide, i++, v);
                }
            }
            if (pos > endBit) return false;
        }
        return true;
    }

    /// <summary>
    /// One thread per frame: parse the header and every subframe; the samples of channel c go to the channel-major region
    /// at frameSamples[f] * channels + c * blockSize (the side channel of a 32-bit stream to wide[frameSamples[f] + i]).
    /// A frame that fails (reserved subframe type, invalid residual) comes out as silence with a non-zero status.
    /// </summary>
    public static void DecodeFrameKernel(Index1D index, ArrayView<byte> data, ArrayView<int> frameOffsets, ArrayView<int> frameEnds, ArrayView<long> frameSamples,
        ArrayView<int> frameInfo, ArrayView<int> buf, ArrayView<long> wide, ArrayView<int> meta, int firstFrame)
    {
        int f = firstFrame + index;
        if (f >= meta[MFrames]) return;
        long at = frameOffsets[f];
        long sample0 = frameSamples[f];
        int channels = meta[MChannels];
        int hl = ParseHeader(data, at, data.Length, channels, meta[MBps], meta[MRate], out int n, out int assign, out int bps, out _, out _);
        frameInfo[f * FrameInts + FBlockSize] = n;
        frameInfo[f * FrameInts + FAssign] = assign;
        long endBit = (frameEnds[f] - 2L) * 8;   // the subframes end before the CRC-16
        long pos = (at + hl) * 8;
        int side = SideChannel(assign);
        int status = 0;
        for (int c = 0; c < channels && status == 0; c++)
        {
            int sbps = bps + (c == side ? 1 : 0);
            bool isWide = sbps > 32;
            long chBase = sample0 * channels + (long)c * n;
            long wideBase = sample0;
            if (Read24(data, ref pos, 1) != 0) { status = 1; break; }
            int type = (int)Read24(data, ref pos, 6);
            int wasted = 0;
            if (Read24(data, ref pos, 1) != 0)
            {
                int u = Unary(data, ref pos, endBit);
                if (u < 0) { status = 2; break; }
                wasted = u + 1;
            }
            int ebps = sbps - wasted;
            if (ebps < 1) { status = 3; break; }
            if (type == 0)
            {
                long v = ReadSignedLong(data, ref pos, ebps);
                for (int i = 0; i < n; i++) Store(buf, wide, chBase, wideBase, isWide, i, v);
            }
            else if (type == 1)
            {
                for (int i = 0; i < n; i++) Store(buf, wide, chBase, wideBase, isWide, i, ReadSignedLong(data, ref pos, ebps));
            }
            else if (type >= 8 && type <= 12)
            {
                int order = type - 8;
                if (order > n) { status = 4; break; }
                for (int i = 0; i < order; i++) Store(buf, wide, chBase, wideBase, isWide, i, ReadSignedLong(data, ref pos, ebps));
                if (!Residual(data, ref pos, endBit, n, order, buf, wide, chBase, wideBase, isWide)) { status = 5; break; }
                for (int i = order; i < n; i++)
                {
                    long r = Load(buf, wide, chBase, wideBase, isWide, i);
                    long p = 0;
                    if (order == 1) p = Load(buf, wide, chBase, wideBase, isWide, i - 1);
                    else if (order == 2) p = 2 * Load(buf, wide, chBase, wideBase, isWide, i - 1) - Load(buf, wide, chBase, wideBase, isWide, i - 2);
                    else if (order == 3)
                        p = 3 * Load(buf, wide, chBase, wideBase, isWide, i - 1) - 3 * Load(buf, wide, chBase, wideBase, isWide, i - 2)
                            + Load(buf, wide, chBase, wideBase, isWide, i - 3);
                    else if (order == 4)
                        p = 4 * Load(buf, wide, chBase, wideBase, isWide, i - 1) - 6 * Load(buf, wide, chBase, wideBase, isWide, i - 2)
                            + 4 * Load(buf, wide, chBase, wideBase, isWide, i - 3) - Load(buf, wide, chBase, wideBase, isWide, i - 4);
                    Store(buf, wide, chBase, wideBase, isWide, i, r + p);
                }
            }
            else if (type >= 32)
            {
                int order = type - 31;
                if (order > n) { status = 4; break; }
                for (int i = 0; i < order; i++) Store(buf, wide, chBase, wideBase, isWide, i, ReadSignedLong(data, ref pos, ebps));
                int precision = (int)Read24(data, ref pos, 4) + 1;
                if (precision == 16) { status = 6; break; }
                int shift = ReadSigned(data, ref pos, 5);
                if (shift < 0) { status = 7; break; }
                // the coefficients: kept in the frame's region of the int view is not possible (it holds samples), so
                // they are read again from the bitstream for each sample - their bit position is known
                long coefPos = pos;
                pos += (long)order * precision;
                if (!Residual(data, ref pos, endBit, n, order, buf, wide, chBase, wideBase, isWide)) { status = 5; break; }
                for (int i = order; i < n; i++)
                {
                    long cp = coefPos;
                    long sum = 0;
                    for (int j = 0; j < order; j++)
                        sum += (long)ReadSigned(data, ref cp, precision) * Load(buf, wide, chBase, wideBase, isWide, i - 1 - j);
                    Store(buf, wide, chBase, wideBase, isWide, i, Load(buf, wide, chBase, wideBase, isWide, i) + (sum >> shift));
                }
            }
            else { status = 8; break; }
            if (wasted > 0)
                for (int i = 0; i < n; i++)
                    Store(buf, wide, chBase, wideBase, isWide, i, Load(buf, wide, chBase, wideBase, isWide, i) << wasted);
            if (pos > endBit) status = 9;
        }
        if (status != 0)
        {
            for (long i = 0; i < (long)n * channels; i++) buf[sample0 * channels + i] = 0;
            if (side >= 0 && bps == 32) for (int i = 0; i < n; i++) wide[sample0 + i] = 0;
            Atomic.Add(ref meta[MBadFrames], 1);
        }
        frameInfo[f * FrameInts + FStatus] = status;
    }

    // ------------------------------------------------------------------ 6. decorrelation + interleave

    /// <summary>One thread per sample (per channel set): undo the stereo decorrelation, write interleaved int32.</summary>
    public static void InterleaveKernel(Index1D index, ArrayView<long> frameSamples, ArrayView<int> frameInfo, ArrayView<int> buf,
        ArrayView<long> wide, ArrayView<int> output, ArrayView<int> meta, long firstSample)
    {
        long s = firstSample + index;
        int frames = meta[MFrames];
        int channels = meta[MChannels];
        if (frames == 0 || s >= frameSamples[frames] || s < frameSamples[0])
        {
            if (s < output.Length / channels) for (int c = 0; c < channels; c++) output[s * channels + c] = 0;
            return;
        }
        // the frame holding sample s
        int lo = 0, hi = frames - 1;
        while (lo < hi)
        {
            int mid = (lo + hi + 1) >> 1;
            if (frameSamples[mid] <= s) lo = mid; else hi = mid - 1;
        }
        long s0 = frameSamples[lo];
        int n = frameInfo[lo * FrameInts + FBlockSize];
        int assign = frameInfo[lo * FrameInts + FAssign];
        int i = (int)(s - s0);
        if (i >= n)
        {
            for (int c = 0; c < channels; c++) output[s * channels + c] = 0;   // a gap left by a lost frame
            return;
        }
        long regionBase = s0 * channels;
        if (assign < 8)
        {
            for (int c = 0; c < channels; c++) output[s * channels + c] = buf[regionBase + (long)c * n + i];
            return;
        }
        bool wideSide = meta[MBps] == 32;
        long a = buf[regionBase + i], b = buf[regionBase + n + i];
        if (wideSide)
        {
            if (assign == 9) a = wide[s0 + i];
            else b = wide[s0 + i];
        }
        long left, right;
        if (assign == 8) { left = a; right = a - b; }
        else if (assign == 9) { right = b; left = a + b; }
        else
        {
            long mid = (a << 1) | (b & 1);
            left = (mid + b) >> 1;
            right = (mid - b) >> 1;
        }
        output[s * 2] = (int)left;
        output[s * 2 + 1] = (int)right;
    }
}
