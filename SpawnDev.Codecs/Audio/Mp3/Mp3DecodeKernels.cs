// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt). See NOTICE.md for upstream attributions.
//
// The MP3 (MPEG-1 / 2 / 2.5 Layer III) decoder's kernels (port of minimp3, CC0 / public domain, github.com/lieff/minimp3).
// Every function that touches codec data is minimp3's, line for line: header tests, frame sync (mp3d_find_frame /
// mp3d_match_frame, free format included), the Xing / Info tag (minimp3_ex's mp3dec_check_vbrtag), side info
// (L3_read_side_info), scalefactors (L3_decode_scalefactors, L3_read_scalefactors, L3_ldexp_q2), Huffman + requantize
// (L3_huffman, L3_pow_43), stereo (L3_intensity_stereo, L3_stereo_process, L3_midside_stereo), short-block reorder,
// alias reduction, IMDCT (L3_imdct36, L3_dct3_9, L3_imdct_short, L3_imdct12), sign change, synthesis (mp3d_DCT_II,
// mp3d_synth, mp3d_synth_pair) and MINIMP3_FLOAT_OUTPUT's scale. Where minimp3 has an SSE path and a scalar path that
// round differently (mp3d_DCT_II's output sums), this follows the SSE path, which x64 builds of minimp3 always take.
//
// minimp3 decodes frame after frame on one thread; here the stages run as kernels and the sequential state is split:
//   Walk          one thread (resumable, bounded steps per launch): sync, the Info frame, every frame's offset / size and
//                 where its main data lands in one contiguous main-data stream (frame sizes from the header fields).
//   SideInfo      one thread per frame: header + side info.
//   Reservoir     one thread (bounded frames per launch): main_data_begin becomes a plain offset into the stream;
//                 minimp3's reservoir rules (a frame whose bytes are missing is not decoded) as offsets.
//   Gather        one thread per frame: its main data into the stream.
//   Granule       one thread per (frame, granule, channel): scalefactors (granule 1 re-reads granule 0's for scfsi),
//                 Huffman, requantize into 576 lines.
//   Stereo        one thread per (frame, granule): M/S, intensity.
//   ImdctPrep     one thread per (granule, channel, subband): reorder / alias reduction (the 18 lines), and the
//                 subband's overlap for the next granule (it depends only on this granule's lines).
//   ImdctOut      one thread per (granule, channel, subband): the 18 time samples from the lines and the previous
//                 granule's overlap, sign change of odd samples in odd subbands.
//   Dct           one thread per (time slot, channel): mp3d_DCT_II of the slot's 32 subband samples.
//   Synth         one thread per output sample: minimp3's 512-tap window over the last 16 slots' DCT outputs, read
//                 through minimp3's lins layout (a row per slot, see V), then the 1/32768 scale and the gapless trim.
// Each output value comes from the same operations in the same order as minimp3's, so a clean stream decodes bit for
// bit as minimp3 x64 (float output) decodes it. The parallel split changes no arithmetic; it changes only what happens
// on streams minimp3 itself handles by resetting (see the notes at Walk and Reservoir): a damaged or missing frame
// comes out as a frame of zero lines (the previous frame's overlap fades out, then silence) and keeps its time.
// Other differences, all on malformed or unusual input: a frame whose next header is damaged is still decoded
// (minimp3 drops it); MPEG-2.5 8 kHz mixed blocks, whose reorder runs past 576 lines in minimp3, stop at 576; a
// granule-0 scfsi (from nonzero private bits) copies zeros where minimp3 copies stale scratch.

using ILGPU;
using T = SpawnDev.Codecs.Audio.Mp3.Mp3DecodeTables;

namespace SpawnDev.Codecs.Audio.Mp3;

/// <summary>Kernels of the MP3 decoder (port of minimp3, CC0; see the file header).</summary>
public static class Mp3DecodeKernels
{
    /// <summary>The meta view (ints): walker state, stream facts, counts.</summary>
    public const int MLen = 0, MState = 1, MPos = 2, MRef = 3, MFree = 4, MFrames = 5, MCap = 6, MStream = 7, MLastBytes = 8,
        MGapStart = 9, MXing = 10, MXFrames = 11, MXDelay = 12, MXPadding = 13, MChannels = 14, MRate = 15, MGranules = 16,
        MLayer = 17, MLost = 18, MConsumed = 19, MBreak = 20, MSrIdx = 21, MUndecoded = 22, MetaSize = 23;

    /// <summary>Walker states.</summary>
    public const int StateStart = 0, StateWalk = 1, StateResync = 2, StateDone = 3;

    /// <summary>Per-frame side ints: status, main_data_begin, header, main data bytes, window start (stream byte where
    /// the frame's main data window begins), decode flag, stream end of the frame's own main data.</summary>
    public const int FStatus = 0, FMainBegin = 1, FHeader = 2, FMainBytes = 3, FWindow = 4, FDecode = 5, FStreamEnd = 6, FrameHead = 8;

    /// <summary>Frame status: decodable side info / lost in a resync gap / side info rejected / channel mode differs.</summary>
    public const int StatusOk = 0, StatusLost = 1, StatusBadSide = 2, StatusMismatch = 3;

    /// <summary>Per-(granule, channel) side ints (minimp3's L3_gr_info_t) and the unit's first bit in the stream.</summary>
    public const int UPart23 = 0, UBigValues = 1, UGlobalGain = 2, UScfCompress = 3, UBlockType = 4, UMixed = 5, UNLong = 6,
        UNShort = 7, USfbTab = 8, UTable0 = 9, URegion0 = 12, USubblockGain0 = 15, UPreflag = 18, UScfScale = 19, UCount1Table = 20,
        UScfsi = 21, UBitStart = 22, UnitInts = 24;

    /// <summary>Ints per frame in the side view.</summary>
    public const int SideInts = FrameHead + 4 * UnitInts;

    /// <summary>Ints per (granule, channel) in the intensity-position view.</summary>
    public const int IstInts = 40;

    /// <summary>Synthesis history slots kept ahead of a batch's slots in the DCT view.</summary>
    public const int HistorySlots = 16;

    // ------------------------------------------------------------------ header (minimp3's HDR_* macros, hdr_* functions)

    private static int B1(uint h) => (int)((h >> 16) & 0xFF);
    private static int B2(uint h) => (int)((h >> 8) & 0xFF);
    private static int B3(uint h) => (int)(h & 0xFF);
    private static bool IsMono(uint h) => (B3(h) & 0xC0) == 0xC0;
    private static bool IsMsStereo(uint h) => (B3(h) & 0xE0) == 0x60;
    private static bool IsFree(uint h) => (B2(h) & 0xF0) == 0;
    private static bool IsCrc(uint h) => (B1(h) & 1) == 0;
    private static bool TestPadding(uint h) => (B2(h) & 0x2) != 0;
    private static bool TestMpeg1(uint h) => (B1(h) & 0x8) != 0;
    private static bool TestNotMpeg25(uint h) => (B1(h) & 0x10) != 0;
    private static bool TestIStereo(uint h) => (B3(h) & 0x10) != 0;
    private static bool TestMsStereo(uint h) => (B3(h) & 0x20) != 0;
    private static int GetLayer(uint h) => (B1(h) >> 1) & 3;
    private static int GetBitrate(uint h) => B2(h) >> 4;
    private static int GetSampleRate(uint h) => (B2(h) >> 2) & 3;
    /// <summary>HDR_GET_MY_SAMPLE_RATE: 0..8 over MPEG-2.5, MPEG-2, MPEG-1.</summary>
    public static int MySampleRate(uint h) => GetSampleRate(h) + (((B1(h) >> 3) & 1) + ((B1(h) >> 4) & 1)) * 3;
    private static bool IsFrame576(uint h) => (B1(h) & 14) == 2;
    private static bool IsLayer1(uint h) => (B1(h) & 6) == 6;

    /// <summary>The 4 header bytes at <paramref name="at"/> as one int (big-endian), or 0 past the end.</summary>
    private static uint HdrAt(ArrayView<byte> d, long at, long len)
    {
        if (at < 0 || at + 4 > len) return 0;
        return ((uint)d[at] << 24) | ((uint)d[at + 1] << 16) | ((uint)d[at + 2] << 8) | d[at + 3];
    }

    private static bool HdrValid(uint h) =>
        (h >> 24) == 0xFF && ((B1(h) & 0xF0) == 0xF0 || (B1(h) & 0xFE) == 0xE2) && GetLayer(h) != 0 && GetBitrate(h) != 15
        && GetSampleRate(h) != 3;

    private static bool HdrCompare(uint h1, uint h2) =>
        HdrValid(h2) && ((B1(h1) ^ B1(h2)) & 0xFE) == 0 && ((B2(h1) ^ B2(h2)) & 0x0C) == 0 && IsFree(h1) == IsFree(h2);

    private static int BitrateKbps(uint h, ArrayView<int> ints) =>
        2 * ints[T.IHalfRate + (TestMpeg1(h) ? 45 : 0) + (GetLayer(h) - 1) * 15 + GetBitrate(h)];

    /// <summary>The sample rate in Hz.</summary>
    public static int SampleRateHz(uint h)
    {
        int sr = GetSampleRate(h);
        int hz = sr == 0 ? 44100 : sr == 1 ? 48000 : 32000;
        return hz >> (TestMpeg1(h) ? 0 : 1) >> (TestNotMpeg25(h) ? 0 : 1);
    }

    private static int FrameSamples(uint h) => IsLayer1(h) ? 384 : (1152 >> (IsFrame576(h) ? 1 : 0));

    private static int FrameBytes(uint h, int freeFormatSize, ArrayView<int> ints)
    {
        int frameBytes = FrameSamples(h) * BitrateKbps(h, ints) * 125 / SampleRateHz(h);
        if (IsLayer1(h)) frameBytes &= ~3;
        return frameBytes != 0 ? frameBytes : freeFormatSize;
    }

    private static int Padding(uint h) => TestPadding(h) ? (IsLayer1(h) ? 4 : 1) : 0;

    /// <summary>Side info bytes after the header (and CRC) of a Layer III frame.</summary>
    private static int SideBytes(uint h) => TestMpeg1(h) ? (IsMono(h) ? 17 : 32) : (IsMono(h) ? 9 : 17);

    /// <summary>Main data bytes of a Layer III frame of <paramref name="frameBytes"/> bytes (0 when it has none).</summary>
    private static int MainBytes(uint h, int frameBytes)
    {
        int n = frameBytes - 4 - (IsCrc(h) ? 2 : 0) - SideBytes(h);
        return n < 0 ? 0 : n;
    }

    // ------------------------------------------------------------------ bit reading

    private static int ByteAt(ArrayView<byte> d, long at) => at >= 0 && at < d.Length ? d[at] : 0;

    /// <summary>minimp3's get_bits: n bits at bit <paramref name="pos"/> of the bytes from <paramref name="baseAt"/>;
    /// 0 when the read would pass <paramref name="limit"/> (the position still advances).</summary>
    private static int GetBits(ArrayView<byte> d, long baseAt, ref int pos, int limit, int n)
    {
        int s = pos & 7;
        int shl = n + s;
        long p = baseAt + (pos >> 3);
        if ((pos += n) > limit) return 0;
        uint next = (uint)(ByteAt(d, p++) & (255 >> s));
        uint cache = 0;
        while ((shl -= 8) > 0)
        {
            cache |= next << shl;
            next = (uint)ByteAt(d, p++);
        }
        return (int)(cache | (next >> -shl));
    }

    // ------------------------------------------------------------------ frame sync (mp3d_match_frame / mp3d_find_frame)

    private static bool MatchFrame(ArrayView<byte> d, long at, long len, int frameBytes, ArrayView<int> ints)
    {
        long bytes = len - at;
        uint h0 = HdrAt(d, at, len);
        long i = 0;
        for (int nmatch = 0; nmatch < 10; nmatch++)
        {
            uint hi = HdrAt(d, at + i, len);
            i += FrameBytes(hi, frameBytes, ints) + Padding(hi);
            if (i + 4 > bytes) return nmatch > 0;
            if (!HdrCompare(h0, HdrAt(d, at + i, len))) return false;
        }
        return true;
    }

    /// <summary>
    /// mp3d_find_frame's test of one position <paramref name="i"/> of a buffer that starts at <paramref name="start"/>:
    /// a valid header whose frame chain matches (10 frames, or to the end), or a single frame filling the buffer. Returns
    /// the frame's bytes (padding included) or 0; free-format frames find their size here (<paramref name="free"/>).
    /// </summary>
    private static int FindFrameAt(ArrayView<byte> d, long i, long start, long len, ref int free, ArrayView<int> ints)
    {
        uint h = HdrAt(d, i, len);
        if (!HdrValid(h)) return 0;
        long bytes = len - start, rel = i - start;
        int frameBytes = FrameBytes(h, free, ints);
        int frameAndPadding = frameBytes + Padding(h);
        for (int k = 4; frameBytes == 0 && k < 2304 && rel + 2 * k < bytes - 4; k++)
        {
            if (HdrCompare(h, HdrAt(d, i + k, len)))
            {
                int fb = k - Padding(h);
                int nextfb = fb + Padding(HdrAt(d, i + k, len));
                if (rel + k + nextfb + 4 > bytes || !HdrCompare(h, HdrAt(d, i + k + nextfb, len))) continue;
                frameAndPadding = k;
                frameBytes = fb;
                free = fb;
            }
        }
        if ((frameBytes != 0 && rel + frameAndPadding <= bytes && MatchFrame(d, i, len, frameBytes, ints))
            || (rel == 0 && frameAndPadding == bytes))
            return frameAndPadding;
        free = 0;
        return 0;
    }

    // ------------------------------------------------------------------ side info (L3_read_side_info)

    /// <summary>
    /// Layer III side info of the frame at <paramref name="frameAt"/> into side[sideAt ..] (units gr * nch + ch);
    /// returns main_data_begin or -1 (minimp3's rejections), and the bit position after the side info.
    /// </summary>
    private static int ReadSideInfo(ArrayView<byte> d, long frameAt, int frameBytes, uint hdr, ArrayView<int> side, long sideAt,
        ArrayView<int> ints, out int bitPos, out int bitLimit)
    {
        long baseAt = frameAt + 4;
        int pos = 0, limit = (frameBytes - 4) * 8;
        bitLimit = limit;
        if (IsCrc(hdr)) GetBits(d, baseAt, ref pos, limit, 16);
        uint scfsi = 0;
        int mainDataBegin, part23Sum = 0;
        int srIdx = MySampleRate(hdr);
        srIdx -= srIdx != 0 ? 1 : 0;
        int grCount = IsMono(hdr) ? 1 : 2;
        bool mpeg1 = TestMpeg1(hdr);
        if (mpeg1)
        {
            grCount *= 2;
            mainDataBegin = GetBits(d, baseAt, ref pos, limit, 9);
            scfsi = (uint)GetBits(d, baseAt, ref pos, limit, 7 + grCount);
        }
        else
        {
            mainDataBegin = GetBits(d, baseAt, ref pos, limit, 8 + grCount) >> grCount;
        }
        for (int u = 0; u < grCount; u++)
        {
            long at = sideAt + FrameHead + u * UnitInts;
            if (IsMono(hdr)) scfsi <<= 4;
            int part23 = GetBits(d, baseAt, ref pos, limit, 12);
            side[at + UPart23] = part23;
            part23Sum += part23;
            int bigValues = GetBits(d, baseAt, ref pos, limit, 9);
            side[at + UBigValues] = bigValues;
            if (bigValues > 288) { bitPos = pos; return -1; }
            side[at + UGlobalGain] = GetBits(d, baseAt, ref pos, limit, 8);
            int sfc = GetBits(d, baseAt, ref pos, limit, mpeg1 ? 4 : 9);
            side[at + UScfCompress] = sfc;
            int sfbTab = T.IScfLong + 23 * srIdx, nLong = 22, nShort = 0;
            int blockType, mixed, rc0, rc1, rc2 = 255, sbg0 = 0, sbg1 = 0, sbg2 = 0;
            uint tables;
            if (GetBits(d, baseAt, ref pos, limit, 1) != 0)
            {
                blockType = GetBits(d, baseAt, ref pos, limit, 2);
                if (blockType == 0) { bitPos = pos; return -1; }
                mixed = GetBits(d, baseAt, ref pos, limit, 1);
                rc0 = 7;
                rc1 = 255;
                if (blockType == 2)
                {
                    scfsi &= 0x0F0F;
                    if (mixed == 0)
                    {
                        rc0 = 8;
                        sfbTab = T.IScfShort + 40 * srIdx;
                        nLong = 0;
                        nShort = 39;
                    }
                    else
                    {
                        sfbTab = T.IScfMixed + 40 * srIdx;
                        nLong = mpeg1 ? 8 : 6;
                        nShort = 30;
                    }
                }
                tables = (uint)GetBits(d, baseAt, ref pos, limit, 10);
                tables <<= 5;
                sbg0 = GetBits(d, baseAt, ref pos, limit, 3);
                sbg1 = GetBits(d, baseAt, ref pos, limit, 3);
                sbg2 = GetBits(d, baseAt, ref pos, limit, 3);
            }
            else
            {
                blockType = 0;
                mixed = 0;
                tables = (uint)GetBits(d, baseAt, ref pos, limit, 15);
                rc0 = GetBits(d, baseAt, ref pos, limit, 4);
                rc1 = GetBits(d, baseAt, ref pos, limit, 3);
                rc2 = 255;
            }
            side[at + UBlockType] = blockType;
            side[at + UMixed] = mixed;
            side[at + UNLong] = nLong;
            side[at + UNShort] = nShort;
            side[at + USfbTab] = sfbTab;
            side[at + UTable0] = (int)(tables >> 10);
            side[at + UTable0 + 1] = (int)((tables >> 5) & 31);
            side[at + UTable0 + 2] = (int)(tables & 31);
            side[at + URegion0] = rc0;
            side[at + URegion0 + 1] = rc1;
            side[at + URegion0 + 2] = rc2;
            side[at + USubblockGain0] = sbg0;
            side[at + USubblockGain0 + 1] = sbg1;
            side[at + USubblockGain0 + 2] = sbg2;
            side[at + UPreflag] = mpeg1 ? GetBits(d, baseAt, ref pos, limit, 1) : (sfc >= 500 ? 1 : 0);
            side[at + UScfScale] = GetBits(d, baseAt, ref pos, limit, 1);
            side[at + UCount1Table] = GetBits(d, baseAt, ref pos, limit, 1);
            side[at + UScfsi] = (int)((scfsi >> 12) & 15);
            scfsi <<= 4;
        }
        bitPos = pos;
        if (part23Sum + pos > limit + mainDataBegin * 8) return -1;
        return mainDataBegin;
    }

    // ------------------------------------------------------------------ Xing / Info (minimp3_ex's mp3dec_check_vbrtag)

    /// <summary>1: a Xing / Info tag with a frame count (frames, delay + 529, padding - 529 into meta); -1: a tag
    /// without a frame count (the frame is still not audio); 0: an audio frame.</summary>
    private static int CheckVbrTag(ArrayView<byte> d, long at, int frameSize, uint hdr, ArrayView<int> scratch, ArrayView<int> meta,
        ArrayView<int> ints)
    {
        if (frameSize < 4 + 8) return 0;
        if (ReadSideInfo(d, at, frameSize, hdr, scratch, 0, ints, out int bitPos, out _) < 0) return 0;
        long tagOff = 4 + bitPos / 8;
        if (tagOff > frameSize || frameSize - tagOff < 8) return 0;
        long t = at + tagOff;
        bool xing = d[t] == 'X' && d[t + 1] == 'i' && d[t + 2] == 'n' && d[t + 3] == 'g';
        bool info = d[t] == 'I' && d[t + 1] == 'n' && d[t + 2] == 'f' && d[t + 3] == 'o';
        if (!xing && !info) return 0;
        int flags = d[t + 7];
        if ((flags & 1) == 0) return -1;
        tagOff += 8;
        if (frameSize - tagOff < 4) return 0;
        t = at + tagOff;
        meta[MXFrames] = (d[t] << 24) | (d[t + 1] << 16) | (d[t + 2] << 8) | d[t + 3];
        tagOff += 4;
        if ((flags & 2) != 0)
        {
            if (frameSize - tagOff < 4) return 0;
            tagOff += 4;
        }
        if ((flags & 4) != 0)
        {
            if (frameSize - tagOff < 100) return 0;
            tagOff += 100;
        }
        if ((flags & 8) != 0)
        {
            if (frameSize - tagOff < 4) return 0;
            tagOff += 4;
        }
        meta[MXDelay] = 0;
        meta[MXPadding] = 0;
        if (frameSize - tagOff < 1) return 0;
        t = at + tagOff;
        if (d[t] != 0)
        {
            // extension (LAME, Lavc, ...): the same structure
            if (frameSize - tagOff <= 35) return 0;
            t += 21;
            meta[MXDelay] = ((d[t] << 4) | (d[t + 1] >> 4)) + (528 + 1);
            meta[MXPadding] = (((d[t + 1] & 0xF) << 8) | d[t + 2]) - (528 + 1);
        }
        return 1;
    }

    // ------------------------------------------------------------------ 1. Walk

    /// <summary>
    /// One thread, at most <paramref name="budget"/> steps per launch (state in meta): find the first frame as minimp3
    /// does (mp3d_find_frame), skip it when it is a Xing / Info frame (recording its frame count, delay and padding),
    /// then follow the frames by their header sizes. Each frame gets its offset, size and where its main data starts
    /// in the main-data stream. A frame is taken when its header matches the stream's (version, layer, rate, free
    /// format) and it fits in the file; a truncated last frame is dropped (as minimp3 drops it). At a damaged header
    /// the walk resyncs with mp3d_find_frame (a matching header followed by matching frames) and records the frames the
    /// skipped bytes held (bytes / last frame size, rounded) as lost: they come out as silence and keep their time.
    /// </summary>
    public static void WalkKernel(Index1D index, ArrayView<byte> d, ArrayView<int> fOff, ArrayView<int> fBytes, ArrayView<int> fMain,
        ArrayView<int> meta, ArrayView<int> ints, ArrayView<int> scratch, int budget)
    {
        int state = meta[MState];
        if (state == StateDone) return;
        long len = meta[MLen];
        int pos = meta[MPos], free = meta[MFree], frames = meta[MFrames], cap = meta[MCap], stream = meta[MStream];
        int lastBytes = meta[MLastBytes], gapStart = meta[MGapStart];
        uint refHdr = (uint)meta[MRef];
        int steps = 0;
        while (steps < budget && state != StateDone)
        {
            steps++;
            if (state == StateStart || state == StateResync)
            {
                if (pos >= len - 4) { state = StateDone; break; }
                if (state == StateResync && !HdrCompare(refHdr, HdrAt(d, pos, len))) { pos++; continue; }
                int fap = FindFrameAt(d, pos, state == StateStart ? 0 : gapStart, len, ref free, ints);
                if (fap == 0) { pos++; continue; }
                if (state == StateStart)
                {
                    refHdr = HdrAt(d, pos, len);
                    meta[MLayer] = 4 - GetLayer(refHdr);
                    meta[MChannels] = IsMono(refHdr) ? 1 : 2;
                    meta[MRate] = SampleRateHz(refHdr);
                    meta[MGranules] = TestMpeg1(refHdr) ? 2 : 1;
                    meta[MSrIdx] = MySampleRate(refHdr);
                    if (GetLayer(refHdr) != 1) { state = StateDone; break; }   // Layer I / II: not decoded here
                    int tag = CheckVbrTag(d, pos, fap, refHdr, scratch, meta, ints);
                    meta[MXing] = tag;
                    if (tag != 0) pos += fap;
                }
                else
                {
                    int gap = pos - gapStart;
                    int lost = lastBytes > 0 ? (gap + lastBytes / 2) / lastBytes : 0;
                    for (int k = 0; k < lost && frames < cap; k++)
                    {
                        fOff[frames] = -1;
                        fBytes[frames] = 0;
                        fMain[frames] = stream;
                        frames++;
                        meta[MLost] = meta[MLost] + 1;
                    }
                }
                state = StateWalk;
                continue;
            }
            // walking
            uint h = HdrAt(d, pos, len);
            if (!HdrCompare(refHdr, h))
            {
                if (pos + 4 > len) { state = StateDone; break; }
                gapStart = pos;
                free = IsFree(refHdr) ? 0 : free;
                state = StateResync;
                continue;
            }
            int size = FrameBytes(h, free, ints) + Padding(h);
            if (size <= 0 || pos + size > len || frames >= cap) { state = StateDone; break; }
            fOff[frames] = pos;
            fBytes[frames] = size;
            fMain[frames] = stream;
            stream += MainBytes(h, size);
            frames++;
            lastBytes = size;
            pos += size;
        }
        meta[MState] = state;
        meta[MPos] = pos;
        meta[MRef] = (int)refHdr;
        meta[MFree] = free;
        meta[MFrames] = frames;
        meta[MStream] = stream;
        meta[MLastBytes] = lastBytes;
        meta[MGapStart] = gapStart;
    }

    // ------------------------------------------------------------------ 2. SideInfo

    /// <summary>One thread per frame: header and side info into side[f * SideInts ..] (see the F* / U* layouts).</summary>
    public static void SideInfoKernel(Index1D index, ArrayView<byte> d, ArrayView<int> fOff, ArrayView<int> fBytes, ArrayView<int> side,
        ArrayView<int> meta, ArrayView<int> ints, int first)
    {
        int f = first + index;
        long at = (long)f * SideInts;
        int off = fOff[f];
        side[at + FDecode] = 0;
        side[at + FWindow] = 0;
        if (off < 0)
        {
            side[at + FStatus] = StatusLost;
            side[at + FMainBytes] = 0;
            side[at + FHeader] = 0;
            return;
        }
        long len = meta[MLen];
        uint h = HdrAt(d, off, len);
        int size = fBytes[f];
        side[at + FHeader] = (int)h;
        side[at + FMainBytes] = MainBytes(h, size);
        if (IsMono(h) != (meta[MChannels] == 1))
        {
            side[at + FStatus] = StatusMismatch;
            return;
        }
        int mdb = ReadSideInfo(d, off, size, h, side, at, ints, out int bitPos, out int bitLimit);
        side[at + FMainBegin] = mdb;
        side[at + FStatus] = mdb < 0 || bitPos > bitLimit ? StatusBadSide : StatusOk;
    }

    // ------------------------------------------------------------------ 3. Reservoir

    /// <summary>
    /// One thread over frames [first, first + count) in order (state in meta): minimp3's bit reservoir as stream offsets.
    /// A frame's main data window starts main_data_begin bytes before its own main data; minimp3 keeps only the bytes
    /// after what the previous frame consumed (at most 511) and none across a reset (a lost frame or rejected side info
    /// reset minimp3's decoder), so a frame whose window reaches before either is not decoded (zero lines), as minimp3
    /// skips it. Decoded frames get their window start and each granule-channel's first bit (part2_3 lengths in order).
    /// </summary>
    public static void ReservoirKernel(Index1D index, ArrayView<int> fMain, ArrayView<int> side, ArrayView<int> meta, int first, int count)
    {
        int consumed = meta[MConsumed], brk = meta[MBreak], undecoded = meta[MUndecoded];
        int units = meta[MGranules] * meta[MChannels];
        for (int f = first; f < first + count; f++)
        {
            long at = (long)f * SideInts;
            int status = side[at + FStatus];
            int s = fMain[f], mb = side[at + FMainBytes];
            side[at + FStreamEnd] = s + mb;
            side[at + FDecode] = 0;
            if (status == StatusLost || status == StatusBadSide)
            {
                brk = s + (status == StatusBadSide ? mb : 0);
                if (consumed < brk) consumed = brk;
                undecoded++;
                continue;
            }
            if (status == StatusOk)
            {
                int w = s - side[at + FMainBegin];
                if (w >= brk && w >= consumed)
                {
                    side[at + FDecode] = 1;
                    side[at + FWindow] = w;
                    int bit = w * 8;
                    for (int u = 0; u < units; u++)
                    {
                        long ua = at + FrameHead + u * UnitInts;
                        side[ua + UBitStart] = bit;
                        bit += side[ua + UPart23];
                    }
                    consumed = w + (bit - w * 8 + 7) / 8;
                }
                else
                {
                    if (consumed < brk) consumed = brk;
                    undecoded++;
                }
            }
            else undecoded++;
            if (consumed < s + mb - 511) consumed = s + mb - 511;
        }
        meta[MConsumed] = consumed;
        meta[MBreak] = brk;
        meta[MUndecoded] = undecoded;
    }

    // ------------------------------------------------------------------ 4. Gather

    /// <summary>One thread per frame: its main data bytes (after header, CRC, side info) into the main-data stream.</summary>
    public static void GatherKernel(Index1D index, ArrayView<byte> d, ArrayView<int> fOff, ArrayView<int> fBytes, ArrayView<int> fMain,
        ArrayView<byte> stream, ArrayView<int> meta, int first)
    {
        int f = first + index;
        int off = fOff[f];
        if (off < 0) return;
        uint h = HdrAt(d, off, meta[MLen]);
        int mb = MainBytes(h, fBytes[f]);
        long src = off + 4 + (IsCrc(h) ? 2 : 0) + SideBytes(h);
        long dst = fMain[f];
        for (int i = 0; i < mb; i++) stream[dst + i] = d[src + i];
    }

    // ------------------------------------------------------------------ 5. Granule: scalefactors + Huffman

    /// <summary>L3_ldexp_q2: y * 2^(exp_q2 / 4) in steps of at most 2^30.</summary>
    private static float LdexpQ2(float y, int expQ2, ArrayView<float> floats)
    {
        int e;
        do
        {
            e = expQ2 < 30 * 4 ? expQ2 : 30 * 4;
            y *= floats[T.FExpFrac + (e & 3)] * (1 << 30 >> (e >> 2));
        } while ((expQ2 -= e) > 0);
        return y;
    }

    /// <summary>L3_read_scalefactors (values 0..255 as minimp3's uint8_t; an intensity position's "illegal" -1 is 255).</summary>
    private static void ReadScalefactors(ArrayView<int> scf, ArrayView<int> istPos, ArrayView<int> scfSize, int scfCount,
        ArrayView<byte> s, ref int pos, int limit, int scfsi, ArrayView<int> ints)
    {
        int at = 0;
        for (int i = 0; i < 4 && ints[scfCount + i] != 0; i++, scfsi *= 2)
        {
            int cnt = ints[scfCount + i];
            if ((scfsi & 8) != 0)
            {
                for (int k = 0; k < cnt; k++) scf[at + k] = istPos[at + k];
            }
            else
            {
                int bits = scfSize[i];
                if (bits == 0)
                {
                    for (int k = 0; k < cnt; k++)
                    {
                        scf[at + k] = 0;
                        istPos[at + k] = 0;
                    }
                }
                else
                {
                    int maxScf = scfsi < 0 ? (1 << bits) - 1 : -1;
                    for (int k = 0; k < cnt; k++)
                    {
                        int v = GetBits(s, 0, ref pos, limit, bits);
                        istPos[at + k] = v == maxScf ? 255 : v;
                        scf[at + k] = v;
                    }
                }
            }
            at += cnt;
        }
        scf[at] = 0;
        scf[at + 1] = 0;
        scf[at + 2] = 0;
    }

    /// <summary>L3_decode_scalefactors: the scalefactors of unit <paramref name="ua"/> (iscf, istPos) and, unless
    /// <paramref name="readOnly"/>, the dequantizer gain of each band (scf).</summary>
    private static void DecodeScalefactors(uint hdr, ArrayView<int> istPos, ArrayView<byte> s, ref int pos, int limit,
        ArrayView<int> side, long ua, ArrayView<float> scf, int ch, ArrayView<int> ints, ArrayView<float> floats,
        ArrayView<int> iscf, ArrayView<int> scfSize, bool readOnly)
    {
        int nLong = side[ua + UNLong], nShort = side[ua + UNShort];
        int scfPartition = T.IScfPartitions + 28 * ((nShort != 0 ? 1 : 0) + (nLong == 0 ? 1 : 0));
        int scfShift = side[ua + UScfScale] + 1, scfsi = side[ua + UScfsi];
        if (TestMpeg1(hdr))
        {
            int part = ints[T.IScfcDecode + side[ua + UScfCompress]];
            scfSize[1] = scfSize[0] = part >> 2;
            scfSize[3] = scfSize[2] = part & 3;
        }
        else
        {
            int ist = TestIStereo(hdr) && ch != 0 ? 1 : 0;
            int sfc = side[ua + UScfCompress] >> ist;
            int k, modprod;
            for (k = ist * 3 * 4; sfc >= 0; sfc -= modprod, k += 4)
            {
                modprod = 1;
                for (int i = 3; i >= 0; i--)
                {
                    scfSize[i] = sfc / modprod % ints[T.IMod + k + i];
                    modprod *= ints[T.IMod + k + i];
                }
            }
            scfPartition += k;
            scfsi = -16;
        }
        ReadScalefactors(iscf, istPos, scfSize, scfPartition, s, ref pos, limit, scfsi, ints);
        if (readOnly) return;
        if (nShort != 0)
        {
            int sh = 3 - scfShift;
            for (int i = 0; i < nShort; i += 3)
            {
                iscf[nLong + i + 0] = (iscf[nLong + i + 0] + (side[ua + USubblockGain0] << sh)) & 0xFF;
                iscf[nLong + i + 1] = (iscf[nLong + i + 1] + (side[ua + USubblockGain0 + 1] << sh)) & 0xFF;
                iscf[nLong + i + 2] = (iscf[nLong + i + 2] + (side[ua + USubblockGain0 + 2] << sh)) & 0xFF;
            }
        }
        else if (side[ua + UPreflag] != 0)
        {
            for (int i = 0; i < 10; i++) iscf[11 + i] = (iscf[11 + i] + ints[T.IPreamp + i]) & 0xFF;
        }
        // BITS_DEQUANTIZER_OUT = -1, MAX_SCFI = 44
        int gainExp = side[ua + UGlobalGain] + (-1) * 4 - 210 - (IsMsStereo(hdr) ? 2 : 0);
        float gain = LdexpQ2(1 << (44 / 4), 44 - gainExp, floats);
        for (int i = 0; i < nLong + nShort; i++) scf[i] = LdexpQ2(gain, iscf[i] << scfShift, floats);
    }

    /// <summary>L3_pow_43: x^(4/3) from the table below 129, else a table value times a 2-term series.</summary>
    private static float Pow43(int x, ArrayView<float> floats)
    {
        float frac;
        int sign, mult = 256;
        if (x < 129) return floats[T.FPow43 + 16 + x];
        if (x < 1024)
        {
            mult = 16;
            x <<= 3;
        }
        sign = 2 * x & 64;
        frac = (float)((x & 63) - sign) / ((x & ~63) + sign);
        return floats[T.FPow43 + 16 + ((x + sign) >> 6)] * (1f + frac * ((4f / 3) + frac * (2f / 9))) * mult;
    }

    /// <summary>L3_huffman: big values (with linbits), count1 quads, each line times its band's gain, into
    /// dst[dstAt ..dstAt + 576). <paramref name="pos"/> and <paramref name="grLimit"/> are absolute stream bits.</summary>
    private static void Huffman(ArrayView<float> dst, long dstAt, ArrayView<byte> s, int pos, ArrayView<int> side, long ua,
        ArrayView<float> scf, int grLimit, ArrayView<int> ints, ArrayView<float> floats)
    {
        float one = 0f;
        int ireg = 0, bigValCnt = side[ua + UBigValues];
        int sfb = side[ua + USfbTab], sc = 0, di = 0;
        int next = pos >> 3;
        uint cache = ((((uint)ByteAt(s, next) * 256u + (uint)ByteAt(s, next + 1)) * 256u + (uint)ByteAt(s, next + 2)) * 256u
            + (uint)ByteAt(s, next + 3)) << (pos & 7);
        int sh = (pos & 7) - 8;
        next += 4;
        int np, pairs;

        while (bigValCnt > 0 && ireg < 3)
        {
            int tabNum = side[ua + UTable0 + ireg];
            int sfbCnt = side[ua + URegion0 + ireg];
            ireg++;
            int codebook = T.ITabs + ints[T.ITabIndex + tabNum];
            int linbits = ints[T.ILinbits + tabNum];
            do
            {
                np = ints[sfb++] / 2;
                if (np == 0) { bigValCnt = 0; break; }   // past the band table (not in valid streams: big_values <= 288)
                pairs = bigValCnt < np ? bigValCnt : np;
                one = scf[sc++];
                do
                {
                    int w = 5;
                    int leaf = ints[codebook + (int)(cache >> (32 - w))];
                    while (leaf < 0)
                    {
                        cache <<= w;
                        sh += w;
                        w = leaf & 7;
                        leaf = ints[codebook + (int)(cache >> (32 - w)) - (leaf >> 3)];
                    }
                    cache <<= leaf >> 8;
                    sh += leaf >> 8;
                    for (int j = 0; j < 2; j++, di++, leaf >>= 4)
                    {
                        int lsb = leaf & 0x0F;
                        float v;
                        if (linbits != 0 && lsb == 15)
                        {
                            lsb += (int)(cache >> (32 - linbits));
                            cache <<= linbits;
                            sh += linbits;
                            while (sh >= 0) { cache |= (uint)ByteAt(s, next++) << sh; sh -= 8; }
                            v = one * Pow43(lsb, floats) * ((int)cache < 0 ? -1 : 1);
                        }
                        else
                        {
                            v = floats[T.FPow43 + 16 + lsb - 16 * (int)(cache >> 31)] * one;
                        }
                        if (di < 576) dst[dstAt + di] = v;
                        if (lsb != 0)
                        {
                            cache <<= 1;
                            sh += 1;
                        }
                    }
                    while (sh >= 0) { cache |= (uint)ByteAt(s, next++) << sh; sh -= 8; }
                } while (--pairs > 0);
            } while ((bigValCnt -= np) > 0 && --sfbCnt >= 0);
        }

        int cb = side[ua + UCount1Table] != 0 ? T.ITab33 : T.ITab32;
        for (np = 1 - bigValCnt; ; di += 4)
        {
            int leaf = ints[cb + (int)(cache >> 28)];
            if ((leaf & 8) == 0) leaf = ints[cb + (leaf >> 3) + (int)(cache << 4 >> (32 - (leaf & 3)))];
            cache <<= leaf & 7;
            sh += leaf & 7;
            if (next * 8 - 24 + sh > grLimit) break;
            if (--np == 0)
            {
                np = ints[sfb++] / 2;
                if (np == 0) break;
                one = scf[sc++];
            }
            if ((leaf & 128) != 0)
            {
                if (di < 576) dst[dstAt + di] = (int)cache < 0 ? -one : one;
                cache <<= 1;
                sh++;
            }
            if ((leaf & 64) != 0)
            {
                if (di + 1 < 576) dst[dstAt + di + 1] = (int)cache < 0 ? -one : one;
                cache <<= 1;
                sh++;
            }
            if (--np == 0)
            {
                np = ints[sfb++] / 2;
                if (np == 0) break;
                one = scf[sc++];
            }
            if ((leaf & 32) != 0)
            {
                if (di + 2 < 576) dst[dstAt + di + 2] = (int)cache < 0 ? -one : one;
                cache <<= 1;
                sh++;
            }
            if ((leaf & 16) != 0)
            {
                if (di + 3 < 576) dst[dstAt + di + 3] = (int)cache < 0 ? -one : one;
                cache <<= 1;
                sh++;
            }
            while (sh >= 0) { cache |= (uint)ByteAt(s, next++) << sh; sh -= 8; }
        }
    }

    /// <summary>
    /// One thread per (frame in batch, granule, channel): scalefactors (granule 1 with scfsi first re-reads granule 0's
    /// scalefactors of the channel, whose values minimp3 keeps in ist_pos), Huffman + requantize into
    /// spec[(g * nch + ch) * 576 ..] (g = batch granule), the intensity positions into ist[(g * nch + ch) * 40 ..].
    /// Frames not decoded get zero lines.
    /// </summary>
    public static void GranuleKernel(Index1D index, ArrayView<byte> stream, ArrayView<int> side, ArrayView<float> spec, ArrayView<int> ist,
        ArrayView<int> ints, ArrayView<float> floats, ArrayView<int> meta, int firstFrame, int first)
    {
        int id = first + index;
        int nch = meta[MChannels], ngr = meta[MGranules];
        int ch = id % nch, rest = id / nch, gr = rest % ngr, lf = rest / ngr;
        int g = lf * ngr + gr;
        long specAt = ((long)g * nch + ch) * 576, istAt = ((long)g * nch + ch) * IstInts;
        for (int i = 0; i < 576; i++) spec[specAt + i] = 0f;
        long at = (long)(firstFrame + lf) * SideInts;
        if (side[at + FDecode] == 0)
        {
            for (int i = 0; i < IstInts; i++) ist[istAt + i] = 0;
            return;
        }
        uint hdr = (uint)side[at + FHeader];
        long ua = at + FrameHead + (gr * nch + ch) * UnitInts;
        int limit = side[at + FStreamEnd] * 8;
        var iscf = LocalMemory.Allocate<int>(IstInts);
        var istPos = LocalMemory.Allocate<int>(IstInts);
        var scf = LocalMemory.Allocate<float>(IstInts);
        var scfSize = LocalMemory.Allocate<int>(4);
        for (int i = 0; i < IstInts; i++)
        {
            istPos[i] = 0;
            iscf[i] = 0;
            scf[i] = 0f;
        }
        if (gr == 1 && side[ua + UScfsi] != 0)
        {
            // granule 0's scalefactors of this channel (scfsi bands copy them)
            long ua0 = at + FrameHead + ch * UnitInts;
            int pos0 = side[ua0 + UBitStart];
            DecodeScalefactors(hdr, istPos, stream, ref pos0, limit, side, ua0, scf, ch, ints, floats, iscf, scfSize, true);
        }
        int pos = side[ua + UBitStart];
        DecodeScalefactors(hdr, istPos, stream, ref pos, limit, side, ua, scf, ch, ints, floats, iscf, scfSize, false);
        Huffman(spec, specAt, stream, pos, side, ua, scf, side[ua + UBitStart] + side[ua + UPart23], ints, floats);
        for (int i = 0; i < IstInts; i++) ist[istAt + i] = istPos[i];
    }

    // ------------------------------------------------------------------ 6. Stereo

    private static void MidSide(ArrayView<float> spec, long left, int n)
    {
        for (int i = 0; i < n; i++)
        {
            float a = spec[left + i];
            float b = spec[left + 576 + i];
            spec[left + i] = a + b;
            spec[left + 576 + i] = a - b;
        }
    }

    /// <summary>
    /// One thread per (frame in batch, granule) of a stereo stream: L3_intensity_stereo (top non-zero band of the right
    /// channel per window, the positions, L3_stereo_process) or L3_midside_stereo over the granule's two channels.
    /// </summary>
    public static void StereoKernel(Index1D index, ArrayView<int> side, ArrayView<float> spec, ArrayView<int> ist, ArrayView<int> ints,
        ArrayView<float> floats, ArrayView<int> meta, int firstFrame, int first)
    {
        int g = first + index;
        int ngr = meta[MGranules];
        int gr = g % ngr, lf = g / ngr;
        long at = (long)(firstFrame + lf) * SideInts;
        if (side[at + FDecode] == 0) return;
        uint hdr = (uint)side[at + FHeader];
        long left = (long)g * 2 * 576;
        if (TestIStereo(hdr))
        {
            long ua0 = at + FrameHead + (gr * 2) * UnitInts, ua1 = ua0 + UnitInts;
            long istAt = ((long)g * 2 + 1) * IstInts;
            int nLong = side[ua0 + UNLong], nShort = side[ua0 + UNShort];
            int sfb = side[ua0 + USfbTab];
            int nSfb = nLong + nShort;
            int maxBlocks = nShort != 0 ? 3 : 1;
            var maxBand = LocalMemory.Allocate<int>(3);
            // L3_stereo_top_band
            maxBand[0] = maxBand[1] = maxBand[2] = -1;
            long right = left + 576;
            for (int i = 0; i < nSfb; i++)
            {
                int w = ints[sfb + i];
                for (int k = 0; k < w; k += 2)
                {
                    if (spec[right + k] != 0 || spec[right + k + 1] != 0)
                    {
                        maxBand[i % 3] = i;
                        break;
                    }
                }
                right += w;
            }
            if (nLong != 0)
            {
                int m = maxBand[0] > maxBand[1] ? maxBand[0] : maxBand[1];
                m = m > maxBand[2] ? m : maxBand[2];
                maxBand[0] = maxBand[1] = maxBand[2] = m;
            }
            for (int i = 0; i < maxBlocks; i++)
            {
                int defaultPos = TestMpeg1(hdr) ? 3 : 0;
                int itop = nSfb - maxBlocks + i;
                int prev = itop - maxBlocks;
                ist[istAt + itop] = maxBand[i] >= prev ? defaultPos : ist[istAt + prev];
            }
            // L3_stereo_process
            int mpeg2Sh = side[ua1 + UScfCompress] & 1;
            int maxPos = TestMpeg1(hdr) ? 7 : 64;
            long l = left;
            for (int i = 0; ints[sfb + i] != 0; i++)
            {
                int w = ints[sfb + i];
                int ipos = ist[istAt + i];
                if (i > maxBand[i % 3] && ipos < maxPos)
                {
                    float kl, kr, sc = TestMsStereo(hdr) ? 1.41421356f : 1;
                    if (TestMpeg1(hdr))
                    {
                        kl = floats[T.FPan + 2 * ipos];
                        kr = floats[T.FPan + 2 * ipos + 1];
                    }
                    else
                    {
                        kl = 1;
                        kr = LdexpQ2(1, (ipos + 1) >> 1 << mpeg2Sh, floats);
                        if ((ipos & 1) != 0)
                        {
                            kl = kr;
                            kr = 1;
                        }
                    }
                    float fl = kl * sc, fr = kr * sc;
                    for (int k = 0; k < w && l + k < left + 576; k++)
                    {
                        spec[l + 576 + k] = spec[l + k] * fr;
                        spec[l + k] = spec[l + k] * fl;
                    }
                }
                else if (TestMsStereo(hdr))
                {
                    MidSide(spec, l, w);
                }
                l += w;
            }
        }
        else if (IsMsStereo(hdr))
        {
            MidSide(spec, left, 576);
        }
    }

    // ------------------------------------------------------------------ 7-8. reorder, alias reduction, IMDCT

    /// <summary>Line j (0..575) of a granule-channel after L3_reorder of the short region (from line r0 with the
    /// short band widths at ints[sfb ..], in threes): lines past what minimp3 reorders keep their place.</summary>
    private static float Reordered(ArrayView<float> spec, long baseAt, int j, int r0, int sfb, ArrayView<int> ints)
    {
        if (j < r0) return spec[baseAt + j];
        int start = r0;
        for (int p = sfb; ; p += 3)
        {
            int len = ints[p];
            if (len == 0) break;
            if (j < start + 3 * len)
            {
                int o = j - start;
                int src = start + (o % 3) * len + o / 3;
                return src < 576 ? spec[baseAt + src] : 0f;
            }
            start += 3 * len;
        }
        return spec[baseAt + j];
    }

    /// <summary>The block shape of a granule-channel: block type, long subbands at the bottom (mixed blocks), short
    /// band table start (or -1 for long blocks), alias-reduction boundaries.</summary>
    private static void BlockShape(ArrayView<int> side, long at, long ua, int srIdx, out int blockType, out int nLongBands, out int shortSfb,
        out int aaBands)
    {
        blockType = 0;
        nLongBands = 0;
        shortSfb = -1;
        aaBands = 31;
        if (side[at + FDecode] == 0) return;
        blockType = side[ua + UBlockType];
        nLongBands = (side[ua + UMixed] != 0 ? 2 : 0) << (srIdx == 2 ? 1 : 0);
        if (side[ua + UNShort] != 0)
        {
            aaBands = nLongBands - 1;
            shortSfb = side[ua + USfbTab] + side[ua + UNLong];
        }
    }

    /// <summary>L3_dct3_9 in place on y[o .. o + 9).</summary>
    private static void Dct3_9(ArrayView<float> y)
    {
        float s0, s1, s2, s3, s4, s5, s6, s7, s8, t0, t2, t4;
        s0 = y[0]; s2 = y[2]; s4 = y[4]; s6 = y[6]; s8 = y[8];
        t0 = s0 + s6 * 0.5f;
        s0 -= s6;
        t4 = (s4 + s2) * 0.93969262f;
        t2 = (s8 + s2) * 0.76604444f;
        s6 = (s4 - s8) * 0.17364818f;
        s4 += s8 - s2;

        s2 = s0 - s4 * 0.5f;
        y[4] = s4 + s0;
        s8 = t0 - t2 + s6;
        s0 = t0 - t4 + t2;
        s4 = t0 + t4 - s6;

        s1 = y[1]; s3 = y[3]; s5 = y[5]; s7 = y[7];

        s3 *= 0.86602540f;
        t0 = (s5 + s1) * 0.98480775f;
        t4 = (s5 - s7) * 0.34202014f;
        t2 = (s1 + s7) * 0.64278761f;
        s1 = (s1 - s5 - s7) * 0.86602540f;

        s5 = t0 - s3 - t2;
        s7 = t4 - s3 - t0;
        s3 = t4 + s3 - t2;

        y[0] = s4 - s7;
        y[1] = s2 + s1;
        y[2] = s0 - s3;
        y[3] = s8 + s5;
        y[5] = s8 - s5;
        y[6] = s0 + s3;
        y[7] = s2 - s1;
        y[8] = s4 + s7;
    }

    /// <summary>L3_imdct36 of one subband: x (18 lines), ovlIn (9) -> out (18 samples), ovlOut (9).</summary>
    private static void Imdct36(ArrayView<float> x, ArrayView<float> ovlIn, ArrayView<float> output, ArrayView<float> ovlOut, int window,
        ArrayView<float> floats, ArrayView<float> co, ArrayView<float> si)
    {
        co[0] = -x[0];
        si[0] = x[17];
        for (int i = 0; i < 4; i++)
        {
            si[8 - 2 * i] = x[4 * i + 1] - x[4 * i + 2];
            co[1 + 2 * i] = x[4 * i + 1] + x[4 * i + 2];
            si[7 - 2 * i] = x[4 * i + 4] - x[4 * i + 3];
            co[2 + 2 * i] = -(x[4 * i + 3] + x[4 * i + 4]);
        }
        Dct3_9(co);
        Dct3_9(si);
        si[1] = -si[1];
        si[3] = -si[3];
        si[5] = -si[5];
        si[7] = -si[7];
        for (int i = 0; i < 9; i++)
        {
            float ovl = ovlIn[i];
            float sum = co[i] * floats[T.FTwid9 + 9 + i] + si[i] * floats[T.FTwid9 + i];
            ovlOut[i] = co[i] * floats[T.FTwid9 + i] - si[i] * floats[T.FTwid9 + 9 + i];
            output[i] = ovl * floats[window + i] - sum * floats[window + 9 + i];
            output[17 - i] = ovl * floats[window + 9 + i] + sum * floats[window + i];
        }
    }

    /// <summary>L3_imdct12 of the short window starting at x[xo] (lines xo, xo + 3, ...): dst[dOff .. + 6), state ov (3).</summary>
    private static void Imdct12(ArrayView<float> x, int xo, ArrayView<float> dst, int dOff, ArrayView<float> ov, ArrayView<float> floats,
        ArrayView<float> co, ArrayView<float> si)
    {
        // L3_idct3(-x[0], x[6] + x[3], x[12] + x[9], co)
        float x0 = -x[xo], x1 = x[xo + 6] + x[xo + 3], x2 = x[xo + 12] + x[xo + 9];
        float m1 = x1 * 0.86602540f, a1 = x0 - x2 * 0.5f;
        co[1] = x0 + x2;
        co[0] = a1 + m1;
        co[2] = a1 - m1;
        // L3_idct3(x[15], x[12] - x[9], x[6] - x[3], si)
        x0 = x[xo + 15];
        x1 = x[xo + 12] - x[xo + 9];
        x2 = x[xo + 6] - x[xo + 3];
        m1 = x1 * 0.86602540f;
        a1 = x0 - x2 * 0.5f;
        si[1] = x0 + x2;
        si[0] = a1 + m1;
        si[2] = a1 - m1;
        si[1] = -si[1];
        for (int i = 0; i < 3; i++)
        {
            float ovl = ov[i];
            float sum = co[i] * floats[T.FTwid3 + 3 + i] + si[i] * floats[T.FTwid3 + i];
            ov[i] = co[i] * floats[T.FTwid3 + i] - si[i] * floats[T.FTwid3 + 3 + i];
            dst[dOff + i] = ovl * floats[T.FTwid3 + 2 - i] - sum * floats[T.FTwid3 + 5 - i];
            dst[dOff + 5 - i] = ovl * floats[T.FTwid3 + 5 - i] + sum * floats[T.FTwid3 + 2 - i];
        }
    }

    /// <summary>L3_imdct_gr for one subband: long (window by block type; mixed blocks' low subbands normal) or short.
    /// ovlOut depends only on x; output also on ovlIn (the previous granule's ovlOut).</summary>
    private static void Imdct(ArrayView<float> x, ArrayView<float> ovlIn, ArrayView<float> output, ArrayView<float> ovlOut, int sb,
        int blockType, int nLongBands, ArrayView<float> floats, ArrayView<float> co, ArrayView<float> si, ArrayView<float> ov)
    {
        if (sb < nLongBands) Imdct36(x, ovlIn, output, ovlOut, T.FMdctWindow, floats, co, si);
        else if (blockType == 2)
        {
            for (int i = 0; i < 6; i++) output[i] = ovlIn[i];
            ov[0] = ovlIn[6];
            ov[1] = ovlIn[7];
            ov[2] = ovlIn[8];
            Imdct12(x, 0, output, 6, ov, floats, co, si);
            Imdct12(x, 1, output, 12, ov, floats, co, si);
            Imdct12(x, 2, ovlOut, 0, ov, floats, co, si);
            ovlOut[6] = ov[0];
            ovlOut[7] = ov[1];
            ovlOut[8] = ov[2];
        }
        else Imdct36(x, ovlIn, output, ovlOut, T.FMdctWindow + (blockType == 3 ? 18 : 0), floats, co, si);
    }

    /// <summary>
    /// One thread per (batch granule, channel, subband): the subband's 18 lines after L3_reorder (short blocks) or
    /// L3_antialias (the 8 butterflies at each boundary it shares; each line changes at most once, from the values
    /// before any butterfly, so the threads are independent) into xbuf, and its IMDCT overlap for the next granule into
    /// ovl[((g + 1) * nch + ch) * 288 + sb * 9 ..] (slot 0 holds the previous batch's last granule).
    /// </summary>
    public static void ImdctPrepKernel(Index1D index, ArrayView<int> side, ArrayView<float> spec, ArrayView<float> xbuf, ArrayView<float> ovl,
        ArrayView<int> ints, ArrayView<float> floats, ArrayView<int> meta, int firstFrame, int first)
    {
        int id = first + index;
        int nch = meta[MChannels], ngr = meta[MGranules];
        int sb = id % 32, rest = id / 32, ch = rest % nch, g = rest / nch;
        long at = (long)(firstFrame + g / ngr) * SideInts;
        long ua = at + FrameHead + ((g % ngr) * nch + ch) * UnitInts;
        BlockShape(side, at, ua, meta[MSrIdx], out int blockType, out int nLongBands, out int shortSfb, out int aaBands);
        long baseAt = ((long)g * nch + ch) * 576;
        var x = LocalMemory.Allocate<float>(18);
        if (shortSfb >= 0)
            for (int i = 0; i < 18; i++) x[i] = Reordered(spec, baseAt, sb * 18 + i, nLongBands * 18, shortSfb, ints);
        else
            for (int i = 0; i < 18; i++) x[i] = spec[baseAt + sb * 18 + i];
        // L3_antialias: boundary b between subbands b - 1 and b, b = 1 .. aaBands
        if (sb >= 1 && sb <= aaBands)
        {
            for (int i = 0; i < 8; i++)
            {
                float u = x[i], dn = spec[baseAt + sb * 18 - 1 - i];
                x[i] = u * floats[T.FAa + i] - dn * floats[T.FAa + 8 + i];
            }
        }
        if (sb + 1 <= aaBands)
        {
            for (int i = 0; i < 8; i++)
            {
                float u = spec[baseAt + (sb + 1) * 18 + i], dn = x[17 - i];
                x[17 - i] = u * floats[T.FAa + 8 + i] + dn * floats[T.FAa + i];
            }
        }
        for (int i = 0; i < 18; i++) xbuf[baseAt + sb * 18 + i] = x[i];
        var zero = LocalMemory.Allocate<float>(9);
        var output = LocalMemory.Allocate<float>(18);
        var ovlOut = LocalMemory.Allocate<float>(9);
        var co = LocalMemory.Allocate<float>(9);
        var si = LocalMemory.Allocate<float>(9);
        var ov = LocalMemory.Allocate<float>(3);
        for (int i = 0; i < 9; i++) zero[i] = 0f;
        Imdct(x, zero, output, ovlOut, sb, blockType, nLongBands, floats, co, si, ov);
        long o = (((long)g + 1) * nch + ch) * 288 + sb * 9;
        for (int i = 0; i < 9; i++) ovl[o + i] = ovlOut[i];
    }

    /// <summary>
    /// One thread per (batch granule, channel, subband): IMDCT of the subband's lines with the previous granule's
    /// overlap, L3_change_sign (odd samples of odd subbands), into tbuf[(g * nch + ch) * 576 + sb * 18 ..].
    /// </summary>
    public static void ImdctOutKernel(Index1D index, ArrayView<int> side, ArrayView<float> xbuf, ArrayView<float> ovl, ArrayView<float> tbuf,
        ArrayView<float> floats, ArrayView<int> meta, int firstFrame, int first)
    {
        int id = first + index;
        int nch = meta[MChannels], ngr = meta[MGranules];
        int sb = id % 32, rest = id / 32, ch = rest % nch, g = rest / nch;
        long at = (long)(firstFrame + g / ngr) * SideInts;
        long ua = at + FrameHead + ((g % ngr) * nch + ch) * UnitInts;
        BlockShape(side, at, ua, meta[MSrIdx], out int blockType, out int nLongBands, out _, out _);
        long baseAt = ((long)g * nch + ch) * 576 + sb * 18;
        var x = LocalMemory.Allocate<float>(18);
        var ovlIn = LocalMemory.Allocate<float>(9);
        var output = LocalMemory.Allocate<float>(18);
        var ovlOut = LocalMemory.Allocate<float>(9);
        var co = LocalMemory.Allocate<float>(9);
        var si = LocalMemory.Allocate<float>(9);
        var ov = LocalMemory.Allocate<float>(3);
        for (int i = 0; i < 18; i++) x[i] = xbuf[baseAt + i];
        long o = ((long)g * nch + ch) * 288 + sb * 9;
        for (int i = 0; i < 9; i++) ovlIn[i] = ovl[o + i];
        Imdct(x, ovlIn, output, ovlOut, sb, blockType, nLongBands, floats, co, si, ov);
        for (int i = 0; i < 18; i++) tbuf[baseAt + i] = (sb & 1) != 0 && (i & 1) != 0 ? -output[i] : output[i];
    }

    /// <summary>One thread per element: buf[to + i] = buf[from + i] (carries state across batches).</summary>
    public static void CarryKernel(Index1D index, ArrayView<float> buf, long from, long to) => buf[to + index] = buf[from + index];

    // ------------------------------------------------------------------ 9-10. synthesis

    /// <summary>
    /// One thread per (batch time slot, channel): mp3d_DCT_II of the slot's 32 subband samples (SSE path's sums) into
    /// dct[((16 + slot) * nch + ch) * 32 ..] (16 history slots ahead of the batch's).
    /// </summary>
    public static void DctKernel(Index1D index, ArrayView<float> tbuf, ArrayView<float> dct, ArrayView<float> floats, ArrayView<int> meta, int first)
    {
        int id = first + index;
        int nch = meta[MChannels];
        int ch = id % nch, slot = id / nch;
        long y = ((long)(slot / 18) * nch + ch) * 576 + slot % 18;
        var t = LocalMemory.Allocate<float>(32);
        for (int i = 0; i < 8; i++)
        {
            float x0 = tbuf[y + i * 18];
            float x1 = tbuf[y + (15 - i) * 18];
            float x2 = tbuf[y + (16 + i) * 18];
            float x3 = tbuf[y + (31 - i) * 18];
            float t0 = x0 + x3;
            float t1 = x1 + x2;
            float t2 = (x1 - x2) * floats[T.FSec + 3 * i + 0];
            float t3 = (x0 - x3) * floats[T.FSec + 3 * i + 1];
            t[i] = t0 + t1;
            t[8 + i] = (t0 - t1) * floats[T.FSec + 3 * i + 2];
            t[16 + i] = t3 + t2;
            t[24 + i] = (t3 - t2) * floats[T.FSec + 3 * i + 2];
        }
        for (int r = 0; r < 32; r += 8)
        {
            float x0 = t[r], x1 = t[r + 1], x2 = t[r + 2], x3 = t[r + 3], x4 = t[r + 4], x5 = t[r + 5], x6 = t[r + 6], x7 = t[r + 7], xt;
            xt = x0 - x7; x0 += x7;
            x7 = x1 - x6; x1 += x6;
            x6 = x2 - x5; x2 += x5;
            x5 = x3 - x4; x3 += x4;
            x4 = x0 - x3; x0 += x3;
            x3 = x1 - x2; x1 += x2;
            t[r] = x0 + x1;
            t[r + 4] = (x0 - x1) * 0.70710677f;
            x5 = x5 + x6;
            x6 = (x6 + x7) * 0.70710677f;
            x7 = x7 + xt;
            x3 = (x3 + x4) * 0.70710677f;
            x5 -= x7 * 0.198912367f; // rotate by PI/8
            x7 += x5 * 0.382683432f;
            x5 -= x7 * 0.198912367f;
            x0 = xt - x6; xt += x6;
            t[r + 1] = (xt + x7) * 0.50979561f;
            t[r + 2] = (x4 + x3) * 0.54119611f;
            t[r + 3] = (x0 - x5) * 0.60134488f;
            t[r + 5] = (x0 + x5) * 0.89997619f;
            t[r + 6] = (x4 - x3) * 1.30656302f;
            t[r + 7] = (xt - x7) * 2.56291556f;
        }
        long o = ((long)(HistorySlots + slot) * nch + ch) * 32;
        for (int i = 0; i < 7; i++)
        {
            float s = t[24 + i] + t[24 + i + 1];   // SSE path: t2 + (t3[i] + t3[i + 1])
            dct[o + 4 * i + 0] = t[i];
            dct[o + 4 * i + 1] = t[16 + i] + s;
            dct[o + 4 * i + 2] = t[8 + i] + t[8 + i + 1];
            dct[o + 4 * i + 3] = t[16 + i + 1] + s;
        }
        dct[o + 28] = t[7];
        dct[o + 29] = t[16 + 7] + t[24 + 7];
        dct[o + 30] = t[8 + 7];
        dct[o + 31] = t[24 + 7];
    }

    /// <summary>
    /// minimp3's lins value at synthesis row <paramref name="row"/> (row r holds slot r - 15), position 0..15, lane 0..3.
    /// mp3d_synth fills a row with two slots' DCT outputs: lanes 0 / 1 = (left, right) of the row's slot s, lanes 2 / 3
    /// of slot s + 1, at position p (0..14) output 31 - p of an even slot or 1 + p of an odd one; position 15 holds
    /// slot s's output 16 (lanes 0 / 1) and 0 (lanes 2 / 3).
    /// </summary>
    private static float V(ArrayView<float> dct, int nch, int row, int p, int lane)
    {
        int s = row - 15;
        int c = nch == 1 ? 0 : (lane & 1);
        int slot, idx;
        if (p == 15)
        {
            slot = s;
            idx = lane < 2 ? 16 : 0;
        }
        else
        {
            slot = lane < 2 ? s : s + 1;
            idx = (s & 1) == 0 ? 31 - p : 1 + p;
        }
        return dct[((long)(slot + HistorySlots) * nch + c) * 32 + idx];
    }

    /// <summary>
    /// One thread per output sample (batch slot, channel, n 0..31): mp3d_synth / mp3d_synth_pair for that sample, times
    /// 1/32768 (MINIMP3_FLOAT_OUTPUT), written at stream sample firstSample + slot * 32 + n - skip when inside
    /// [0, total): interleaved (planar = 0) or planar [channel][sample].
    /// </summary>
    public static void SynthKernel(Index1D index, ArrayView<float> dct, ArrayView<float> output, ArrayView<float> floats, ArrayView<int> meta,
        long firstSample, long skip, long total, int planar, long first)
    {
        long id = first + index;
        int nch = meta[MChannels];
        int n = (int)(id % 32);
        long rest = id / 32;
        int ch = (int)(rest % nch);
        int slot = (int)(rest / nch);
        long t = firstSample + (long)slot * 32 + n - skip;
        if (t < 0 || t >= total) return;
        int pair = slot & ~1, odd = slot & 1;
        float a;
        if (n == 0)
        {
            int b0 = pair + odd, l = ch;
            a = (V(dct, nch, b0 + 14, 15, l) - V(dct, nch, b0, 15, l)) * 29;
            a += (V(dct, nch, b0 + 1, 15, l) + V(dct, nch, b0 + 13, 15, l)) * 213;
            a += (V(dct, nch, b0 + 12, 15, l) - V(dct, nch, b0 + 2, 15, l)) * 459;
            a += (V(dct, nch, b0 + 3, 15, l) + V(dct, nch, b0 + 11, 15, l)) * 2037;
            a += (V(dct, nch, b0 + 10, 15, l) - V(dct, nch, b0 + 4, 15, l)) * 5153;
            a += (V(dct, nch, b0 + 5, 15, l) + V(dct, nch, b0 + 9, 15, l)) * 6574;
            a += (V(dct, nch, b0 + 8, 15, l) - V(dct, nch, b0 + 6, 15, l)) * 37489;
            a += V(dct, nch, b0 + 7, 15, l) * 75038;
        }
        else if (n == 16)
        {
            int b0 = pair + odd, l = ch + 2;
            a = V(dct, nch, b0 + 14, 15, l) * 104;
            a += V(dct, nch, b0 + 12, 15, l) * 1567;
            a += V(dct, nch, b0 + 10, 15, l) * 9727;
            a += V(dct, nch, b0 + 8, 15, l) * 64019;
            a += V(dct, nch, b0 + 6, 15, l) * -9975;
            a += V(dct, nch, b0 + 4, 15, l) * -45;
            a += V(dct, nch, b0 + 2, 15, l) * 146;
            a += V(dct, nch, b0 + 0, 15, l) * -5;
        }
        else
        {
            int z = pair + 15, lane = ch + 2 * odd;
            int j = n < 16 ? 15 - n : n - 17;
            int w = T.FWin + (14 - j) * 16;
            float sa = 0, sb = 0;
            for (int k = 0; k < 8; k++)
            {
                float w0 = floats[w + 2 * k], w1 = floats[w + 2 * k + 1];
                float vz = V(dct, nch, z - k, j, lane), vy = V(dct, nch, z - (15 - k), j, lane);
                if (k == 0)
                {
                    sb = vz * w1 + vy * w0;
                    sa = vz * w0 - vy * w1;
                }
                else if ((k & 1) != 0)
                {
                    sb = sb + (vz * w1 + vy * w0);
                    sa = sa + (vy * w1 - vz * w0);
                }
                else
                {
                    sb = sb + (vz * w1 + vy * w0);
                    sa = sa + (vz * w0 - vy * w1);
                }
            }
            a = n < 16 ? sa : sb;
        }
        float v = a * (1f / 32768f);
        if (planar != 0) output[ch * total + t] = v;
        else output[t * nch + ch] = v;
    }
}
