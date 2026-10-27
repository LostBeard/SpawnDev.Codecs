// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GDeflate tile emission for DeflateGpu (Microsoft DirectStorage GDeflate; format and writer from libdeflate's
// GDEFLATE paths, deflate_compress.c, NVIDIA / Microsoft, Apache 2.0). A tile (at most 64 KB of input) is parsed by
// ParseStep exactly like a DEFLATE block; PlanGDeflateLanes builds its codes (Deflate64 alphabet: length 258 is symbol 284
// with 5 extra bits, since 285 means 3 + 16 extra bits) and decides stored or dynamic; EmitGDeflateLanes writes the
// tile as one GDeflate stream (32 lanes, the bytes of libdeflate's serial writer):
//   - 32 sub-streams, each a 64-bit bit buffer flushed in 32-bit packets at a reserved position;
//   - the writer EMULATES the decoder's bit count per sub-stream: when the decoder would hold fewer than 32 bits after a
//     symbol it refills from the next packet, so the writer reserves that packet position right then (in the decoder's
//     refill order, which is the stream order);
//   - a match's distance goes to the same sub-stream one round later (deferred copies), and the copies still pending at
//     the end of the block are written last (tail copies);
//   - stored blocks: LEN (16 bits) then one byte per sub-stream slot.

using ILGPU;

namespace SpawnDev.Codecs.Compression;

public static partial class DeflateGpu
{
    private const int GdStreams = 32;

    // GDeflate writer state (ints), after the block's scratch: per sub-stream bit buffer (lo, hi), bit count, packet
    // write position, next packet position (-1 none), emulated decoder bit count, deferred copy bits / count / round;
    // then the stream index, the next free output byte and the overflow flag.
    private const int GdLo = 0, GdHi = GdLo + GdStreams, GdCount = GdHi + GdStreams, GdWrite = GdCount + GdStreams,
        GdNext = GdWrite + GdStreams, GdInput = GdNext + GdStreams, GdCopyBits = GdInput + GdStreams,
        GdCopyCount = GdCopyBits + GdStreams, GdCopyRound = GdCopyCount + GdStreams, GdIdx = GdCopyRound + GdStreams,
        GdNextPos = GdIdx + 1, GdEnd = GdIdx + 2, GdOverflow = GdIdx + 3;
    /// <summary>Ints of GDeflate writer state per tile (after <see cref="ScratchInts"/>).</summary>
    public const int GDeflateWriterInts = GdIdx + 4;

    /// <summary>Upper bound of one GDeflate tile's output for <paramref name="len"/> input bytes (stored, plus headers and the final packets).</summary>
    public static int MaxGDeflateTileOutput(int len) => len + 3 * (len / 65535 + 1) + 8 * GdStreams + 64;

    /// <summary>Shared ints <see cref="PlanGDeflateLanes"/> needs: the literal/length and distance counts, the extra bits.</summary>
    public const int GDeflatePlanSharedInts = 288 + 32 + 1;

    /// <summary>
    /// Level 4 (Huffman only) without a parse: thread <paramref name="k"/> (0 .. blockSize - 1) stores byte k as its
    /// literal token; thread 0 marks the block parsed. The tokens are the ones <see cref="ParseStep"/> writes.
    /// </summary>
    public static void LiteralTokensThread(ArrayView<byte> data, int start, int len, int k, ArrayView<int> s, int sb, int blockSize)
    {
        int tokAt = sb + HashSize + blockSize;
        int st = tokAt + blockSize + StateOff;
        if (k < len) s[tokAt + k] = data[start + k];
        if ((k == 0) & (s[st + StPhase] == PhaseParse))
        {
            s[st + StPos] = len;
            s[st + StTokens] = len;
            s[st + StPhase] = PhaseParsed;
        }
    }

    /// <summary>
    /// After the parse, by the <paramref name="lanes"/> threads of a group (<paramref name="lane"/> = Group.IdxX, with
    /// <paramref name="h"/> = <see cref="GDeflatePlanSharedInts"/> ints of shared memory): the tile's GDeflate codes
    /// (Deflate64 length symbols) and the stored / static / dynamic decision, from the exact sizes of the encodings.
    /// Lane 0 merges consecutive matches at one distance, every lane counts symbols, lane 0 builds the codes.
    /// </summary>
    public static void PlanGDeflateLanes(int len, int level, ArrayView<int> s, int sb, int blockSize, ArrayView<int> h, int lane, int lanes)
    {
        int tokAt = sb + HashSize + blockSize;
        int tail = tokAt + blockSize;
        int st = tail + StateOff;
        bool active = Group.BarrierAnd(s[st + StPhase] == PhaseParsed);
        if (active)
        {
            bool stored = level <= 0;
            if ((lane == 0) & (!stored) & (level != LevelHuffmanOnly)) MergeGDeflateMatches(s, tokAt, st);
            for (int i = lane; i < GDeflatePlanSharedInts; i += lanes) h[i] = 0;
            Group.Barrier();
            if (!stored)
            {
                int count = s[st + StTokens];
                int extra = 0;
                for (int t = lane; t < count; t += lanes)
                {
                    int tok = s[tokAt + t];
                    if (tok < 256) Atomic.Add(ref h[tok], 1);
                    else
                    {
                        int ml = tok >> 16;
                        int md = tok & 0xFFFF;
                        Atomic.Add(ref h[GdLengthSymbol(ml)], 1);
                        Atomic.Add(ref h[288 + DistSymbol(md)], 1);
                        extra += GdLengthExtraBits(ml) + DistExtraBits(md);
                    }
                }
                if (extra != 0) Atomic.Add(ref h[320], extra);
            }
            Group.Barrier();
            if (lane == 0) PlanGDeflateCodes(len, stored, s, tail, st, h);
        }
    }

    /// <summary>
    /// Deflate64 matches reach 65538 bytes: consecutive matches at the same distance are one copy (byte-wise copy
    /// semantics), merged up to GdMaxMerged (the token keeps the length in its upper 15 bits).
    /// </summary>
    private static void MergeGDeflateMatches(ArrayView<int> s, int tokAt, int st)
    {
        int count = s[st + StTokens];
        int w = 0;
        for (int t = 0; t < count; t++)
        {
            int tok = s[tokAt + t];
            bool merge = (w > 0) & (tok >= 256);
            if (merge)
            {
                int prev = s[tokAt + w - 1];
                merge = (prev >= 256) & ((prev & 0xFFFF) == (tok & 0xFFFF)) & ((prev >> 16) + (tok >> 16) <= GdMaxMerged);
                if (merge) s[tokAt + w - 1] = (((prev >> 16) + (tok >> 16)) << 16) | (tok & 0xFFFF);
            }
            if (!merge)
            {
                s[tokAt + w] = tok;
                w++;
            }
        }
        s[st + StTokens] = w;
    }

    /// <summary>Lane 0 of the plan: the counts into the block's tables, the codes, the encoding with the fewest bits.</summary>
    private static void PlanGDeflateCodes(int len, bool stored, ArrayView<int> s, int tail, int st, ArrayView<int> h)
    {
        if (!stored)
        {
            for (int i = 0; i < 288; i++) s[tail + LitFreqOff + i] = h[i];
            for (int i = 0; i < 32; i++) s[tail + DistFreqOff + i] = h[288 + i];
            int extra = h[320];
            s[st + StExtra] = extra;
            // A static tile before (same scratch) left fixed lengths for symbols 286 / 287 and distances 30 / 31, which
            // the dynamic code builder never writes but AssignCodes reads: clear them.
            s[tail + LitLenOff + 286] = 0;
            s[tail + LitLenOff + 287] = 0;
            s[tail + DistLenOff + 30] = 0;
            s[tail + DistLenOff + 31] = 0;
            long dynBits = BuildBlockCodes(s, tail, st);
            // Static block: the RFC 1951 fixed codes, no code tables (small tiles, long runs).
            long staticBits = 3 + extra;
            for (int sym = 0; sym < 286; sym++) staticBits += (long)s[tail + LitFreqOff + sym] * FixedLitLength(sym);
            for (int sym = 0; sym < 30; sym++) staticBits += 5L * s[tail + DistFreqOff + sym];
            // Stored GDeflate blocks: 3 header bits + LEN (16) + 8 bits a byte, at most 65535 bytes a block.
            long storedBits = 8L * len + 19L * ((len + 65534) / 65535);
            int mode = GdModeDynamic;
            long best = dynBits;
            if (staticBits < best)
            {
                mode = GdModeStatic;
                best = staticBits;
            }
            if (storedBits <= best) mode = GdModeStored;
            if (mode == GdModeStatic)
            {
                // The emitter reads the codes from the block's tables: put the fixed ones there.
                for (int sym = 0; sym < 288; sym++)
                {
                    int l = FixedLitLength(sym);
                    s[tail + LitLenOff + sym] = l;
                    s[tail + LitCodeOff + sym] = (int)ReverseBits((uint)FixedLitCode(sym), l);
                }
                for (int sym = 0; sym < 32; sym++)
                {
                    s[tail + DistLenOff + sym] = 5;
                    s[tail + DistCodeOff + sym] = (int)ReverseBits((uint)sym, 5);
                }
            }
            s[st + StGdStored] = mode;
        }
        else s[st + StGdStored] = GdModeStored;
        s[st + StPhase] = PhaseEmitTokens;
    }

    private const int GdModeDynamic = 0, GdModeStored = 1, GdModeStatic = 2;

    /// <summary>RFC 1951 fixed literal/length code length.</summary>
    private static int FixedLitLength(int sym) => sym < 144 ? 8 : sym < 256 ? 9 : sym < 280 ? 7 : 8;

    /// <summary>RFC 1951 fixed literal/length code (most significant bit first).</summary>
    private static int FixedLitCode(int sym) =>
        sym < 144 ? 0x30 + sym : sym < 256 ? 0x190 + (sym - 144) : sym < 280 ? sym - 256 : 0xC0 + (sym - 280);

    /// <summary>Longest merged match (its length sits in the token's upper 15 bits).</summary>
    private const int GdMaxMerged = 32767;

    /// <summary>Deflate64 length symbol: 257..283 as in DEFLATE up to 257, 258 is 284 (227 + 31), longer is 285 (3 + 16 bits).</summary>
    private static int GdLengthSymbol(int length) => length < 258 ? LengthSymbol(length) : length == 258 ? 284 : 285;

    private static int GdLengthExtraBits(int length) => length < 258 ? LengthExtraBits(length) : length == 258 ? 5 : 16;

    private static void GdInit(ArrayView<int> s, int g, ArrayView<byte> o, int outStart, int outEnd)
    {
        int next = outStart;
        s[g + GdOverflow] = 0;
        s[g + GdEnd] = outEnd;
        for (int n = 0; n < GdStreams; n++)
        {
            s[g + GdLo + n] = 0;
            s[g + GdHi + n] = 0;
            s[g + GdCount + n] = 0;
            s[g + GdWrite + n] = next;
            s[g + GdNext + n] = -1;
            // The decoder starts with one packet in every sub-stream.
            s[g + GdInput + n] = 32;
            if (next + 4 <= outEnd)
            {
                o[next] = 0;
                o[next + 1] = 0;
                o[next + 2] = 0;
                o[next + 3] = 0;
            }
            else s[g + GdOverflow] = 1;
            next += 4;
        }
        s[g + GdNextPos] = next;
        s[g + GdIdx] = 0;
    }

    /// <summary>Moves to the next sub-stream; 1 when that wraps to sub-stream 0 (a new round).</summary>
    private static int GdAdvance(ArrayView<int> s, int g)
    {
        int idx = (s[g + GdIdx] + 1) & (GdStreams - 1);
        s[g + GdIdx] = idx;
        return idx == 0 ? 1 : 0;
    }

    /// <summary>Stores the current sub-stream's low 32 bits at its packet position and moves to its next packet.</summary>
    private static void GdFlushPacket(ArrayView<int> s, int g, ArrayView<byte> o)
    {
        int i = s[g + GdIdx];
        int at = s[g + GdWrite + i];
        uint lo = (uint)s[g + GdLo + i];
        if ((at >= 0) & (at + 4 <= s[g + GdEnd]))
        {
            o[at] = (byte)lo;
            o[at + 1] = (byte)(lo >> 8);
            o[at + 2] = (byte)(lo >> 16);
            o[at + 3] = (byte)(lo >> 24);
        }
        else s[g + GdOverflow] = 1;
        s[g + GdLo + i] = s[g + GdHi + i];
        s[g + GdHi + i] = 0;
        s[g + GdCount + i] = s[g + GdCount + i] - 32;
        s[g + GdWrite + i] = s[g + GdNext + i];
        s[g + GdNext + i] = -1;
    }

    /// <summary>
    /// Adds <paramref name="n"/> (at most 32) bits to the current sub-stream: a full 32-bit packet is flushed, and when the
    /// emulated decoder would now hold fewer than 32 bits of this sub-stream, its next packet position is reserved.
    /// </summary>
    private static void GdAddBits(ArrayView<int> s, int g, ArrayView<byte> o, uint bits, int n)
    {
        int i = s[g + GdIdx];
        int count = s[g + GdCount + i];
        uint lo = (uint)s[g + GdLo + i];
        uint hi = (uint)s[g + GdHi + i];
        if (count == 0) lo |= bits;
        else
        {
            lo |= bits << count;
            hi |= bits >> (32 - count);
        }
        s[g + GdLo + i] = (int)lo;
        s[g + GdHi + i] = (int)hi;
        s[g + GdCount + i] = count + n;
        int input = s[g + GdInput + i] - n;
        s[g + GdInput + i] = input;
        if (count + n >= 32) GdFlushPacket(s, g, o);
        if ((input < 32) & (s[g + GdNext + i] < 0))
        {
            int next = s[g + GdNextPos];
            s[g + GdNext + i] = next;
            if (next + 4 <= s[g + GdEnd])
            {
                o[next] = 0;
                o[next + 1] = 0;
                o[next + 2] = 0;
                o[next + 3] = 0;
            }
            else s[g + GdOverflow] = 1;
            s[g + GdNextPos] = next + 4;
            s[g + GdInput + i] = input + 32;
        }
    }

    // ------------------------------------------------------------------
    // The writer with 32 lanes per tile (lane i owns sub-stream i), the bytes of the serial writer. That writer places the
    // symbols in rounds (every round visits sub-streams 0..31 once: a sub-stream that took a match last round writes its
    // distance, the others take the next symbols in order), and only a reservation (the emulated decoder's refill) touches
    // shared state: the next free packet position. So a round is: a prefix sum hands out the symbols, every lane adds its
    // bits to its own sub-stream twice (a dry pass that counts its reservations, a prefix sum over those counts, the real
    // pass that writes at the positions it got), and the cursor moves past them all. The block headers stay serial
    // (lane 0, GdAddBits on the shared copy of the writer state). Every packet is one aligned 32-bit word that only its
    // lane writes (the caller's slot must start at a multiple of 4).

    /// <summary>Ints of shared memory <see cref="EmitGDeflateLanes"/> needs (writer state and one per-lane scan area).</summary>
    public const int GDeflateLanesSharedInts = GDeflateWriterInts + GdStreams;
    private const int GdScanArea = GDeflateWriterInts;

    /// <summary>One lane's sub-stream writer state, held in registers during a slot.</summary>
    private struct GdLane
    {
        public uint Lo, Hi;
        public int Count, Input, Write, Next;
    }

    /// <summary>
    /// Writes the planned tile as one GDeflate stream at <paramref name="outStart"/> (at most <paramref name="outCap"/>
    /// bytes, a multiple of 4 from a 4-aligned start) by the 32 lanes of a group (<paramref name="lane"/> = Group.IdxX),
    /// with <paramref name="w"/> = <see cref="GDeflateLanesSharedInts"/> ints of shared memory; its length (or -1 on
    /// overflow) goes to the block output slot (<see cref="BlockOutput"/>). The tile must be planned by an earlier launch
    /// (<see cref="PlanGDeflateLanes"/>).
    /// </summary>
    public static void EmitGDeflateLanes(ArrayView<byte> data, int start, int len, ArrayView<byte> o, int outStart, int outCap,
        ArrayView<int> s, int sb, int blockSize, ArrayView<int> w, int lane)
    {
        int tokAt = sb + HashSize + blockSize;
        int tail = tokAt + blockSize;
        int st = tail + StateOff;
        int end = outStart + outCap;
        bool active = Group.BarrierAnd(s[st + StPhase] == PhaseEmitTokens);
        if (active)
        {
            if (lane == 0) GdInit(w, 0, o, outStart, end);
            w[GdCopyRound + lane] = 0;
            Group.Barrier();
            bool stored = Group.BarrierOr(s[st + StGdStored] == GdModeStored);
            if (stored) GdStoredLanes(data, start, len, o, w, lane, end);
            else
            {
                if (lane == 0) GdHeaderSerial(o, s, tail, st, w);
                Group.Barrier();
                GdSymbolLanes(o, s, tokAt, tail, s[st + StTokens], w, lane, end);
            }

            // The last packet of every sub-stream.
            Group.Barrier();
            int at = w[GdWrite + lane];
            int ovf = 0;
            GdPutPacket(o, at, (uint)w[GdLo + lane], end, ref ovf);
            if (ovf != 0) w[GdOverflow] = 1;
            Group.Barrier();
            if (lane == 0)
            {
                int outLen = w[GdNextPos] - outStart;
                s[st + StOutLen] = w[GdOverflow] != 0 ? -1 : outLen;
                s[st + StPhase] = PhaseDone;
            }
        }
    }

    /// <summary>Lane 0: the block header bits (final, type) and a dynamic block's Huffman header (precode lengths one per sub-stream, then the code-length symbols).</summary>
    private static void GdHeaderSerial(ArrayView<byte> o, ArrayView<int> s, int tail, int st, ArrayView<int> w)
    {
        int mode = s[st + StGdStored];
        w[GdIdx] = 0;
        GdAddBits(w, 0, o, 1, 1);
        GdAddBits(w, 0, o, mode == GdModeStatic ? 1u : 2u, 2);
        if (mode == GdModeDynamic)
        {
            int hlit = s[st + StHlit];
            int hdist = s[st + StHdist];
            int hclen = s[st + StHclen];
            int seqCount = s[st + StSeq];
            w[GdIdx] = 0;
            GdAddBits(w, 0, o, (uint)(hlit - 257), 5);
            GdAddBits(w, 0, o, (uint)(hdist - 1), 5);
            GdAddBits(w, 0, o, (uint)(hclen - 4), 4);
            for (int i = 0; i < hclen; i++)
            {
                GdAddBits(w, 0, o, (uint)s[tail + ClLenOff + InflateGpu.CodeLengthOrder(i)], 3);
                GdAdvance(w, 0);
            }
            w[GdIdx] = 0;
            for (int i = 0; i < seqCount; i++)
            {
                int packed = s[tail + ClSeqOff + i];
                int sym = packed & 31;
                GdAddBits(w, 0, o, (uint)s[tail + ClCodeOff + sym], s[tail + ClLenOff + sym]);
                int eb = (packed >> 12) & 7;
                if (eb > 0) GdAddBits(w, 0, o, (uint)((packed >> 5) & 127), eb);
                GdAdvance(w, 0);
            }
        }
        w[GdIdx] = 0;
    }

    /// <summary>The symbol rounds of a static or dynamic block, then the end-of-block symbol and the deferred distances.</summary>
    private static void GdSymbolLanes(ArrayView<byte> o, ArrayView<int> s, int tokAt, int tail, int tokCount, ArrayView<int> w,
        int lane, int end)
    {
        int round = 1;
        int taken = 0;
        bool going = true;
        while (going)
        {
            bool eobPlaced = taken > tokCount;
            bool hasPrev = (round > 1) & (w[GdCopyRound + lane] == round - 1);
            bool free = (!hasPrev) & (!eobPlaced);
            int freeCount = 0;
            int t = taken + GdScan(w, lane, free ? 1 : 0, ref freeCount);
            taken += freeCount;

            uint b1 = 0, b2 = 0, copyBits = 0;
            int n1 = 0, n2 = 0, copyCount = 0;
            bool setCopy = false;
            if (hasPrev)
            {
                b1 = (uint)w[GdCopyBits + lane];
                n1 = w[GdCopyCount + lane];
            }
            else if (free & (t == tokCount))
            {
                b1 = (uint)s[tail + LitCodeOff + 256];
                n1 = s[tail + LitLenOff + 256];
            }
            else if (free & (t < tokCount))
            {
                int tok = s[tokAt + t];
                if (tok < 256)
                {
                    b1 = (uint)s[tail + LitCodeOff + tok];
                    n1 = s[tail + LitLenOff + tok];
                }
                else
                {
                    int ml = tok >> 16;
                    int md = tok & 0xFFFF;
                    int ls = GdLengthSymbol(ml);
                    int leb = GdLengthExtraBits(ml);
                    int lev = ml - 3;
                    if (ls < 285) lev = LengthExtraValue(ml);
                    if (ml == 258) lev = 258 - 227;
                    b1 = (uint)s[tail + LitCodeOff + ls];
                    n1 = s[tail + LitLenOff + ls];
                    b2 = (uint)lev;
                    n2 = leb;
                    int ds = DistSymbol(md);
                    int dlen = s[tail + DistLenOff + ds];
                    copyBits = (uint)s[tail + DistCodeOff + ds] | ((uint)DistExtraValue(md) << dlen);
                    copyCount = dlen + DistExtraBits(md);
                    setCopy = true;
                }
            }
            GdLaneSlot(o, w, lane, end, b1, n1, b2, n2);
            if (hasPrev) w[GdCopyRound + lane] = 0;
            if (setCopy)
            {
                w[GdCopyBits + lane] = (int)copyBits;
                w[GdCopyCount + lane] = copyCount;
                w[GdCopyRound + lane] = round;
            }
            round++;
            going = Group.BarrierOr((taken <= tokCount) | (w[GdCopyRound + lane] == round - 1));
        }
    }

    /// <summary>Stored blocks (at most 65535 bytes each): lane 0 writes the header, then one byte per lane per round.</summary>
    private static void GdStoredLanes(ArrayView<byte> data, int start, int len, ArrayView<byte> o, ArrayView<int> w, int lane, int end)
    {
        int done = 0;
        bool more = Group.BarrierOr(done < len);
        while (more)
        {
            int chunk = len - done;
            if (chunk > 65535) chunk = 65535;
            if (lane == 0)
            {
                w[GdIdx] = 0;
                GdAddBits(w, 0, o, done + chunk == len ? 1u : 0u, 1);
                GdAddBits(w, 0, o, 0, 2);
                GdAddBits(w, 0, o, (uint)chunk, 16);
            }
            Group.Barrier();
            int k = 0;
            bool rounds = Group.BarrierOr(k < chunk);
            while (rounds)
            {
                bool act = k + lane < chunk;
                uint b = 0;
                if (act) b = data[start + done + k + lane];
                GdLaneSlot(o, w, lane, end, b, act ? 8 : 0, 0, 0);
                k += GdStreams;
                rounds = Group.BarrierOr(k < chunk);
            }
            done += chunk;
            more = Group.BarrierOr(done < len);
        }
    }

    /// <summary>
    /// Every lane adds its slot's bits (up to two adds, <paramref name="n1"/> / <paramref name="n2"/> = 0 for none) to its
    /// own sub-stream: a dry pass counts its reservations, a prefix sum in lane order places them after the cursor, and
    /// the real pass writes; then the cursor moves past all of them.
    /// </summary>
    private static void GdLaneSlot(ArrayView<byte> o, ArrayView<int> w, int lane, int end, uint b1, int n1, uint b2, int n2)
    {
        var l = new GdLane
        {
            Lo = (uint)w[GdLo + lane], Hi = (uint)w[GdHi + lane], Count = w[GdCount + lane],
            Input = w[GdInput + lane], Write = w[GdWrite + lane], Next = w[GdNext + lane],
        };
        var dry = l;
        int k = 0;
        int ovf = 0;
        if (n1 > 0) GdLaneAdd(ref dry, o, b1, n1, false, ref k, 0, end, ref ovf);
        if (n2 > 0) GdLaneAdd(ref dry, o, b2, n2, false, ref k, 0, end, ref ovf);
        int cursor = w[GdNextPos];
        int total = 0;
        int before = GdScan(w, lane, k, ref total);
        k = 0;
        if (n1 > 0) GdLaneAdd(ref l, o, b1, n1, true, ref k, cursor + 4 * before, end, ref ovf);
        if (n2 > 0) GdLaneAdd(ref l, o, b2, n2, true, ref k, cursor + 4 * before, end, ref ovf);
        w[GdLo + lane] = (int)l.Lo;
        w[GdHi + lane] = (int)l.Hi;
        w[GdCount + lane] = l.Count;
        w[GdInput + lane] = l.Input;
        w[GdWrite + lane] = l.Write;
        w[GdNext + lane] = l.Next;
        if (ovf != 0) w[GdOverflow] = 1;
        if (lane == 0) w[GdNextPos] = cursor + 4 * total;
    }

    /// <summary>GdAddBits on one lane's registers; reservation k of this slot is at <paramref name="basePos"/> + 4 k.</summary>
    private static void GdLaneAdd(ref GdLane l, ArrayView<byte> o, uint bits, int n, bool emit, ref int k, int basePos, int end, ref int ovf)
    {
        if (l.Count == 0) l.Lo |= bits;
        else
        {
            l.Lo |= bits << l.Count;
            l.Hi |= bits >> (32 - l.Count);
        }
        l.Count += n;
        l.Input -= n;
        if (l.Count >= 32)
        {
            if (emit) GdPutPacket(o, l.Write, l.Lo, end, ref ovf);
            l.Lo = l.Hi;
            l.Hi = 0;
            l.Count -= 32;
            l.Write = l.Next;
            l.Next = -1;
        }
        if ((l.Input < 32) & (l.Next < 0))
        {
            int pos = basePos + 4 * k;
            k++;
            l.Next = pos;
            if (emit) GdPutPacket(o, pos, 0, end, ref ovf);
            l.Input += 32;
        }
    }

    /// <summary>One 32-bit packet (little-endian) at <paramref name="at"/>, or the overflow flag.</summary>
    private static void GdPutPacket(ArrayView<byte> o, int at, uint v, int end, ref int ovf)
    {
        if ((at >= 0) & (at + 4 <= end))
        {
            o[at] = (byte)v;
            o[at + 1] = (byte)(v >> 8);
            o[at + 2] = (byte)(v >> 16);
            o[at + 3] = (byte)(v >> 24);
        }
        else ovf = 1;
    }

    /// <summary>Exclusive prefix sum over the 32 lanes; <paramref name="total"/> = the sum (two barriers).</summary>
    private static int GdScan(ArrayView<int> w, int lane, int v, ref int total)
    {
        w[GdScanArea + lane] = v;
        Group.Barrier();
        int before = 0, all = 0;
        for (int i = 0; i < GdStreams; i++)
        {
            int x = w[GdScanArea + i];
            all += x;
            if (i < lane) before += x;
        }
        Group.Barrier();
        total = all;
        return before;
    }
}
