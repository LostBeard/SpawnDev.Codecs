// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GDeflate tile emission for DeflateGpu (Microsoft DirectStorage GDeflate; format and writer from libdeflate's
// GDEFLATE paths, deflate_compress.c, NVIDIA / Microsoft, Apache 2.0). A tile (at most 64 KB of input) is parsed by
// ParseStep exactly like a DEFLATE block; PlanGDeflate builds its codes (Deflate64 alphabet: length 258 is symbol 284
// with 5 extra bits, since 285 means 3 + 16 extra bits) and decides stored or dynamic; EmitGDeflate writes the tile as
// one GDeflate stream:
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

    /// <summary>
    /// After the parse: the tile's GDeflate codes (Deflate64 length symbols) and the stored-or-dynamic decision, from the
    /// exact sizes of both encodings.
    /// </summary>
    public static void PlanGDeflate(int len, int level, ArrayView<int> s, int sb, int blockSize)
    {
        int tail = sb + HashSize + 2 * blockSize;
        int st = tail + StateOff;
        if (s[st + StPhase] != PhaseParsed) return;
        bool stored = level <= 0;
        if (!stored)
        {
            // Deflate64 matches reach 65538 bytes: consecutive matches at the same distance are one copy (byte-wise
            // copy semantics), merged up to GdMaxMerged (the token keeps the length in its upper 15 bits). Then the
            // frequencies are counted again with the Deflate64 length symbols.
            int tokAt = sb + HashSize + blockSize;
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
            for (int i = 0; i < 288; i++) s[tail + LitFreqOff + i] = 0;
            for (int i = 0; i < 32; i++) s[tail + DistFreqOff + i] = 0;
            int extra = 0;
            for (int t = 0; t < w; t++)
            {
                int tok = s[tokAt + t];
                if (tok < 256) s[tail + LitFreqOff + tok] = s[tail + LitFreqOff + tok] + 1;
                else
                {
                    int ml = tok >> 16;
                    int md = tok & 0xFFFF;
                    int ls = GdLengthSymbol(ml);
                    s[tail + LitFreqOff + ls] = s[tail + LitFreqOff + ls] + 1;
                    int ds = DistSymbol(md);
                    s[tail + DistFreqOff + ds] = s[tail + DistFreqOff + ds] + 1;
                    extra += GdLengthExtraBits(ml) + DistExtraBits(md);
                }
            }
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

    /// <summary>
    /// Writes the planned tile as one GDeflate stream at <paramref name="outStart"/> (at most <paramref name="outCap"/>
    /// bytes); its length (or -1 on overflow) goes to the block output slot (<see cref="BlockOutput"/>).
    /// </summary>
    public static void EmitGDeflate(ArrayView<byte> data, int start, int len, ArrayView<byte> o, int outStart, int outCap,
        ArrayView<int> s, int sb, int blockSize)
    {
        int tokAt = sb + HashSize + blockSize;
        int tail = tokAt + blockSize;
        int st = tail + StateOff;
        if (s[st + StPhase] != PhaseEmitTokens) return;
        int g = sb + ScratchInts(blockSize);
        GdInit(s, g, o, outStart, outStart + outCap);

        int mode = s[st + StGdStored];
        if (mode == GdModeStored)
        {
            int done = 0;
            bool more = true;
            while (more)
            {
                int chunk = len - done;
                if (chunk > 65535) chunk = 65535;
                bool last = done + chunk == len;
                s[g + GdIdx] = 0;
                GdAddBits(s, g, o, last ? 1u : 0u, 1);
                GdAddBits(s, g, o, 0, 2);
                GdAddBits(s, g, o, (uint)chunk, 16);
                for (int i = 0; i < chunk; i++)
                {
                    GdAddBits(s, g, o, data[start + done + i], 8);
                    GdAdvance(s, g);
                }
                done += chunk;
                more = done < len;
            }
        }
        else
        {
            // Block header: final, static (1) or dynamic (2).
            s[g + GdIdx] = 0;
            GdAddBits(s, g, o, 1, 1);
            GdAddBits(s, g, o, mode == GdModeStatic ? 1u : 2u, 2);
            if (mode == GdModeDynamic)
            {
                // The Huffman header: precode lengths one per sub-stream, then the code-length symbols.
                int hlit = s[st + StHlit];
                int hdist = s[st + StHdist];
                int hclen = s[st + StHclen];
                int seqCount = s[st + StSeq];
                s[g + GdIdx] = 0;
                GdAddBits(s, g, o, (uint)(hlit - 257), 5);
                GdAddBits(s, g, o, (uint)(hdist - 1), 5);
                GdAddBits(s, g, o, (uint)(hclen - 4), 4);
                for (int i = 0; i < hclen; i++)
                {
                    GdAddBits(s, g, o, (uint)s[tail + ClLenOff + InflateGpu.CodeLengthOrder(i)], 3);
                    GdAdvance(s, g);
                }
                s[g + GdIdx] = 0;
                for (int i = 0; i < seqCount; i++)
                {
                    int packed = s[tail + ClSeqOff + i];
                    int sym = packed & 31;
                    GdAddBits(s, g, o, (uint)s[tail + ClCodeOff + sym], s[tail + ClLenOff + sym]);
                    int eb = (packed >> 12) & 7;
                    if (eb > 0) GdAddBits(s, g, o, (uint)((packed >> 5) & 127), eb);
                    GdAdvance(s, g);
                }
            }

            // The symbols: a match's distance goes to its sub-stream one round later.
            for (int n = 0; n < GdStreams; n++) s[g + GdCopyRound + n] = 0;
            int round = 1;
            s[g + GdIdx] = 0;
            int tokCount = s[st + StTokens];
            for (int t = 0; t < tokCount; t++)
            {
                round = GdPrevRoundCopies(s, g, o, round);
                int tok = s[tokAt + t];
                if (tok < 256)
                {
                    GdAddBits(s, g, o, (uint)s[tail + LitCodeOff + tok], s[tail + LitLenOff + tok]);
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
                    GdAddBits(s, g, o, (uint)s[tail + LitCodeOff + ls], s[tail + LitLenOff + ls]);
                    if (leb > 0) GdAddBits(s, g, o, (uint)lev, leb);
                    int ds = DistSymbol(md);
                    int idx = s[g + GdIdx];
                    int dlen = s[tail + DistLenOff + ds];
                    s[g + GdCopyBits + idx] = (int)((uint)s[tail + DistCodeOff + ds] | ((uint)DistExtraValue(md) << dlen));
                    s[g + GdCopyCount + idx] = dlen + DistExtraBits(md);
                    s[g + GdCopyRound + idx] = round;
                }
                round += GdAdvance(s, g);
            }
            round = GdPrevRoundCopies(s, g, o, round);
            GdAddBits(s, g, o, (uint)s[tail + LitCodeOff + 256], s[tail + LitLenOff + 256]);
            round += GdAdvance(s, g);
            GdTailCopies(s, g, o, round);
        }

        // The last packet of every sub-stream.
        for (int n = 0; n < GdStreams; n++)
        {
            GdFlushPacket(s, g, o);
            GdAdvance(s, g);
        }
        int outLen = s[g + GdNextPos] - outStart;
        s[st + StOutLen] = s[g + GdOverflow] != 0 ? -1 : outLen;
        s[st + StPhase] = PhaseDone;
    }

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

    /// <summary>Writes the distances deferred from the previous round while the current sub-stream has one (each takes a slot).</summary>
    private static int GdPrevRoundCopies(ArrayView<int> s, int g, ArrayView<byte> o, int round)
    {
        bool going = true;
        while (going)
        {
            int idx = s[g + GdIdx];
            going = (round > 1) & (s[g + GdCopyRound + idx] == round - 1);
            if (going)
            {
                s[g + GdCopyRound + idx] = 0;
                GdAddBits(s, g, o, (uint)s[g + GdCopyBits + idx], s[g + GdCopyCount + idx]);
                round += GdAdvance(s, g);
            }
        }
        return round;
    }

    /// <summary>The distances still pending after the end-of-block symbol: the previous round's from the current sub-stream on, then the current round's.</summary>
    private static void GdTailCopies(ArrayView<int> s, int g, ArrayView<byte> o, int round)
    {
        int split = s[g + GdIdx];
        for (int n = split; n < GdStreams; n++)
        {
            bool prev = (round > 1) & (s[g + GdCopyRound + n] == round - 1);
            if (prev)
            {
                s[g + GdIdx] = n;
                GdAddBits(s, g, o, (uint)s[g + GdCopyBits + n], s[g + GdCopyCount + n]);
            }
        }
        for (int n = 0; n < split; n++)
        {
            if (s[g + GdCopyRound + n] == round)
            {
                s[g + GdIdx] = n;
                GdAddBits(s, g, o, (uint)s[g + GdCopyBits + n], s[g + GdCopyCount + n]);
            }
        }
    }
}
