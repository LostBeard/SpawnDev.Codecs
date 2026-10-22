// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// DEFLATE (RFC 1951) block compressor, GPU-callable form. One GPU thread compresses one independent block
// of input: LZ77 (hash chains, greedy or lazy) -> dynamic Huffman codes -> bit-packed output, falling back to
// stored blocks when that is smaller. Blocks are independent (no match crosses a block boundary) so any
// number of them compress in parallel, one thread each; a non-final block ends with an empty stored block
// (a "sync flush"), which leaves the stream byte-aligned so the blocks concatenate with a plain byte copy.
//
// The Huffman tree builder is a two-queue merge with a Kraft-sum repair to cap code lengths (15 for the
// literal/length and distance alphabets, 7 for the code-length alphabet). No managed allocations, no tables
// as static arrays: length / distance symbol arithmetic is computed, all scratch is an ArrayView<int> the
// caller owns (see ScratchInts).

using ILGPU;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Compression;

/// <summary>Bit-packing output cursor of one block.</summary>
public struct DeflateBitWriter
{
    /// <summary>Pending bits (low bits are written first).</summary>
    public ulong Acc;
    /// <summary>Number of pending bits in <see cref="Acc"/> (always &lt; 8 between calls).</summary>
    public int Count;
    /// <summary>Next output byte index.</summary>
    public int Pos;
    /// <summary>One past the last writable output byte index.</summary>
    public int End;
    /// <summary>Set to 1 when the output slot was too small.</summary>
    public int Overflow;
}

/// <summary>GPU-callable DEFLATE block compressor.</summary>
public static class DeflateGpu
{
    /// <summary>Hash table size (entries) of the LZ77 matcher.</summary>
    public const int HashSize = 1 << 13;
    /// <summary>Largest distance a match may use (the DEFLATE window).</summary>
    public const int WindowSize = 32768;
    /// <summary>Longest match.</summary>
    public const int MaxMatch = 258;
    /// <summary>Largest allowed block size in input bytes (distances must fit the window).</summary>
    public const int MaxBlockSize = 1 << 20;

    // ---- tail scratch layout (ints), after the hash heads, prev chain and token array ----
    private const int LitFreqOff = 0;        // [288]
    private const int DistFreqOff = 288;     // [32]
    private const int LitLenOff = 320;       // [288]
    private const int DistLenOff = 608;      // [32]
    private const int LitCodeOff = 640;      // [288]
    private const int DistCodeOff = 928;     // [32]
    private const int ClFreqOff = 960;       // [24]
    private const int ClLenOff = 984;        // [24]
    private const int ClCodeOff = 1008;      // [24]
    private const int ClSeqOff = 1032;       // [320] packed run-length symbols
    private const int TreeSortedOff = 1352;  // [288]
    private const int TreeFreqOff = 1640;    // [576]
    private const int TreeParentOff = 2216;  // [576]
    private const int TreeDepthOff = 2792;   // [576]
    private const int TreeCountOff = 3368;   // [16]
    private const int TreeNextOff = 3384;    // [16]
    private const int TailInts = 3400;

    /// <summary>Ints of scratch one block compression needs for blocks of <paramref name="blockSize"/> input bytes.</summary>
    public static int ScratchInts(int blockSize) => HashSize + 2 * blockSize + TailInts;

    /// <summary>Upper bound of the output bytes of one block of <paramref name="blockSize"/> input bytes.</summary>
    public static int MaxBlockOutput(int blockSize) => blockSize + 5 * (blockSize / 65535 + 2) + 16;

    // ------------------------------------------------------------------
    // Bit writer
    // ------------------------------------------------------------------

    private static void PutBits(ref DeflateBitWriter w, ArrayView<byte> o, uint value, int n)
    {
        w.Acc |= (ulong)value << w.Count;
        w.Count += n;
        while (w.Count >= 8)
        {
            if (w.Pos < w.End) o[w.Pos] = (byte)w.Acc;
            else w.Overflow = 1;
            w.Pos++;
            w.Acc >>= 8;
            w.Count -= 8;
        }
    }

    private static void PutByte(ref DeflateBitWriter w, ArrayView<byte> o, int value)
    {
        if (w.Pos < w.End) o[w.Pos] = (byte)value;
        else w.Overflow = 1;
        w.Pos++;
    }

    private static void AlignToByte(ref DeflateBitWriter w, ArrayView<byte> o)
    {
        if (w.Count > 0) PutBits(ref w, o, 0, 8 - w.Count);
    }

    // ------------------------------------------------------------------
    // Symbol arithmetic
    // ------------------------------------------------------------------

    private static int Log2Floor(int x)
    {
        int r = 0;
        while ((x >> (r + 1)) != 0) r++;
        return r;
    }

    /// <summary>Length symbol (257..285) of match length 3..258; extra bits / value via the two helpers.</summary>
    private static int LengthSymbol(int length)
    {
        int l = length - 3;
        int sym;
        if (length == 258)
        {
            sym = 285;
        }
        else if (l < 8)
        {
            sym = 257 + l;
        }
        else
        {
            int e = Log2Floor(l) - 2;
            sym = 257 + 4 * (e + 1) + ((l >> e) & 3);
        }
        return sym;
    }

    private static int LengthExtraBits(int length)
    {
        int l = length - 3;
        int e = 0;
        bool ext = (length != 258) & (l >= 8);
        if (ext) e = Log2Floor(l) - 2;
        return e;
    }

    private static int LengthExtraValue(int length)
    {
        int l = length - 3;
        int v = 0;
        bool ext = (length != 258) & (l >= 8);
        if (ext)
        {
            int e = Log2Floor(l) - 2;
            v = l & ((1 << e) - 1);
        }
        return v;
    }

    private static int DistSymbol(int dist)
    {
        int x = dist - 1;
        int sym;
        if (x < 4)
        {
            sym = x;
        }
        else
        {
            int e = Log2Floor(x) - 1;
            sym = 2 * e + 2 + ((x >> e) & 1);
        }
        return sym;
    }

    private static int DistExtraBits(int dist)
    {
        int x = dist - 1;
        int e = 0;
        if (x >= 4) e = Log2Floor(x) - 1;
        return e;
    }

    private static int DistExtraValue(int dist)
    {
        int x = dist - 1;
        int v = 0;
        if (x >= 4)
        {
            int e = Log2Floor(x) - 1;
            v = x & ((1 << e) - 1);
        }
        return v;
    }

    private static uint ReverseBits(uint code, int length)
    {
        uint r = 0;
        for (int i = 0; i < length; i++)
        {
            r = (r << 1) | (code & 1u);
            code >>= 1;
        }
        return r;
    }

    // ------------------------------------------------------------------
    // Huffman code construction
    // ------------------------------------------------------------------

    /// <summary>
    /// Length-limited Huffman code lengths for <paramref name="n"/> symbols whose frequencies start at
    /// <paramref name="freqAt"/>; lengths are written to <paramref name="lensAt"/> (0 = unused symbol).
    /// </summary>
    private static void BuildLengths(
        ArrayView<int> s, int freqAt, int n, int maxBits, int lensAt, int tail)
    {
        int sortedAt = tail + TreeSortedOff;
        int nodeFreqAt = tail + TreeFreqOff;
        int parentAt = tail + TreeParentOff;
        int depthAt = tail + TreeDepthOff;
        int countAt = tail + TreeCountOff;

        int m = 0;
        for (int sym = 0; sym < n; sym++)
        {
            s[lensAt + sym] = 0;
            if (s[freqAt + sym] > 0)
            {
                s[sortedAt + m] = sym;
                m++;
            }
        }
        if (m == 0) return;
        if (m == 1)
        {
            s[lensAt + s[sortedAt]] = 1;
            return;
        }

        // Insertion sort, ascending by (freq, symbol).
        for (int i = 1; i < m; i++)
        {
            int key = s[sortedAt + i];
            int kf = s[freqAt + key];
            int j = i - 1;
            int moving = 1;
            while (moving != 0)
            {
                if (j < 0)
                {
                    moving = 0;
                }
                else
                {
                    int other = s[sortedAt + j];
                    int of = s[freqAt + other];
                    bool greater = (of > kf) | ((of == kf) & (other > key));
                    if (greater)
                    {
                        s[sortedAt + j + 1] = other;
                        j--;
                    }
                    else
                    {
                        moving = 0;
                    }
                }
            }
            s[sortedAt + j + 1] = key;
        }

        // Two-queue merge: leaves are nodes 0..m-1 (ascending weight), internal nodes m..2m-2 are
        // created in non-decreasing weight order.
        for (int i = 0; i < m; i++) s[nodeFreqAt + i] = s[freqAt + s[sortedAt + i]];
        int leaf = 0;
        int inner = m;
        int next = m;
        for (int k = 0; k < m - 1; k++)
        {
            int a;
            bool takeLeafA = (leaf < m) & ((inner >= next) | (s[nodeFreqAt + leaf] <= s[nodeFreqAt + inner]));
            if (takeLeafA) { a = leaf; leaf++; } else { a = inner; inner++; }
            int b;
            bool takeLeafB = (leaf < m) & ((inner >= next) | (s[nodeFreqAt + leaf] <= s[nodeFreqAt + inner]));
            if (takeLeafB) { b = leaf; leaf++; } else { b = inner; inner++; }
            s[nodeFreqAt + next] = s[nodeFreqAt + a] + s[nodeFreqAt + b];
            s[parentAt + a] = next;
            s[parentAt + b] = next;
            next++;
        }

        // Depths (parents have higher indices than their children).
        int root = 2 * m - 2;
        s[depthAt + root] = 0;
        for (int i = root - 1; i >= 0; i--)
            s[depthAt + i] = s[depthAt + s[parentAt + i]] + 1;

        // Count leaves per depth, clamping to maxBits, then repair the Kraft sum.
        for (int len = 0; len <= 15; len++) s[countAt + len] = 0;
        for (int i = 0; i < m; i++)
        {
            int d = s[depthAt + i];
            if (d > maxBits) d = maxBits;
            s[countAt + d] = s[countAt + d] + 1;
        }
        int total = 0;
        for (int len = maxBits; len > 0; len--)
            total += s[countAt + len] << (maxBits - len);
        while (total != (1 << maxBits))
        {
            s[countAt + maxBits] = s[countAt + maxBits] - 1;
            int fixedUp = 0;
            for (int len = maxBits - 1; len > 0; len--)
            {
                bool canFix = (fixedUp == 0) & (s[countAt + len] != 0);
                if (canFix)
                {
                    s[countAt + len] = s[countAt + len] - 1;
                    s[countAt + len + 1] = s[countAt + len + 1] + 2;
                    fixedUp = 1;
                }
            }
            total--;
        }

        // Most frequent symbols (the end of the ascending order) get the shortest codes.
        int idx = m - 1;
        for (int len = 1; len <= maxBits; len++)
        {
            int c = s[countAt + len];
            for (int k = 0; k < c; k++)
            {
                s[lensAt + s[sortedAt + idx]] = len;
                idx--;
            }
        }
    }

    /// <summary>Canonical codes (bit-reversed, ready to emit LSB first) for the lengths at <paramref name="lensAt"/>.</summary>
    private static void AssignCodes(ArrayView<int> s, int lensAt, int n, int codesAt, int tail)
    {
        int countAt = tail + TreeCountOff;
        int nextAt = tail + TreeNextOff;
        for (int len = 0; len <= 15; len++) s[countAt + len] = 0;
        for (int sym = 0; sym < n; sym++)
        {
            int l = s[lensAt + sym];
            s[countAt + l] = s[countAt + l] + 1;
        }
        s[countAt] = 0;
        int code = 0;
        s[nextAt] = 0;
        for (int len = 1; len <= 15; len++)
        {
            code = (code + s[countAt + len - 1]) << 1;
            s[nextAt + len] = code;
        }
        for (int sym = 0; sym < n; sym++)
        {
            int l = s[lensAt + sym];
            if (l != 0)
            {
                int c = s[nextAt + l];
                s[nextAt + l] = c + 1;
                s[codesAt + sym] = (int)ReverseBits((uint)c, l);
            }
            else
            {
                s[codesAt + sym] = 0;
            }
        }
    }

    // ------------------------------------------------------------------
    // Block compression
    // ------------------------------------------------------------------

    private static int Hash3(ArrayView<byte> d, int p)
    {
        uint v = ((uint)d[p] << 16) | ((uint)d[p + 1] << 8) | d[p + 2];
        return (int)((v * 2654435761u) >> (32 - 13));
    }

    /// <summary>
    /// Compress input bytes [<paramref name="start"/>, <paramref name="start"/> + <paramref name="len"/>) of
    /// <paramref name="data"/> into <paramref name="o"/> at <paramref name="outStart"/> (at most
    /// <paramref name="outCap"/> bytes). Returns the output byte count, or -1 when the slot was too small.
    /// <paramref name="level"/>: 0 = stored only, 1 = greedy (chain 8), 2 = greedy (chain 32), 3 = lazy (chain 64).
    /// </summary>
    public static int CompressBlock(
        ArrayView<byte> data, int start, int len, bool isFinal, int level,
        ArrayView<byte> o, int outStart, int outCap,
        ArrayView<int> s, int sb, int blockSize)
    {
        var w = new DeflateBitWriter { Acc = 0, Count = 0, Pos = outStart, End = outStart + outCap, Overflow = 0 };

        if (len == 0)
        {
            // An empty final block: fixed Huffman, only the end-of-block code (7 zero bits).
            PutBits(ref w, o, isFinal ? 1u : 0u, 1);
            PutBits(ref w, o, 1, 2);
            PutBits(ref w, o, 0, 7);
            if (isFinal) AlignToByte(ref w, o); else EmitSyncFlush(ref w, o);
            return w.Overflow != 0 ? -1 : w.Pos - outStart;
        }

        bool stored = level <= 0;
        int headAt = sb;
        int prevAt = sb + HashSize;
        int tokAt = prevAt + blockSize;
        int tail = tokAt + blockSize;
        int tokCount = 0;
        long dynBits = 0;
        int hlit = 257;
        int hdist = 1;
        int hclen = 4;
        int seqCount = 0;

        if (!stored)
        {
            tokCount = Lz77Parse(data, start, len, level, s, headAt, prevAt, tokAt, tail);
            long extraBits = 0;
            for (int t = 0; t < tokCount; t++)
            {
                int tok = s[tokAt + t];
                if (tok >= 256)
                    extraBits += LengthExtraBits(tok >> 16) + DistExtraBits(tok & 0xFFFF);
            }
            s[tail + LitFreqOff + 256] = 1;

            // ---- code lengths ----
            BuildLengths(s, tail + LitFreqOff, 286, 15, tail + LitLenOff, tail);
            BuildLengths(s, tail + DistFreqOff, 30, 15, tail + DistLenOff, tail);
            hlit = 286;
            bool trimLit = (hlit > 257) & (s[tail + LitLenOff + hlit - 1] == 0);
            while (trimLit)
            {
                hlit--;
                trimLit = (hlit > 257) & (s[tail + LitLenOff + hlit - 1] == 0);
            }
            hdist = 30;
            bool trimDist = (hdist > 1) & (s[tail + DistLenOff + hdist - 1] == 0);
            while (trimDist)
            {
                hdist--;
                trimDist = (hdist > 1) & (s[tail + DistLenOff + hdist - 1] == 0);
            }

            // ---- run-length code the length sequences ----
            for (int i = 0; i < 24; i++) s[tail + ClFreqOff + i] = 0;
            int total = hlit + hdist;
            int ix = 0;
            while (ix < total)
            {
                int cur = ix < hlit ? s[tail + LitLenOff + ix] : s[tail + DistLenOff + (ix - hlit)];
                int run = 1;
                int scanning = 1;
                while (scanning != 0)
                {
                    int at = ix + run;
                    if (at >= total)
                    {
                        scanning = 0;
                    }
                    else
                    {
                        int v = at < hlit ? s[tail + LitLenOff + at] : s[tail + DistLenOff + (at - hlit)];
                        if (v == cur) run++; else scanning = 0;
                    }
                }
                int r = run;
                if (cur == 0)
                {
                    while (r >= 11)
                    {
                        int n = r < 138 ? r : 138;
                        seqCount = AddClSymbol(s, tail, seqCount, 18, n - 11, 7);
                        r -= n;
                    }
                    if (r >= 3)
                    {
                        seqCount = AddClSymbol(s, tail, seqCount, 17, r - 3, 3);
                        r = 0;
                    }
                    while (r > 0)
                    {
                        seqCount = AddClSymbol(s, tail, seqCount, 0, 0, 0);
                        r--;
                    }
                }
                else
                {
                    seqCount = AddClSymbol(s, tail, seqCount, cur, 0, 0);
                    r--;
                    while (r >= 3)
                    {
                        int n = r < 6 ? r : 6;
                        seqCount = AddClSymbol(s, tail, seqCount, 16, n - 3, 2);
                        r -= n;
                    }
                    while (r > 0)
                    {
                        seqCount = AddClSymbol(s, tail, seqCount, cur, 0, 0);
                        r--;
                    }
                }
                ix += run;
            }

            // The code-length code must be complete: never leave it with a single used symbol.
            int used = 0;
            for (int i = 0; i < 19; i++) if (s[tail + ClFreqOff + i] > 0) used++;
            if (used == 1)
            {
                int dummy = s[tail + ClFreqOff + 0] > 0 ? 1 : 0;
                s[tail + ClFreqOff + dummy] = 1;
            }
            BuildLengths(s, tail + ClFreqOff, 19, 7, tail + ClLenOff, tail);
            hclen = 19;
            bool trimCl = (hclen > 4) & (s[tail + ClLenOff + InflateGpu.CodeLengthOrder(hclen - 1)] == 0);
            while (trimCl)
            {
                hclen--;
                trimCl = (hclen > 4) & (s[tail + ClLenOff + InflateGpu.CodeLengthOrder(hclen - 1)] == 0);
            }

            AssignCodes(s, tail + LitLenOff, 288, tail + LitCodeOff, tail);
            AssignCodes(s, tail + DistLenOff, 32, tail + DistCodeOff, tail);
            AssignCodes(s, tail + ClLenOff, 19, tail + ClCodeOff, tail);

            // ---- exact dynamic-block size in bits ----
            dynBits = 3 + 5 + 5 + 4 + 3L * hclen;
            for (int i = 0; i < seqCount; i++)
            {
                int packed = s[tail + ClSeqOff + i];
                dynBits += s[tail + ClLenOff + (packed & 31)] + ((packed >> 12) & 7);
            }
            for (int sym = 0; sym < 286; sym++)
                dynBits += (long)s[tail + LitFreqOff + sym] * s[tail + LitLenOff + sym];
            for (int sym = 0; sym < 30; sym++)
                dynBits += (long)s[tail + DistFreqOff + sym] * s[tail + DistLenOff + sym];
            dynBits += extraBits;

            long storedBits = 8L * (len + 5L * ((len + 65534) / 65535));
            if (dynBits >= storedBits) stored = true;
        }

        if (stored)
        {
            int done = 0;
            while (done < len)
            {
                int chunk = len - done;
                if (chunk > 65535) chunk = 65535;
                bool last = (done + chunk == len);
                PutBits(ref w, o, (isFinal & last) ? 1u : 0u, 1);
                PutBits(ref w, o, 0, 2);
                AlignToByte(ref w, o);
                PutByte(ref w, o, chunk & 0xFF);
                PutByte(ref w, o, chunk >> 8);
                PutByte(ref w, o, ~chunk & 0xFF);
                PutByte(ref w, o, (~chunk >> 8) & 0xFF);
                for (int i = 0; i < chunk; i++) PutByte(ref w, o, data[start + done + i]);
                done += chunk;
            }
            // A stored block already ends byte-aligned; nothing more is needed to concatenate.
            return w.Overflow != 0 ? -1 : w.Pos - outStart;
        }

        // ---- emit the dynamic block ----
        PutBits(ref w, o, isFinal ? 1u : 0u, 1);
        PutBits(ref w, o, 2, 2);
        PutBits(ref w, o, (uint)(hlit - 257), 5);
        PutBits(ref w, o, (uint)(hdist - 1), 5);
        PutBits(ref w, o, (uint)(hclen - 4), 4);
        for (int i = 0; i < hclen; i++)
            PutBits(ref w, o, (uint)s[tail + ClLenOff + InflateGpu.CodeLengthOrder(i)], 3);
        for (int i = 0; i < seqCount; i++)
        {
            int packed = s[tail + ClSeqOff + i];
            int sym = packed & 31;
            PutBits(ref w, o, (uint)s[tail + ClCodeOff + sym], s[tail + ClLenOff + sym]);
            int eb = (packed >> 12) & 7;
            if (eb > 0) PutBits(ref w, o, (uint)((packed >> 5) & 127), eb);
        }
        for (int t = 0; t < tokCount; t++)
        {
            int tok = s[tokAt + t];
            if (tok < 256)
            {
                PutBits(ref w, o, (uint)s[tail + LitCodeOff + tok], s[tail + LitLenOff + tok]);
            }
            else
            {
                int ml = tok >> 16;
                int md = tok & 0xFFFF;
                int ls = LengthSymbol(ml);
                PutBits(ref w, o, (uint)s[tail + LitCodeOff + ls], s[tail + LitLenOff + ls]);
                int leb = LengthExtraBits(ml);
                if (leb > 0) PutBits(ref w, o, (uint)LengthExtraValue(ml), leb);
                int ds = DistSymbol(md);
                PutBits(ref w, o, (uint)s[tail + DistCodeOff + ds], s[tail + DistLenOff + ds]);
                int deb = DistExtraBits(md);
                if (deb > 0) PutBits(ref w, o, (uint)DistExtraValue(md), deb);
            }
        }
        PutBits(ref w, o, (uint)s[tail + LitCodeOff + 256], s[tail + LitLenOff + 256]);
        if (isFinal) AlignToByte(ref w, o);
        else EmitSyncFlush(ref w, o);
        return w.Overflow != 0 ? -1 : w.Pos - outStart;
    }

    // ------------------------------------------------------------------
    // LZ77
    // ------------------------------------------------------------------

    private static void InsertHash(ArrayView<byte> data, int start, int p, int len, ArrayView<int> s, int headAt, int prevAt)
    {
        if (p + 3 <= len)
        {
            int h = Hash3(data, start + p);
            s[prevAt + p] = s[headAt + h];
            s[headAt + h] = p;
        }
    }

    /// <summary>Best match at <paramref name="pos"/>: (length &lt;&lt; 16) | distance, or 0 when none (length &lt; 3).</summary>
    private static int FindMatch(
        ArrayView<byte> data, int start, int pos, int len, ArrayView<int> s, int headAt, int prevAt,
        int maxChain, int niceLen)
    {
        int bestLen = 0;
        int bestDist = 0;
        if (pos + 3 <= len)
        {
            int cand = s[headAt + Hash3(data, start + pos)];
            int chain = maxChain;
            int maxl = len - pos;
            if (maxl > MaxMatch) maxl = MaxMatch;
            int searching = 1;
            while (searching != 0)
            {
                bool stop = (cand < 0) | (chain <= 0);
                if (stop)
                {
                    searching = 0;
                }
                else
                {
                    int dist = pos - cand;
                    if (dist > WindowSize)
                    {
                        searching = 0;
                    }
                    else
                    {
                        int ml = 0;
                        int go = 1;
                        while (go != 0)
                        {
                            if (ml >= maxl) go = 0;
                            else if (data[start + cand + ml] != data[start + pos + ml]) go = 0;
                            else ml++;
                        }
                        // A 3-byte match from far away costs more bits than three literals.
                        bool worthIt = (ml > bestLen) & !((ml == 3) & (dist > 4096));
                        if (worthIt)
                        {
                            bestLen = ml;
                            bestDist = dist;
                        }
                        if (bestLen >= niceLen) searching = 0;
                        cand = s[prevAt + cand];
                        chain--;
                    }
                }
            }
        }
        if (bestLen < 3) return 0;
        return (bestLen << 16) | bestDist;
    }

    private static int AddLiteral(ArrayView<int> s, int tokAt, int tail, int n, int lit)
    {
        s[tokAt + n] = lit;
        s[tail + LitFreqOff + lit] = s[tail + LitFreqOff + lit] + 1;
        return n + 1;
    }

    private static int AddMatch(ArrayView<int> s, int tokAt, int tail, int n, int length, int dist)
    {
        s[tokAt + n] = (length << 16) | dist;
        int ls = LengthSymbol(length);
        int ds = DistSymbol(dist);
        s[tail + LitFreqOff + ls] = s[tail + LitFreqOff + ls] + 1;
        s[tail + DistFreqOff + ds] = s[tail + DistFreqOff + ds] + 1;
        return n + 1;
    }

    /// <summary>
    /// LZ77-parses the block into the token array (literal = value &lt; 256, match = (length &lt;&lt; 16) | distance)
    /// and counts the literal/length and distance symbol frequencies. Returns the token count.
    /// </summary>
    private static int Lz77Parse(
        ArrayView<byte> data, int start, int len, int level, ArrayView<int> s,
        int headAt, int prevAt, int tokAt, int tail)
    {
        int maxChain = level == 1 ? 8 : (level == 2 ? 32 : 64);
        bool lazy = level >= 3;
        int niceLen = level == 1 ? 32 : (level == 2 ? 128 : 258);

        for (int i = 0; i < HashSize; i++) s[headAt + i] = -1;
        for (int i = 0; i < 288; i++) s[tail + LitFreqOff + i] = 0;
        for (int i = 0; i < 32; i++) s[tail + DistFreqOff + i] = 0;

        int n = 0;
        int pos = 0;
        int pendingLen = 0;   // lazy: a match found at pos - 1 that has not been emitted yet
        int pendingDist = 0;
        while (pos < len)
        {
            int found = FindMatch(data, start, pos, len, s, headAt, prevAt, maxChain, niceLen);
            int bestLen = found >> 16;
            int bestDist = found & 0xFFFF;
            if (!lazy)
            {
                if (bestLen >= 3)
                {
                    n = AddMatch(s, tokAt, tail, n, bestLen, bestDist);
                    for (int p = pos; p < pos + bestLen; p++) InsertHash(data, start, p, len, s, headAt, prevAt);
                    pos += bestLen;
                }
                else
                {
                    n = AddLiteral(s, tokAt, tail, n, data[start + pos]);
                    InsertHash(data, start, pos, len, s, headAt, prevAt);
                    pos++;
                }
            }
            else if (pendingLen > 0)
            {
                if (bestLen > pendingLen)
                {
                    // A longer match starts here: the previous byte becomes a literal.
                    n = AddLiteral(s, tokAt, tail, n, data[start + pos - 1]);
                    pendingLen = bestLen;
                    pendingDist = bestDist;
                    InsertHash(data, start, pos, len, s, headAt, prevAt);
                    pos++;
                }
                else
                {
                    // Take the pending match (it started at pos - 1).
                    n = AddMatch(s, tokAt, tail, n, pendingLen, pendingDist);
                    int matchEnd = pos - 1 + pendingLen;
                    for (int p = pos; p < matchEnd; p++) InsertHash(data, start, p, len, s, headAt, prevAt);
                    pos = matchEnd;
                    pendingLen = 0;
                }
            }
            else if (bestLen >= 3)
            {
                pendingLen = bestLen;
                pendingDist = bestDist;
                InsertHash(data, start, pos, len, s, headAt, prevAt);
                pos++;
            }
            else
            {
                n = AddLiteral(s, tokAt, tail, n, data[start + pos]);
                InsertHash(data, start, pos, len, s, headAt, prevAt);
                pos++;
            }
        }
        if (pendingLen > 0)
            n = AddMatch(s, tokAt, tail, n, pendingLen, pendingDist);
        return n;
    }

    /// <summary>Empty stored block: byte-aligns the stream so the next block can be appended by a byte copy.</summary>
    private static void EmitSyncFlush(ref DeflateBitWriter w, ArrayView<byte> o)
    {
        PutBits(ref w, o, 0, 1);   // BFINAL = 0
        PutBits(ref w, o, 0, 2);   // BTYPE = stored
        AlignToByte(ref w, o);
        PutByte(ref w, o, 0x00);
        PutByte(ref w, o, 0x00);
        PutByte(ref w, o, 0xFF);
        PutByte(ref w, o, 0xFF);
    }

    private static int AddClSymbol(ArrayView<int> s, int tail, int count, int sym, int extraValue, int extraBits)
    {
        s[tail + ClSeqOff + count] = sym | (extraValue << 5) | (extraBits << 12);
        s[tail + ClFreqOff + sym] = s[tail + ClFreqOff + sym] + 1;
        return count + 1;
    }

    /// <summary>Adler-32 components (a, b; both &lt; 65521) of a byte range, from the initial state a = 1, b = 0.</summary>
    public static uint AdlerOfRange(ArrayView<byte> data, int start, int len)
    {
        uint a = 1;
        uint b = 0;
        int pos = start;
        int remaining = len;
        while (remaining > 0)
        {
            int n = remaining < 5552 ? remaining : 5552;
            for (int i = 0; i < n; i++)
            {
                a += data[pos + i];
                b += a;
            }
            a %= 65521u;
            b %= 65521u;
            pos += n;
            remaining -= n;
        }
        return (b << 16) | a;
    }
}
