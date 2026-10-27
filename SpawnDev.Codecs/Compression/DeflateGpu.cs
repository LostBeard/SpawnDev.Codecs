// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// DEFLATE (RFC 1951) block compressor, GPU-callable form. One GPU thread compresses one independent block
// of input: LZ77 (hash chains, greedy or lazy) -> dynamic Huffman codes -> bit-packed output, falling back to
// stored blocks when that is smaller. Blocks are independent (no match crosses a block boundary) so any
// number of them compress in parallel, one thread each; a non-final block ends with an empty stored block
// (a "sync flush"), which leaves the stream byte-aligned so the blocks concatenate with a plain byte copy.
//
// A block is compressed in RESUMABLE phases whose state lives in the block's scratch (see the St* slots), so no
// launch holds the GPU long whatever the block size (a long launch stalls the desktop compositor and trips the
// OS GPU watchdog):
//   InitBlockThread  (1 thread per hash slot)  clears the hash heads and the frequencies.
//   ParseStep        (resumable)  the LZ77 parse, ParseChunk positions per launch; greedy matching (levels 1-2) costs
//                    at most 4 * chain compares per position, lazy matching (level 3) also stops a launch after
//                    ParseCompareBudget compares (its host loop then checks for unfinished blocks).
//   PlanBlock        (1 thread per block)  Huffman code lengths, the stored-or-dynamic decision, the block header.
//   EmitStep         (resumable)  EmitChunk tokens (or stored bytes) per launch, then the end of the block.
// CompressBlock runs the same phases in one call. The bytes produced are the same either way.
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
public static partial class DeflateGpu
{
    /// <summary>Hash table size (entries) of the LZ77 matcher.</summary>
    public const int HashSize = 1 << 13;
    /// <summary>Largest distance a match may use (the DEFLATE window).</summary>
    public const int WindowSize = 32768;
    /// <summary>Longest match.</summary>
    public const int MaxMatch = 258;
    /// <summary>
    /// Level of a Huffman-only block (zlib's Z_HUFFMAN_ONLY strategy): every byte a literal, no LZ77, dynamic Huffman codes
    /// of the byte frequencies (or stored when that is smaller). The right coder for data whose redundancy is only in the
    /// byte distribution, e.g. a byte plane of float weights.
    /// </summary>
    public const int LevelHuffmanOnly = 4;

    /// <summary>Largest allowed block size in input bytes (distances must fit the window).</summary>
    public const int MaxBlockSize = 1 << 20;

    /// <summary>Input positions one <see cref="ParseStep"/> launch parses at most (plus the rest of a match).</summary>
    public const int ParseChunk = 512;
    /// <summary>
    /// Byte compares after which one lazy (level 3) <see cref="ParseStep"/> launch stops even before
    /// <see cref="ParseChunk"/> positions (a few milliseconds of compares on a desktop GPU). Greedy matching needs no such
    /// stop: it makes at most 4 * chain compares per position it advances.
    /// </summary>
    public const int ParseCompareBudget = 1 << 18;
    /// <summary>Tokens (or stored bytes) one <see cref="EmitStep"/> launch writes at most.</summary>
    public const int EmitChunk = 4096;

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
    private const int StateOff = 3400;       // [24]  resumable block state (St* slots)
    private const int TailInts = 3424;

    // Slots of the resumable block state (relative to the tail's StateOff).
    private const int StPos = 0, StTokens = 1, StPendLen = 2, StPendDist = 3, StExtra = 4, StPhase = 5, StSeq = 6,
        StCursor = 7, StAcc = 8, StCount = 9, StWPos = 10, StOverflow = 11, StOutLen = 12, StWEnd = 13,
        StHlit = 14, StHdist = 15, StHclen = 16, StGdStored = 17;
    // Block phases: parsing, parsed (codes not built yet), emitting tokens, emitting stored bytes, done.
    private const int PhaseParse = 0, PhaseParsed = 1, PhaseEmitTokens = 2, PhaseEmitStored = 3, PhaseDone = 4;

    /// <summary>Ints of scratch one block compression needs for blocks of <paramref name="blockSize"/> input bytes.</summary>
    public static int ScratchInts(int blockSize) => HashSize + 2 * blockSize + TailInts;

    /// <summary>Upper bound of the output bytes of one block of <paramref name="blockSize"/> input bytes.</summary>
    public static int MaxBlockOutput(int blockSize) => blockSize + 5 * (blockSize / 65535 + 2) + 16;

    /// <summary>Launches of <see cref="ParseStep"/> that finish every greedy (level 1-2) parse of blocks of <paramref name="blockSize"/> bytes.</summary>
    public static int ParseSteps(int blockSize) => (blockSize + ParseChunk - 1) / ParseChunk;

    /// <summary>Launches of <see cref="EmitStep"/> that finish every block of <paramref name="blockSize"/> bytes.</summary>
    public static int EmitSteps(int blockSize) => (blockSize + EmitChunk - 1) / EmitChunk;

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
    /// <paramref name="level"/>: 0 = stored only, 1 = greedy (chain 8), 2 = greedy (chain 32), 3 = lazy (chain 64),
    /// <see cref="LevelHuffmanOnly"/> = literals only.
    /// The whole block in one call (one thread); the kernels run the same phases as bounded launches.
    /// </summary>
    public static int CompressBlock(
        ArrayView<byte> data, int start, int len, bool isFinal, int level,
        ArrayView<byte> o, int outStart, int outCap,
        ArrayView<int> s, int sb, int blockSize)
    {
        for (int k = 0; k < HashSize; k++) InitBlockThread(k, len, level, s, sb, blockSize);
        ParseStep(data, start, len, level, s, sb, blockSize, int.MaxValue, int.MaxValue);
        PlanBlock(data, start, len, isFinal, level, o, outStart, outCap, s, sb, blockSize);
        EmitStep(data, start, len, isFinal, o, outStart, s, sb, blockSize, int.MaxValue);
        return BlockOutput(s, sb, blockSize);
    }

    /// <summary>The block's output byte count (or -1 when the slot was too small) once <see cref="EmitStep"/> has finished it.</summary>
    public static int BlockOutput(ArrayView<int> s, int sb, int blockSize) => s[sb + HashSize + 2 * blockSize + StateOff + StOutLen];

    /// <summary>
    /// Thread <paramref name="k"/> (0 .. <see cref="HashSize"/> - 1) of the block set-up: clears hash head k and the
    /// frequencies, and (thread 0) resets the resumable state. Levels &lt;= 0 and empty blocks skip the parse.
    /// </summary>
    public static void InitBlockThread(int k, int len, int level, ArrayView<int> s, int sb, int blockSize)
    {
        int tail = sb + HashSize + 2 * blockSize;
        s[sb + k] = -1;
        if (k < 288) s[tail + LitFreqOff + k] = 0;
        if (k < 32) s[tail + DistFreqOff + k] = 0;
        if (k == 0)
        {
            int st = tail + StateOff;
            s[st + StPos] = 0;
            s[st + StTokens] = 0;
            s[st + StPendLen] = 0;
            s[st + StPendDist] = 0;
            s[st + StExtra] = 0;
            bool skipParse = (level <= 0) | (len == 0);
            s[st + StPhase] = skipParse ? PhaseParsed : PhaseParse;
            s[st + StOutLen] = -1;
        }
    }

    /// <summary>
    /// Continues the LZ77 parse of the block from its saved state: at most <paramref name="positionBudget"/> positions
    /// (plus the rest of the match it is in) or until <paramref name="compareBudget"/> byte compares, then saves the state.
    /// Tokens go to the token array (literal = value &lt; 256, match = (length &lt;&lt; 16) | distance) and the
    /// literal/length and distance frequencies are counted. A call after the parse ended does nothing.
    /// </summary>
    public static void ParseStep(
        ArrayView<byte> data, int start, int len, int level, ArrayView<int> s, int sb, int blockSize,
        int positionBudget, int compareBudget)
    {
        int headAt = sb;
        int prevAt = sb + HashSize;
        int tokAt = prevAt + blockSize;
        int tail = tokAt + blockSize;
        int st = tail + StateOff;
        if (s[st + StPhase] != PhaseParse) return;

        bool huffmanOnly = level == LevelHuffmanOnly;
        int maxChain = level == 1 ? 8 : (level == 2 ? 32 : 64);
        bool lazy = level == 3;
        int niceLen = level == 1 ? 32 : (level == 2 ? 128 : 258);

        int pos = s[st + StPos];
        int n = s[st + StTokens];
        int pendingLen = s[st + StPendLen];   // lazy: a match found at pos - 1 that has not been emitted yet
        int pendingDist = s[st + StPendDist];
        int extraBits = s[st + StExtra];
        int posLimit = len;
        if (positionBudget < len - pos) posLimit = pos + positionBudget;
        int compares = 0;
        bool going = pos < posLimit;
        while (going)
        {
            if (huffmanOnly)
            {
                n = AddLiteral(s, tokAt, tail, n, data[start + pos]);
                pos++;
                going = pos < posLimit;
                continue;
            }
            int found = FindMatch(data, start, pos, len, s, headAt, prevAt, maxChain, niceLen, ref compares);
            int bestLen = found >> 16;
            int bestDist = found & 0xFFFF;
            if (!lazy)
            {
                if (bestLen >= 3)
                {
                    n = AddMatch(s, tokAt, tail, n, bestLen, bestDist, ref extraBits);
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
                    n = AddMatch(s, tokAt, tail, n, pendingLen, pendingDist, ref extraBits);
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
            going = (pos < posLimit) & (compares < compareBudget);
        }
        if (pos >= len)
        {
            if (pendingLen > 0)
                n = AddMatch(s, tokAt, tail, n, pendingLen, pendingDist, ref extraBits);
            pendingLen = 0;
            s[st + StPhase] = PhaseParsed;
        }
        s[st + StPos] = pos;
        s[st + StTokens] = n;
        s[st + StPendLen] = pendingLen;
        s[st + StPendDist] = pendingDist;
        s[st + StExtra] = extraBits;
    }

    /// <summary>1 while the block's parse has not finished (a lazy parse can stop a launch on its compare budget).</summary>
    public static int ParsePending(ArrayView<int> s, int sb, int blockSize) =>
        s[sb + HashSize + 2 * blockSize + StateOff + StPhase] == PhaseParse ? 1 : 0;

    /// <summary>
    /// After the parse: builds the Huffman codes, decides between a dynamic and a stored block (exact sizes) and writes
    /// the block header (an empty block is written whole). The tokens / stored bytes follow in <see cref="EmitStep"/>.
    /// </summary>
    public static void PlanBlock(
        ArrayView<byte> data, int start, int len, bool isFinal, int level,
        ArrayView<byte> o, int outStart, int outCap,
        ArrayView<int> s, int sb, int blockSize)
    {
        int tokAt = sb + HashSize + blockSize;
        int tail = tokAt + blockSize;
        int st = tail + StateOff;
        if (s[st + StPhase] != PhaseParsed) return;
        var w = new DeflateBitWriter { Acc = 0, Count = 0, Pos = outStart, End = outStart + outCap, Overflow = 0 };

        if (len == 0)
        {
            // An empty final block: fixed Huffman, only the end-of-block code (7 zero bits).
            PutBits(ref w, o, isFinal ? 1u : 0u, 1);
            PutBits(ref w, o, 1, 2);
            PutBits(ref w, o, 0, 7);
            if (isFinal) AlignToByte(ref w, o); else EmitSyncFlush(ref w, o);
            s[st + StOutLen] = w.Overflow != 0 ? -1 : w.Pos - outStart;
            s[st + StPhase] = PhaseDone;
            return;
        }

        bool stored = level <= 0;
        int tokCount = s[st + StTokens];
        long dynBits = 0;
        int hlit = 257;
        int hdist = 1;
        int hclen = 4;
        int seqCount = 0;

        if (!stored)
        {
            dynBits = BuildBlockCodes(s, tail, st);
            hlit = s[st + StHlit];
            hdist = s[st + StHdist];
            hclen = s[st + StHclen];
            seqCount = s[st + StSeq];

            long storedBits = 8L * (len + 5L * ((len + 65534) / 65535));
            if (dynBits >= storedBits) stored = true;
        }

        if (stored)
        {
            // The stored sub-blocks (header + bytes) are written by EmitStep.
            s[st + StPhase] = PhaseEmitStored;
        }
        else
        {
            // ---- the dynamic block header ----
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
            s[st + StPhase] = PhaseEmitTokens;
        }
        s[st + StSeq] = seqCount;
        s[st + StCursor] = 0;
        SaveWriter(ref w, s, st);
    }

    /// <summary>
    /// The block's dynamic Huffman codes from its symbol frequencies: literal/length and distance code lengths (15-bit
    /// limit, trailing zeros trimmed), the run-length coded length sequence and the code-length code, the canonical
    /// codes; HLIT / HDIST / HCLEN and the sequence count go to the state slots. Returns the dynamic block's exact size in
    /// bits (header included).
    /// </summary>
    private static long BuildBlockCodes(ArrayView<int> s, int tail, int st)
    {
        int hlit;
        int hdist;
        int hclen;
        int seqCount = 0;
        long dynBits;
        long extraBits = s[st + StExtra];
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
        s[st + StHlit] = hlit;
        s[st + StHdist] = hdist;
        s[st + StHclen] = hclen;
        s[st + StSeq] = seqCount;
        return dynBits;
    }

    /// <summary>
    /// Continues writing the block from its saved state: at most <paramref name="budget"/> tokens (dynamic block) or
    /// stored bytes (stored sub-blocks of at most 65535 bytes, each with its header), then the end of the block when
    /// they are all out (end-of-block code + byte alignment or sync flush). A call after the end does nothing.
    /// </summary>
    public static void EmitStep(
        ArrayView<byte> data, int start, int len, bool isFinal,
        ArrayView<byte> o, int outStart,
        ArrayView<int> s, int sb, int blockSize, int budget)
    {
        int tokAt = sb + HashSize + blockSize;
        int tail = tokAt + blockSize;
        int st = tail + StateOff;
        int phase = s[st + StPhase];
        bool emitting = (phase == PhaseEmitTokens) | (phase == PhaseEmitStored);
        if (!emitting) return;

        var w = new DeflateBitWriter
        {
            Acc = (ulong)(uint)s[st + StAcc],
            Count = s[st + StCount],
            Pos = s[st + StWPos],
            End = s[st + StWEnd],
            Overflow = s[st + StOverflow],
        };
        int cursor = s[st + StCursor];

        if (phase == PhaseEmitTokens)
        {
            int tokCount = s[st + StTokens];
            int end = tokCount;
            if (budget < tokCount - cursor) end = cursor + budget;
            for (int t = cursor; t < end; t++)
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
            cursor = end;
            if (cursor == tokCount)
            {
                PutBits(ref w, o, (uint)s[tail + LitCodeOff + 256], s[tail + LitLenOff + 256]);
                if (isFinal) AlignToByte(ref w, o);
                else EmitSyncFlush(ref w, o);
                s[st + StOutLen] = w.Overflow != 0 ? -1 : w.Pos - outStart;
                s[st + StPhase] = PhaseDone;
            }
        }
        else
        {
            int done = cursor;
            int limit = len;
            if (budget < len - done) limit = done + budget;
            bool going = done < limit;
            while (going)
            {
                // Stored sub-blocks of at most 65535 bytes; each starts with its header.
                int chunkStart = (done / 65535) * 65535;
                int chunk = len - chunkStart;
                if (chunk > 65535) chunk = 65535;
                if (done == chunkStart)
                {
                    bool last = chunkStart + chunk == len;
                    PutBits(ref w, o, (isFinal & last) ? 1u : 0u, 1);
                    PutBits(ref w, o, 0, 2);
                    AlignToByte(ref w, o);
                    PutByte(ref w, o, chunk & 0xFF);
                    PutByte(ref w, o, chunk >> 8);
                    PutByte(ref w, o, ~chunk & 0xFF);
                    PutByte(ref w, o, (~chunk >> 8) & 0xFF);
                }
                int stop = chunkStart + chunk;
                if (stop > limit) stop = limit;
                for (int i = done; i < stop; i++) PutByte(ref w, o, data[start + i]);
                done = stop;
                going = done < limit;
            }
            cursor = done;
            if (done == len)
            {
                // A stored block already ends byte-aligned; nothing more is needed to concatenate.
                s[st + StOutLen] = w.Overflow != 0 ? -1 : w.Pos - outStart;
                s[st + StPhase] = PhaseDone;
            }
        }
        s[st + StCursor] = cursor;
        SaveWriter(ref w, s, st);
    }

    /// <summary>Saves the bit writer (pending bits &lt; 8, so the accumulator fits an int) into the block state.</summary>
    private static void SaveWriter(ref DeflateBitWriter w, ArrayView<int> s, int st)
    {
        s[st + StAcc] = (int)(uint)w.Acc;
        s[st + StCount] = w.Count;
        s[st + StWPos] = w.Pos;
        s[st + StWEnd] = w.End;
        s[st + StOverflow] = w.Overflow;
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

    /// <summary>
    /// Best match at <paramref name="pos"/>: (length &lt;&lt; 16) | distance, or 0 when none (length &lt; 3).
    /// Adds the byte compares it made to <paramref name="compares"/>.
    /// </summary>
    private static int FindMatch(
        ArrayView<byte> data, int start, int pos, int len, ArrayView<int> s, int headAt, int prevAt,
        int maxChain, int niceLen, ref int compares)
    {
        int cand = -1;
        if (pos + 3 <= len) cand = s[headAt + Hash3(data, start + pos)];
        return FindMatchFrom(data, start, pos, len, s, prevAt, cand, maxChain, niceLen, ref compares);
    }

    /// <summary>
    /// <see cref="FindMatch"/> from the chain's first candidate <paramref name="cand"/> (the latest earlier position with
    /// the hash of <paramref name="pos"/>, -1 for none).
    /// </summary>
    private static int FindMatchFrom(
        ArrayView<byte> data, int start, int pos, int len, ArrayView<int> s, int prevAt, int cand,
        int maxChain, int niceLen, ref int compares)
    {
        int bestLen = 0;
        int bestDist = 0;
        if (pos + 3 <= len)
        {
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
                        compares += ml + 1;
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

    private static int AddMatch(ArrayView<int> s, int tokAt, int tail, int n, int length, int dist, ref int extraBits)
    {
        s[tokAt + n] = (length << 16) | dist;
        int ls = LengthSymbol(length);
        int ds = DistSymbol(dist);
        s[tail + LitFreqOff + ls] = s[tail + LitFreqOff + ls] + 1;
        s[tail + DistFreqOff + ds] = s[tail + DistFreqOff + ds] + 1;
        // Extra bits of the block's matches (for the exact dynamic-block size).
        extraBits += LengthExtraBits(length) + DistExtraBits(dist);
        return n + 1;
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
