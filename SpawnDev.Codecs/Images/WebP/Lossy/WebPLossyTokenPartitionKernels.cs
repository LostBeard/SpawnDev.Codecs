// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// The lossy encoder's token partition (VP8EmitTokens into one VP8BitWriter) in parallel, byte-identical with the one-thread
// FinishTokensKernel walk. The bool coder's output is one big number: a token with bit 1 adds (split + 1) shifted left by
// its own normalisation shift and every later one; the written bytes are that number's base-256 digits (the delayed 0xff
// run and the carry into the previous byte are its carries). Checked on a model of the writer before building
// (Plans/PLAN-Image-Codecs.md, the token partition entry): after any token, int(bytes ++ 0xff x run) << (16 + nb) + value
// equals the big number, and a writer rebuilt from the digits finishes to the same bytes.
//
//   MapKernel      chunk c of ChunkTokens tokens, run from each of the 128 ranges a normalised writer can hold (127..254):
//                  the end range and the chunk's shift total per start range.
//   ComposeKernel  one thread: the maps composed in order from the writer's first range (254) -> each chunk's true start
//                  range and the shifts before it; the total shift count and the final range.
//   AddKernel      chunk c from its true start: each token's addition atomically added (with carry propagation: order-free)
//                  into the big number at bit position total - (shifts before the token).
//   DigitsKernel   a thread per output byte: digit j of the number above bit 16 + nb.
//   TailKernel     one thread: the writer rebuilt (bytes up to the trailing 0xff run, run, value = the low 16 + nb bits,
//                  nb, the final range) for FinishHeaderKernel's VP8BitWriterFinish, unchanged.
//
// The int buffer `tp`: [0] total shifts, [1] final range, maps (2 ints per chunk and start range), the chunks' start
// ranges, the chunks' shift prefixes, then the big number (32-bit words, little-endian).

using ILGPU;
using ILGPU.Runtime;
using M = SpawnDev.Codecs.Images.WebP.Lossy.WebPLossyEncMeta;
using T = SpawnDev.Codecs.Images.WebP.Lossy.WebPLossyEncTables;

namespace SpawnDev.Codecs.Images.WebP.Lossy;

internal static class WebPLossyTokenPartitionKernels
{
    /// <summary>Tokens per chunk of the range maps.</summary>
    public const int ChunkTokens = 256;

    private const int States = 128, MinRange = 127, HeaderInts = 16;

    public static int Chunks(int tokenCount) => (tokenCount + ChunkTokens - 1) / ChunkTokens;

    /// <summary>The big number's words for <paramref name="tokenCount"/> tokens (a token shifts at most 7 bits).</summary>
    public static long NumberWords(int tokenCount) => (7L * tokenCount + 64) / 32 + 4;

    private static int MapsOff() => HeaderInts;
    private static int StartsOff(int chunks) => HeaderInts + chunks * States * 2;
    private static int PrefixOff(int chunks) => StartsOff(chunks) + chunks;
    public static int NumberOff(int chunks) => (PrefixOff(chunks) + chunks + 15) & ~15;

    /// <summary>Ints of the <c>tp</c> buffer for <paramref name="tokenCount"/> tokens.</summary>
    public static long BufferInts(int tokenCount) => NumberOff(Chunks(tokenCount)) + NumberWords(tokenCount);

    private static int Prob(ArrayView<int> coeffs, int token) =>
        (token & (1 << 14)) != 0 ? token & 0xff : coeffs[token & 0x3fff];

    // one token on a writer range (the Range field of WebPBoolWriter: 254 at the start): the new range, the addition, the shift
    private static int Step(ArrayView<int> t, int range, int bit, int prob, out int add, out int shift)
    {
        int split = (range * prob) >> 8;
        add = 0;
        if (bit != 0) { add = split + 1; range -= split + 1; }
        else range = split;
        shift = 0;
        if (range < 127)
        {
            shift = t[T.Norm + range];
            range = t[T.NewRange + range];
        }
        return range;
    }

    public static void MapKernel(Index1D i, ArrayView<int> state, ArrayView<int> meta, ArrayView<int> t, ArrayView<int> tokens, ArrayView<int> tp)
    {
        if (meta[M.Error] != 0) return;
        int n = meta[M.TokenCount];
        int chunks = Chunks(n);
        if (i >= chunks * States) return;
        int c = i / States, s = i - c * States;
        var coeffs = state.SubView(meta[M.StCoeffsOff], WebPLossySetupKernel.ProbaCount);
        int range = MinRange + s, shifts = 0;
        int end = Min(n, (c + 1) * ChunkTokens);
        for (int k = c * ChunkTokens; k < end; ++k)
        {
            int token = tokens[k];
            range = Step(t, range, (token >> 15) & 1, Prob(coeffs, token), out _, out int sh);
            shifts += sh;
        }
        int o = MapsOff() + 2 * (c * States + s);
        tp[o] = range;
        tp[o + 1] = shifts;
    }

    public static void ComposeKernel(Index1D _, ArrayView<int> state, ArrayView<int> meta, ArrayView<int> t, ArrayView<int> tokens, ArrayView<int> tp)
    {
        if (meta[M.Error] != 0) return;
        int chunks = Chunks(meta[M.TokenCount]);
        int range = 254, total = 0;
        int so = StartsOff(chunks), po = PrefixOff(chunks);
        for (int c = 0; c < chunks; ++c)
        {
            tp[so + c] = range;
            tp[po + c] = total;
            int o = MapsOff() + 2 * (c * States + range - MinRange);
            total += tp[o + 1];
            range = tp[o];
        }
        tp[0] = total;
        tp[1] = range;
    }

    // number += v << pos (v < 2^24), carries propagated by atomics: the sum does not depend on the order of the additions
    private static void AddAt(ArrayView<int> num, int pos, int v)
    {
        int w = pos >> 5, b = pos & 31;
        ulong x = (ulong)(uint)v << b;
        AddWord(num, w, (uint)x);
        AddWord(num, w + 1, (uint)(x >> 32));
    }

    private static void AddWord(ArrayView<int> num, int w, uint v)
    {
        while (v != 0)
        {
            uint old = (uint)Atomic.Add(ref num[w], (int)v);
            v = old + v < old ? 1u : 0u;   // the carry out of this word
            w++;
        }
    }

    public static void AddKernel(Index1D c, ArrayView<int> state, ArrayView<int> meta, ArrayView<int> t, ArrayView<int> tokens, ArrayView<int> tp)
    {
        if (meta[M.Error] != 0) return;
        int n = meta[M.TokenCount];
        int chunks = Chunks(n);
        if (c >= chunks) return;
        var coeffs = state.SubView(meta[M.StCoeffsOff], WebPLossySetupKernel.ProbaCount);
        var num = tp.SubView(NumberOff(chunks), NumberWords(n));
        int total = tp[0];
        int range = tp[StartsOff(chunks) + c];
        int before = tp[PrefixOff(chunks) + c];
        int end = Min(n, (c + 1) * ChunkTokens);
        // consecutive additions at the same position (no shift between them) are summed before the atomic add
        int pending = 0, pendingPos = 0;
        for (int k = c * ChunkTokens; k < end; ++k)
        {
            int token = tokens[k];
            range = Step(t, range, (token >> 15) & 1, Prob(coeffs, token), out int add, out int sh);
            if (add != 0)
            {
                int pos = total - before;   // the token's own shift and every later one
                if (pending != 0 && pos != pendingPos) { AddAt(num, pendingPos, pending); pending = 0; }
                pending += add;
                pendingPos = pos;
            }
            before += sh;
        }
        if (pending != 0) AddAt(num, pendingPos, pending);
    }

    // with t = total - 8: the bytes (written + the pending 0xff run) and nb of the writer after every token
    private static int ByteCount(int total) { int tt = total - 8; return tt <= 0 ? 0 : (tt + 7) >> 3; }
    private static int NbBits(int total) => total - 8 - 8 * ByteCount(total);

    private static int DigitAt(ArrayView<int> num, int pos)
    {
        int w = pos >> 5, b = pos & 31;
        ulong x = ((ulong)(uint)num[w + 1] << 32) | (uint)num[w];
        return (int)((x >> b) & 0xff);
    }

    public static void DigitsKernel(Index1D j, ArrayView<int> state, ArrayView<int> meta, ArrayView<int> t, ArrayView<int> tokens, ArrayView<int> tp,
        ArrayView<byte> scratch, int baseOff, int cap)
    {
        if (meta[M.Error] != 0) return;
        int n = meta[M.TokenCount];
        int total = tp[0];
        int bytes = ByteCount(total), nb = NbBits(total);
        if (j >= bytes || j >= cap) return;
        var num = tp.SubView(NumberOff(Chunks(n)), NumberWords(n));
        scratch[baseOff + j] = (byte)DigitAt(num, 16 + nb + 8 * (bytes - 1 - j));
    }

    public static void TailKernel(Index1D _, ArrayView<int> state, ArrayView<int> meta, ArrayView<int> t, ArrayView<int> tokens, ArrayView<int> tp,
        ArrayView<byte> scratch, int baseOff, int cap)
    {
        if (meta[M.Error] != 0) return;
        int n = meta[M.TokenCount];
        var num = tp.SubView(NumberOff(Chunks(n)), NumberWords(n));
        int total = n == 0 ? 0 : tp[0];
        int range = n == 0 ? 254 : tp[1];
        int bytes = ByteCount(total), nb = NbBits(total);
        int k = bytes;
        while (k > 0 && DigitAt(num, 16 + nb + 8 * (bytes - k)) == 0xff) k--;   // the trailing 0xff run stays pending
        int low = 16 + nb;   // value = the number's bits below the digits (8 < low <= 16)
        int value = n == 0 ? 0 : (int)(((uint)num[0]) & ((1u << low) - 1));   // (no tokens: the number was never cleared)
        int at = M.TokWriter;
        meta[at + 0] = range;
        meta[at + 1] = value;
        meta[at + 2] = bytes - k;   // Run
        meta[at + 3] = nb;
        meta[at + 4] = k;           // Pos
        // Base, Cap: FinishProbasKernel's
        meta[at + 7] = bytes > cap ? 1 : 0;   // Overflow: the digits did not fit (the walk's Put sets it the same way)
        meta[M.TokNext] = n;
    }

    private static int Min(int a, int b) => a < b ? a : b;
}
