// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP8L tokenisation in parallel: the tokens, histograms and extra bits Vp8lEncodeGpu.TokenizeStep (one thread, greedy LZ77
// with hash chains and a colour cache) produces, without its serial walk over every pixel. Lore's report (2026-11-14,
// SpawnGen Save Animated WEBP): the encode took ~14 s a frame on an RX 590 (4.3 s on an RTX 4070), ~85% of it this
// tokeniser, run once per strategy trial, per colour-cache size and once more for real.
//
// Why it can run in parallel with the same result. The serial walk inserts EVERY position into the hash chain (literal or
// inside a copy) and EVERY pixel into the colour cache, in pixel order. So
//   - the chain at position p (positions j < p with the same pair hash, newest first) is fixed by the pixels alone:
//     cand0 = the previous position with p's key, then that position's previous, ...;
//   - the best (length, distance) at p is fixed by the pixels alone (run, row above, then the chain, strictly longer wins,
//     the same depth / distance / early-stop rules);
//   - a literal at p hits the cache iff the previous pixel with p's cache key has p's colour (with none, the cache's
//     initial 0: a 0 pixel hits; found by the test on predictor residuals and cleared transparent pixels).
// Only the parse (which positions start a token: jump by the copy length or 1) depends on the order, and with the decisions
// precomputed it is one load per token. Passes:
//   Links      one thread per block of BlockPositions: the previous position with the same key inside the block (-2 if none
//              there) and the block's last position per key (pair hash 65536 keys, cache key up to 1024);
//   Carry      one thread per key: the last position of that key before each block (in place);
//   Fixup      one thread per position: links that left their block take the carry;
//   Search     one thread per position (bounded launches): the best copy;
//   Parse      one thread, bounded resumable launches: marks token starts (1 literal / cache hit, 2 copy);
//   Count / Scan / Emit  per block of EmitBlock positions: tokens in order, histograms and extra bits by atomics.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Images.WebP;

/// <summary>Layout and kernels of the parallel VP8L tokeniser (see the file header).</summary>
public static class Vp8lTokenizeParallel
{
    /// <summary>Positions per block of the link pass.</summary>
    public const int BlockPositions = 16384;
    /// <summary>Positions per thread of the emit passes.</summary>
    public const int EmitBlock = 1024;
    /// <summary>Positions per launch of the search pass.</summary>
    public const int SearchChunk = 1 << 18;
    /// <summary>Tokens per launch of the parse.</summary>
    public const int ParseChunk = 1 << 16;

    public const int PairKeys = 1 << Vp8lEncLayout.HashBits;
    public const int CacheKeys = 1 << 10;

    // state ints
    public const int SParsePos = 0, STotal = 1, SExtra = 2, StateInts = 4;

    /// <summary>Offsets of the parallel tokeniser's ints for <paramref name="n"/> pixels (one buffer).</summary>
    public readonly struct Layout
    {
        public readonly int N, Blocks, EmitBlocks;
        public readonly long KeyPrev, CachePrev, Len, Dist, Flag, PairTable, CacheTable, BlockCount, BlockExtra, State, Total;

        public Layout(int n)
        {
            N = n;
            Blocks = (n + BlockPositions - 1) / BlockPositions;
            EmitBlocks = (n + EmitBlock - 1) / EmitBlock;
            long p = 0;
            KeyPrev = p; p += n;
            CachePrev = p; p += n;
            Len = p; p += n;
            Dist = p; p += n;
            Flag = p; p += n;
            PairTable = p; p += (long)Blocks * PairKeys;
            CacheTable = p; p += (long)Blocks * CacheKeys;
            BlockCount = p; p += EmitBlocks;
            BlockExtra = p; p += EmitBlocks;
            State = p; p += StateInts;
            Total = p;
        }
    }

    /// <summary>Kernel arguments that fit in one struct (WebGPU binding budget): sizes and offsets.</summary>
    public struct Args
    {
        public int N, XSize, CacheBits, ChainDepth, Blocks, EmitBlocks, Base;
        public int KeyPrev, CachePrev, Len, Dist, Flag, PairTable, CacheTable, BlockCount, BlockExtra, State;
    }

    public static Args MakeArgs(in Layout l, int xsize, int cacheBits, int chainDepth) => new Args
    {
        N = l.N, XSize = xsize, CacheBits = cacheBits, ChainDepth = chainDepth, Blocks = l.Blocks, EmitBlocks = l.EmitBlocks,
        KeyPrev = checked((int)l.KeyPrev), CachePrev = checked((int)l.CachePrev), Len = checked((int)l.Len), Dist = checked((int)l.Dist),
        Flag = checked((int)l.Flag), PairTable = checked((int)l.PairTable), CacheTable = checked((int)l.CacheTable),
        BlockCount = checked((int)l.BlockCount), BlockExtra = checked((int)l.BlockExtra), State = checked((int)l.State),
    };

    /// <summary>Thread i: s[start + i] = value (the link tables start at -1, the flags at 0).</summary>
    public static void FillKernel(Index1D i, ArrayView<int> s, int start, int value)
    {
        s[start + i] = value;
    }

    // ---- links ----

    /// <summary>Thread b: block b's in-block links and its last position per key (tables pre-filled with -1).</summary>
    public static void LinksKernel(Index1D b, ArrayView<uint> pix, ArrayView<int> s, Args a)
    {
        int start = b * BlockPositions;
        int end = start + BlockPositions;
        if (end > a.N) end = a.N;
        int pt = a.PairTable + b * PairKeys;
        int ct = a.CacheTable + b * CacheKeys;
        int cacheShift = 32 - a.CacheBits;
        for (int p = start; p < end; p++)
        {
            int kp = -1;
            if (p + 1 < a.N)
            {
                int h = Vp8lEncodeGpu.PairHash(pix[p], pix[p + 1]);
                int last = s[pt + h];
                kp = last >= 0 ? last : -2;   // -2: look before the block (Fixup)
                s[pt + h] = p;
            }
            s[a.KeyPrev + p] = kp;
            int cp = -1;
            if (a.CacheBits > 0)
            {
                int k = (int)Vp8lEncodeGpu.CacheHash(pix[p], cacheShift);
                int last = s[ct + k];
                cp = last >= 0 ? last : -2;
                s[ct + k] = p;
            }
            s[a.CachePrev + p] = cp;
        }
    }

    /// <summary>Thread k (pair keys, then cache keys): each block's entry becomes the key's last position before the block.</summary>
    public static void CarryKernel(Index1D k, ArrayView<int> s, Args a)
    {
        bool pair = k < PairKeys;
        int key = pair ? (int)k : k - PairKeys;
        int table = pair ? a.PairTable : a.CacheTable;
        int stride = pair ? PairKeys : CacheKeys;
        bool live = pair | (key < CacheKeys & a.CacheBits > 0);
        if (!live) return;
        int run = -1;
        for (int b = 0; b < a.Blocks; b++)
        {
            int at = table + b * stride + key;
            int last = s[at];
            s[at] = run;
            if (last >= 0) run = last;
        }
    }

    /// <summary>Thread p: a link that left its block takes the key's last position before the block.</summary>
    public static void FixupKernel(Index1D p, ArrayView<uint> pix, ArrayView<int> s, Args a)
    {
        int b = p / BlockPositions;
        if (s[a.KeyPrev + p] == -2)
            s[a.KeyPrev + p] = s[a.PairTable + b * PairKeys + Vp8lEncodeGpu.PairHash(pix[p], pix[p + 1])];
        if (s[a.CachePrev + p] == -2)
            s[a.CachePrev + p] = s[a.CacheTable + b * CacheKeys + (int)Vp8lEncodeGpu.CacheHash(pix[p], 32 - a.CacheBits)];
    }

    // ---- search ----

    /// <summary>Thread i: position a.Base + i's best copy, exactly as TokenizeStep searches at a token start.</summary>
    public static void SearchKernel(Index1D i, ArrayView<uint> pix, ArrayView<int> s, Args a)
    {
        const int MaxCopy = 4096;
        const int MaxDist = (1 << 20) - 120;
        int pos = a.Base + i;
        if (pos >= a.N) return;
        int n = a.N, xsize = a.XSize;
        int maxLen = n - pos;
        if (maxLen > MaxCopy) maxLen = MaxCopy;
        int bestLen = 0, bestDist = 0;
        if (pos > 0)
        {
            int l = Vp8lEncodeGpu.MatchLength(pix, pos - 1, pos, maxLen);
            if (l > bestLen) { bestLen = l; bestDist = 1; }
            if (pos >= xsize)
            {
                l = Vp8lEncodeGpu.MatchLength(pix, pos - xsize, pos, maxLen);
                if (l > bestLen) { bestLen = l; bestDist = xsize; }
            }
            if (a.ChainDepth > 0 & pos + 1 < n)
            {
                int cand = s[a.KeyPrev + pos];
                int depth = 0;
                // as TokenizeStep: the first candidate is tried even when bestLen == maxLen (it cannot be longer)
                bool go = (cand >= 0) & (depth < a.ChainDepth);
                while (go)
                {
                    int dist = pos - cand;
                    if (dist > MaxDist)
                    {
                        go = false;
                    }
                    else
                    {
                        l = Vp8lEncodeGpu.MatchLength(pix, cand, pos, maxLen);
                        if (l > bestLen) { bestLen = l; bestDist = dist; }
                        cand = s[a.KeyPrev + cand];
                        depth++;
                        go = (cand >= 0) & (depth < a.ChainDepth) & (bestLen < maxLen);
                    }
                }
            }
        }
        s[a.Len + pos] = bestLen;
        s[a.Dist + pos] = bestDist;
    }

    // ---- parse ----

    /// <summary>One thread: the next ParseChunk token starts from the resumable position (flags: 1 literal, 2 copy).</summary>
    public static void ParseKernel(Index1D _, ArrayView<int> s, ArrayView<int> tables, Args a)
    {
        int pos = s[a.State + SParsePos];
        int steps = 0;
        bool go = pos < a.N;
        while (go)
        {
            int len = s[a.Len + pos];
            bool copy = len >= 4;
            if (len == 3) copy = Vp8lEncodeGpu.DistanceToPlaneCode(a.XSize, s[a.Dist + pos], tables) <= 120;
            s[a.Flag + pos] = copy ? 2 : 1;
            pos += copy ? len : 1;
            steps++;
            go = (pos < a.N) & (steps < ParseChunk);
        }
        s[a.State + SParsePos] = pos;
    }

    // ---- emit ----

    /// <summary>Thread b: the tokens starting in emit block b.</summary>
    public static void CountKernel(Index1D b, ArrayView<int> s, Args a)
    {
        int start = b * EmitBlock;
        int end = start + EmitBlock;
        if (end > a.N) end = a.N;
        int c = 0;
        for (int p = start; p < end; p++) c += s[a.Flag + p] != 0 ? 1 : 0;
        s[a.BlockCount + b] = c;
    }

    /// <summary>One thread: the blocks' counts into exclusive offsets; the total.</summary>
    public static void ScanKernel(Index1D _, ArrayView<int> s, Args a)
    {
        int run = 0;
        for (int b = 0; b < a.EmitBlocks; b++)
        {
            int c = s[a.BlockCount + b];
            s[a.BlockCount + b] = run;
            run += c;
        }
        s[a.State + STotal] = run;
    }

    /// <summary>
    /// Thread b: the tokens of emit block b in order at its offset (TokenizeStep's token format), their histogram counts
    /// and extra bits (atomics: blocks share the histograms).
    /// </summary>
    public static void EmitKernel(Index1D b, ArrayView<uint> pix, ArrayView<int> s, ArrayView<int> w, ArrayView<int> tok,
        ArrayView<int> meta, ArrayView<int> tables, Args a)
    {
        int start = b * EmitBlock;
        int end = start + EmitBlock;
        if (end > a.N) end = a.N;
        int t = s[a.BlockCount + b];
        int cacheSize = a.CacheBits > 0 ? 1 << a.CacheBits : 0;
        int set = meta[WebPEncMeta.RegionMain];
        int gOff = set + Vp8lEncodeGpu.AlphaOff(0, cacheSize);
        int rOff = set + Vp8lEncodeGpu.AlphaOff(1, cacheSize);
        int bOff = set + Vp8lEncodeGpu.AlphaOff(2, cacheSize);
        int aOff = set + Vp8lEncodeGpu.AlphaOff(3, cacheSize);
        int dOff = set + Vp8lEncodeGpu.AlphaOff(4, cacheSize);
        int extra = 0;
        for (int p = start; p < end; p++)
        {
            int f = s[a.Flag + p];
            if (f == 2)
            {
                int len = s[a.Len + p], dist = s[a.Dist + p];
                int plane = Vp8lEncodeGpu.DistanceToPlaneCode(a.XSize, dist, tables);
                int lp = Vp8lEncodeGpu.PrefixEncode(len);
                int dp = Vp8lEncodeGpu.PrefixEncode(plane);
                tok[t * 2] = 2 | (len << 2);
                tok[t * 2 + 1] = dist;
                t++;
                Atomic.Add(ref w[gOff + 256 + (lp & 0xFF)], 1);
                Atomic.Add(ref w[dOff + (dp & 0xFF)], 1);
                extra += ((lp >> 8) & 0x1F) + ((dp >> 8) & 0x1F);
            }
            else if (f == 1)
            {
                uint px = pix[p];
                bool hit = false;
                int key = 0;
                if (a.CacheBits > 0)
                {
                    key = (int)Vp8lEncodeGpu.CacheHash(px, 32 - a.CacheBits);
                    // the cache starts zero-filled (TokenizeInit, and the decoder's): with no earlier pixel of this key the
                    // entry still holds 0, so a 0 pixel hits it
                    int prev = s[a.CachePrev + p];
                    if (prev >= 0) hit = pix[prev] == px;
                    else hit = px == 0u;
                }
                if (hit)
                {
                    tok[t * 2] = 1 | (key << 2);
                    tok[t * 2 + 1] = 0;
                    Atomic.Add(ref w[gOff + 280 + key], 1);
                }
                else
                {
                    tok[t * 2] = 0;
                    tok[t * 2 + 1] = (int)px;
                    Atomic.Add(ref w[gOff + (int)((px >> 8) & 0xFFu)], 1);
                    Atomic.Add(ref w[rOff + (int)((px >> 16) & 0xFFu)], 1);
                    Atomic.Add(ref w[bOff + (int)(px & 0xFFu)], 1);
                    Atomic.Add(ref w[aOff + (int)(px >> 24)], 1);
                }
                t++;
            }
        }
        if (extra != 0) Atomic.Add(ref s[a.State + SExtra], extra);
    }

    /// <summary>One thread: the token count and extra bits into the meta block (as TokenizeStep leaves them).</summary>
    public static void FinishKernel(Index1D _, ArrayView<int> s, ArrayView<int> meta, Args a)
    {
        meta[WebPEncMeta.TokenCount] = s[a.State + STotal];
        meta[WebPEncMeta.ExtraBits] = s[a.State + SExtra];
    }
}
