// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP8L (WebP lossless) encoder building blocks, GPU-callable form. Written from the WebP Lossless Bitstream
// Specification (RFC 9649); the heuristics (what to try, in which order, how deep to search) are our own.
//
// The pixel transforms (palette, subtract green, predictor, cross colour) are parallel kernels in
// WebPEncodeKernels. What lives here is the sequential part - the bitstream - which runs as one GPU thread
// (or one thread per independent prefix code): LZ77 hash-chain matching with the colour cache, Huffman code
// construction (length limited to 15), code-length coding and bit packing. Same shape as the decoder: a state
// struct passed by ref, no managed allocations, no static tables, scratch regions of a caller-owned int buffer
// ("work"), conditions in loops and branches written as single bool locals built with non-short-circuit
// operators (a compound `a && b` loop header hung the WebGPU device).

using ILGPU;

namespace SpawnDev.Codecs.Images.WebP;

/// <summary>Slot indices of the encoder's shared int block.</summary>
public static class WebPEncMeta
{
    /// <summary>Total ints.</summary>
    public const int Size = 64;

    // ---- written by the encode kernels (slots below HostStart) ----
    /// <summary>Non-zero when the output buffer was too small.</summary>
    public const int Error = 0;
    /// <summary>1 when any pixel has alpha != 255.</summary>
    public const int HasAlpha = 1;
    /// <summary>Number of distinct colours when there are at most 256 (a palette is possible), else 0.</summary>
    public const int PaletteAvailable = 2;
    /// <summary>Number of tokens of the main image.</summary>
    public const int TokenCount = 3;
    /// <summary>Extra bits (length / distance) of all copy tokens.</summary>
    public const int ExtraBits = 4;
    /// <summary>Colour cache bits of the main image (0 = none).</summary>
    public const int CacheBits = 5;
    /// <summary>Per-code cost in bits of the five main codes (6..10), code definitions estimated in.</summary>
    public const int CodeCost = 6;
    /// <summary>Bytes of the VP8L payload.</summary>
    public const int PayloadLength = 11;
    /// <summary>Bytes of the whole file.</summary>
    public const int Total = 12;
    /// <summary>File offset the EXIF payload is copied to.</summary>
    public const int ExifDst = 13;
    /// <summary>File offset the XMP payload is copied to.</summary>
    public const int XmpDst = 14;
    /// <summary>File offset the ICCP payload is copied to.</summary>
    public const int IccDst = 15;

    // ---- written by the host (slots HostStart .. Size) ----
    /// <summary>First host-written slot.</summary>
    public const int HostStart = 24;
    /// <summary>Palette size when the palette transform is used, else 0.</summary>
    public const int PaletteCount = 24;
    /// <summary>Pixel bundling width bits of the palette transform (0..3).</summary>
    public const int PaletteBits = 25;
    /// <summary>Width of the image the entropy coder sees (packed width for a bundled palette).</summary>
    public const int PackedWidth = 26;
    /// <summary>1 = subtract green.</summary>
    public const int UseSubtractGreen = 27;
    /// <summary>1 = predictor transform.</summary>
    public const int UsePredictor = 28;
    /// <summary>Predictor block size bits.</summary>
    public const int PredictorBits = 29;
    /// <summary>1 = cross-colour transform.</summary>
    public const int UseCrossColor = 30;
    /// <summary>Cross-colour block size bits.</summary>
    public const int CrossColorBits = 31;
    /// <summary>1 = collect a palette (the colour analysis).</summary>
    public const int AllowPalette = 32;
    /// <summary>Longest hash chain walked per position (0 = only run / 2-D neighbour candidates).</summary>
    public const int ChainDepth = 33;
    /// <summary>Predictor mode set: 0 = few modes, 1 = all 14.</summary>
    public const int ModeSet = 34;
    /// <summary>File offset of the VP8L payload.</summary>
    public const int PayloadStart = 35;
    /// <summary>Capacity of the output buffer.</summary>
    public const int OutputCapacity = 36;
    /// <summary>1 = the file needs a VP8X chunk.</summary>
    public const int ExtendedFile = 37;
    /// <summary>ICC blob length (0 = absent).</summary>
    public const int IccLength = 38;
    /// <summary>EXIF blob length (0 = absent).</summary>
    public const int ExifLength = 39;
    /// <summary>XMP blob length (0 = absent).</summary>
    public const int XmpLength = 40;
    /// <summary>Image width.</summary>
    public const int Width = 41;
    /// <summary>Image height.</summary>
    public const int Height = 42;
    /// <summary>Work-buffer region offsets (see <see cref="Vp8lEncLayout"/>).</summary>
    public const int RegionPredictor = 43, RegionCross = 44, RegionMain = 45, RegionBuild = 46, RegionSub = 47,
        RegionWriter = 48, RegionHead = 49, RegionPrev = 50, RegionCache = 51;
}

/// <summary>Layout of the encoder's int work buffer: a pure function of the image size and transform block sizes.</summary>
public static class Vp8lEncLayout
{
    /// <summary>Largest colour cache.</summary>
    public const int MaxCacheSize = 2048;
    /// <summary>Hash table bits of the LZ77 matcher.</summary>
    public const int HashBits = 16;
    /// <summary>Upper bound on an alphabet size (280 + 2048 = 2328).</summary>
    public const int MaxAlpha = 2400;
    /// <summary>Sum of the five alphabets with the largest colour cache.</summary>
    public const int SetAlphaMax = 280 + MaxCacheSize + 256 * 3 + 40;
    /// <summary>Ints of a code set (histograms + lengths + packed codes).</summary>
    public const int SetInts = 3 * SetAlphaMax;
    /// <summary>Sum of the five alphabets without a colour cache.</summary>
    public const int SubSetAlpha = 280 + 256 * 3 + 40;
    /// <summary>Ints of a code set without colour cache.</summary>
    public const int SubSetInts = 3 * SubSetAlpha;
    /// <summary>Scratch of one Huffman length construction.</summary>
    public const int BuildScratch = 9 * MaxAlpha;
    /// <summary>Scratch of the code-length writer (tokens + the small code-length code) on top of one build scratch.</summary>
    public const int WriterScratch = 2 * MaxAlpha + 256 + BuildScratch;

    /// <summary>Palette (256 ints) and its delta-coded form (256 ints).</summary>
    public const int Palette = 0, PaletteDelta = 256;
    /// <summary>Hash table of the colour analysis (1024 keys + 1024 occupied flags).</summary>
    public const int Analysis = 512;
    /// <summary>First region after the fixed ones.</summary>
    public const int FirstDynamic = 512 + 2048;

    /// <summary>Tiles of a transform with block size 2^bits.</summary>
    public static int Tiles(int width, int height, int bits) => ((width + (1 << bits) - 1) >> bits) * ((height + (1 << bits) - 1) >> bits);

    /// <summary>Fill the region offsets for an image; returns the total ints needed.</summary>
    public static long Compute(int width, int height, int predBits, int crossBits, int[] meta)
    {
        long n = (long)width * height;
        long p = FirstDynamic;
        meta[WebPEncMeta.RegionPredictor] = (int)p; p += Tiles(width, height, predBits);
        meta[WebPEncMeta.RegionCross] = (int)p; p += Tiles(width, height, crossBits);
        meta[WebPEncMeta.RegionCache] = (int)p; p += MaxCacheSize;
        meta[WebPEncMeta.RegionMain] = (int)p; p += SetInts;
        meta[WebPEncMeta.RegionBuild] = (int)p; p += 5L * BuildScratch;
        meta[WebPEncMeta.RegionSub] = (int)p; p += SubSetInts;
        meta[WebPEncMeta.RegionWriter] = (int)p; p += WriterScratch;
        meta[WebPEncMeta.RegionHead] = (int)p; p += 1 << HashBits;
        meta[WebPEncMeta.RegionPrev] = (int)p; p += n;
        return p;
    }
}

/// <summary>Running state of the VP8L bit writer.</summary>
public struct Vp8lBitWriter
{
    /// <summary>Next output byte index.</summary>
    public int Pos;
    /// <summary>Bit accumulator (low bits are the next bits written).</summary>
    public uint Acc;
    /// <summary>Number of valid bits in <see cref="Acc"/> (always &lt; 8 between calls).</summary>
    public int Nbits;
    /// <summary>One past the last writable byte.</summary>
    public int Cap;
    /// <summary>1 when a byte did not fit.</summary>
    public int Overflow;
}

/// <summary>GPU-callable VP8L encoder building blocks.</summary>
public static class Vp8lEncodeGpu
{
    private const int MaxLen = 15;
    // BuildLengths scratch (relative offsets, each M = MaxAlpha ints unless noted)
    private const int BsIdx = 0, BsTmp = 1, BsNw = 2, BsLeft = 4, BsRight = 5, BsDep = 6;
    // WriteOneCode scratch inside WriterScratch
    private const int WsTokSym = 0, WsTokExtra = Vp8lEncLayout.MaxAlpha, WsSmall = 2 * Vp8lEncLayout.MaxAlpha, WsBuild = 2 * Vp8lEncLayout.MaxAlpha + 256;
    // small block: hist19 [0..19), lens19 [32..51), codes19 [64..83), counts [96..112), next [112..128)
    private const int SmHist = 0, SmLens = 32, SmCodes = 64, SmCnt = 96, SmNext = 112;

    // ------------------------------------------------------------------
    // Bit writer
    // ------------------------------------------------------------------

    /// <summary>Appends the low <paramref name="n"/> (0..24) bits of <paramref name="value"/>, least significant first.</summary>
    public static void PutBits(ref Vp8lBitWriter bw, ArrayView<byte> output, uint value, int n)
    {
        if (n > 0)
        {
            bw.Acc |= value << bw.Nbits;
            bw.Nbits += n;
            while (bw.Nbits >= 8)
            {
                if (bw.Pos < bw.Cap) output[bw.Pos] = (byte)(bw.Acc & 0xFFu);
                else bw.Overflow = 1;
                bw.Pos++;
                bw.Acc >>= 8;
                bw.Nbits -= 8;
            }
        }
    }

    /// <summary>Writes the partial last byte.</summary>
    public static void Flush(ref Vp8lBitWriter bw, ArrayView<byte> output)
    {
        if (bw.Nbits > 0)
        {
            if (bw.Pos < bw.Cap) output[bw.Pos] = (byte)(bw.Acc & 0xFFu);
            else bw.Overflow = 1;
            bw.Pos++;
            bw.Acc = 0;
            bw.Nbits = 0;
        }
    }

    // ------------------------------------------------------------------
    // Alphabets of a code set
    // ------------------------------------------------------------------

    /// <summary>Alphabet size of code <paramref name="k"/> (0 green + length + cache, 1 red, 2 blue, 3 alpha, 4 distance).</summary>
    public static int Alpha(int k, int cacheSize)
    {
        int a = 256;
        if (k == 0) a = 280 + cacheSize;
        else if (k == 4) a = 40;
        return a;
    }

    /// <summary>Offset of code <paramref name="k"/> inside the histogram / lengths / codes arrays of a set.</summary>
    public static int AlphaOff(int k, int cacheSize)
    {
        int o = 0;
        if (k >= 1) o += 280 + cacheSize;
        if (k >= 2) o += 256;
        if (k >= 3) o += 256;
        if (k >= 4) o += 256;
        return o;
    }

    /// <summary>Sum of the five alphabets.</summary>
    public static int SetAlpha(int cacheSize) => 280 + cacheSize + 256 * 3 + 40;

    // ------------------------------------------------------------------
    // Huffman construction
    // ------------------------------------------------------------------

    /// <summary>
    /// Code lengths (at most <paramref name="limit"/> bits) for the <paramref name="n"/> histogram counts at
    /// <c>w[hb ..]</c>, written to <c>w[lb ..]</c>. A single used symbol gets length 1 (it is then coded in zero bits).
    /// <paramref name="sb"/> is a <see cref="Vp8lEncLayout.BuildScratch"/> scratch region.
    /// </summary>
    public static void BuildLengths(ArrayView<int> w, int hb, int n, int lb, int sb, int limit)
    {
        const int M = Vp8lEncLayout.MaxAlpha;
        for (int i = 0; i < n; i++) w[lb + i] = 0;
        int m = 0;
        for (int i = 0; i < n; i++)
        {
            if (w[hb + i] > 0)
            {
                w[sb + BsIdx * M + m] = i;
                m++;
            }
        }
        if (m == 1)
        {
            w[lb + w[sb + BsIdx * M]] = 1;
        }
        else if (m > 1)
        {
            // Bottom-up stable merge sort of the used symbols by count.
            int src = sb + BsIdx * M;
            int dst = sb + BsTmp * M;
            int width = 1;
            while (width < m)
            {
                int start = 0;
                while (start < m)
                {
                    int mid = start + width;
                    if (mid > m) mid = m;
                    int end = start + 2 * width;
                    if (end > m) end = m;
                    int a = start;
                    int b = mid;
                    int o = start;
                    while (o < end)
                    {
                        bool aValid = a < mid;
                        bool bValid = b < end;
                        bool takeLeft = aValid;
                        if (aValid & bValid)
                        {
                            int fa = w[hb + w[src + a]];
                            int fb = w[hb + w[src + b]];
                            takeLeft = fb >= fa;
                        }
                        if (takeLeft)
                        {
                            w[dst + o] = w[src + a];
                            a++;
                        }
                        else
                        {
                            w[dst + o] = w[src + b];
                            b++;
                        }
                        o++;
                    }
                    start += 2 * width;
                }
                int t = src;
                src = dst;
                dst = t;
                width *= 2;
            }

            int nw = sb + BsNw * M;
            int left = sb + BsLeft * M;
            int right = sb + BsRight * M;
            int dep = sb + BsDep * M;
            int countMin = 1;
            bool again = true;
            while (again)
            {
                for (int i = 0; i < m; i++)
                {
                    int f = w[hb + w[src + i]];
                    w[nw + i] = f > countMin ? f : countMin;
                }
                // Two-queue Huffman: leaves in ascending weight, internal nodes are created in ascending weight.
                int li = 0;
                int ii = m;
                int next = m;
                for (int step = 0; step < m - 1; step++)
                {
                    int pick0;
                    bool leaf0 = (li < m) & ((ii >= next) | (w[nw + (li < m ? li : 0)] <= w[nw + (ii < next ? ii : 0)]));
                    if (leaf0) { pick0 = li; li++; } else { pick0 = ii; ii++; }
                    int pick1;
                    bool leaf1 = (li < m) & ((ii >= next) | (w[nw + (li < m ? li : 0)] <= w[nw + (ii < next ? ii : 0)]));
                    if (leaf1) { pick1 = li; li++; } else { pick1 = ii; ii++; }
                    w[nw + next] = w[nw + pick0] + w[nw + pick1];
                    w[left + (next - m)] = pick0;
                    w[right + (next - m)] = pick1;
                    next++;
                }
                w[dep + next - 1] = 0;
                for (int k = next - 1; k >= m; k--)
                {
                    int d = w[dep + k] + 1;
                    w[dep + w[left + (k - m)]] = d;
                    w[dep + w[right + (k - m)]] = d;
                }
                int maxDepth = 0;
                for (int i = 0; i < m; i++)
                {
                    int d = w[dep + i];
                    if (d > maxDepth) maxDepth = d;
                }
                again = maxDepth > limit;
                if (again) countMin <<= 1;
            }
            for (int i = 0; i < m; i++) w[lb + w[src + i]] = w[dep + i];
        }
    }

    /// <summary>
    /// Canonical packed codes for the lengths at <c>w[lb ..]</c>: <c>(length &lt;&lt; 16) | bit-reversed code</c> at <c>w[cb ..]</c>
    /// (the stream carries codes MSB first, our writer LSB first). A code with at most one used symbol is zero bits long.
    /// Returns the number of used symbols. <paramref name="sb"/> needs 32 ints.
    /// </summary>
    public static int MakeCodes(ArrayView<int> w, int lb, int n, int cb, int sb)
    {
        int used = 0;
        for (int i = 0; i < 32; i++) w[sb + i] = 0;
        for (int s = 0; s < n; s++)
        {
            int l = w[lb + s];
            if (l > 0)
            {
                used++;
                w[sb + l] = w[sb + l] + 1;
            }
            w[cb + s] = 0;
        }
        if (used > 1)
        {
            int code = 0;
            for (int len = 1; len <= MaxLen; len++)
            {
                code = (code + w[sb + len - 1]) << 1;
                w[sb + 16 + len] = code;
            }
            for (int s = 0; s < n; s++)
            {
                int l = w[lb + s];
                if (l > 0)
                {
                    int c = w[sb + 16 + l];
                    w[sb + 16 + l] = c + 1;
                    int rev = 0;
                    int tmp = c;
                    for (int b = 0; b < l; b++)
                    {
                        rev = (rev << 1) | (tmp & 1);
                        tmp >>= 1;
                    }
                    w[cb + s] = (l << 16) | rev;
                }
            }
        }
        return used;
    }

    /// <summary>Builds lengths + packed codes of every code of a set; code <paramref name="k"/> uses scratch <paramref name="scratchFor"/>(k).</summary>
    public static void BuildOneCode(ArrayView<int> w, int set, int cacheSize, int k, int sb)
    {
        int setAlpha = SetAlpha(cacheSize);
        int n = Alpha(k, cacheSize);
        int off = AlphaOff(k, cacheSize);
        BuildLengths(w, set + off, n, set + setAlpha + off, sb, MaxLen);
        MakeCodes(w, set + setAlpha + off, n, set + 2 * setAlpha + off, sb + 8 * Vp8lEncLayout.MaxAlpha);
    }

    /// <summary>Total bits the data of code <paramref name="k"/> costs with its built code (histogram x length).</summary>
    public static int CodeCostBits(ArrayView<int> w, int set, int cacheSize, int k)
    {
        int setAlpha = SetAlpha(cacheSize);
        int n = Alpha(k, cacheSize);
        int off = AlphaOff(k, cacheSize);
        int bits = 0;
        int used = 0;
        for (int s = 0; s < n; s++)
        {
            bits += w[set + off + s] * (w[set + 2 * setAlpha + off + s] >> 16);
            if (w[set + setAlpha + off + s] > 0) used++;
        }
        // plus a rough size of the code definition (about 5 bits per used symbol)
        return bits + used * 5 + 30;
    }

    // ------------------------------------------------------------------
    // Writing prefix codes
    // ------------------------------------------------------------------

    /// <summary>Writes one symbol with the packed code at <c>w[cb + sym]</c>.</summary>
    public static void WriteSymbol(ref Vp8lBitWriter bw, ArrayView<byte> output, ArrayView<int> w, int cb, int sym)
    {
        int packed = w[cb + sym];
        PutBits(ref bw, output, (uint)(packed & 0xFFFF), packed >> 16);
    }

    /// <summary>
    /// Writes the definition of the prefix code whose lengths are at <c>w[lb ..]</c> (alphabet <paramref name="n"/>):
    /// the simple form when at most two symbols below 256 are used, the code-length-code form otherwise.
    /// <paramref name="ws"/> is a <see cref="Vp8lEncLayout.WriterScratch"/> region.
    /// </summary>
    public static void WriteCodeDefinition(ref Vp8lBitWriter bw, ArrayView<byte> output, ArrayView<int> w, int lb, int n, int ws)
    {
        int used = 0;
        int s0 = 0;
        int s1 = 0;
        for (int s = 0; s < n; s++)
        {
            if (w[lb + s] > 0)
            {
                if (used == 0) s0 = s;
                else if (used == 1) s1 = s;
                used++;
            }
        }

        bool simple = (used <= 2) & (s0 < 256) & (s1 < 256);
        if (simple)
        {
            // An empty code is written as the one-symbol code for symbol 0 (nothing is ever coded with it).
            PutBits(ref bw, output, 1, 1);
            PutBits(ref bw, output, (uint)(used == 2 ? 1 : 0), 1);
            bool eightBits = (used == 2) | (s0 > 1);
            PutBits(ref bw, output, (uint)(eightBits ? 1 : 0), 1);
            PutBits(ref bw, output, (uint)s0, eightBits ? 8 : 1);
            if (used == 2) PutBits(ref bw, output, (uint)s1, 8);
        }
        else
        {
            // ---- run-length tokens of the lengths ----
            int tokSym = ws + WsTokSym;
            int tokExtra = ws + WsTokExtra;
            int small = ws + WsSmall;
            int nt = 0;
            int prev = 8;
            int i = 0;
            while (i < n)
            {
                int v = w[lb + i];
                int run = 1;
                bool same = (i + run < n);
                while (same)
                {
                    bool eq = w[lb + i + run] == v;
                    if (eq) run++;
                    same = eq & (i + run < n);
                }
                i += run;
                if (v == 0)
                {
                    while (run >= 3)
                    {
                        if (run >= 11)
                        {
                            int r = run > 138 ? 138 : run;
                            w[tokSym + nt] = 18;
                            w[tokExtra + nt] = r - 11;
                            run -= r;
                        }
                        else
                        {
                            int r = run > 10 ? 10 : run;
                            w[tokSym + nt] = 17;
                            w[tokExtra + nt] = r - 3;
                            run -= r;
                        }
                        nt++;
                    }
                    while (run > 0)
                    {
                        w[tokSym + nt] = 0;
                        w[tokExtra + nt] = 0;
                        nt++;
                        run--;
                    }
                }
                else
                {
                    if (v != prev)
                    {
                        w[tokSym + nt] = v;
                        w[tokExtra + nt] = 0;
                        nt++;
                        run--;
                        prev = v;
                    }
                    while (run >= 3)
                    {
                        int r = run > 6 ? 6 : run;
                        w[tokSym + nt] = 16;
                        w[tokExtra + nt] = r - 3;
                        nt++;
                        run -= r;
                    }
                    while (run > 0)
                    {
                        w[tokSym + nt] = v;
                        w[tokExtra + nt] = 0;
                        nt++;
                        run--;
                    }
                }
            }

            // ---- code for the tokens ----
            for (int j = 0; j < 32; j++) w[small + SmHist + j] = 0;
            for (int t = 0; t < nt; t++)
            {
                int sy = w[tokSym + t];
                w[small + SmHist + sy] = w[small + SmHist + sy] + 1;
            }
            BuildLengths(w, small + SmHist, 19, small + SmLens, ws + WsBuild, 7);
            MakeCodes(w, small + SmLens, 19, small + SmCodes, small + SmCnt);

            // Trailing zero tokens need not be written: the decoder stops after max_symbol tokens.
            int trimmed = nt;
            bool trailing = trimmed > 0;
            while (trailing)
            {
                int sy = w[tokSym + trimmed - 1];
                bool isZero = (sy == 0) | (sy == 17) | (sy == 18);
                if (isZero) trimmed--;
                trailing = isZero & (trimmed > 0);
            }
            if (trimmed < 2) trimmed = nt < 2 ? nt : 2;

            // Number of code-length-code lengths to store: up to the last nonzero one in the fixed order.
            int ncl = 4;
            for (int j = 0; j < 19; j++)
            {
                int sy = CodeLengthOrder(j);
                if (w[small + SmLens + sy] > 0) ncl = j + 1 > 4 ? j + 1 : 4;
            }

            PutBits(ref bw, output, 0, 1);
            PutBits(ref bw, output, (uint)(ncl - 4), 4);
            for (int j = 0; j < ncl; j++)
                PutBits(ref bw, output, (uint)w[small + SmLens + CodeLengthOrder(j)], 3);

            int tokenCount = nt;
            bool useMax = (trimmed < nt) & (trimmed >= 2);
            if (useMax)
            {
                int nbits = 2;
                int vmax = trimmed - 2;
                bool grow = vmax >= (1 << nbits);
                while (grow)
                {
                    nbits += 2;
                    grow = vmax >= (1 << nbits);
                }
                PutBits(ref bw, output, 1, 1);
                PutBits(ref bw, output, (uint)((nbits - 2) / 2), 3);
                PutBits(ref bw, output, (uint)vmax, nbits);
                tokenCount = trimmed;
            }
            else
            {
                PutBits(ref bw, output, 0, 1);
            }
            for (int t = 0; t < tokenCount; t++)
            {
                int sy = w[tokSym + t];
                WriteSymbol(ref bw, output, w, small + SmCodes, sy);
                if (sy == 16) PutBits(ref bw, output, (uint)w[tokExtra + t], 2);
                else if (sy == 17) PutBits(ref bw, output, (uint)w[tokExtra + t], 3);
                else if (sy == 18) PutBits(ref bw, output, (uint)w[tokExtra + t], 7);
            }
        }
    }

    private static int CodeLengthOrder(int i)
    {
        int v;
        switch (i)
        {
            case 0: v = 17; break;
            case 1: v = 18; break;
            case 2: v = 0; break;
            case 3: v = 1; break;
            case 4: v = 2; break;
            case 5: v = 3; break;
            case 6: v = 4; break;
            case 7: v = 5; break;
            case 8: v = 16; break;
            case 9: v = 6; break;
            case 10: v = 7; break;
            case 11: v = 8; break;
            case 12: v = 9; break;
            case 13: v = 10; break;
            case 14: v = 11; break;
            case 15: v = 12; break;
            case 16: v = 13; break;
            case 17: v = 14; break;
            default: v = 15; break;
        }
        return v;
    }

    /// <summary>Writes the five code definitions of a built set.</summary>
    public static void WriteCodeSet(ref Vp8lBitWriter bw, ArrayView<byte> output, ArrayView<int> w, int set, int cacheSize, int ws)
    {
        int setAlpha = SetAlpha(cacheSize);
        for (int k = 0; k < 5; k++)
            WriteCodeDefinition(ref bw, output, w, set + setAlpha + AlphaOff(k, cacheSize), Alpha(k, cacheSize), ws);
    }

    /// <summary>
    /// Writes the entropy-coded image of <paramref name="count"/> pixels at <c>w[src ..]</c> (literals only, no colour cache,
    /// one prefix group): the form of every transform sub-image.
    /// </summary>
    public static void WriteSubImage(
        ref Vp8lBitWriter bw, ArrayView<byte> output, ArrayView<int> w, int src, int count, int sub, int build, int ws)
    {
        int setAlpha = SetAlpha(0);
        for (int i = 0; i < setAlpha; i++) w[sub + i] = 0;
        for (int i = 0; i < count; i++)
        {
            uint px = (uint)w[src + i];
            int g = sub + AlphaOff(0, 0) + (int)((px >> 8) & 0xFFu);
            int r = sub + AlphaOff(1, 0) + (int)((px >> 16) & 0xFFu);
            int b = sub + AlphaOff(2, 0) + (int)(px & 0xFFu);
            int a = sub + AlphaOff(3, 0) + (int)(px >> 24);
            w[g] = w[g] + 1;
            w[r] = w[r] + 1;
            w[b] = w[b] + 1;
            w[a] = w[a] + 1;
        }
        for (int k = 0; k < 5; k++) BuildOneCode(w, sub, 0, k, build);
        PutBits(ref bw, output, 0, 1); // no colour cache
        WriteCodeSet(ref bw, output, w, sub, 0, ws);
        int cg = sub + 2 * setAlpha + AlphaOff(0, 0);
        int cr = sub + 2 * setAlpha + AlphaOff(1, 0);
        int cb = sub + 2 * setAlpha + AlphaOff(2, 0);
        int ca = sub + 2 * setAlpha + AlphaOff(3, 0);
        for (int i = 0; i < count; i++)
        {
            uint px = (uint)w[src + i];
            WriteSymbol(ref bw, output, w, cg, (int)((px >> 8) & 0xFFu));
            WriteSymbol(ref bw, output, w, cr, (int)((px >> 16) & 0xFFu));
            WriteSymbol(ref bw, output, w, cb, (int)(px & 0xFFu));
            WriteSymbol(ref bw, output, w, ca, (int)(px >> 24));
        }
    }

    // ------------------------------------------------------------------
    // LZ77 prefix coding
    // ------------------------------------------------------------------

    /// <summary>
    /// Prefix code of a length / distance value (RFC 9649 section 5.2.2), packed as
    /// <c>code | (extraBits &lt;&lt; 8) | (extraValue &lt;&lt; 13)</c>.
    /// </summary>
    public static int PrefixEncode(int value)
    {
        int code;
        int extraBits = 0;
        int extraValue = 0;
        if (value > 2)
        {
            int d = value - 1;
            int highest = 0;
            int t = d;
            while (t > 1)
            {
                t >>= 1;
                highest++;
            }
            int second = (d >> (highest - 1)) & 1;
            extraBits = highest - 1;
            extraValue = d & ((1 << extraBits) - 1);
            code = 2 * highest + second;
        }
        else
        {
            code = value - 1;
        }
        return code | (extraBits << 8) | (extraValue << 13);
    }

    /// <summary>Distance code (1-based; above 120 means distance + 120) of a backward distance in an image of width <paramref name="xsize"/>.</summary>
    public static int DistanceToPlaneCode(int xsize, int dist, ArrayView<int> tables)
    {
        int yoffset = dist / xsize;
        int xoffset = dist - yoffset * xsize;
        int code = dist + 120;
        int idx = -1;
        if ((xoffset <= 8) & (yoffset < 8)) idx = yoffset * 16 + 8 - xoffset;
        else if ((xoffset > xsize - 8) & (yoffset < 7)) idx = (yoffset + 1) * 16 + 8 + (xsize - xoffset);
        if (idx >= 0)
        {
            int c = tables[Vp8lTables.PlaneToCode + idx];
            if (c >= 0)
            {
                // Accept the short code only if the decoder maps it back to exactly this distance.
                int dx = tables[Vp8lTables.PlaneCodes + c * 2];
                int dy = tables[Vp8lTables.PlaneCodes + c * 2 + 1];
                int back = dy * xsize + dx;
                if (back < 1) back = 1;
                if (back == dist) code = c + 1;
            }
        }
        return code;
    }

    // ------------------------------------------------------------------
    // Tokenisation: LZ77 + colour cache + histograms
    // ------------------------------------------------------------------

    private static uint CacheHash(uint argb, int shift) => (0x1e35a7bdu * argb) >> shift;

    private static int PairHash(uint a, uint b)
        => (int)(((a * 0x1e35a7bdu) ^ (b * 0x9E3779B1u)) >> (32 - Vp8lEncLayout.HashBits));

    private static int MatchLength(ArrayView<uint> pix, int src, int pos, int maxLen)
    {
        int l = 0;
        bool go = maxLen > 0;
        while (go)
        {
            bool eq = pix[src + l] == pix[pos + l];
            if (eq) l++;
            go = eq & (l < maxLen);
        }
        return l;
    }

    private static void HashInsert(ArrayView<uint> pix, ArrayView<int> w, int head, int prev, int p, int n)
    {
        if (p + 1 < n)
        {
            int h = PairHash(pix[p], pix[p + 1]);
            w[prev + p] = w[head + h];
            w[head + h] = p;
        }
    }

    /// <summary>
    /// Tokenises the <paramref name="n"/> pixels (<paramref name="xsize"/> wide): literals, colour cache hits and LZ77 copies,
    /// into <paramref name="tok"/> (two ints per token) and the histograms of the main code set.
    /// </summary>
    public static void Tokenize(
        ArrayView<uint> pix, int xsize, int ysize, ArrayView<int> w, ArrayView<int> tok, ArrayView<int> meta,
        ArrayView<int> tables, int cacheBits)
    {
        const int MaxCopy = 4096;
        const int MaxDist = (1 << 20) - 120;
        int n = xsize * ysize;
        int cacheSize = cacheBits > 0 ? 1 << cacheBits : 0;
        int cacheShift = 32 - cacheBits;
        int chainDepth = meta[WebPEncMeta.ChainDepth];
        int set = meta[WebPEncMeta.RegionMain];
        int cache = meta[WebPEncMeta.RegionCache];
        int head = meta[WebPEncMeta.RegionHead];
        int prev = meta[WebPEncMeta.RegionPrev];

        int setAlpha = SetAlpha(cacheSize);
        for (int i = 0; i < setAlpha; i++) w[set + i] = 0;
        for (int i = 0; i < cacheSize; i++) w[cache + i] = 0;
        int hashSize = 1 << Vp8lEncLayout.HashBits;
        for (int i = 0; i < hashSize; i++) w[head + i] = -1;

        int gOff = set + AlphaOff(0, cacheSize);
        int rOff = set + AlphaOff(1, cacheSize);
        int bOff = set + AlphaOff(2, cacheSize);
        int aOff = set + AlphaOff(3, cacheSize);
        int dOff = set + AlphaOff(4, cacheSize);

        int pos = 0;
        int tcount = 0;
        int extraTotal = 0;
        while (pos < n)
        {
            uint px = pix[pos];
            int maxLen = n - pos;
            if (maxLen > MaxCopy) maxLen = MaxCopy;
            int bestLen = 0;
            int bestDist = 0;
            if (pos > 0)
            {
                // run of equal pixels, and the pixel row above
                int l = MatchLength(pix, pos - 1, pos, maxLen);
                if (l > bestLen) { bestLen = l; bestDist = 1; }
                if (pos >= xsize)
                {
                    l = MatchLength(pix, pos - xsize, pos, maxLen);
                    if (l > bestLen) { bestLen = l; bestDist = xsize; }
                }
                if (chainDepth > 0)
                {
                    if (pos + 1 < n)
                    {
                        int cand = w[head + PairHash(px, pix[pos + 1])];
                        int depth = 0;
                        bool go = (cand >= 0) & (depth < chainDepth);
                        while (go)
                        {
                            int dist = pos - cand;
                            if (dist > MaxDist)
                            {
                                go = false;
                            }
                            else
                            {
                                l = MatchLength(pix, cand, pos, maxLen);
                                if (l > bestLen) { bestLen = l; bestDist = dist; }
                                cand = w[prev + cand];
                                depth++;
                                go = (cand >= 0) & (depth < chainDepth) & (bestLen < maxLen);
                            }
                        }
                    }
                }
            }

            bool useCopy = bestLen >= 4;
            if (bestLen == 3)
            {
                // A 3-pixel copy only pays when its distance has a short code.
                useCopy = DistanceToPlaneCode(xsize, bestDist, tables) <= 120;
            }

            if (useCopy)
            {
                int plane = DistanceToPlaneCode(xsize, bestDist, tables);
                int lp = PrefixEncode(bestLen);
                int dp = PrefixEncode(plane);
                tok[tcount * 2] = 2 | (bestLen << 2);
                tok[tcount * 2 + 1] = bestDist;
                tcount++;
                int lc = 256 + (lp & 0xFF);
                w[gOff + lc] = w[gOff + lc] + 1;
                int dc = dp & 0xFF;
                w[dOff + dc] = w[dOff + dc] + 1;
                extraTotal += ((lp >> 8) & 0x1F) + ((dp >> 8) & 0x1F);
                for (int i = 0; i < bestLen; i++)
                {
                    uint cp = pix[pos + i];
                    if (cacheBits > 0) w[cache + (int)CacheHash(cp, cacheShift)] = (int)cp;
                    HashInsert(pix, w, head, prev, pos + i, n);
                }
                pos += bestLen;
            }
            else
            {
                bool hit = false;
                int key = 0;
                if (cacheBits > 0)
                {
                    key = (int)CacheHash(px, cacheShift);
                    hit = (uint)w[cache + key] == px;
                }
                if (hit)
                {
                    tok[tcount * 2] = 1 | (key << 2);
                    tok[tcount * 2 + 1] = 0;
                    tcount++;
                    int gi = gOff + 280 + key;
                    w[gi] = w[gi] + 1;
                }
                else
                {
                    tok[tcount * 2] = 0;
                    tok[tcount * 2 + 1] = (int)px;
                    tcount++;
                    int gi = gOff + (int)((px >> 8) & 0xFFu);
                    int ri = rOff + (int)((px >> 16) & 0xFFu);
                    int bi = bOff + (int)(px & 0xFFu);
                    int ai = aOff + (int)(px >> 24);
                    w[gi] = w[gi] + 1;
                    w[ri] = w[ri] + 1;
                    w[bi] = w[bi] + 1;
                    w[ai] = w[ai] + 1;
                }
                if (cacheBits > 0) w[cache + (int)CacheHash(px, cacheShift)] = (int)px;
                HashInsert(pix, w, head, prev, pos, n);
                pos++;
            }
        }
        meta[WebPEncMeta.TokenCount] = tcount;
        meta[WebPEncMeta.ExtraBits] = extraTotal;
        meta[WebPEncMeta.CacheBits] = cacheBits;
    }

    /// <summary>Writes the tokens of the main image with the built main code set.</summary>
    public static void WriteTokens(
        ref Vp8lBitWriter bw, ArrayView<byte> output, ArrayView<int> w, ArrayView<int> tok, ArrayView<int> meta,
        ArrayView<int> tables, int xsize, int cacheBits)
    {
        int cacheSize = cacheBits > 0 ? 1 << cacheBits : 0;
        int set = meta[WebPEncMeta.RegionMain];
        int setAlpha = SetAlpha(cacheSize);
        int cg = set + 2 * setAlpha + AlphaOff(0, cacheSize);
        int cr = set + 2 * setAlpha + AlphaOff(1, cacheSize);
        int cb = set + 2 * setAlpha + AlphaOff(2, cacheSize);
        int ca = set + 2 * setAlpha + AlphaOff(3, cacheSize);
        int cd = set + 2 * setAlpha + AlphaOff(4, cacheSize);
        int count = meta[WebPEncMeta.TokenCount];
        for (int t = 0; t < count; t++)
        {
            int t0 = tok[t * 2];
            int type = t0 & 3;
            if (type == 0)
            {
                uint px = (uint)tok[t * 2 + 1];
                WriteSymbol(ref bw, output, w, cg, (int)((px >> 8) & 0xFFu));
                WriteSymbol(ref bw, output, w, cr, (int)((px >> 16) & 0xFFu));
                WriteSymbol(ref bw, output, w, cb, (int)(px & 0xFFu));
                WriteSymbol(ref bw, output, w, ca, (int)(px >> 24));
            }
            else if (type == 1)
            {
                WriteSymbol(ref bw, output, w, cg, 280 + (t0 >> 2));
            }
            else
            {
                int len = t0 >> 2;
                int dist = tok[t * 2 + 1];
                int lp = PrefixEncode(len);
                WriteSymbol(ref bw, output, w, cg, 256 + (lp & 0xFF));
                PutBits(ref bw, output, (uint)(lp >> 13), (lp >> 8) & 0x1F);
                int plane = DistanceToPlaneCode(xsize, dist, tables);
                int dp = PrefixEncode(plane);
                WriteSymbol(ref bw, output, w, cd, dp & 0xFF);
                PutBits(ref bw, output, (uint)(dp >> 13), (dp >> 8) & 0x1F);
            }
        }
    }
}
