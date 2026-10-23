// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP8L (WebP lossless) bitstream decoder, GPU-callable form. Written from the WebP Lossless Bitstream
// Specification (RFC 9649). The bitstream is one sequential dependency chain (variable-length prefix codes,
// LZ77 copies that read earlier output, a colour cache updated per pixel), so per the library's cardinal rule
// it runs as ONE GPU thread: still accelerator-resident, the compressed bytes and the decoded pixels never
// touch the host. That thread's work is cut into bounded launches (DecodeStep: a resumable state machine with a
// per-launch budget), so no launch holds the GPU long whatever the image size. The inverse transforms that
// follow are separate kernels.
//
// Shape: the same as InflateGpu - a state struct passed by ref into static helpers, no managed allocations,
// no static tables (the constant tables are an int buffer the host uploads once), loops that need an early
// exit use a flag instead of returning from inside, every scratch array is a region of a buffer the caller
// owns. Memory the decode needs:
//
//   work (int)  transform data, entropy image, colour cache, prefix-group map, code-length scratch
//               (layout: Vp8lLayout, a pure function of the image size)
//   pool (int)  the built prefix codes of the groups that are actually referenced, plus a group table. Its size
//               depends on the file, so the host passes a capacity; when it is too small the kernel stops with
//               WebPError.PoolTooSmall and reports the size it needs, and the host decodes again.
//   pix  (uint) the decoded main image (ARGB, one pixel per uint, width = the width AFTER colour indexing)
//
// A prefix code in the pool is: [0] header (-1 = ordinary code, >= 0 = the one symbol of a single-symbol code,
// which costs zero bits), [1..15] number of codes per length, [16..271] fast table indexed by the next 8 stream
// bits (entry = length << 16 | symbol, 0 = longer code), [272..] the symbols sorted by (length, value).

using ILGPU;

namespace SpawnDev.Codecs.Images.WebP;

/// <summary>Running state of the VP8L bit reader.</summary>
public struct Vp8lBitState
{
    /// <summary>Next byte index in the file.</summary>
    public int Pos;
    /// <summary>One past the last payload byte.</summary>
    public int End;
    /// <summary>Bit buffer: the low bits are the next bits of the stream (LSB first).</summary>
    public uint Buf;
    /// <summary>Number of bits in <see cref="Buf"/>, virtual zero bits past the end included.</summary>
    public int Cnt;
    /// <summary>Number of virtual zero bits appended past the end of the payload (they sit at the top of the buffer).</summary>
    public int Virt;
    /// <summary>First <see cref="WebPError"/> raised (0 = none). Once set the decode unwinds.</summary>
    public int Err;
}

/// <summary>Layout of the int work buffer: a pure function of the image size, shared by the host and the kernels.</summary>
public static class Vp8lLayout
{
    /// <summary>Colour cache entries reserved (11 bits).</summary>
    public const int CacheMax = 2048;
    /// <summary>Entries of the prefix-group map (16-bit group indices).</summary>
    public const int MapSize = 65536;
    /// <summary>Scratch ints: code lengths (4096), code-length-code block (304), analysis (64), code-length lengths (32).</summary>
    public const int ScratchInts = 4096 + 304 + 64 + 32;

    /// <summary>Pixels of the largest transform / entropy sub-image (block size 4).</summary>
    public static int Tiles4(int width, int height) => ((width + 3) >> 2) * ((height + 3) >> 2);

    /// <summary>Capacity of the transform data region.</summary>
    public static int TransformCapacity(int width, int height) => 2 * Tiles4(width, height) + 256;

    /// <summary>Start of the entropy image region.</summary>
    public static int EntropyBase(int width, int height) => TransformCapacity(width, height);

    /// <summary>Start of the colour cache region.</summary>
    public static int CacheBase(int width, int height) => EntropyBase(width, height) + Tiles4(width, height);

    /// <summary>Start of the prefix-group map.</summary>
    public static int MapBase(int width, int height) => CacheBase(width, height) + CacheMax;

    /// <summary>Start of the scratch region.</summary>
    public static int ScratchBase(int width, int height) => MapBase(width, height) + MapSize;

    /// <summary>Ints of the resumable decode state (see Vp8lDecodeGpu.DecodeStep).</summary>
    public const int StepInts = 32;

    /// <summary>Start of the resumable decode state.</summary>
    public static int StepBase(int width, int height) => ScratchBase(width, height) + ScratchInts;

    /// <summary>Total ints of the work buffer.</summary>
    public static long WorkSize(int width, int height) => (long)StepBase(width, height) + StepInts;
}

/// <summary>Constant tables the VP8L decoder reads (uploaded once by the host).</summary>
public static class Vp8lTables
{
    /// <summary>Offset of the 120 (dx, dy) pairs of the short distance codes.</summary>
    public const int PlaneCodes = 0;
    /// <summary>Offset of the code-length-code order (19 ints).</summary>
    public const int CodeLengthOrder = 240;
    /// <summary>Offset of the encoder's inverse map: <c>dy * 16 + 8 - dx</c> to short distance code - 1 (or -1), 128 ints.</summary>
    public const int PlaneToCode = 264;
    /// <summary>Total ints.</summary>
    public const int Size = 392;

    // RFC 9649 section 5.2.2: (xi, yi) of distance codes 1..120.
    private static readonly sbyte[] Planes =
    {
        0, 1, 1, 0, 1, 1, -1, 1, 0, 2, 2, 0, 1, 2, -1, 2, 2, 1, -2, 1, 2, 2, -2, 2,
        0, 3, 3, 0, 1, 3, -1, 3, 3, 1, -3, 1, 2, 3, -2, 3, 3, 2, -3, 2, 0, 4, 4, 0,
        1, 4, -1, 4, 4, 1, -4, 1, 3, 3, -3, 3, 2, 4, -2, 4, 4, 2, -4, 2, 0, 5, 3, 4,
        -3, 4, 4, 3, -4, 3, 5, 0, 1, 5, -1, 5, 5, 1, -5, 1, 2, 5, -2, 5, 5, 2, -5, 2,
        4, 4, -4, 4, 3, 5, -3, 5, 5, 3, -5, 3, 0, 6, 6, 0, 1, 6, -1, 6, 6, 1, -6, 1,
        2, 6, -2, 6, 6, 2, -6, 2, 4, 5, -4, 5, 5, 4, -5, 4, 3, 6, -3, 6, 6, 3, -6, 3,
        0, 7, 7, 0, 1, 7, -1, 7, 5, 5, -5, 5, 7, 1, -7, 1, 4, 6, -4, 6, 6, 4, -6, 4,
        2, 7, -2, 7, 7, 2, -7, 2, 3, 7, -3, 7, 7, 3, -7, 3, 5, 6, -5, 6, 6, 5, -6, 5,
        8, 0, 4, 7, -4, 7, 7, 4, -7, 4, 8, 1, 8, 2, 6, 6, -6, 6, 8, 3, 5, 7, -5, 7,
        7, 5, -7, 5, 8, 4, 6, 7, -6, 7, 7, 6, -7, 6, 8, 5, 7, 7, -7, 7, 8, 6, 8, 7,
    };

    private static readonly int[] LengthOrder = { 17, 18, 0, 1, 2, 3, 4, 5, 16, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };

    /// <summary>Builds the table buffer content.</summary>
    public static int[] Build()
    {
        var t = new int[Size];
        for (int i = 0; i < 240; i++) t[PlaneCodes + i] = Planes[i];
        for (int i = 0; i < 19; i++) t[CodeLengthOrder + i] = LengthOrder[i];
        for (int i = 0; i < 128; i++) t[PlaneToCode + i] = -1;
        for (int c = 0; c < 120; c++)
            t[PlaneToCode + Planes[2 * c + 1] * 16 + 8 - Planes[2 * c]] = c;
        return t;
    }
}

/// <summary>GPU-callable VP8L bitstream decoder.</summary>
public static class Vp8lDecodeGpu
{
    private const int FastOff = 16;
    private const int SymOff = 272;
    private const int MaxLen = 15;

    // scratch offsets, relative to the scratch base
    private const int ScLen = 0;
    private const int ScCl = 4096;
    private const int ScAna = 4096 + 304;
    private const int ScClLens = ScAna + 64;

    // ------------------------------------------------------------------
    // Bit reader
    // ------------------------------------------------------------------

    private static void Fill(ref Vp8lBitState st, ArrayView<byte> file)
    {
        while (st.Cnt <= 24)
        {
            uint b = 0;
            if (st.Pos < st.End)
            {
                b = file[st.Pos];
                st.Pos++;
            }
            else
            {
                st.Virt += 8;
            }
            st.Buf |= b << st.Cnt;
            st.Cnt += 8;
        }
    }

    private static void Consume(ref Vp8lBitState st, int n)
    {
        st.Buf >>= n;
        st.Cnt -= n;
        // The virtual bits are the topmost ones: having fewer bits than that left means they were consumed.
        bool overrun = (st.Cnt < st.Virt) & (st.Err == 0);
        if (overrun)
            st.Err = (int)WebPError.TruncatedBitstream;
    }

    /// <summary>Reads <paramref name="n"/> (0..24) bits, first bit read = least significant.</summary>
    private static int ReadBits(ref Vp8lBitState st, ArrayView<byte> file, int n)
    {
        Fill(ref st, file);
        int v = (int)(st.Buf & ((1u << n) - 1u));
        Consume(ref st, n);
        return v;
    }

    // ------------------------------------------------------------------
    // Prefix codes
    // ------------------------------------------------------------------

    /// <summary>
    /// Counts the lengths of <paramref name="n"/> code lengths at <paramref name="lb"/> into the analysis
    /// scratch at <paramref name="ab"/> and validates the code. Returns the ints a stored code needs (>= 1),
    /// or the negated <see cref="WebPError"/>.
    /// </summary>
    private static int AnalyzeCode(ArrayView<int> work, int lb, int n, int ab)
    {
        for (int i = 0; i < 16; i++) work[ab + i] = 0;
        int used = 0;
        int lastSym = 0;
        for (int s = 0; s < n; s++)
        {
            int l = work[lb + s];
            if (l > 0)
            {
                work[ab + l] = work[ab + l] + 1;
                used++;
                lastSym = s;
            }
        }
        int result;
        if (used == 0)
        {
            result = -(int)WebPError.BadPrefixCode;
        }
        else if (used == 1)
        {
            // A one-symbol code costs zero bits whatever length it was given.
            work[ab + 16] = lastSym;
            result = 1;
        }
        else
        {
            int left = 1;
            for (int len = 1; len <= MaxLen; len++)
            {
                // An over-subscribed code stays negative from here on, so the sign needs no early exit.
                left <<= 1;
                left -= work[ab + len];
            }
            if (left != 0)
            {
                result = -(int)WebPError.BadPrefixCode;
            }
            else
            {
                work[ab + 16] = -1;
                result = SymOff + used;
            }
        }
        return result;
    }

    /// <summary>Writes the code analysed by <see cref="AnalyzeCode"/> as a block at <paramref name="db"/> of <paramref name="dst"/>.</summary>
    private static void StoreCode(ArrayView<int> work, int lb, int n, int ab, ArrayView<int> dst, int db)
    {
        int hdr = work[ab + 16];
        dst[db] = hdr;
        if (hdr < 0)
        {
            for (int len = 1; len <= MaxLen; len++) dst[db + len] = work[ab + len];
            for (int i = 0; i < 256; i++) dst[db + FastOff + i] = 0;

            // First index of each length in the sorted symbol list, and the first canonical code of each length.
            int off = 0;
            int code = 0;
            for (int len = 1; len <= MaxLen; len++)
            {
                work[ab + 32 + len] = off;
                off += work[ab + len];
                code = (code + work[ab + len - 1]) << 1;
                work[ab + 48 + len] = code;
            }

            for (int s = 0; s < n; s++)
            {
                int l = work[lb + s];
                if (l > 0)
                {
                    int c = work[ab + 48 + l];
                    work[ab + 48 + l] = c + 1;
                    int pos = work[ab + 32 + l];
                    work[ab + 32 + l] = pos + 1;
                    dst[db + SymOff + pos] = s;
                    if (l <= 8)
                    {
                        // The stream delivers a code MSB first, so the table is indexed by the bit-reversed code.
                        int rev = 0;
                        int tmp = c;
                        for (int b = 0; b < l; b++)
                        {
                            rev = (rev << 1) | (tmp & 1);
                            tmp >>= 1;
                        }
                        int entry = (l << 16) | s;
                        for (int k = rev; k < 256; k += 1 << l) dst[db + FastOff + k] = entry;
                    }
                }
            }
        }
    }

    /// <summary>Decodes one symbol of the code stored at <paramref name="cb"/> in <paramref name="blk"/>.</summary>
    private static int ReadSymbol(ref Vp8lBitState st, ArrayView<byte> file, ArrayView<int> blk, int cb)
    {
        int hdr = blk[cb];
        int result = hdr;
        if (hdr < 0)
        {
            Fill(ref st, file);
            int e = blk[cb + FastOff + (int)(st.Buf & 255u)];
            if (e != 0)
            {
                result = e & 0xFFFF;
                Consume(ref st, e >> 16);
            }
            else
            {
                // Longer than 8 bits: walk the canonical code one bit at a time (puff.c style).
                int code = 0;
                int first = 0;
                int index = 0;
                int len = 1;
                int found = 0;
                result = 0;
                bool walk = len <= MaxLen;
                while (walk)
                {
                    code |= (int)((st.Buf >> (len - 1)) & 1u);
                    int count = blk[cb + len];
                    if (code - count < first)
                    {
                        result = blk[cb + SymOff + index + (code - first)];
                        found = 1;
                        Consume(ref st, len);
                    }
                    else
                    {
                        index += count;
                        first = (first + count) << 1;
                        code <<= 1;
                        len++;
                    }
                    walk = (found == 0) & (len <= MaxLen);
                }
                bool unresolved = (found == 0) & (st.Err == 0);
                if (unresolved)
                    st.Err = (int)WebPError.BadPrefixCode;
            }
        }
        return result;
    }

    /// <summary>Reads the definition of one prefix code (RFC 9649 section 5.2.1); the lengths land at scratch + <see cref="ScLen"/>.</summary>
    private static void ReadCodeLengths(
        ref Vp8lBitState st, ArrayView<byte> file, ArrayView<int> work, ArrayView<int> tables, int alpha, int sb)
    {
        int lenBase = sb + ScLen;
        for (int i = 0; i < alpha; i++) work[lenBase + i] = 0;

        int simple = ReadBits(ref st, file, 1);
        if (simple == 1)
        {
            int nsym = ReadBits(ref st, file, 1) + 1;
            int first8 = ReadBits(ref st, file, 1);
            int s0 = ReadBits(ref st, file, first8 == 0 ? 1 : 8);
            // A symbol outside the alphabet is ignored, exactly like libwebp (its length array is only read up to the alphabet).
            if (s0 < alpha) work[lenBase + s0] = 1;
            if (nsym == 2)
            {
                int s1 = ReadBits(ref st, file, 8);
                if (s1 < alpha) work[lenBase + s1] = 1;
            }
        }
        else
        {
            int ncl = ReadBits(ref st, file, 4) + 4;
            int clb = sb + ScClLens;
            for (int i = 0; i < 19; i++) work[clb + i] = 0;
            for (int i = 0; i < ncl; i++)
                work[clb + tables[Vp8lTables.CodeLengthOrder + i]] = ReadBits(ref st, file, 3);

            int size = AnalyzeCode(work, clb, 19, sb + ScAna);
            if (size < 0)
            {
                if (st.Err == 0) st.Err = -size;
            }
            else if (st.Err == 0)
            {
                StoreCode(work, clb, 19, sb + ScAna, work, sb + ScCl);
                int maxSymbol = alpha;
                if (ReadBits(ref st, file, 1) == 1)
                {
                    int lengthNbits = 2 + 2 * ReadBits(ref st, file, 3);
                    maxSymbol = 2 + ReadBits(ref st, file, lengthNbits);
                }
                bool tooMany = (maxSymbol > alpha) & (st.Err == 0);
                if (tooMany)
                    st.Err = (int)WebPError.BadCodeLengths;

                int sym = 0;
                int prev = 8;
                bool more = (sym < alpha) & (maxSymbol > 0) & (st.Err == 0);
                while (more)
                {
                    maxSymbol--;
                    int c = ReadSymbol(ref st, file, work, sb + ScCl);
                    if (st.Err == 0)
                    {
                        if (c < 16)
                        {
                            work[lenBase + sym] = c;
                            sym++;
                            if (c != 0) prev = c;
                        }
                        else
                        {
                            int extra = 7;
                            int offset = 11;
                            if (c == 16) { extra = 2; offset = 3; }
                            else if (c == 17) { extra = 3; offset = 3; }
                            int rep = ReadBits(ref st, file, extra) + offset;
                            if (sym + rep > alpha)
                            {
                                if (st.Err == 0) st.Err = (int)WebPError.BadCodeLengths;
                            }
                            else
                            {
                                int v = c == 16 ? prev : 0;
                                for (int r = 0; r < rep; r++) work[lenBase + sym + r] = v;
                                sym += rep;
                            }
                        }
                    }
                    more = (sym < alpha) & (maxSymbol > 0) & (st.Err == 0);
                }
            }
        }
    }

    /// <summary>
    /// Reads the five prefix codes of one group. When <paramref name="store"/> is set they are built into the pool
    /// and the pool offset of each is written to <c>pool[groupTab + k]</c>; otherwise they are only validated.
    /// </summary>
    private static void ReadGroup(
        ref Vp8lBitState st, ArrayView<byte> file, ArrayView<int> work, ArrayView<int> pool, ArrayView<int> meta,
        ArrayView<int> tables, int sb, int cacheSize, int store, int groupTab, int poolCap)
    {
        int k = 0;
        bool go = st.Err == 0;
        while (go)
        {
            int alpha = 256;
            if (k == 0) alpha = 256 + 24 + cacheSize;
            else if (k == 4) alpha = 40;

            ReadCodeLengths(ref st, file, work, tables, alpha, sb);
            if (st.Err == 0)
            {
                int size = AnalyzeCode(work, sb + ScLen, alpha, sb + ScAna);
                if (size < 0)
                {
                    st.Err = -size;
                }
                else if (store != 0)
                {
                    int used = meta[WebPMeta.PoolUsed];
                    int newUsed = used + size;
                    meta[WebPMeta.PoolUsed] = newUsed;
                    if (newUsed > meta[WebPMeta.PoolNeeded]) meta[WebPMeta.PoolNeeded] = newUsed;
                    if (newUsed <= poolCap)
                    {
                        StoreCode(work, sb + ScLen, alpha, sb + ScAna, pool, used);
                        pool[groupTab + k] = used;
                    }
                    else
                    {
                        st.Err = (int)WebPError.PoolTooSmall;
                    }
                }
            }
            k++;
            go = (k < 5) & (st.Err == 0);
        }
    }

    // ------------------------------------------------------------------
    // Pixels
    // ------------------------------------------------------------------

    /// <summary>Prefix-coded value of a length or distance symbol (RFC 9649 section 5.2.2).</summary>
    private static int ReadPrefixValue(ref Vp8lBitState st, ArrayView<byte> file, int sym)
    {
        int v;
        if (sym < 4)
        {
            v = sym + 1;
        }
        else
        {
            int extra = (sym - 2) >> 1;
            int offset = (2 + (sym & 1)) << extra;
            v = offset + ReadBits(ref st, file, extra) + 1;
        }
        return v;
    }

    private static uint CacheHash(uint argb, int shift)
    {
        return (0x1e35a7bdu * argb) >> shift;
    }

    /// <summary>
    /// Decodes pixels <paramref name="pos"/> .. of an <paramref name="xsize"/> x <paramref name="ysize"/> image with the groups
    /// at <paramref name="groupTab"/> until the image is complete, an error is raised or <paramref name="budget"/> is spent
    /// (one unit per pixel written). <paramref name="pos"/>, <paramref name="x"/>, <paramref name="y"/> carry over to the next
    /// launch. The destination is <paramref name="pix"/> (the main image) when <paramref name="toMain"/> is set, otherwise
    /// <c>work[dstBase ..]</c> (a sub-image). The colour cache is cleared by the caller before the first pixel.
    /// </summary>
    private static void DecodePixels(
        ref Vp8lBitState st, ArrayView<byte> file, ArrayView<int> work, ArrayView<int> pool, ArrayView<uint> pix,
        ArrayView<int> tables, int toMain, int dstBase, int xsize, int ysize, int cacheBits, int cacheBase,
        int metaBits, int entBase, int tilesPerRow, int groupTab, ref int pos, ref int x, ref int y, ref int budget)
    {
        int n = xsize * ysize;
        int cacheShift = 32 - cacheBits;
        bool running = (pos < n) & (budget > 0) & (st.Err == 0);
        while (running)
        {
            int g = 0;
            if (metaBits > 0) g = work[entBase + (y >> metaBits) * tilesPerRow + (x >> metaBits)];
            int gt = groupTab + g * 5;

            int s = ReadSymbol(ref st, file, pool, pool[gt]);
            if (st.Err != 0)
            {
                // truncated or corrupt: leave the loop through the loop condition
            }
            else if (s < 256)
            {
                int r = ReadSymbol(ref st, file, pool, pool[gt + 1]);
                int b = ReadSymbol(ref st, file, pool, pool[gt + 2]);
                int a = ReadSymbol(ref st, file, pool, pool[gt + 3]);
                if (st.Err == 0)
                {
                    uint px = ((uint)a << 24) | ((uint)r << 16) | ((uint)s << 8) | (uint)b;
                    if (toMain != 0) pix[pos] = px;
                    else work[dstBase + pos] = (int)px;
                    if (cacheBits > 0) work[cacheBase + (int)CacheHash(px, cacheShift)] = (int)px;
                    pos++;
                    budget--;
                    x++;
                    if (x == xsize)
                    {
                        x = 0;
                        y++;
                    }
                }
            }
            else if (s < 256 + 24)
            {
                int length = ReadPrefixValue(ref st, file, s - 256);
                int dsym = ReadSymbol(ref st, file, pool, pool[gt + 4]);
                int dcode = ReadPrefixValue(ref st, file, dsym);
                if (st.Err == 0)
                {
                    int dist;
                    if (dcode > 120)
                    {
                        dist = dcode - 120;
                    }
                    else
                    {
                        int dx = tables[Vp8lTables.PlaneCodes + (dcode - 1) * 2];
                        int dy = tables[Vp8lTables.PlaneCodes + (dcode - 1) * 2 + 1];
                        dist = dy * xsize + dx;
                        if (dist < 1) dist = 1;
                    }
                    bool bad = (dist > pos) | (length > n - pos);
                    if (bad)
                    {
                        st.Err = (int)WebPError.BadBackReference;
                    }
                    else
                    {
                        // Byte-for-byte: source and destination ranges may overlap (dist < length).
                        for (int i = 0; i < length; i++)
                        {
                            uint px;
                            if (toMain != 0)
                            {
                                px = pix[pos - dist];
                                pix[pos] = px;
                            }
                            else
                            {
                                px = (uint)work[dstBase + pos - dist];
                                work[dstBase + pos] = (int)px;
                            }
                            if (cacheBits > 0) work[cacheBase + (int)CacheHash(px, cacheShift)] = (int)px;
                            pos++;
                        }
                        budget -= length;
                        int nx = x + length;
                        y += nx / xsize;
                        x = nx - (nx / xsize) * xsize;
                    }
                }
            }
            else
            {
                int idx = s - 256 - 24;
                if (cacheBits == 0)
                {
                    st.Err = (int)WebPError.BadCacheIndex;
                }
                else
                {
                    uint px = (uint)work[cacheBase + idx];
                    if (toMain != 0) pix[pos] = px;
                    else work[dstBase + pos] = (int)px;
                    work[cacheBase + (int)CacheHash(px, cacheShift)] = (int)px;
                    pos++;
                    budget--;
                    x++;
                    if (x == xsize)
                    {
                        x = 0;
                        y++;
                    }
                }
            }
            running = (pos < n) & (budget > 0) & (st.Err == 0);
        }
    }

    // ------------------------------------------------------------------
    // Whole image: a resumable state machine
    // ------------------------------------------------------------------
    //
    // The decode runs as a chain of launches. Each launch continues from the state in work[Vp8lLayout.StepBase ..] and
    // stops when it has spent StepBudget units (a decoded pixel = 1, a prefix-code group = GroupCost, a step of the entropy
    // map = 1), so no launch holds the GPU long whatever the image size; a launch after the end does nothing. The phases:
    //   Start -> Transform* (each predictor / cross-colour / colour-indexing transform decodes a sub-image:
    //   SubBegin -> SubPixels -> SubEnd) -> Cache (colour cache bits, meta prefix codes: the entropy sub-image, then the
    //   group map in EntMax / EntClear / EntMark / EntCompact / EntRemap) -> GroupsBegin -> Groups -> MainBegin ->
    //   MainPixels -> Done -> Finished.
    // An error (st.Err) jumps to Done. The host zeroes the state before the first launch (Phase 0 = Start).

    /// <summary>Work units one launch of <see cref="DecodeStep"/> may spend.</summary>
    public const int StepBudget = 4096;
    /// <summary>Units a prefix-code group costs (five code definitions, up to 2328 code lengths, table builds).</summary>
    private const int GroupCost = 512;

    private const int PhStart = 0, PhTransform = 1, PhSubBegin = 2, PhSubPixels = 3, PhSubEnd = 4, PhCache = 5,
        PhEntMax = 6, PhEntClear = 7, PhEntMark = 8, PhEntCompact = 9, PhEntRemap = 10, PhGroupsBegin = 11, PhGroups = 12,
        PhMainBegin = 13, PhMainPixels = 14, PhDone = 15;

    /// <summary>Phase value once the decode (or its error) is final.</summary>
    public const int PhaseFinished = 16;

    // state slots, relative to Vp8lLayout.StepBase
    /// <summary>Slot of the phase in the step state (the host polls it: <see cref="PhaseFinished"/> = done).</summary>
    public const int SPhase = 0;
    private const int SPos = 1, SEnd = 2, SBuf = 3, SCnt = 4, SVirt = 5, SErr = 6,
        SXsize = 7, SXfPos = 8, SNx = 9, SSeen = 10,
        SSubKind = 11, SSubW = 12, SSubH = 13, SSubDst = 14, SSubCacheBits = 15, SSubGroupTab = 16, SSubMark = 17, SSubRec = 18,
        SPixPos = 19, SPixX = 20, SPixY = 21,
        SCacheBits = 22, SMetaBits = 23, SGroups = 24, SGroupsUsed = 25, STilesPerRow = 26, SGroupTab = 27, SGi = 28,
        SEntI = 29, SEntMax = 30;

    private const int SubBlockTransform = 0, SubPalette = 1, SubEntropy = 2;

    private static Vp8lBitState LoadBits(ArrayView<int> work, int b) => new Vp8lBitState
    {
        Pos = work[b + SPos],
        End = work[b + SEnd],
        Buf = (uint)work[b + SBuf],
        Cnt = work[b + SCnt],
        Virt = work[b + SVirt],
        Err = work[b + SErr],
    };

    private static void SaveBits(ArrayView<int> work, int b, ref Vp8lBitState st)
    {
        work[b + SPos] = st.Pos;
        work[b + SEnd] = st.End;
        work[b + SBuf] = (int)st.Buf;
        work[b + SCnt] = st.Cnt;
        work[b + SVirt] = st.Virt;
        work[b + SErr] = st.Err;
    }

    /// <summary>Starts decoding a sub-image (colour cache bits + its one group of prefix codes) into work[dst ..].</summary>
    private static void BeginSubImage(ArrayView<int> work, int b, int kind, int w, int h, int dst, int rec)
    {
        work[b + SSubKind] = kind;
        work[b + SSubW] = w;
        work[b + SSubH] = h;
        work[b + SSubDst] = dst;
        work[b + SSubRec] = rec;
        work[b + SPhase] = PhSubBegin;
    }

    private static int CeilDiv(int a, int b) => (a + b - 1) / b;

    /// <summary>
    /// One launch of the VP8L decode of <c>file[payOff .. payOff + payLen)</c> (the 5-byte header included, already
    /// validated by the container scan): continues the state machine for up to <see cref="StepBudget"/> units. Writes the
    /// main image to <paramref name="pix"/> and, once finished, the facts the host needs (transform list, errors, pool use)
    /// to <paramref name="meta"/>.
    /// </summary>
    public static void DecodeStep(
        ArrayView<byte> file, int payOff, int payLen, int width, int height, int poolCap,
        ArrayView<int> meta, ArrayView<int> work, ArrayView<int> pool, ArrayView<uint> pix, ArrayView<int> tables)
    {
        int b = Vp8lLayout.StepBase(width, height);
        int entBase = Vp8lLayout.EntropyBase(width, height);
        int cacheBase = Vp8lLayout.CacheBase(width, height);
        int mapBase = Vp8lLayout.MapBase(width, height);
        int sb = Vp8lLayout.ScratchBase(width, height);

        int phase = work[b + SPhase];
        if (phase == PhStart)
        {
            for (int i = WebPMeta.DecodeError; i < WebPMeta.Size; i++) meta[i] = 0;
            work[b + SPos] = payOff + 5;
            work[b + SEnd] = payOff + payLen;
            work[b + SBuf] = 0;
            work[b + SCnt] = 0;
            work[b + SVirt] = 0;
            work[b + SErr] = 0;
            work[b + SXsize] = width;
            work[b + SXfPos] = 0;
            work[b + SNx] = 0;
            work[b + SSeen] = 0;
            phase = PhTransform;
            work[b + SPhase] = phase;
        }
        var st = LoadBits(work, b);
        int budget = StepBudget;
        bool go = (phase != PhaseFinished) & (budget > 0);
        while (go)
        {
            if (phase == PhTransform)
            {
                budget--;
                if (ReadBits(ref st, file, 1) == 0)
                {
                    work[b + SPhase] = PhCache;
                }
                else
                {
                    int type = ReadBits(ref st, file, 2);
                    int seen = work[b + SSeen];
                    int nx = work[b + SNx];
                    int xsize = work[b + SXsize];
                    if (((seen >> type) & 1) != 0)
                    {
                        if (st.Err == 0) st.Err = (int)WebPError.BadTransform;
                    }
                    else
                    {
                        work[b + SSeen] = seen | (1 << type);
                        int rec = WebPMeta.Transforms + nx * WebPMeta.TransformStride;
                        meta[rec + WebPMeta.TrKind] = type;
                        meta[rec + WebPMeta.TrWidth] = xsize;
                        bool blockTransform = (type == WebPMeta.KindPredictor) | (type == WebPMeta.KindCrossColor);
                        if (blockTransform)
                        {
                            int bits = ReadBits(ref st, file, 3) + 2;
                            meta[rec + WebPMeta.TrBits] = bits;
                            meta[rec + WebPMeta.TrData] = work[b + SXfPos];
                            BeginSubImage(work, b, SubBlockTransform, CeilDiv(xsize, 1 << bits), CeilDiv(height, 1 << bits), work[b + SXfPos], rec);
                        }
                        else if (type == WebPMeta.KindColorIndexing)
                        {
                            int count = ReadBits(ref st, file, 8) + 1;
                            meta[rec + WebPMeta.TrData] = work[b + SXfPos];
                            meta[rec + WebPMeta.TrCount] = count;
                            BeginSubImage(work, b, SubPalette, count, 1, work[b + SXfPos], rec);
                        }
                        else
                        {
                            work[b + SNx] = nx + 1; // subtract green: no data
                        }
                    }
                }
            }
            else if (phase == PhSubBegin)
            {
                budget--;
                int cacheBits = 0;
                if (ReadBits(ref st, file, 1) == 1)
                {
                    cacheBits = ReadBits(ref st, file, 4);
                    bool badBits = ((cacheBits < 1) | (cacheBits > 11)) & (st.Err == 0);
                    if (badBits)
                        st.Err = (int)WebPError.BadCacheBits;
                }
                if (st.Err == 0)
                {
                    int cacheSize = cacheBits > 0 ? 1 << cacheBits : 0;
                    int mark = meta[WebPMeta.PoolUsed];
                    int after = mark + 5;
                    meta[WebPMeta.PoolUsed] = after;
                    if (after > meta[WebPMeta.PoolNeeded]) meta[WebPMeta.PoolNeeded] = after;
                    work[b + SSubCacheBits] = cacheBits;
                    work[b + SSubGroupTab] = mark;
                    work[b + SSubMark] = mark;
                    if (after > poolCap)
                    {
                        st.Err = (int)WebPError.PoolTooSmall;
                    }
                    else
                    {
                        ReadGroup(ref st, file, work, pool, meta, tables, sb, cacheSize, 1, mark, poolCap);
                        budget -= GroupCost;
                        for (int i = 0; i < cacheSize; i++) work[cacheBase + i] = 0;
                        work[b + SPixPos] = 0;
                        work[b + SPixX] = 0;
                        work[b + SPixY] = 0;
                        work[b + SPhase] = PhSubPixels;
                    }
                }
            }
            else if (phase == PhSubPixels)
            {
                int pos = work[b + SPixPos];
                int x = work[b + SPixX];
                int y = work[b + SPixY];
                int w = work[b + SSubW];
                int h = work[b + SSubH];
                DecodePixels(ref st, file, work, pool, pix, tables, 0, work[b + SSubDst], w, h, work[b + SSubCacheBits], cacheBase,
                    0, 0, 0, work[b + SSubGroupTab], ref pos, ref x, ref y, ref budget);
                work[b + SPixPos] = pos;
                work[b + SPixX] = x;
                work[b + SPixY] = y;
                if (pos >= w * h) work[b + SPhase] = PhSubEnd;
            }
            else if (phase == PhSubEnd)
            {
                budget--;
                // the sub-image's prefix codes live in the pool only while it decodes
                meta[WebPMeta.PoolUsed] = work[b + SSubMark];
                int kind = work[b + SSubKind];
                int w = work[b + SSubW];
                int h = work[b + SSubH];
                if (kind == SubEntropy)
                {
                    work[b + SEntI] = 0;
                    work[b + SEntMax] = 0;
                    work[b + SPhase] = PhEntMax;
                }
                else
                {
                    int xfPos = work[b + SXfPos];
                    if (kind == SubPalette)
                    {
                        int count = w;
                        // The palette is delta coded: each entry adds to the previous one, channel by channel.
                        for (int i = 1; i < count; i++)
                        {
                            uint cur = (uint)work[xfPos + i];
                            uint prv = (uint)work[xfPos + i - 1];
                            uint ag = (cur & 0xff00ff00u) + (prv & 0xff00ff00u);
                            uint rb = (cur & 0x00ff00ffu) + (prv & 0x00ff00ffu);
                            work[xfPos + i] = (int)((ag & 0xff00ff00u) | (rb & 0x00ff00ffu));
                        }
                        int wbits = 0;
                        if (count <= 2) wbits = 3;
                        else if (count <= 4) wbits = 2;
                        else if (count <= 16) wbits = 1;
                        meta[work[b + SSubRec] + WebPMeta.TrBits] = wbits;
                        work[b + SXsize] = CeilDiv(work[b + SXsize], 1 << wbits);
                    }
                    work[b + SXfPos] = xfPos + w * h;
                    work[b + SNx] = work[b + SNx] + 1;
                    work[b + SPhase] = PhTransform;
                }
            }
            else if (phase == PhCache)
            {
                budget--;
                int xsize = work[b + SXsize];
                meta[WebPMeta.TransformCount] = work[b + SNx];
                meta[WebPMeta.PackedWidth] = xsize;
                int cacheBits = 0;
                if (ReadBits(ref st, file, 1) == 1)
                {
                    cacheBits = ReadBits(ref st, file, 4);
                    bool badBits = ((cacheBits < 1) | (cacheBits > 11)) & (st.Err == 0);
                    if (badBits)
                        st.Err = (int)WebPError.BadCacheBits;
                }
                meta[WebPMeta.CacheBits] = cacheBits;
                work[b + SCacheBits] = cacheBits;
                work[b + SMetaBits] = 0;
                work[b + SGroups] = 1;
                work[b + SGroupsUsed] = 1;
                work[b + STilesPerRow] = 0;
                if (st.Err == 0)
                {
                    if (ReadBits(ref st, file, 1) == 1)
                    {
                        int metaBits = ReadBits(ref st, file, 3) + 2;
                        int ew = CeilDiv(xsize, 1 << metaBits);
                        work[b + SMetaBits] = metaBits;
                        work[b + STilesPerRow] = ew;
                        BeginSubImage(work, b, SubEntropy, ew, CeilDiv(height, 1 << metaBits), entBase, 0);
                    }
                    else
                    {
                        work[b + SPhase] = PhGroupsBegin;
                    }
                }
            }
            else if (phase == PhEntMax)
            {
                // The group index is red + green of each entropy pixel. Only referenced groups get tables; the others
                // are parsed (they take bits) and validated but not stored.
                int entN = work[b + SSubW] * work[b + SSubH];
                int i = work[b + SEntI];
                int maxIdx = work[b + SEntMax];
                int end = i + budget;
                if (end > entN) end = entN;
                budget -= end - i;
                for (; i < end; i++)
                {
                    int gi = (work[entBase + i] >> 8) & 0xFFFF;
                    if (gi > maxIdx) maxIdx = gi;
                }
                work[b + SEntMax] = maxIdx;
                work[b + SEntI] = i;
                if (i >= entN)
                {
                    work[b + SGroups] = maxIdx + 1;
                    work[b + SEntI] = 0;
                    work[b + SPhase] = PhEntClear;
                }
            }
            else if (phase == PhEntClear)
            {
                int groups = work[b + SGroups];
                int i = work[b + SEntI];
                int end = i + budget;
                if (end > groups) end = groups;
                budget -= end - i;
                for (; i < end; i++) work[mapBase + i] = -1;
                work[b + SEntI] = i;
                if (i >= groups)
                {
                    work[b + SEntI] = 0;
                    work[b + SPhase] = PhEntMark;
                }
            }
            else if (phase == PhEntMark)
            {
                int entN = work[b + SSubW] * work[b + SSubH];
                int i = work[b + SEntI];
                int end = i + budget;
                if (end > entN) end = entN;
                budget -= end - i;
                for (; i < end; i++) work[mapBase + ((work[entBase + i] >> 8) & 0xFFFF)] = -2;
                work[b + SEntI] = i;
                if (i >= entN)
                {
                    work[b + SEntI] = 0;
                    work[b + SGroupsUsed] = 0;
                    work[b + SPhase] = PhEntCompact;
                }
            }
            else if (phase == PhEntCompact)
            {
                int groups = work[b + SGroups];
                int used = work[b + SGroupsUsed];
                int i = work[b + SEntI];
                int end = i + budget;
                if (end > groups) end = groups;
                budget -= end - i;
                for (; i < end; i++)
                {
                    if (work[mapBase + i] == -2)
                    {
                        work[mapBase + i] = used;
                        used++;
                    }
                }
                work[b + SGroupsUsed] = used;
                work[b + SEntI] = i;
                if (i >= groups)
                {
                    work[b + SEntI] = 0;
                    work[b + SPhase] = PhEntRemap;
                }
            }
            else if (phase == PhEntRemap)
            {
                int entN = work[b + SSubW] * work[b + SSubH];
                int i = work[b + SEntI];
                int end = i + budget;
                if (end > entN) end = entN;
                budget -= end - i;
                for (; i < end; i++) work[entBase + i] = work[mapBase + ((work[entBase + i] >> 8) & 0xFFFF)];
                work[b + SEntI] = i;
                if (i >= entN) work[b + SPhase] = PhGroupsBegin;
            }
            else if (phase == PhGroupsBegin)
            {
                budget--;
                meta[WebPMeta.MetaBits] = work[b + SMetaBits];
                meta[WebPMeta.Groups] = work[b + SGroups];
                meta[WebPMeta.GroupsUsed] = work[b + SGroupsUsed];
                int groupTab = meta[WebPMeta.PoolUsed];
                int after = groupTab + 5 * work[b + SGroupsUsed];
                meta[WebPMeta.PoolUsed] = after;
                if (after > meta[WebPMeta.PoolNeeded]) meta[WebPMeta.PoolNeeded] = after;
                if (after > poolCap) st.Err = (int)WebPError.PoolTooSmall;
                work[b + SGroupTab] = groupTab;
                work[b + SGi] = 0;
                work[b + SPhase] = work[b + SGroups] > 0 ? PhGroups : PhMainBegin;
            }
            else if (phase == PhGroups)
            {
                int gi = work[b + SGi];
                int id = 0;
                if (work[b + SMetaBits] > 0) id = work[mapBase + gi];
                int store = id >= 0 ? 1 : 0;
                int tab = work[b + SGroupTab] + (id >= 0 ? id : 0) * 5;
                int cacheSizeMain = work[b + SCacheBits] > 0 ? 1 << work[b + SCacheBits] : 0;
                ReadGroup(ref st, file, work, pool, meta, tables, sb, cacheSizeMain, store, tab, poolCap);
                budget -= GroupCost;
                gi++;
                work[b + SGi] = gi;
                if (gi >= work[b + SGroups]) work[b + SPhase] = PhMainBegin;
            }
            else if (phase == PhMainBegin)
            {
                budget--;
                int cacheBits = work[b + SCacheBits];
                int cacheSize = cacheBits > 0 ? 1 << cacheBits : 0;
                for (int i = 0; i < cacheSize; i++) work[cacheBase + i] = 0;
                work[b + SPixPos] = 0;
                work[b + SPixX] = 0;
                work[b + SPixY] = 0;
                work[b + SPhase] = PhMainPixels;
            }
            else if (phase == PhMainPixels)
            {
                int pos = work[b + SPixPos];
                int x = work[b + SPixX];
                int y = work[b + SPixY];
                int xsize = work[b + SXsize];
                DecodePixels(ref st, file, work, pool, pix, tables, 1, 0, xsize, height, work[b + SCacheBits], cacheBase,
                    work[b + SMetaBits], entBase, work[b + STilesPerRow], work[b + SGroupTab], ref pos, ref x, ref y, ref budget);
                work[b + SPixPos] = pos;
                work[b + SPixX] = x;
                work[b + SPixY] = y;
                if (pos >= xsize * height) work[b + SPhase] = PhDone;
            }
            else if (phase == PhDone)
            {
                budget--;
                meta[WebPMeta.BitsConsumed] = (st.Pos - payOff) * 8 - st.Cnt + st.Virt;
                meta[WebPMeta.DecodeError] = st.Err;
                work[b + SPhase] = PhaseFinished;
            }
            // an error ends the decode (Done records it), from whatever phase raised it
            bool failed = (st.Err != 0) & (work[b + SPhase] != PhaseFinished);
            if (failed) work[b + SPhase] = PhDone;
            phase = work[b + SPhase];
            go = (phase != PhaseFinished) & (budget > 0);
        }
        SaveBits(work, b, ref st);
    }
}
