// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP8L (WebP lossless) bitstream decoder, GPU-callable form. Written from the WebP Lossless Bitstream
// Specification (RFC 9649). The bitstream is one sequential dependency chain (variable-length prefix codes,
// LZ77 copies that read earlier output, a colour cache updated per pixel), so per the library's cardinal rule
// it runs as ONE GPU thread: still accelerator-resident, the compressed bytes and the decoded pixels never
// touch the host. The inverse transforms that follow are separate kernels (Vp8lInverseKernels).
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

    /// <summary>Total ints of the work buffer.</summary>
    public static long WorkSize(int width, int height) => (long)ScratchBase(width, height) + ScratchInts;
}

/// <summary>Constant tables the VP8L decoder reads (uploaded once by the host).</summary>
public static class Vp8lTables
{
    /// <summary>Offset of the 120 (dx, dy) pairs of the short distance codes.</summary>
    public const int PlaneCodes = 0;
    /// <summary>Offset of the code-length-code order (19 ints).</summary>
    public const int CodeLengthOrder = 240;
    /// <summary>Total ints.</summary>
    public const int Size = 264;

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
    /// Decodes <paramref name="xsize"/> x <paramref name="ysize"/> pixels with the groups at
    /// <paramref name="groupTab"/>. The destination is <paramref name="pix"/> (the main image) when
    /// <paramref name="toMain"/> is set, otherwise <c>work[dstBase ..]</c> (a sub-image).
    /// </summary>
    private static void DecodePixels(
        ref Vp8lBitState st, ArrayView<byte> file, ArrayView<int> work, ArrayView<int> pool, ArrayView<uint> pix,
        ArrayView<int> tables, int toMain, int dstBase, int xsize, int ysize, int cacheBits, int cacheBase,
        int metaBits, int entBase, int tilesPerRow, int groupTab)
    {
        int n = xsize * ysize;
        int cacheShift = 32 - cacheBits;
        if (cacheBits > 0)
        {
            int cacheSize = 1 << cacheBits;
            for (int i = 0; i < cacheSize; i++) work[cacheBase + i] = 0;
        }

        int pos = 0;
        int x = 0;
        int y = 0;
        bool running = n > 0;
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
                    x++;
                    if (x == xsize)
                    {
                        x = 0;
                        y++;
                    }
                }
            }
            running = (pos < n) & (st.Err == 0);
        }
    }

    /// <summary>
    /// An entropy-coded image without meta prefix codes (transform data, palette, entropy image): colour cache
    /// info, one group of prefix codes, the pixels into <c>work[dstBase ..]</c>. Its prefix codes live in the pool
    /// only while it decodes.
    /// </summary>
    private static void DecodeSubImage(
        ref Vp8lBitState st, ArrayView<byte> file, ArrayView<int> meta, ArrayView<int> work, ArrayView<int> pool,
        ArrayView<uint> pix, ArrayView<int> tables, int sb, int w, int h, int dstBase, int cacheBase, int poolCap)
    {
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
            int groupTab = mark;
            int after = mark + 5;
            meta[WebPMeta.PoolUsed] = after;
            if (after > meta[WebPMeta.PoolNeeded]) meta[WebPMeta.PoolNeeded] = after;
            if (after > poolCap)
            {
                st.Err = (int)WebPError.PoolTooSmall;
            }
            else
            {
                ReadGroup(ref st, file, work, pool, meta, tables, sb, cacheSize, 1, groupTab, poolCap);
                if (st.Err == 0)
                    DecodePixels(ref st, file, work, pool, pix, tables, 0, dstBase, w, h, cacheBits, cacheBase, 0, 0, 0, groupTab);
            }
            meta[WebPMeta.PoolUsed] = mark;
        }
    }

    // ------------------------------------------------------------------
    // Whole image
    // ------------------------------------------------------------------

    private static int CeilDiv(int a, int b) => (a + b - 1) / b;

    /// <summary>
    /// Decodes the VP8L stream at <c>file[payOff .. payOff + payLen)</c> (the 5-byte header included, already
    /// validated by the container scan). Writes the main image to <paramref name="pix"/> and the facts the host
    /// needs (transform list, errors, pool use) to <paramref name="meta"/>.
    /// </summary>
    public static void Decode(
        ArrayView<byte> file, int payOff, int payLen, int width, int height, int poolCap,
        ArrayView<int> meta, ArrayView<int> work, ArrayView<int> pool, ArrayView<uint> pix, ArrayView<int> tables)
    {
        for (int i = WebPMeta.DecodeError; i < WebPMeta.Size; i++) meta[i] = 0;

        var st = new Vp8lBitState
        {
            Pos = payOff + 5,
            End = payOff + payLen,
            Buf = 0,
            Cnt = 0,
            Virt = 0,
            Err = 0,
        };

        int tiles4 = Vp8lLayout.Tiles4(width, height);
        int entBase = Vp8lLayout.EntropyBase(width, height);
        int cacheBase = Vp8lLayout.CacheBase(width, height);
        int mapBase = Vp8lLayout.MapBase(width, height);
        int sb = Vp8lLayout.ScratchBase(width, height);

        // ---- transforms ----
        int xsize = width;
        int xfPos = 0;
        int nx = 0;
        int seen = 0;
        int more = 1;
        bool tloop = true;
        while (tloop)
        {
            if (ReadBits(ref st, file, 1) == 0)
            {
                more = 0;
            }
            else
            {
                int type = ReadBits(ref st, file, 2);
                if (((seen >> type) & 1) != 0)
                {
                    if (st.Err == 0) st.Err = (int)WebPError.BadTransform;
                }
                else
                {
                    seen |= 1 << type;
                    int rec = WebPMeta.Transforms + nx * WebPMeta.TransformStride;
                    meta[rec + WebPMeta.TrKind] = type;
                    meta[rec + WebPMeta.TrWidth] = xsize;
                    bool blockTransform = (type == WebPMeta.KindPredictor) | (type == WebPMeta.KindCrossColor);
                    if (blockTransform)
                    {
                        int bits = ReadBits(ref st, file, 3) + 2;
                        int bw = CeilDiv(xsize, 1 << bits);
                        int bh = CeilDiv(height, 1 << bits);
                        meta[rec + WebPMeta.TrBits] = bits;
                        meta[rec + WebPMeta.TrData] = xfPos;
                        if (st.Err == 0)
                        {
                            DecodeSubImage(ref st, file, meta, work, pool, pix, tables, sb, bw, bh, xfPos, cacheBase, poolCap);
                            xfPos += bw * bh;
                        }
                    }
                    else if (type == WebPMeta.KindColorIndexing)
                    {
                        int count = ReadBits(ref st, file, 8) + 1;
                        meta[rec + WebPMeta.TrData] = xfPos;
                        meta[rec + WebPMeta.TrCount] = count;
                        if (st.Err == 0)
                        {
                            DecodeSubImage(ref st, file, meta, work, pool, pix, tables, sb, count, 1, xfPos, cacheBase, poolCap);
                            if (st.Err == 0)
                            {
                                // The palette is delta coded: each entry adds to the previous one, channel by channel.
                                for (int i = 1; i < count; i++)
                                {
                                    uint cur = (uint)work[xfPos + i];
                                    uint prv = (uint)work[xfPos + i - 1];
                                    uint ag = (cur & 0xff00ff00u) + (prv & 0xff00ff00u);
                                    uint rb = (cur & 0x00ff00ffu) + (prv & 0x00ff00ffu);
                                    work[xfPos + i] = (int)((ag & 0xff00ff00u) | (rb & 0x00ff00ffu));
                                }
                            }
                            int wbits = 0;
                            if (count <= 2) wbits = 3;
                            else if (count <= 4) wbits = 2;
                            else if (count <= 16) wbits = 1;
                            meta[rec + WebPMeta.TrBits] = wbits;
                            xsize = CeilDiv(xsize, 1 << wbits);
                            xfPos += count;
                        }
                    }
                    nx++;
                }
            }
            tloop = (more == 1) & (st.Err == 0);
        }
        meta[WebPMeta.TransformCount] = nx;
        meta[WebPMeta.PackedWidth] = xsize;

        // ---- colour cache + meta prefix codes ----
        int cacheBits = 0;
        if (st.Err == 0)
        {
            if (ReadBits(ref st, file, 1) == 1)
            {
                cacheBits = ReadBits(ref st, file, 4);
                bool badBits = ((cacheBits < 1) | (cacheBits > 11)) & (st.Err == 0);
                if (badBits)
                    st.Err = (int)WebPError.BadCacheBits;
            }
        }
        meta[WebPMeta.CacheBits] = cacheBits;
        int cacheSizeMain = cacheBits > 0 ? 1 << cacheBits : 0;

        int metaBits = 0;
        int groups = 1;
        int groupsUsed = 1;
        int tilesPerRow = 0;
        if (st.Err == 0)
        {
            if (ReadBits(ref st, file, 1) == 1)
            {
                metaBits = ReadBits(ref st, file, 3) + 2;
                int ew = CeilDiv(xsize, 1 << metaBits);
                int eh = CeilDiv(height, 1 << metaBits);
                tilesPerRow = ew;
                if (st.Err == 0)
                {
                    DecodeSubImage(ref st, file, meta, work, pool, pix, tables, sb, ew, eh, entBase, cacheBase, poolCap);
                }
                if (st.Err == 0)
                {
                    // The group index is red + green of each entropy pixel. Only referenced groups get tables; the others
                    // are parsed (they take bits) and validated but not stored.
                    int maxIdx = 0;
                    int entN = ew * eh;
                    for (int i = 0; i < entN; i++)
                    {
                        int gi = (work[entBase + i] >> 8) & 0xFFFF;
                        if (gi > maxIdx) maxIdx = gi;
                    }
                    groups = maxIdx + 1;
                    for (int i = 0; i < groups; i++) work[mapBase + i] = -1;
                    for (int i = 0; i < entN; i++)
                        work[mapBase + ((work[entBase + i] >> 8) & 0xFFFF)] = -2;
                    groupsUsed = 0;
                    for (int i = 0; i < groups; i++)
                    {
                        if (work[mapBase + i] == -2)
                        {
                            work[mapBase + i] = groupsUsed;
                            groupsUsed++;
                        }
                    }
                    for (int i = 0; i < entN; i++)
                        work[entBase + i] = work[mapBase + ((work[entBase + i] >> 8) & 0xFFFF)];
                }
            }
        }
        meta[WebPMeta.MetaBits] = metaBits;
        meta[WebPMeta.Groups] = groups;
        meta[WebPMeta.GroupsUsed] = groupsUsed;

        // ---- prefix codes of every group, then the pixels ----
        if (st.Err == 0)
        {
            int groupTab = meta[WebPMeta.PoolUsed];
            int after = groupTab + 5 * groupsUsed;
            meta[WebPMeta.PoolUsed] = after;
            if (after > meta[WebPMeta.PoolNeeded]) meta[WebPMeta.PoolNeeded] = after;
            if (after > poolCap) st.Err = (int)WebPError.PoolTooSmall;

            int gi = 0;
            bool gloop = (groups > 0) & (st.Err == 0);
            while (gloop)
            {
                int id = 0;
                if (metaBits > 0) id = work[mapBase + gi];
                int store = id >= 0 ? 1 : 0;
                int tab = groupTab + (id >= 0 ? id : 0) * 5;
                ReadGroup(ref st, file, work, pool, meta, tables, sb, cacheSizeMain, store, tab, poolCap);
                gi++;
                gloop = (gi < groups) & (st.Err == 0);
            }
            if (st.Err == 0)
            {
                DecodePixels(ref st, file, work, pool, pix, tables, 1, 0, xsize, height, cacheBits, cacheBase,
                    metaBits, entBase, tilesPerRow, groupTab);
            }
        }

        meta[WebPMeta.BitsConsumed] = (st.Pos - payOff) * 8 - st.Cnt + st.Virt;
        meta[WebPMeta.DecodeError] = st.Err;
    }
}
