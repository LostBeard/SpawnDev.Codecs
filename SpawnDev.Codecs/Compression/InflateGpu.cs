// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// DEFLATE (RFC 1951) / zlib (RFC 1950) decoder, GPU-callable form. A port of Mark Adler's puff.c
// (zlib/contrib/puff, public-domain-style permissive licence) into ILGPU-compatible C#: no managed
// allocations, no tables as static arrays (length / distance bases are computed arithmetically),
// all scratch lives in an ArrayView<int> the caller owns.
//
// A DEFLATE stream is one sequential bit-serial dependency chain, so per the library's cardinal
// rule it runs as ONE GPU thread (still accelerator-resident: the compressed bytes and the
// inflated bytes never cross to the host). Streams that the container splits into independent
// pieces (PNG has one; some formats have many) are parallel across streams.
//
// State is a struct passed by ref into static helpers, the same shape as Vp8BoolDecoderGpu, which
// is proven bit-exact on all six backends. Loops that need an early exit use a flag in the loop
// body rather than an early return from inside the loop.

using ILGPU;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Compression;

/// <summary>Error codes written by <see cref="InflateGpu"/> (negated into the result slot).</summary>
public enum InflateError
{
    /// <summary>No error.</summary>
    None = 0,
    /// <summary>Compressed input ended before the stream did.</summary>
    InputTruncated = 1,
    /// <summary>Inflated data would exceed the output capacity.</summary>
    OutputOverflow = 2,
    /// <summary>Block type 3 (reserved).</summary>
    BadBlockType = 3,
    /// <summary>Stored block LEN / NLEN mismatch.</summary>
    BadStoredLength = 4,
    /// <summary>HLIT / HDIST out of range.</summary>
    BadCodeCounts = 5,
    /// <summary>Code-length code is over-subscribed or incomplete.</summary>
    BadCodeLengthCode = 6,
    /// <summary>A repeat code is invalid (no previous length, or runs past the lengths).</summary>
    BadRepeat = 7,
    /// <summary>No end-of-block code in the literal/length code.</summary>
    MissingEndOfBlock = 8,
    /// <summary>Literal/length or distance code is over-subscribed or illegally incomplete.</summary>
    BadHuffmanCode = 9,
    /// <summary>A symbol did not decode (ran out of codes).</summary>
    BadSymbol = 10,
    /// <summary>Distance is greater than the data produced so far (or the length/distance symbol is invalid).</summary>
    BadDistance = 11,
    /// <summary>zlib header invalid (not deflate, bad FCHECK, or a preset dictionary).</summary>
    BadZlibHeader = 12,
    /// <summary>zlib Adler-32 trailer does not match the inflated data.</summary>
    AdlerMismatch = 13,
}

/// <summary>Running state of one inflate: input cursor, bit buffer, output cursor, error.</summary>
public struct InflateState
{
    /// <summary>Next input byte index.</summary>
    public int InPos;
    /// <summary>One past the last input byte index.</summary>
    public int InEnd;
    /// <summary>Bit buffer (low bits are the next bits of the stream).</summary>
    public uint BitBuf;
    /// <summary>Number of valid bits in <see cref="BitBuf"/>.</summary>
    public int BitCnt;
    /// <summary>First output byte index of this stream (back-references may not reach before it).</summary>
    public int OutBase;
    /// <summary>Next output byte index.</summary>
    public int OutPos;
    /// <summary>One past the last writable output byte index.</summary>
    public int OutEnd;
    /// <summary>First <see cref="InflateError"/> raised (0 = none). Once set the decode unwinds.</summary>
    public int Err;
}

/// <summary>GPU-callable DEFLATE / zlib decoder (puff.c port).</summary>
public static class InflateGpu
{
    private const int MaxBits = 15;

    // Scratch layout (ints). One inflate uses ScratchInts ints.
    private const int LitCountOff = 0;                    // [16]
    private const int LitSymOff = LitCountOff + 16;       // [288]
    private const int DistCountOff = LitSymOff + 288;     // [32]  (30 symbols)
    private const int DistSymOff = DistCountOff + 16;     // [32]
    private const int LcCountOff = DistSymOff + 32;       // [16]  code-length code
    private const int LcSymOff = LcCountOff + 16;         // [24]  (19 symbols)
    private const int LengthsOff = LcSymOff + 24;         // [320] (286 + 30 + slack)
    private const int OffsOff = LengthsOff + 320;         // [16]  construct() working offsets

    /// <summary>Number of <c>int</c>s of scratch one inflate needs.</summary>
    public const int ScratchInts = OffsOff + 16;

    /// <summary>
    /// Inflate a raw DEFLATE stream. Returns the number of bytes written to <paramref name="dst"/>
    /// (&gt;= 0) or the negated <see cref="InflateError"/>.
    /// </summary>
    public static int InflateRaw(
        ArrayView<byte> src, int srcOff, int srcLen,
        ArrayView<byte> dst, int dstOff, int dstCap,
        ArrayView<int> scratch, int scratchBase)
    {
        var st = NewState(srcOff, srcLen, dstOff, dstCap);
        InflateStream(ref st, src, dst, scratch, scratchBase);
        if (st.Err != 0) return -st.Err;
        return st.OutPos - dstOff;
    }

    private static InflateState NewState(int srcOff, int srcLen, int dstOff, int dstCap)
    {
        return new InflateState
        {
            InPos = srcOff,
            InEnd = srcOff + srcLen,
            BitBuf = 0,
            BitCnt = 0,
            OutBase = dstOff,
            OutPos = dstOff,
            OutEnd = dstOff + dstCap,
            Err = 0,
        };
    }

    /// <summary>
    /// Inflate a zlib (RFC 1950) stream: 2-byte header, DEFLATE data, big-endian Adler-32 trailer.
    /// Returns the inflated length or the negated <see cref="InflateError"/>.
    /// </summary>
    public static int InflateZlib(
        ArrayView<byte> src, int srcOff, int srcLen,
        ArrayView<byte> dst, int dstOff, int dstCap,
        ArrayView<int> scratch, int scratchBase,
        bool verifyAdler)
    {
        if (srcLen < 6) return -(int)InflateError.InputTruncated;
        int cmf = src[srcOff];
        int flg = src[srcOff + 1];
        // CM must be 8 (deflate), CINFO <= 7, header checksum, no preset dictionary.
        bool badHeader = (cmf & 0x0F) != 8;
        badHeader = badHeader || ((cmf >> 4) > 7);
        badHeader = badHeader || (((cmf << 8) | flg) % 31 != 0);
        badHeader = badHeader || ((flg & 0x20) != 0);
        if (badHeader) return -(int)InflateError.BadZlibHeader;

        var st = NewState(srcOff + 2, srcLen - 2, dstOff, dstCap);
        InflateStream(ref st, src, dst, scratch, scratchBase);
        if (st.Err != 0) return -st.Err;
        int outLen = st.OutPos - dstOff;

        if (verifyAdler)
        {
            // The bit reader loads a byte only when it needs one, so InPos is exactly the byte
            // boundary after the last (possibly partially used) deflate byte.
            int trailer = st.InPos;
            if (trailer + 4 > srcOff + srcLen) return -(int)InflateError.InputTruncated;
            uint expected = ((uint)src[trailer] << 24) | ((uint)src[trailer + 1] << 16)
                          | ((uint)src[trailer + 2] << 8) | src[trailer + 3];
            if (Adler32(dst, dstOff, outLen) != expected) return -(int)InflateError.AdlerMismatch;
        }
        return outLen;
    }

    /// <summary>Adler-32 of <paramref name="length"/> bytes at <paramref name="offset"/>.</summary>
    public static uint Adler32(ArrayView<byte> data, int offset, int length)
    {
        uint a = 1;
        uint b = 0;
        int pos = offset;
        int remaining = length;
        while (remaining > 0)
        {
            // 5552 is the largest block for which the sums cannot overflow 32 bits.
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

    // ------------------------------------------------------------------
    // Stream / block level
    // ------------------------------------------------------------------

    private static void InflateStream(
        ref InflateState st, ArrayView<byte> src, ArrayView<byte> dst,
        ArrayView<int> scratch, int sb)
    {
        int last = 0;
        while (last == 0 && st.Err == 0)
        {
            last = Bits(ref st, src, 1);
            int type = Bits(ref st, src, 2);
            if (st.Err == 0)
            {
                if (type == 0)
                    Stored(ref st, src, dst);
                else if (type == 1)
                    Fixed(ref st, src, dst, scratch, sb);
                else if (type == 2)
                    Dynamic(ref st, src, dst, scratch, sb);
                else
                    st.Err = (int)InflateError.BadBlockType;
            }
        }
    }

    private static void Stored(ref InflateState st, ArrayView<byte> src, ArrayView<byte> dst)
    {
        // Discard the rest of the current byte.
        st.BitBuf = 0;
        st.BitCnt = 0;
        if (st.InPos + 4 > st.InEnd)
        {
            st.Err = (int)InflateError.InputTruncated;
        }
        else
        {
            int len = src[st.InPos] | (src[st.InPos + 1] << 8);
            int nlen = src[st.InPos + 2] | (src[st.InPos + 3] << 8);
            st.InPos += 4;
            if (len != (~nlen & 0xFFFF))
            {
                st.Err = (int)InflateError.BadStoredLength;
            }
            else if (st.InPos + len > st.InEnd)
            {
                st.Err = (int)InflateError.InputTruncated;
            }
            else if (st.OutPos + len > st.OutEnd)
            {
                st.Err = (int)InflateError.OutputOverflow;
            }
            else
            {
                for (int i = 0; i < len; i++)
                    dst[st.OutPos + i] = src[st.InPos + i];
                st.InPos += len;
                st.OutPos += len;
            }
        }
    }

    private static void Fixed(
        ref InflateState st, ArrayView<byte> src, ArrayView<byte> dst, ArrayView<int> scratch, int sb)
    {
        for (int sym = 0; sym < 144; sym++) scratch[sb + LengthsOff + sym] = 8;
        for (int sym = 144; sym < 256; sym++) scratch[sb + LengthsOff + sym] = 9;
        for (int sym = 256; sym < 280; sym++) scratch[sb + LengthsOff + sym] = 7;
        for (int sym = 280; sym < 288; sym++) scratch[sb + LengthsOff + sym] = 8;
        Construct(scratch, sb, sb + LengthsOff, 288, sb + LitCountOff, sb + LitSymOff);

        for (int sym = 0; sym < 30; sym++) scratch[sb + LengthsOff + sym] = 5;
        Construct(scratch, sb, sb + LengthsOff, 30, sb + DistCountOff, sb + DistSymOff);

        Codes(ref st, src, dst, scratch, sb);
    }

    private static void Dynamic(
        ref InflateState st, ArrayView<byte> src, ArrayView<byte> dst, ArrayView<int> scratch, int sb)
    {
        int nlen = Bits(ref st, src, 5) + 257;
        int ndist = Bits(ref st, src, 5) + 1;
        int ncode = Bits(ref st, src, 4) + 4;
        if (st.Err != 0) return;
        if (nlen > 286 || ndist > 30)
        {
            st.Err = (int)InflateError.BadCodeCounts;
            return;
        }

        int lengthsAt = sb + LengthsOff;
        for (int i = 0; i < 19; i++) scratch[lengthsAt + i] = 0;
        for (int i = 0; i < ncode; i++)
            scratch[lengthsAt + CodeLengthOrder(i)] = Bits(ref st, src, 3);
        if (st.Err != 0) return;

        // The code-length code must be complete.
        int left = Construct(scratch, sb, lengthsAt, 19, sb + LcCountOff, sb + LcSymOff);
        if (left != 0)
        {
            st.Err = (int)InflateError.BadCodeLengthCode;
            return;
        }

        int index = 0;
        int total = nlen + ndist;
        while (index < total && st.Err == 0)
        {
            int sym = Decode(ref st, src, scratch, sb + LcCountOff, sb + LcSymOff);
            if (st.Err == 0)
            {
                if (sym < 16)
                {
                    scratch[lengthsAt + index] = sym;
                    index++;
                }
                else
                {
                    int prev = 0;
                    int rep;
                    if (sym == 16)
                    {
                        if (index == 0)
                        {
                            st.Err = (int)InflateError.BadRepeat;
                            rep = 0;
                        }
                        else
                        {
                            prev = scratch[lengthsAt + index - 1];
                            rep = 3 + Bits(ref st, src, 2);
                        }
                    }
                    else if (sym == 17)
                    {
                        rep = 3 + Bits(ref st, src, 3);
                    }
                    else
                    {
                        rep = 11 + Bits(ref st, src, 7);
                    }
                    if (st.Err == 0)
                    {
                        if (index + rep > total)
                        {
                            st.Err = (int)InflateError.BadRepeat;
                        }
                        else
                        {
                            for (int r = 0; r < rep; r++)
                            {
                                scratch[lengthsAt + index] = prev;
                                index++;
                            }
                        }
                    }
                }
            }
        }
        if (st.Err != 0) return;

        if (scratch[lengthsAt + 256] == 0)
        {
            st.Err = (int)InflateError.MissingEndOfBlock;
            return;
        }

        // Literal/length code: complete, or incomplete only when it has a single code.
        left = Construct(scratch, sb, lengthsAt, nlen, sb + LitCountOff, sb + LitSymOff);
        bool litBad = left < 0;
        litBad = litBad || (left > 0 && nlen != scratch[sb + LitCountOff + 0] + scratch[sb + LitCountOff + 1]);
        if (litBad)
        {
            st.Err = (int)InflateError.BadHuffmanCode;
            return;
        }

        left = Construct(scratch, sb, lengthsAt + nlen, ndist, sb + DistCountOff, sb + DistSymOff);
        bool distBad = left < 0;
        distBad = distBad || (left > 0 && ndist != scratch[sb + DistCountOff + 0] + scratch[sb + DistCountOff + 1]);
        if (distBad)
        {
            st.Err = (int)InflateError.BadHuffmanCode;
            return;
        }

        Codes(ref st, src, dst, scratch, sb);
    }

    private static void Codes(
        ref InflateState st, ArrayView<byte> src, ArrayView<byte> dst, ArrayView<int> scratch, int sb)
    {
        bool done = false;
        while (!done && st.Err == 0)
        {
            int sym = Decode(ref st, src, scratch, sb + LitCountOff, sb + LitSymOff);
            if (st.Err != 0)
            {
                done = true;
            }
            else if (sym < 256)
            {
                if (st.OutPos >= st.OutEnd)
                {
                    st.Err = (int)InflateError.OutputOverflow;
                }
                else
                {
                    dst[st.OutPos] = (byte)sym;
                    st.OutPos++;
                }
            }
            else if (sym == 256)
            {
                done = true;
            }
            else
            {
                int ls = sym - 257;
                if (ls >= 29)
                {
                    st.Err = (int)InflateError.BadDistance;
                }
                else
                {
                    int len;
                    if (ls < 8)
                    {
                        len = 3 + ls;
                    }
                    else if (ls == 28)
                    {
                        len = 258;
                    }
                    else
                    {
                        int e = ((ls - 8) >> 2) + 1;
                        len = 3 + ((4 + ((ls - 8) & 3)) << e) + Bits(ref st, src, e);
                    }

                    int dsym = Decode(ref st, src, scratch, sb + DistCountOff, sb + DistSymOff);
                    if (st.Err == 0)
                    {
                        if (dsym >= 30)
                        {
                            st.Err = (int)InflateError.BadDistance;
                        }
                        else
                        {
                            int dist;
                            if (dsym < 4)
                            {
                                dist = dsym + 1;
                            }
                            else
                            {
                                int de = (dsym >> 1) - 1;
                                dist = (((2 + (dsym & 1)) << de) + 1) + Bits(ref st, src, de);
                            }
                            if (st.Err == 0)
                            {
                                if (dist > st.OutPos - st.OutBase)
                                {
                                    st.Err = (int)InflateError.BadDistance;
                                }
                                else if (st.OutPos + len > st.OutEnd)
                                {
                                    st.Err = (int)InflateError.OutputOverflow;
                                }
                                else
                                {
                                    // Byte-wise: the source and destination ranges may overlap (dist < len).
                                    for (int i = 0; i < len; i++)
                                    {
                                        dst[st.OutPos] = dst[st.OutPos - dist];
                                        st.OutPos++;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Bit / Huffman primitives
    // ------------------------------------------------------------------

    /// <summary>Reads <paramref name="need"/> (0..16) bits, LSB first.</summary>
    private static int Bits(ref InflateState st, ArrayView<byte> src, int need)
    {
        uint val = st.BitBuf;
        while (st.BitCnt < need && st.Err == 0)
        {
            if (st.InPos >= st.InEnd)
            {
                st.Err = (int)InflateError.InputTruncated;
            }
            else
            {
                val |= (uint)src[st.InPos] << st.BitCnt;
                st.InPos++;
                st.BitCnt += 8;
            }
        }
        if (st.Err != 0) return 0;
        st.BitBuf = val >> need;
        st.BitCnt -= need;
        return (int)(val & ((1u << need) - 1u));
    }

    /// <summary>
    /// Decodes one symbol with the canonical code whose per-length counts start at
    /// <paramref name="countAt"/> and whose sorted symbols start at <paramref name="symAt"/>.
    /// </summary>
    private static int Decode(
        ref InflateState st, ArrayView<byte> src, ArrayView<int> scratch, int countAt, int symAt)
    {
        int code = 0;
        int first = 0;
        int index = 0;
        int result = -1;
        int found = 0;
        for (int len = 1; len <= MaxBits; len++)
        {
            if (found == 0 && st.Err == 0)
            {
                code |= Bits(ref st, src, 1);
                int count = scratch[countAt + len];
                if (code - count < first)
                {
                    result = scratch[symAt + index + (code - first)];
                    found = 1;
                }
                else
                {
                    index += count;
                    first += count;
                    first <<= 1;
                    code <<= 1;
                }
            }
        }
        if (found == 0 && st.Err == 0)
            st.Err = (int)InflateError.BadSymbol;
        return result;
    }

    /// <summary>
    /// Builds the canonical decoding tables for <paramref name="n"/> code lengths at
    /// <paramref name="lengthsAt"/>. Returns 0 for a complete code, &gt; 0 for an incomplete one,
    /// &lt; 0 for an over-subscribed one.
    /// </summary>
    private static int Construct(
        ArrayView<int> scratch, int sb, int lengthsAt, int n, int countAt, int symAt)
    {
        int offsAt = sb + OffsOff;
        for (int len = 0; len <= MaxBits; len++) scratch[countAt + len] = 0;
        for (int sym = 0; sym < n; sym++)
        {
            int slot = countAt + scratch[lengthsAt + sym];
            scratch[slot] = scratch[slot] + 1;
        }

        int left = 1;
        if (scratch[countAt] == n)
        {
            left = 0;
        }
        else
        {
            for (int len = 1; len <= MaxBits; len++)
            {
                // An over-subscribed code stays negative from here on (counts are >= 0), so the
                // sign needs no early exit.
                left <<= 1;
                left -= scratch[countAt + len];
            }
        }

        if (left >= 0 && scratch[countAt] != n)
        {
            scratch[offsAt + 1] = 0;
            for (int len = 1; len < MaxBits; len++)
                scratch[offsAt + len + 1] = scratch[offsAt + len] + scratch[countAt + len];
            for (int sym = 0; sym < n; sym++)
            {
                int l = scratch[lengthsAt + sym];
                if (l != 0)
                {
                    int at = scratch[offsAt + l];
                    scratch[symAt + at] = sym;
                    scratch[offsAt + l] = at + 1;
                }
            }
        }
        return left;
    }

    /// <summary>RFC 1951 3.2.7: the order in which code-length code lengths are stored.</summary>
    private static int CodeLengthOrder(int i)
    {
        int v;
        switch (i)
        {
            case 0: v = 16; break;
            case 1: v = 17; break;
            case 2: v = 18; break;
            case 3: v = 0; break;
            case 4: v = 8; break;
            case 5: v = 7; break;
            case 6: v = 9; break;
            case 7: v = 6; break;
            case 8: v = 10; break;
            case 9: v = 5; break;
            case 10: v = 11; break;
            case 11: v = 4; break;
            case 12: v = 12; break;
            case 13: v = 3; break;
            case 14: v = 13; break;
            case 15: v = 2; break;
            case 16: v = 14; break;
            case 17: v = 1; break;
            default: v = 15; break;
        }
        return v;
    }

}
