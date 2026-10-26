// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// DEFLATE (RFC 1951) / zlib (RFC 1950) decoder, GPU-callable form. A port of Mark Adler's puff.c
// (zlib/contrib/puff, public-domain-style permissive licence) into ILGPU-compatible C#: no managed
// allocations, no tables as static arrays (length / distance bases are computed arithmetically),
// all scratch lives in an ArrayView<int> the caller owns.
//
// A DEFLATE stream is one sequential bit-serial dependency chain, so per the library's cardinal
// rule it runs on ONE GPU thread (still accelerator-resident: the compressed bytes and the
// inflated bytes never cross to the host). Streams that the container splits into independent
// pieces (PNG has one; some formats have many) are parallel across streams.
//
// The decode is RESUMABLE: InitZlib / InitRaw prepare it, and each Step continues from the state kept
// in the scratch (bit reader, block phase, the current block's Huffman tables, stored-block remainder,
// output cursor, running Adler-32) for at most StepInputBytes compressed / StepOutputBytes inflated
// bytes, so no launch holds the GPU long however big the stream is (a long launch stalls the desktop
// compositor and trips the OS GPU watchdog). MaxSteps is an upper bound of the launches a stream needs,
// known on the host from the two sizes: no readback is needed to drive it.
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
    private const int FixLitCountOff = OffsOff + 16;      // [16]  fixed literal/length code, built once per stream
    private const int FixLitSymOff = FixLitCountOff + 16; // [288]
    private const int FixDistCountOff = FixLitSymOff + 288; // [16] fixed distance code
    private const int FixDistSymOff = FixDistCountOff + 16; // [32]
    private const int StateOff = FixDistSymOff + 32;      // [24]  resumable decode state (slots below)

    /// <summary>Number of <c>int</c>s of scratch one inflate needs.</summary>
    public const int ScratchInts = StateOff + 24;

    // Slots of the resumable state (relative to StateOff).
    private const int SInPos = 0, SInEnd = 1, SBitBuf = 2, SBitCnt = 3, SOutBase = 4, SOutPos = 5, SOutEnd = 6,
        SErr = 7, SPhase = 8, SLast = 9, SStored = 10, STables = 11, SMode = 12, SAdlerA = 13, SAdlerB = 14,
        SAdlerPos = 15, SResult = 16;

    // Decode phases: next block header, inside a stored block, inside a Huffman block, after the last block, done.
    private const int PhaseHeader = 0, PhaseStored = 1, PhaseCodes = 2, PhaseEnd = 3, PhaseFinished = 4;
    // Stream modes: raw DEFLATE, zlib, zlib with the Adler-32 trailer verified.
    private const int ModeRaw = 0, ModeZlib = 1, ModeZlibVerify = 2;

    /// <summary>
    /// Compressed bytes one <see cref="Step"/> launch consumes at most (it stops at the first symbol boundary past this).
    /// One GPU thread decodes bit by bit; with the dynamic block headers an adversarial stream can pack, 2048 bytes
    /// stay a few milliseconds on a desktop GPU.
    /// </summary>
    public const int StepInputBytes = 2048;

    /// <summary>Inflated bytes one <see cref="Step"/> launch produces at most (plus one back-reference, at most 258 bytes).</summary>
    public const int StepOutputBytes = 32768;

    /// <summary>
    /// Launches of <see cref="Step"/> that always suffice for a stream of <paramref name="srcLen"/> compressed bytes
    /// inflating into at most <paramref name="dstCap"/> bytes: every launch but the last stops only after consuming
    /// <see cref="StepInputBytes"/> or producing <see cref="StepOutputBytes"/>. Launches after the end do nothing.
    /// </summary>
    public static int MaxSteps(long srcLen, long dstCap) =>
        (int)Math.Min(int.MaxValue, srcLen / StepInputBytes + dstCap / StepOutputBytes + 2);

    /// <summary>
    /// Inflate a raw DEFLATE stream. Returns the number of bytes written to <paramref name="dst"/>
    /// (&gt;= 0) or the negated <see cref="InflateError"/>. The whole stream in one call (one thread, unbounded);
    /// launch <see cref="InitRaw"/> + <see cref="Step"/> to bound the work per launch.
    /// </summary>
    public static int InflateRaw(
        ArrayView<byte> src, int srcOff, int srcLen,
        ArrayView<byte> dst, int dstOff, int dstCap,
        ArrayView<int> scratch, int scratchBase)
    {
        InitRaw(srcOff, srcLen, dstOff, dstCap, scratch, scratchBase);
        Step(src, dst, scratch, scratchBase, int.MaxValue, int.MaxValue);
        return Result(scratch, scratchBase);
    }

    /// <summary>
    /// Inflate a zlib (RFC 1950) stream: 2-byte header, DEFLATE data, big-endian Adler-32 trailer.
    /// Returns the inflated length or the negated <see cref="InflateError"/>. The whole stream in one call (one
    /// thread, unbounded); launch <see cref="InitZlib"/> + <see cref="Step"/> to bound the work per launch.
    /// </summary>
    public static int InflateZlib(
        ArrayView<byte> src, int srcOff, int srcLen,
        ArrayView<byte> dst, int dstOff, int dstCap,
        ArrayView<int> scratch, int scratchBase,
        bool verifyAdler)
    {
        InitZlib(src, srcOff, srcLen, dstOff, dstCap, scratch, scratchBase, verifyAdler);
        Step(src, dst, scratch, scratchBase, int.MaxValue, int.MaxValue);
        return Result(scratch, scratchBase);
    }

    /// <summary>
    /// Prepares the resumable decode of a raw DEFLATE stream in <paramref name="scratch"/> (builds the fixed codes,
    /// resets the state); <see cref="Step"/> then decodes it.
    /// </summary>
    public static void InitRaw(int srcOff, int srcLen, int dstOff, int dstCap, ArrayView<int> scratch, int scratchBase)
    {
        BuildFixed(scratch, scratchBase);
        SetState(scratch, scratchBase, srcOff, srcOff + srcLen, dstOff, dstCap, ModeRaw);
    }

    /// <summary>
    /// Prepares the resumable decode of a zlib stream: checks the 2-byte header (a bad one finishes the decode at once
    /// with its error), builds the fixed codes and resets the state; <see cref="Step"/> then decodes it.
    /// </summary>
    public static void InitZlib(
        ArrayView<byte> src, int srcOff, int srcLen, int dstOff, int dstCap,
        ArrayView<int> scratch, int scratchBase, bool verifyAdler)
    {
        int so = scratchBase + StateOff;
        if (srcLen < 6)
        {
            scratch[so + SPhase] = PhaseFinished;
            scratch[so + SResult] = -(int)InflateError.InputTruncated;
            return;
        }
        int cmf = src[srcOff];
        int flg = src[srcOff + 1];
        // CM must be 8 (deflate), CINFO <= 7, header checksum, no preset dictionary.
        bool badHeader = ((cmf & 0x0F) != 8) | ((cmf >> 4) > 7) | ((((cmf << 8) | flg) % 31) != 0) | ((flg & 0x20) != 0);
        if (badHeader)
        {
            scratch[so + SPhase] = PhaseFinished;
            scratch[so + SResult] = -(int)InflateError.BadZlibHeader;
            return;
        }
        BuildFixed(scratch, scratchBase);
        SetState(scratch, scratchBase, srcOff + 2, srcOff + srcLen, dstOff, dstCap, verifyAdler ? ModeZlibVerify : ModeZlib);
    }

    /// <summary>
    /// The decode's result: the inflated length (&gt;= 0) or the negated <see cref="InflateError"/> once it has finished;
    /// <see cref="InflateError.InputTruncated"/> (negated) while it has not.
    /// </summary>
    public static int Result(ArrayView<int> scratch, int scratchBase) => scratch[scratchBase + StateOff + SResult];

    private static void SetState(ArrayView<int> scratch, int sb, int inPos, int inEnd, int dstOff, int dstCap, int mode)
    {
        int so = sb + StateOff;
        scratch[so + SInPos] = inPos;
        scratch[so + SInEnd] = inEnd;
        scratch[so + SBitBuf] = 0;
        scratch[so + SBitCnt] = 0;
        scratch[so + SOutBase] = dstOff;
        scratch[so + SOutPos] = dstOff;
        scratch[so + SOutEnd] = dstOff + dstCap;
        scratch[so + SErr] = 0;
        scratch[so + SPhase] = PhaseHeader;
        scratch[so + SLast] = 0;
        scratch[so + SStored] = 0;
        scratch[so + STables] = 0;
        scratch[so + SMode] = mode;
        scratch[so + SAdlerA] = 1;
        scratch[so + SAdlerB] = 0;
        scratch[so + SAdlerPos] = dstOff;
        scratch[so + SResult] = -(int)InflateError.InputTruncated;
    }

    /// <summary>Adler-32 of <paramref name="length"/> bytes at <paramref name="offset"/>.</summary>
    public static uint Adler32(ArrayView<byte> data, int offset, int length)
    {
        uint a = 1;
        uint b = 0;
        AdlerFold(data, offset, offset + length, ref a, ref b);
        return (b << 16) | a;
    }

    /// <summary>Folds bytes [<paramref name="from"/>, <paramref name="to"/>) into the Adler-32 sums (both kept &lt; 65521).</summary>
    private static void AdlerFold(ArrayView<byte> data, int from, int to, ref uint a, ref uint b)
    {
        int pos = from;
        bool more = pos < to;
        while (more)
        {
            // 5552 is the largest block for which the sums cannot overflow 32 bits.
            int n = to - pos;
            if (n > 5552) n = 5552;
            for (int i = 0; i < n; i++)
            {
                a += data[pos + i];
                b += a;
            }
            a %= 65521u;
            b %= 65521u;
            pos += n;
            more = pos < to;
        }
    }

    // ------------------------------------------------------------------
    // Resumable stream decode
    // ------------------------------------------------------------------

    /// <summary>
    /// Continues the decode prepared by <see cref="InitZlib"/> / <see cref="InitRaw"/> from the state in
    /// <paramref name="scratch"/>: decodes symbols until <paramref name="inputBudget"/> compressed bytes were consumed or
    /// <paramref name="outputBudget"/> bytes produced (checked at symbol boundaries; a stored block is copied in pieces of
    /// at most <paramref name="outputBudget"/> bytes), the stream ended, or an error occurred, then saves the state. The
    /// Adler-32 of the produced bytes is folded as they are produced, so verifying it adds no pass over the output.
    /// A call after the end does nothing.
    /// </summary>
    public static void Step(
        ArrayView<byte> src, ArrayView<byte> dst, ArrayView<int> scratch, int sb, int inputBudget, int outputBudget)
    {
        int so = sb + StateOff;
        int phase = scratch[so + SPhase];
        if (phase >= PhaseFinished) return;

        var st = new InflateState
        {
            InPos = scratch[so + SInPos],
            InEnd = scratch[so + SInEnd],
            BitBuf = (uint)scratch[so + SBitBuf],
            BitCnt = scratch[so + SBitCnt],
            OutBase = scratch[so + SOutBase],
            OutPos = scratch[so + SOutPos],
            OutEnd = scratch[so + SOutEnd],
            Err = scratch[so + SErr],
        };
        int last = scratch[so + SLast];
        int stored = scratch[so + SStored];
        int tables = scratch[so + STables];
        int mode = scratch[so + SMode];
        uint adlerA = (uint)scratch[so + SAdlerA];
        uint adlerB = (uint)scratch[so + SAdlerB];
        int adlerPos = scratch[so + SAdlerPos];
        int result = scratch[so + SResult];
        int inStart = st.InPos;
        int outStart = st.OutPos;

        // Loop conditions are single bool locals built with non-short-circuit &: compound && / || in a
        // loop header has hung the WebGPU device (the WGSL structurizer never leaves the loop).
        bool going = true;
        while (going)
        {
            if (phase == PhaseHeader)
            {
                last = Bits(ref st, src, 1);
                int type = Bits(ref st, src, 2);
                if (st.Err == 0)
                {
                    if (type == 0)
                    {
                        stored = StoredHeader(ref st, src);
                        if (st.Err == 0) phase = PhaseStored;
                    }
                    else if (type == 1)
                    {
                        tables = 1;
                        phase = PhaseCodes;
                    }
                    else if (type == 2)
                    {
                        Dynamic(ref st, src, scratch, sb);
                        if (st.Err == 0)
                        {
                            tables = 0;
                            phase = PhaseCodes;
                        }
                    }
                    else
                    {
                        st.Err = (int)InflateError.BadBlockType;
                    }
                }
            }
            else if (phase == PhaseStored)
            {
                int n = stored < outputBudget ? stored : outputBudget;
                for (int i = 0; i < n; i++)
                    dst[st.OutPos + i] = src[st.InPos + i];
                st.InPos += n;
                st.OutPos += n;
                stored -= n;
                if (stored == 0) phase = last != 0 ? PhaseEnd : PhaseHeader;
            }
            else if (phase == PhaseCodes)
            {
                int litCount = sb + LitCountOff;
                int litSym = sb + LitSymOff;
                int distCount = sb + DistCountOff;
                int distSym = sb + DistSymOff;
                if (tables != 0)
                {
                    litCount = sb + FixLitCountOff;
                    litSym = sb + FixLitSymOff;
                    distCount = sb + FixDistCountOff;
                    distSym = sb + FixDistSymOff;
                }
                int endOfBlock = CodeSymbol(ref st, src, dst, scratch, litCount, litSym, distCount, distSym);
                bool blockDone = (endOfBlock != 0) & (st.Err == 0);
                if (blockDone) phase = last != 0 ? PhaseEnd : PhaseHeader;
            }
            else
            {
                // After the last block: the result, and for a verified zlib stream the Adler-32 trailer. The bit
                // reader loads a byte only when it needs one, so InPos is exactly the byte boundary after the last
                // (possibly partially used) deflate byte.
                result = st.OutPos - st.OutBase;
                if (mode == ModeZlibVerify)
                {
                    AdlerFold(dst, adlerPos, st.OutPos, ref adlerA, ref adlerB);
                    adlerPos = st.OutPos;
                    int trailer = st.InPos;
                    if (trailer + 4 > st.InEnd)
                    {
                        result = -(int)InflateError.InputTruncated;
                    }
                    else
                    {
                        uint expected = ((uint)src[trailer] << 24) | ((uint)src[trailer + 1] << 16)
                                      | ((uint)src[trailer + 2] << 8) | src[trailer + 3];
                        if (((adlerB << 16) | adlerA) != expected) result = -(int)InflateError.AdlerMismatch;
                    }
                }
                phase = PhaseFinished;
            }
            if (st.Err != 0)
            {
                result = -st.Err;
                phase = PhaseFinished;
            }
            going = (phase < PhaseFinished) & (st.InPos - inStart < inputBudget) & (st.OutPos - outStart < outputBudget);
        }

        bool fold = (mode == ModeZlibVerify) & (phase < PhaseFinished);
        if (fold)
        {
            AdlerFold(dst, adlerPos, st.OutPos, ref adlerA, ref adlerB);
            adlerPos = st.OutPos;
        }

        scratch[so + SInPos] = st.InPos;
        scratch[so + SBitBuf] = (int)st.BitBuf;
        scratch[so + SBitCnt] = st.BitCnt;
        scratch[so + SOutPos] = st.OutPos;
        scratch[so + SErr] = st.Err;
        scratch[so + SPhase] = phase;
        scratch[so + SLast] = last;
        scratch[so + SStored] = stored;
        scratch[so + STables] = tables;
        scratch[so + SAdlerA] = (int)adlerA;
        scratch[so + SAdlerB] = (int)adlerB;
        scratch[so + SAdlerPos] = adlerPos;
        scratch[so + SResult] = result;
    }

    // ------------------------------------------------------------------
    // Block level
    // ------------------------------------------------------------------

    /// <summary>Stored block header: returns LEN after checking it (the bytes are copied by the stored phase).</summary>
    private static int StoredHeader(ref InflateState st, ArrayView<byte> src)
    {
        // Discard the rest of the current byte.
        st.BitBuf = 0;
        st.BitCnt = 0;
        int len = 0;
        if (st.InPos + 4 > st.InEnd)
        {
            st.Err = (int)InflateError.InputTruncated;
        }
        else
        {
            len = src[st.InPos] | (src[st.InPos + 1] << 8);
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
        }
        return len;
    }

    /// <summary>Builds the fixed literal/length and distance codes (once per stream, in their own scratch region).</summary>
    private static void BuildFixed(ArrayView<int> scratch, int sb)
    {
        for (int sym = 0; sym < 144; sym++) scratch[sb + LengthsOff + sym] = 8;
        for (int sym = 144; sym < 256; sym++) scratch[sb + LengthsOff + sym] = 9;
        for (int sym = 256; sym < 280; sym++) scratch[sb + LengthsOff + sym] = 7;
        for (int sym = 280; sym < 288; sym++) scratch[sb + LengthsOff + sym] = 8;
        Construct(scratch, sb, sb + LengthsOff, 288, sb + FixLitCountOff, sb + FixLitSymOff);

        for (int sym = 0; sym < 30; sym++) scratch[sb + LengthsOff + sym] = 5;
        Construct(scratch, sb, sb + LengthsOff, 30, sb + FixDistCountOff, sb + FixDistSymOff);
    }

    /// <summary>Dynamic block header: reads the code lengths and builds the block's codes into the dynamic tables.</summary>
    private static void Dynamic(ref InflateState st, ArrayView<byte> src, ArrayView<int> scratch, int sb)
    {
        int nlen = Bits(ref st, src, 5) + 257;
        int ndist = Bits(ref st, src, 5) + 1;
        int ncode = Bits(ref st, src, 4) + 4;
        if (st.Err != 0) return;
        bool tooMany = (nlen > 286) | (ndist > 30);
        if (tooMany)
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
        bool reading = (index < total) & (st.Err == 0);
        while (reading)
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
            reading = (index < total) & (st.Err == 0);
        }
        if (st.Err != 0) return;

        if (scratch[lengthsAt + 256] == 0)
        {
            st.Err = (int)InflateError.MissingEndOfBlock;
            return;
        }

        // Literal/length code: complete, or incomplete only when it has a single code.
        left = Construct(scratch, sb, lengthsAt, nlen, sb + LitCountOff, sb + LitSymOff);
        bool litBad = (left < 0) | ((left > 0) & (nlen != scratch[sb + LitCountOff + 0] + scratch[sb + LitCountOff + 1]));
        if (litBad)
        {
            st.Err = (int)InflateError.BadHuffmanCode;
            return;
        }

        left = Construct(scratch, sb, lengthsAt + nlen, ndist, sb + DistCountOff, sb + DistSymOff);
        bool distBad = (left < 0) | ((left > 0) & (ndist != scratch[sb + DistCountOff + 0] + scratch[sb + DistCountOff + 1]));
        if (distBad)
        {
            st.Err = (int)InflateError.BadHuffmanCode;
            return;
        }
    }

    /// <summary>
    /// Decodes one literal/length symbol of a Huffman block and writes its bytes (a literal, or a whole back-reference
    /// of at most 258 bytes). Returns 1 at the end-of-block symbol, else 0.
    /// </summary>
    private static int CodeSymbol(
        ref InflateState st, ArrayView<byte> src, ArrayView<byte> dst, ArrayView<int> scratch,
        int litCount, int litSym, int distCount, int distSym)
    {
        int endOfBlock = 0;
        int sym = Decode(ref st, src, scratch, litCount, litSym);
        if (st.Err != 0)
        {
            endOfBlock = 0;
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
            endOfBlock = 1;
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

                int dsym = Decode(ref st, src, scratch, distCount, distSym);
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
        return endOfBlock;
    }

    // ------------------------------------------------------------------
    // Bit / Huffman primitives
    // ------------------------------------------------------------------

    /// <summary>Reads <paramref name="need"/> (0..16) bits, LSB first.</summary>
    private static int Bits(ref InflateState st, ArrayView<byte> src, int need)
    {
        uint val = st.BitBuf;
        bool loading = (st.BitCnt < need) & (st.Err == 0);
        while (loading)
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
            loading = (st.BitCnt < need) & (st.Err == 0);
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
        // Stops at the code's length (it used to walk all MaxBits lengths with the rest masked off: ~15 table reads per
        // symbol whatever its length). The condition is one bool built with & (a compound loop condition hung WGSL).
        int len = 1;
        bool going = st.Err == 0;
        while (going)
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
            len++;
            going = (found == 0) & (len <= MaxBits) & (st.Err == 0);
        }
        bool exhausted = (found == 0) & (st.Err == 0);
        if (exhausted)
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

        bool build = (left >= 0) & (scratch[countAt] != n);
        if (build)
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
    internal static int CodeLengthOrder(int i)
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
