// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Huffman-only DEFLATE blocks planned from a byte histogram (no LZ77 parse, no per-position token array): the code
// construction of PlanBlock at LevelHuffmanOnly (BuildBlockCodes: the same lengths, the same header), so a caller that
// counts the bytes and writes the codes in parallel produces exactly the bytes PlanBlock + EmitStep write for a final
// Huffman-only block. Working memory: the block "tail" only (code tables + state), PlanTailInts ints.

using ILGPU;

namespace SpawnDev.Codecs.Compression;

public static partial class DeflateGpu
{
    /// <summary>Ints of a Huffman-only plan's working memory (code tables and state).</summary>
    public const int PlanTailInts = TailInts;
    /// <summary>Most header bytes a dynamic block header can take (17 + 3 x 19 + 320 x 14 bits, rounded up).</summary>
    public const int MaxDynamicHeaderBytes = 576;
    /// <summary>Offsets, inside a plan tail, of the literal/length codes (bit-reversed, LSB first) and their lengths.</summary>
    public const int PlanLitCodeOff = LitCodeOff, PlanLitLenOff = LitLenOff;

    private const int HoHeaderBits = StHlit + 4, HoStored = StHlit + 5, HoTotalBits = StHlit + 6;

    /// <summary>
    /// Plans a FINAL Huffman-only block of <paramref name="len"/> bytes (&gt; 0) whose byte counts are at
    /// <paramref name="countsAt"/> (256 ints): builds the codes as PlanBlock does, decides stored or dynamic the same way,
    /// and for a dynamic block writes its header bits at <paramref name="hdrAt"/> (the last partial byte included).
    /// </summary>
    public static void PlanHuffmanOnly(ArrayView<int> counts, int countsAt, int len, ArrayView<int> s, int tail,
        ArrayView<byte> hdr, int hdrAt)
    {
        int st = tail + StateOff;
        if (len == 0)
        {
            // PlanBlock's empty final block: fixed Huffman, only the end-of-block code (bytes 0x03 0x00).
            hdr[hdrAt] = 0x03;
            hdr[hdrAt + 1] = 0x00;
            s[st + HoStored] = 2;
            s[st + StOutLen] = 2;
            return;
        }
        for (int i = 0; i < 288; i++) s[tail + LitFreqOff + i] = i < 256 ? counts[countsAt + i] : 0;
        for (int i = 0; i < 32; i++) s[tail + DistFreqOff + i] = 0;
        s[st + StExtra] = 0;
        long dynBits = BuildBlockCodes(s, tail, st);
        long storedBits = 8L * (len + 5L * ((len + 65534) / 65535));
        bool stored = dynBits >= storedBits;
        s[st + HoStored] = stored ? 1 : 0;
        if (stored)
        {
            s[st + HoHeaderBits] = 0;
            s[st + HoTotalBits] = 0;
            s[st + StOutLen] = len + 5 * ((len + 65534) / 65535);
            return;
        }
        var w = new DeflateBitWriter { Acc = 0, Count = 0, Pos = hdrAt, End = hdrAt + MaxDynamicHeaderBytes, Overflow = 0 };
        int hlit = s[st + StHlit];
        int hdist = s[st + StHdist];
        int hclen = s[st + StHclen];
        int seqCount = s[st + StSeq];
        PutBits(ref w, hdr, 1, 1);
        PutBits(ref w, hdr, 2, 2);
        PutBits(ref w, hdr, (uint)(hlit - 257), 5);
        PutBits(ref w, hdr, (uint)(hdist - 1), 5);
        PutBits(ref w, hdr, (uint)(hclen - 4), 4);
        for (int i = 0; i < hclen; i++)
            PutBits(ref w, hdr, (uint)s[tail + ClLenOff + InflateGpu.CodeLengthOrder(i)], 3);
        for (int i = 0; i < seqCount; i++)
        {
            int packed = s[tail + ClSeqOff + i];
            int sym = packed & 31;
            PutBits(ref w, hdr, (uint)s[tail + ClCodeOff + sym], s[tail + ClLenOff + sym]);
            int eb = (packed >> 12) & 7;
            if (eb > 0) PutBits(ref w, hdr, (uint)((packed >> 5) & 127), eb);
        }
        int headerBits = (w.Pos - hdrAt) * 8 + w.Count;
        if (w.Count > 0) hdr[w.Pos] = (byte)w.Acc;
        s[st + HoHeaderBits] = headerBits;
        s[st + HoTotalBits] = (int)dynBits;
        // A final dynamic block ends byte-aligned (AlignToByte): ceil(bits / 8) bytes.
        s[st + StOutLen] = (int)((dynBits + 7) / 8);
    }

    /// <summary>Planned output bytes of the Huffman-only block (dynamic or stored).</summary>
    public static int PlannedBytes(ArrayView<int> s, int tail) => s[tail + StateOff + StOutLen];

    /// <summary>The planned block's kind: 0 dynamic, 1 stored, 2 empty (its 2 bytes are the header bytes).</summary>
    public static int PlannedKind(ArrayView<int> s, int tail) => s[tail + StateOff + HoStored];

    /// <summary>Bits of the planned dynamic block's header.</summary>
    public static int PlannedHeaderBits(ArrayView<int> s, int tail) => s[tail + StateOff + HoHeaderBits];

    /// <summary>Bits of the planned dynamic block (header, symbols and end-of-block code).</summary>
    public static int PlannedTotalBits(ArrayView<int> s, int tail) => s[tail + StateOff + HoTotalBits];
}
