using SpawnDev.Codecs.Audio.Vorbis;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for <see cref="VorbisHuffmanDecoder"/>. Each test builds a codebook,
/// runs a known sequence of entry indices through an inline encoder that
/// writes the canonical codewords MSB-first onto an LSB-first Vorbis
/// bitstream, and decodes them back through the production decoder.
/// </summary>
public abstract partial class CodecsTestBase
{
    /// <summary>Emit a canonical Huffman codeword MSB-first onto a Vorbis LSB-first stream.</summary>
    private static void WriteHuffmanCodeword(VorbisTestWriter w, uint code, int length)
    {
        for (int bitIdx = length - 1; bitIdx >= 0; bitIdx--)
            w.Write((code >> bitIdx) & 1u, 1);
    }

    [TestMethod]
    public void VorbisHuffmanDecoder_AllEqualLengths_RoundtripsEverySymbol()
    {
        // 4-entry codebook, all length 2.
        var lengths = new[] { 2, 2, 2, 2 };
        var table = VorbisHuffman.Build(lengths);
        var decoder = new VorbisHuffmanDecoder(table);

        // Encode sequence 0, 3, 1, 2, 2, 0.
        var sequence = new[] { 0, 3, 1, 2, 2, 0 };
        var w = new VorbisTestWriter();
        foreach (var s in sequence)
            WriteHuffmanCodeword(w, table.Codewords[s], table.EntryLengths[s]);
        var bytes = w.ToArray();
        var reader = new VorbisBitReader(bytes);
        foreach (var s in sequence)
            Equal(s, decoder.Decode(ref reader));
    }

    [TestMethod]
    public void VorbisHuffmanDecoder_VariableLengths_RoundtripsCanonicalExample()
    {
        // 1 entry at length 1 (idx 0 -> code 0)
        // 2 entries at length 2 (idx 1 -> code 10 = 2, idx 2 -> code 11 = 3)
        var lengths = new[] { 1, 2, 2 };
        var table = VorbisHuffman.Build(lengths);
        var decoder = new VorbisHuffmanDecoder(table);
        var sequence = new[] { 2, 0, 1, 2, 0 };
        var w = new VorbisTestWriter();
        foreach (var s in sequence)
            WriteHuffmanCodeword(w, table.Codewords[s], table.EntryLengths[s]);
        var bytes = w.ToArray();
        var reader = new VorbisBitReader(bytes);
        foreach (var s in sequence)
            Equal(s, decoder.Decode(ref reader));
    }

    [TestMethod]
    public void VorbisHuffmanDecoder_SparseCodebook_SkipsUnusedEntries()
    {
        // Entries 0 and 2 used with length 1; entry 1 unused.
        var lengths = new[] { 1, 0, 1 };
        var table = VorbisHuffman.Build(lengths);
        var decoder = new VorbisHuffmanDecoder(table);
        var sequence = new[] { 0, 2, 2, 0, 0, 2 };
        var w = new VorbisTestWriter();
        foreach (var s in sequence)
            WriteHuffmanCodeword(w, table.Codewords[s], table.EntryLengths[s]);
        var reader = new VorbisBitReader(w.ToArray());
        foreach (var s in sequence)
            Equal(s, decoder.Decode(ref reader));
    }

    [TestMethod]
    public void VorbisHuffmanDecoder_DeepTree_8Entries3Bits()
    {
        // 8-entry codebook, every entry at length 3 -> full binary tree of depth 3.
        var lengths = new int[8];
        for (int i = 0; i < 8; i++) lengths[i] = 3;
        var table = VorbisHuffman.Build(lengths);
        var decoder = new VorbisHuffmanDecoder(table);
        var sequence = new[] { 7, 0, 3, 5, 1, 6, 2, 4 };
        var w = new VorbisTestWriter();
        foreach (var s in sequence)
            WriteHuffmanCodeword(w, table.Codewords[s], table.EntryLengths[s]);
        var reader = new VorbisBitReader(w.ToArray());
        foreach (var s in sequence)
            Equal(s, decoder.Decode(ref reader));
    }

    [TestMethod]
    public void VorbisHuffmanDecoder_MixedLengthEntries_DecodesInOrder()
    {
        // 5 entries with lengths 2, 3, 3, 3, 3. One short code, four long ones.
        // Canonical codes (computed by Build):
        // idx 0 len 2 -> 00 = 0
        // idx 1 len 3 -> counter[3] starts at (0 + 1) << 1 = 2 ... no wait
        //   count[2]=1, nextCode[3] = (nextCode[2] + count[2]) << 1 = (0 + 1) << 1 = 2?
        //   nextCode[2] = 0, so nextCode[3] = 2.
        //   But counter[3] starts at nextCode[3] = 2 only if not already assigned elsewhere.
        //   idx 1: code = 2 (010), counter[3]++
        //   idx 2: code = 3 (011), counter[3]++
        //   idx 3: code = 4 (100), counter[3]++
        //   idx 4: code = 5 (101), counter[3]++
        //   counter[3] ends at 6 - check against (1<<3) = 8, ok.
        //   Unused leaves 110, 111 left in tree - decoder may error if bit pattern lands there.
        var lengths = new[] { 2, 3, 3, 3, 3 };
        var table = VorbisHuffman.Build(lengths);
        var decoder = new VorbisHuffmanDecoder(table);
        var sequence = new[] { 0, 4, 1, 0, 2, 3, 4, 1 };
        var w = new VorbisTestWriter();
        foreach (var s in sequence)
            WriteHuffmanCodeword(w, table.Codewords[s], table.EntryLengths[s]);
        var reader = new VorbisBitReader(w.ToArray());
        foreach (var s in sequence)
            Equal(s, decoder.Decode(ref reader));
    }

    [TestMethod]
    public void VorbisHuffmanDecoder_SingleEntryCodebook_AlwaysDecodesZero()
    {
        // Single-used-entry degenerate: entry 0 at some length. Build assigns code 0.
        // Decoder should succeed after reading `length` zero bits.
        var lengths = new[] { 4 };
        var table = VorbisHuffman.Build(lengths);
        var decoder = new VorbisHuffmanDecoder(table);
        // Encode the single entry twice: write "0000 0000" (8 bits = 2 codewords).
        var w = new VorbisTestWriter();
        w.Write(0, 8);
        var reader = new VorbisBitReader(w.ToArray());
        Equal(0, decoder.Decode(ref reader));
        Equal(0, decoder.Decode(ref reader));
    }
}
