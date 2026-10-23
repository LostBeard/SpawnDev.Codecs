using SpawnDev.Codecs.Audio.Vorbis;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for <see cref="VorbisHuffman"/>.Build - canonical Huffman codeword
/// assignment for Vorbis codebooks per Vorbis I Section 3.2.1.
///
/// Canonical form: sort by (length, entry index) and assign codewords in
/// increasing order; first codeword at length L is
/// <c>(first code at L-1 + count[L-1]) &lt;&lt; 1</c>.
/// </summary>
public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void VorbisHuffman_AllSameLength_AssignsConsecutiveCodes()
    {
        CpuOnlyTest();
        // 4 entries, each length 2 -> codes 00, 01, 10, 11.
        var lengths = new[] { 2, 2, 2, 2 };
        var tbl = VorbisHuffman.Build(lengths);
        Equal(4, tbl.Codewords.Length);
        Equal(0u, tbl.Codewords[0]);
        Equal(1u, tbl.Codewords[1]);
        Equal(2u, tbl.Codewords[2]);
        Equal(3u, tbl.Codewords[3]);
        Equal(2, tbl.MaxLength);
    }

    [TestMethod]
    public void VorbisHuffman_CanonicalIncreasingLengths()
    {
        CpuOnlyTest();
        // Classical canonical case:
        // 2 entries at length 1 -> ERROR (over-specified, 2*2 = 4 slots at length 2 exceeded).
        // Use: 1 entry at length 1, 2 at length 2.
        // code[0] = 0 (length 1); code[1] = 10 = 2; code[2] = 11 = 3.
        var lengths = new[] { 1, 2, 2 };
        var tbl = VorbisHuffman.Build(lengths);
        Equal(0u, tbl.Codewords[0]);
        Equal(2u, tbl.Codewords[1]);
        Equal(3u, tbl.Codewords[2]);
    }

    [TestMethod]
    public void VorbisHuffman_WithUnusedEntries_UnusedGetZeroCode()
    {
        CpuOnlyTest();
        // Entry 1 unused (length 0). Entries 0 and 2 have length 1 -> codes 0 and 1.
        var lengths = new[] { 1, 0, 1 };
        var tbl = VorbisHuffman.Build(lengths);
        Equal(0u, tbl.Codewords[0]);
        Equal(0u, tbl.Codewords[1]); // unused
        Equal(1u, tbl.Codewords[2]);
    }

    [TestMethod]
    public void VorbisHuffman_SingleUsedEntry_CodeIsZero()
    {
        CpuOnlyTest();
        // Degenerate: only one entry is used. Vorbis spec allows this; the sole
        // entry gets code 0 regardless of its declared length.
        var lengths = new[] { 0, 3, 0 };
        var tbl = VorbisHuffman.Build(lengths);
        Equal(0u, tbl.Codewords[1]);
    }

    [TestMethod]
    public void VorbisHuffman_OverSpecified_Throws()
    {
        CpuOnlyTest();
        // 3 entries at length 1: 3 > 2 codes available at that length.
        var lengths = new[] { 1, 1, 1 };
        bool threw = false;
        try { _ = VorbisHuffman.Build(lengths); } catch (InvalidDataException) { threw = true; }
        True(threw);
    }

    [TestMethod]
    public void VorbisHuffman_MixedLengths_AssignLowestFreeCodewordInEntryOrder()
    {
        CpuOnlyTest();
        // Vorbis I spec 3.2.1: Vorbis codebooks are NOT canonical Huffman codes. Entries are taken in ENTRY order and each one
        // gets the numerically lowest codeword of its length that is still free (not a prefix of, and not prefixed by, a
        // codeword already assigned) - libvorbis sharedbook.c _make_words. (This test used to expect canonical codes 6/0/7/2,
        // which no Vorbis decoder produces; it had failed since it was written.)
        // lengths {3, 1, 3, 2}: entry 0 takes 000; entry 1 the lowest free 1-bit code is 1 (0 is a prefix of 000);
        // entry 2 takes 001; entry 3 takes 01.
        var lengths = new[] { 3, 1, 3, 2 };
        var tbl = VorbisHuffman.Build(lengths);
        Equal(0b000u, tbl.Codewords[0]);
        Equal(0b1u, tbl.Codewords[1]);
        Equal(0b001u, tbl.Codewords[2]);
        Equal(0b01u, tbl.Codewords[3]);
    }

    [TestMethod]
    public void VorbisHuffman_SpecExample_Section3_2_1()
    {
        CpuOnlyTest();
        // The worked example of the Vorbis I specification, section 3.2.1: lengths 2 4 4 4 4 2 3 3 give the codewords
        // 00, 0100, 0101, 0110, 0111, 10, 110, 111.
        var tbl = VorbisHuffman.Build(new[] { 2, 4, 4, 4, 4, 2, 3, 3 });
        uint[] expected = { 0b00, 0b0100, 0b0101, 0b0110, 0b0111, 0b10, 0b110, 0b111 };
        for (int i = 0; i < expected.Length; i++) Equal(expected[i], tbl.Codewords[i], $"entry {i}");
    }

    [TestMethod]
    public void VorbisHuffman_AllUnused_EmptyTable()
    {
        CpuOnlyTest();
        var lengths = new[] { 0, 0, 0 };
        var tbl = VorbisHuffman.Build(lengths);
        Equal(0, tbl.MaxLength);
        Equal(3, tbl.Codewords.Length);
    }

    [TestMethod]
    public void VorbisHuffman_MaxLength32_Builds()
    {
        CpuOnlyTest();
        // One entry at length 32 and many unused. Valid but extreme.
        var lengths = new int[2];
        lengths[0] = 32;
        lengths[1] = 0;
        var tbl = VorbisHuffman.Build(lengths);
        Equal(32, tbl.MaxLength);
    }

    [TestMethod]
    public void VorbisHuffman_MaxLength33_Throws()
    {
        CpuOnlyTest();
        var lengths = new[] { 33 };
        bool threw = false;
        try { _ = VorbisHuffman.Build(lengths); } catch (InvalidDataException) { threw = true; }
        True(threw);
    }
}
