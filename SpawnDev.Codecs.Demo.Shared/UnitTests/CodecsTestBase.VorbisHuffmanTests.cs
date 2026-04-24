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
        // Degenerate: only one entry is used. Vorbis spec allows this; the sole
        // entry gets code 0 regardless of its declared length.
        var lengths = new[] { 0, 3, 0 };
        var tbl = VorbisHuffman.Build(lengths);
        Equal(0u, tbl.Codewords[1]);
    }

    [TestMethod]
    public void VorbisHuffman_OverSpecified_Throws()
    {
        // 3 entries at length 1: 3 > 2 codes available at that length.
        var lengths = new[] { 1, 1, 1 };
        bool threw = false;
        try { _ = VorbisHuffman.Build(lengths); } catch (InvalidDataException) { threw = true; }
        True(threw);
    }

    [TestMethod]
    public void VorbisHuffman_MixedLengths_SortsByLengthThenIndex()
    {
        // Entry order irrelevant to canonical form - sorting is by (length, entry idx).
        // Entries: idx 0 = len 3, idx 1 = len 1, idx 2 = len 3, idx 3 = len 2.
        // Sorted: idx 1 (len 1), idx 3 (len 2), idx 0 (len 3), idx 2 (len 3).
        // Canonical codes:
        //   idx 1: code 0 at length 1.
        //   idx 3: code 10 = 2 at length 2. (nextCode[2] = (0 + 1) << 1 = 2)
        //   idx 0: code 110 = 6 at length 3. (nextCode[3] = (2 + 1) << 1 = 6)
        //   idx 2: code 111 = 7 at length 3.
        // BUT our Build assigns in entry-index order, not sorted order.
        // Entry-index order with per-length counters:
        //   i=0 (len 3): counter[3] starts at 6. code = 6. counter[3]++.
        //   i=1 (len 1): counter[1] starts at 0. code = 0. counter[1]++.
        //   i=2 (len 3): counter[3] now 7. code = 7.
        //   i=3 (len 2): counter[2] starts at 2. code = 2.
        var lengths = new[] { 3, 1, 3, 2 };
        var tbl = VorbisHuffman.Build(lengths);
        Equal(6u, tbl.Codewords[0]);
        Equal(0u, tbl.Codewords[1]);
        Equal(7u, tbl.Codewords[2]);
        Equal(2u, tbl.Codewords[3]);
    }

    [TestMethod]
    public void VorbisHuffman_AllUnused_EmptyTable()
    {
        var lengths = new[] { 0, 0, 0 };
        var tbl = VorbisHuffman.Build(lengths);
        Equal(0, tbl.MaxLength);
        Equal(3, tbl.Codewords.Length);
    }

    [TestMethod]
    public void VorbisHuffman_MaxLength32_Builds()
    {
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
        var lengths = new[] { 33 };
        bool threw = false;
        try { _ = VorbisHuffman.Build(lengths); } catch (InvalidDataException) { threw = true; }
        True(threw);
    }
}
