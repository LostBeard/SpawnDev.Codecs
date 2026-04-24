using SpawnDev.Codecs.Audio.Vorbis;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for <see cref="VorbisResidueConfigParser"/>. Hand-built bit patterns
/// verify the 24-bit begin/end/partition-size fields, the cascade decoder
/// (3-bit low + 1-bit flag + optional 5-bit high), and the per-classification
/// per-pass book indices.
/// </summary>
public abstract partial class CodecsTestBase
{
    private static VorbisResidueConfig ParseResidue(byte[] data, VorbisResidueType type)
    {
        var r = new VorbisBitReader(data);
        return VorbisResidueConfigParser.Parse(ref r, type);
    }

    [TestMethod]
    public void VorbisResidue_Type0_SingleClassification_SinglePass_Parses()
    {
        // begin = 0, end = 128, partition_size = 32 (write 31), classifications = 1 (write 0), classbook = 2.
        // cascade[0]: low bits = 0b001 (write 1), bitflag = 0 (no high bits) -> cascade = 0b001 = 1.
        //   Only pass 0 is active. books[0][0] = 5.
        var w = new VorbisTestWriter();
        w.Write(0, 24);            // begin
        w.Write(128, 24);          // end
        w.Write(31, 24);           // partition_size - 1 = 31 -> 32
        w.Write(0, 6);             // classifications - 1 = 0 -> 1
        w.Write(2, 8);             // classbook
        // cascade[0]
        w.Write(1, 3);             // low bits
        w.Write(0, 1);             // bitflag = no
        // books
        w.Write(5, 8);             // books[0][0]
        var cfg = ParseResidue(w.ToArray(), VorbisResidueType.Type0);
        Equal(VorbisResidueType.Type0, cfg.Type);
        Equal(0, cfg.Begin);
        Equal(128, cfg.End);
        Equal(32, cfg.PartitionSize);
        Equal(1, cfg.Classifications);
        Equal(2, cfg.Classbook);
        EqualInts(new[] { 1 }, cfg.Cascade);
        Equal(5, cfg.Books[0][0]);
        Equal(-1, cfg.Books[0][1]);
        Equal(-1, cfg.Books[0][7]);
    }

    [TestMethod]
    public void VorbisResidue_Type1_MultipleCascadePasses()
    {
        // cascade value 0b00001011 = 11 (passes 0, 1, 3).
        // low bits = 011 = 3, bitflag = 1, high bits = 00001 = 1 -> cascade = (1<<3) | 3 = 11.
        var w = new VorbisTestWriter();
        w.Write(16, 24); w.Write(256, 24); w.Write(63, 24); // 64-sample partitions
        w.Write(0, 6);              // 1 classification
        w.Write(7, 8);              // classbook
        w.Write(3, 3);              // low
        w.Write(1, 1);              // bitflag
        w.Write(1, 5);              // high -> cascade = 0b00001011
        // passes 0, 1, 3 active
        w.Write(10, 8);             // books[0][0]
        w.Write(11, 8);             // books[0][1]
        w.Write(12, 8);             // books[0][3]
        var cfg = ParseResidue(w.ToArray(), VorbisResidueType.Type1);
        Equal(11, cfg.Cascade[0]);
        Equal(10, cfg.Books[0][0]);
        Equal(11, cfg.Books[0][1]);
        Equal(-1, cfg.Books[0][2]);
        Equal(12, cfg.Books[0][3]);
    }

    [TestMethod]
    public void VorbisResidue_Type2_MultipleClassifications()
    {
        // 2 classifications, each with different cascade patterns.
        var w = new VorbisTestWriter();
        w.Write(0, 24); w.Write(64, 24); w.Write(15, 24); // 16-sample partitions
        w.Write(1, 6);              // classifications - 1 = 1 -> 2
        w.Write(3, 8);              // classbook
        // classification 0: cascade = 0b0000_0001 (pass 0 only)
        w.Write(1, 3);              // low
        w.Write(0, 1);              // bitflag
        // classification 1: cascade = 0b0000_0101 (passes 0 and 2)
        w.Write(5, 3);              // low = 101
        w.Write(0, 1);              // bitflag = 0 (no high bits)
        // books
        w.Write(20, 8);             // books[0][0]
        w.Write(30, 8);             // books[1][0]
        w.Write(31, 8);             // books[1][2]
        var cfg = ParseResidue(w.ToArray(), VorbisResidueType.Type2);
        Equal(VorbisResidueType.Type2, cfg.Type);
        Equal(2, cfg.Classifications);
        EqualInts(new[] { 1, 5 }, cfg.Cascade);
        Equal(20, cfg.Books[0][0]);
        Equal(30, cfg.Books[1][0]);
        Equal(-1, cfg.Books[1][1]);
        Equal(31, cfg.Books[1][2]);
    }

    [TestMethod]
    public void VorbisResidue_InvalidType_Throws()
    {
        var w = new VorbisTestWriter();
        // Data doesn't matter; parser rejects before reading.
        bool threw = false;
        try { ParseResidue(w.ToArray(), (VorbisResidueType)5); }
        catch (InvalidDataException) { threw = true; }
        True(threw);
    }

    [TestMethod]
    public void VorbisResidue_EndBeforeBegin_Throws()
    {
        var w = new VorbisTestWriter();
        w.Write(100, 24);           // begin
        w.Write(50, 24);            // end < begin
        w.Write(0, 24);             // partition_size - 1
        w.Write(0, 6);              // 1 classification
        w.Write(0, 8);              // classbook
        bool threw = false;
        try { ParseResidue(w.ToArray(), VorbisResidueType.Type0); }
        catch (InvalidDataException) { threw = true; }
        True(threw);
    }

    [TestMethod]
    public void VorbisResidue_CascadeZero_NoBooksRead()
    {
        // Classification with cascade = 0: no books are read for that classification.
        var w = new VorbisTestWriter();
        w.Write(0, 24); w.Write(64, 24); w.Write(15, 24);
        w.Write(0, 6);              // 1 classification
        w.Write(2, 8);              // classbook
        w.Write(0, 3); w.Write(0, 1); // cascade = 0
        // No books to read.
        var cfg = ParseResidue(w.ToArray(), VorbisResidueType.Type0);
        Equal(0, cfg.Cascade[0]);
        for (int j = 0; j < 8; j++) Equal(-1, cfg.Books[0][j]);
    }
}
