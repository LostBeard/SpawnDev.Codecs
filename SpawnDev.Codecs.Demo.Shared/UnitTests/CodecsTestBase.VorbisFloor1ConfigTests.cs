using SpawnDev.Codecs.Audio.Vorbis;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for <see cref="VorbisFloor1ConfigParser"/>. Each test hand-packs a
/// valid floor 1 configuration bit pattern using the LSB-first writer and
/// verifies the parsed result matches field-by-field.
/// </summary>
public abstract partial class CodecsTestBase
{
    private static VorbisFloor1Config ParseFloor1(byte[] data)
    {
        var r = new VorbisBitReader(data);
        return VorbisFloor1ConfigParser.Parse(ref r);
    }

    [TestMethod]
    public void VorbisFloor1_SingleClass_MinimalConfig()
    {
        // 1 partition belonging to class 0.
        // Class 0: dimensions = 1 (write 0), subclasses = 0 (no master book, 1 subclass book).
        //   subclass_books[0][0] = 5 (book index 5 written as 5 + 1 = 6).
        // multiplier = 1 (write 0), rangebits = 4.
        // X list will have 2 + 1 = 3 entries: [0, 16, value].
        var w = new VorbisTestWriter();
        w.Write(1, 5);            // partitions
        w.Write(0, 4);            // partition_class_list[0] = class 0
        w.Write(0, 3);            // class 0 dimensions - 1 = 0 -> 1 posterior
        w.Write(0, 2);            // class 0 subclasses bits = 0
        w.Write(6, 8);            // subclass book 0 index = 5 (written as 5+1 = 6)
        w.Write(0, 2);            // multiplier - 1 = 0 -> 1
        w.Write(4, 4);            // rangebits = 4
        w.Write(7, 4);            // xlist[2] = 7
        var cfg = ParseFloor1(w.ToArray());
        Equal(1, cfg.Partitions);
        EqualInts(new[] { 0 }, cfg.PartitionClassList);
        Equal(1, cfg.ClassDimensions.Length);
        Equal(1, cfg.ClassDimensions[0]);
        Equal(0, cfg.ClassSubclasses[0]);
        Equal(5, cfg.ClassSubclassBooks[0][0]);
        Equal(1, cfg.Multiplier);
        Equal(4, cfg.RangeBits);
        EqualInts(new[] { 0, 16, 7 }, cfg.XList);
    }

    [TestMethod]
    public void VorbisFloor1_TwoClasses_WithMasterbookAndSubclasses()
    {
        // 2 partitions: partition 0 in class 0, partition 1 in class 1.
        // Class 0: dimensions = 2 (write 1), subclasses = 1 (write 1) -> 2 subclass books,
        //          master book = 3 (write 3), subclass books 0 and 1 each = no-book (write 0 -> -1) then a real book 7 (write 8).
        // Class 1: dimensions = 3 (write 2), subclasses = 0 -> 1 subclass book, no master.
        //          subclass book[0] = 2 (write 3).
        var w = new VorbisTestWriter();
        w.Write(2, 5);            // partitions
        w.Write(0, 4);            // partition_class_list[0] = 0
        w.Write(1, 4);            // partition_class_list[1] = 1
        // Class 0
        w.Write(1, 3);            // dimensions - 1 = 1 -> 2
        w.Write(1, 2);            // subclasses = 1 -> 2 subclass books
        w.Write(3, 8);            // masterbook = 3
        w.Write(0, 8);            // subclass book 0 = -1 (unused)
        w.Write(8, 8);            // subclass book 1 = 7 (written as 7 + 1 = 8)
        // Class 1
        w.Write(2, 3);            // dimensions - 1 = 2 -> 3
        w.Write(0, 2);            // subclasses = 0 -> 1 subclass book
        w.Write(3, 8);            // subclass book[0] = 2 (written as 2 + 1 = 3)
        // Global
        w.Write(2, 2);            // multiplier - 1 = 2 -> 3
        w.Write(5, 4);            // rangebits = 5
        // X list: 2 + 2 (class 0 dims) + 3 (class 1 dims) = 7 entries.
        // First 2 are implicit (0 and 32). Next 5 read from bitstream.
        w.Write(4, 5);
        w.Write(8, 5);
        w.Write(12, 5);
        w.Write(16, 5);
        w.Write(20, 5);
        var cfg = ParseFloor1(w.ToArray());
        Equal(2, cfg.Partitions);
        EqualInts(new[] { 0, 1 }, cfg.PartitionClassList);
        Equal(2, cfg.ClassDimensions.Length);
        EqualInts(new[] { 2, 3 }, cfg.ClassDimensions);
        EqualInts(new[] { 1, 0 }, cfg.ClassSubclasses);
        Equal(3, cfg.ClassMasterbooks[0]);
        Equal(-1, cfg.ClassSubclassBooks[0][0]);
        Equal(7, cfg.ClassSubclassBooks[0][1]);
        Equal(2, cfg.ClassSubclassBooks[1][0]);
        Equal(3, cfg.Multiplier);
        Equal(5, cfg.RangeBits);
        EqualInts(new[] { 0, 32, 4, 8, 12, 16, 20 }, cfg.XList);
    }

    [TestMethod]
    public void VorbisFloor1_Multiplier_EncodesMinusOne()
    {
        // Verify that the on-wire 2-bit value encodes (multiplier - 1).
        // Using 3 (written as 3) should decode to 4.
        var w = new VorbisTestWriter();
        w.Write(1, 5);            // 1 partition
        w.Write(0, 4);            // class 0
        w.Write(0, 3);            // class 0 dims = 1
        w.Write(0, 2);            // class 0 subclasses = 0
        w.Write(1, 8);            // subclass book = 0 (1 - 1)
        w.Write(3, 2);            // multiplier = 4
        w.Write(4, 4);            // rangebits = 4
        w.Write(0, 4);            // xlist[2] = 0
        var cfg = ParseFloor1(w.ToArray());
        Equal(4, cfg.Multiplier);
    }

    [TestMethod]
    public void VorbisFloor1_XListOrdering_KeepsInputOrder()
    {
        // X list isn't sorted by the parser - it preserves bitstream order.
        // Write 3 partitions × 1 dim each, so 5 total X entries: [0, 2^rb, x0, x1, x2].
        var w = new VorbisTestWriter();
        w.Write(3, 5);            // partitions = 3
        w.Write(0, 4); w.Write(0, 4); w.Write(0, 4); // all class 0
        // Class 0 config
        w.Write(0, 3);            // dims = 1
        w.Write(0, 2);            // subclasses = 0
        w.Write(1, 8);            // subclass book = 0
        // Globals
        w.Write(0, 2);            // multiplier = 1
        w.Write(5, 4);            // rangebits = 5
        // X entries (intentionally unordered)
        w.Write(20, 5);
        w.Write(5, 5);
        w.Write(15, 5);
        var cfg = ParseFloor1(w.ToArray());
        EqualInts(new[] { 0, 32, 20, 5, 15 }, cfg.XList);
    }

    [TestMethod]
    public void VorbisFloor1_ExceedsSpecMaximum_Throws()
    {
        // Spec maximum floor1_values = 65. Build a config that sums to 66.
        // Use 11 partitions × class-with-dims-6 => 11*6 = 66 + 2 = 68 > 65.
        // (Partitions max is 31 per the 5-bit field, so this is reachable.)
        var w = new VorbisTestWriter();
        w.Write(11, 5);           // partitions = 11
        for (int i = 0; i < 11; i++) w.Write(0, 4); // all class 0
        w.Write(5, 3);            // dims - 1 = 5 -> 6
        w.Write(0, 2);            // subclasses = 0
        w.Write(1, 8);            // subclass book = 0
        w.Write(0, 2); w.Write(4, 4);
        // Don't need to supply X values since parser throws before reading them.
        bool threw = false;
        try { ParseFloor1(w.ToArray()); } catch (InvalidDataException) { threw = true; }
        True(threw, "Floor 1 X list > 65 should throw.");
    }
}
