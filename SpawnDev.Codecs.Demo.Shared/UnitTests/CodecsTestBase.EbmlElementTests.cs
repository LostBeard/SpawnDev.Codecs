using SpawnDev.Codecs.Container.Ebml;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for <see cref="EbmlElementReader"/>. Hand-builds EBML element
/// sequences and verifies the reader picks out IDs, sizes, and offsets.
/// </summary>
public abstract partial class CodecsTestBase
{
    /// <summary>Build a one-byte VINT with the length marker preserved.</summary>
    private static byte OneByteIdVint(byte value) => (byte)(value | 0x80);

    /// <summary>Build a one-byte VINT size (stripped representation; marker at bit 7).</summary>
    private static byte OneByteSizeVint(byte value)
    {
        if (value > 0x7E) throw new ArgumentException("value too large for 1-byte VINT");
        return (byte)(value | 0x80);
    }

    /// <summary>Build a 4-byte EBML header ID 0x1A45DFA3 verbatim.</summary>
    private static byte[] EbmlHeaderId() => new byte[] { 0x1A, 0x45, 0xDF, 0xA3 };

    [TestMethod]
    public void EbmlElement_SingleOneByteIdAndSize()
    {
        // ID = 0x81, Size VINT = 0x82 -> size=2, data = 0xAA 0xBB.
        byte[] bytes = { 0x81, 0x82, 0xAA, 0xBB };
        var el = EbmlElementReader.ReadAt(bytes, 0);
        Equal(0x81UL, el.Id);
        Equal(2UL, el.Size);
        Equal(0, el.Offset);
        Equal(2, el.HeaderBytes);
        Equal(2, el.DataOffset);
    }

    [TestMethod]
    public void EbmlElement_MatroskaRootHeader_DecodesFourByteId()
    {
        // ID = 0x1A45DFA3 (4-byte VINT preserved), size = 0x80 (1-byte stripped to 0).
        var bytes = new List<byte>();
        bytes.AddRange(EbmlHeaderId());
        bytes.Add(0x80); // size 0
        var el = EbmlElementReader.ReadAt(bytes.ToArray(), 0);
        Equal(0x1A45DFA3UL, el.Id);
        Equal(0UL, el.Size);
        Equal(5, el.HeaderBytes);
    }

    [TestMethod]
    public void EbmlElement_UnknownSize_ReturnsSentinelAndStopsEnumeration()
    {
        // Element with all-ones size VINT -> unknown size.
        byte[] bytes = { OneByteIdVint(0x1A), 0xFF, 0x00, 0x00 };
        var el = EbmlElementReader.ReadAt(bytes, 0);
        Equal(EbmlVint.UnknownSize, el.Size);
        // Top-level enumeration should stop cleanly.
        var list = EbmlElementReader.EnumerateTopLevel(bytes).ToList();
        Equal(1, list.Count);
    }

    [TestMethod]
    public void EbmlElement_EnumerateTopLevel_YieldsSequence()
    {
        // Three consecutive 1-byte-id + 1-byte-size elements:
        //   (0x81, size 2, 0xAA 0xBB)
        //   (0x82, size 1, 0xCC)
        //   (0x83, size 0)
        byte[] bytes =
        {
            0x81, 0x82, 0xAA, 0xBB,
            0x82, 0x81, 0xCC,
            0x83, 0x80,
        };
        var list = EbmlElementReader.EnumerateTopLevel(bytes).ToList();
        Equal(3, list.Count);
        Equal(0x81UL, list[0].Id);
        Equal(2UL, list[0].Size);
        Equal(0x82UL, list[1].Id);
        Equal(1UL, list[1].Size);
        Equal(0x83UL, list[2].Id);
        Equal(0UL, list[2].Size);
    }

    [TestMethod]
    public void EbmlElement_EnumerationStops_WhenSizeExtendsPastBuffer()
    {
        // Element claims size 100 but we only supplied 10 bytes of data.
        // EnumerateTopLevel should yield the element and then stop cleanly.
        byte[] bytes = new byte[12];
        bytes[0] = 0x81;
        bytes[1] = 0x80 | 100;
        // Rest is arbitrary...but we only have 10 data bytes available.
        var list = EbmlElementReader.EnumerateTopLevel(bytes).ToList();
        Equal(1, list.Count);
        Equal(100UL, list[0].Size);
    }

    [TestMethod]
    public void EbmlElement_SegmentHeader_ParsesCorrectly()
    {
        // Matroska "Segment" = 0x18538067 (4-byte VINT preserved), size = 0x410000 with
        // 3-byte VINT = 0x40 0x00 0x00... actually let's use 0x82 (1-byte) for simplicity.
        // Build: 0x18 0x53 0x80 0x67 0x82 DATA DATA.
        var bytes = new byte[] { 0x18, 0x53, 0x80, 0x67, 0x82, 0x11, 0x22 };
        var el = EbmlElementReader.ReadAt(bytes, 0);
        Equal(0x18538067UL, el.Id);
        Equal(2UL, el.Size);
    }
}
