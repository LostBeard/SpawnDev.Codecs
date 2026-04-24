using SpawnDev.Codecs.Audio.Flac;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for <see cref="FlacBitReader"/>. FLAC bitstreams are MSB-first and big-endian,
/// the inverse of the Opus range-coded bitstream. Covers:
/// byte-aligned ReadBits, cross-byte reads, signed reads with two's-complement
/// sign-extension, unary prefix reads (Rice coding), and the UTF-8 variable-length
/// integer used in FLAC frame headers.
/// </summary>
public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void FlacBitReader_ReadBits_ByteAligned_ReturnsFullByte()
    {
        byte[] data = { 0xA5, 0x3C };
        var r = new FlacBitReader(data);
        Equal(0xA5u, r.ReadBits(8));
        Equal(0x3Cu, r.ReadBits(8));
        True(r.IsEnd, "Reader should be at end after consuming both bytes.");
    }

    [TestMethod]
    public void FlacBitReader_ReadBits_SingleBitsMsbFirst()
    {
        // 0xA5 = 1010 0101
        byte[] data = { 0xA5 };
        var r = new FlacBitReader(data);
        Equal(1u, r.ReadBit());
        Equal(0u, r.ReadBit());
        Equal(1u, r.ReadBit());
        Equal(0u, r.ReadBit());
        Equal(0u, r.ReadBit());
        Equal(1u, r.ReadBit());
        Equal(0u, r.ReadBit());
        Equal(1u, r.ReadBit());
        True(r.IsEnd, "Reader should be at end after 8 bits.");
    }

    [TestMethod]
    public void FlacBitReader_ReadBits_PartialByte()
    {
        // 0xA5 = 1010 0101; take top 3 bits (101 = 5), then next 5 bits (00101 = 5).
        byte[] data = { 0xA5 };
        var r = new FlacBitReader(data);
        Equal(5u, r.ReadBits(3));
        Equal(5u, r.ReadBits(5));
        True(r.IsEnd);
    }

    [TestMethod]
    public void FlacBitReader_ReadBits_CrossByteBoundary()
    {
        // 0xA5 0x3C = 1010 0101 0011 1100
        // Skip 4 bits (1010), then read 8 bits (0101 0011) = 0x53.
        byte[] data = { 0xA5, 0x3C };
        var r = new FlacBitReader(data);
        Equal(0xAu, r.ReadBits(4));
        Equal(0x53u, r.ReadBits(8));
        Equal(0xCu, r.ReadBits(4));
        True(r.IsEnd);
    }

    [TestMethod]
    public void FlacBitReader_ReadBits_Span_MultiByte()
    {
        // 16-bit field spanning full reader; 0xA53C.
        byte[] data = { 0xA5, 0x3C };
        var r = new FlacBitReader(data);
        Equal(0xA53Cu, r.ReadBits(16));
        True(r.IsEnd);
    }

    [TestMethod]
    public void FlacBitReader_ReadBits_MaxWidth_32Bits()
    {
        // 32-bit big-endian integer 0xDEADBEEF.
        byte[] data = { 0xDE, 0xAD, 0xBE, 0xEF };
        var r = new FlacBitReader(data);
        Equal(0xDEADBEEFu, r.ReadBits(32));
        True(r.IsEnd);
    }

    [TestMethod]
    public void FlacBitReader_ReadBits_Zero_ReturnsZero_NoAdvance()
    {
        byte[] data = { 0xFF };
        var r = new FlacBitReader(data);
        Equal(0u, r.ReadBits(0));
        Equal(0, r.Position);
        Equal(8, r.BitsRemaining);
    }

    [TestMethod]
    public void FlacBitReader_ReadBits_OutOfRange_Throws()
    {
        byte[] data = { 0xFF };
        Throws<ArgumentOutOfRangeException>(() =>
        {
            var r = new FlacBitReader(data);
            r.ReadBits(-1);
        });
        Throws<ArgumentOutOfRangeException>(() =>
        {
            var r = new FlacBitReader(data);
            r.ReadBits(33);
        });
    }

    [TestMethod]
    public void FlacBitReader_ReadBits_NotEnough_Throws()
    {
        byte[] data = { 0xFF };
        Throws<InvalidOperationException>(() =>
        {
            var r = new FlacBitReader(data);
            _ = r.ReadBits(16);
        });
    }

    [TestMethod]
    public void FlacBitReader_BitsRemaining_And_Position_Track()
    {
        byte[] data = { 0xFF, 0xFF };
        var r = new FlacBitReader(data);
        Equal(16, r.BitsRemaining);
        Equal(0, r.Position);
        _ = r.ReadBits(5);
        Equal(11, r.BitsRemaining);
        Equal(5, r.Position);
        _ = r.ReadBits(10);
        Equal(1, r.BitsRemaining);
        Equal(15, r.Position);
        _ = r.ReadBit();
        Equal(0, r.BitsRemaining);
        Equal(16, r.Position);
        True(r.IsEnd);
    }

    [TestMethod]
    public void FlacBitReader_ReadBitsSigned_Positive()
    {
        // 5-bit signed integer 0b01010 = +10.
        byte[] data = { 0b01010_000 };
        var r = new FlacBitReader(data);
        Equal(10, r.ReadBitsSigned(5));
    }

    [TestMethod]
    public void FlacBitReader_ReadBitsSigned_NegativeSignExtend()
    {
        // 5-bit signed integer 0b11010 = -6 (two's complement).
        byte[] data = { 0b11010_000 };
        var r = new FlacBitReader(data);
        Equal(-6, r.ReadBitsSigned(5));
    }

    [TestMethod]
    public void FlacBitReader_ReadBitsSigned_FullWidth32()
    {
        // 0x80000000 as a 32-bit signed int = int.MinValue.
        byte[] data = { 0x80, 0x00, 0x00, 0x00 };
        var r = new FlacBitReader(data);
        Equal(int.MinValue, r.ReadBitsSigned(32));
    }

    [TestMethod]
    public void FlacBitReader_ReadBitsSigned_AllOnes()
    {
        // 4 bits all-ones = -1.
        byte[] data = { 0xF0 };
        var r = new FlacBitReader(data);
        Equal(-1, r.ReadBitsSigned(4));
    }

    [TestMethod]
    public void FlacBitReader_ReadBitsSigned_InvalidWidth_Throws()
    {
        byte[] data = { 0xFF };
        Throws<ArgumentOutOfRangeException>(() =>
        {
            var r = new FlacBitReader(data);
            _ = r.ReadBitsSigned(0);
        });
        Throws<ArgumentOutOfRangeException>(() =>
        {
            var r = new FlacBitReader(data);
            _ = r.ReadBitsSigned(33);
        });
    }

    [TestMethod]
    public void FlacBitReader_ReadUnary_ZeroPrefix()
    {
        // 0b1000_0000: no leading zeros, terminating 1 is the first bit.
        byte[] data = { 0x80 };
        var r = new FlacBitReader(data);
        Equal(0, r.ReadUnary());
        Equal(1, r.Position);
    }

    [TestMethod]
    public void FlacBitReader_ReadUnary_SevenZeros()
    {
        // 0b0000_0001: 7 zero bits then a terminating 1.
        byte[] data = { 0x01 };
        var r = new FlacBitReader(data);
        Equal(7, r.ReadUnary());
        Equal(8, r.Position);
        True(r.IsEnd);
    }

    [TestMethod]
    public void FlacBitReader_ReadUnary_CrossesByteBoundary()
    {
        // 0x00 0x40 = 0000 0000 0100 0000: 9 zero bits then a 1.
        byte[] data = { 0x00, 0x40 };
        var r = new FlacBitReader(data);
        Equal(9, r.ReadUnary());
        Equal(10, r.Position);
    }

    [TestMethod]
    public void FlacBitReader_ReadUnary_ExceedsStream_Throws()
    {
        // All zeros, no terminating 1.
        byte[] data = { 0x00, 0x00 };
        Throws<InvalidOperationException>(() =>
        {
            var r = new FlacBitReader(data);
            _ = r.ReadUnary();
        });
    }

    [TestMethod]
    public void FlacBitReader_AlignToByte_FromMidByte()
    {
        byte[] data = { 0xFF, 0x55 };
        var r = new FlacBitReader(data);
        _ = r.ReadBits(3);
        Equal(3, r.Position);
        r.AlignToByte();
        Equal(8, r.Position);
        Equal(0x55u, r.ReadBits(8));
    }

    [TestMethod]
    public void FlacBitReader_AlignToByte_OnBoundary_NoOp()
    {
        byte[] data = { 0xFF, 0x55 };
        var r = new FlacBitReader(data);
        _ = r.ReadBits(8);
        Equal(8, r.Position);
        r.AlignToByte();
        Equal(8, r.Position);
        Equal(0x55u, r.ReadBits(8));
    }

    // ----- UTF-8 variable-length integer (FLAC frame header sample/frame number) -----

    [TestMethod]
    public void FlacBitReader_ReadUtf8VarLen_AsciiRange_OneByte()
    {
        // Any byte < 0x80 is itself the value.
        byte[] data = { 0x00 };
        var r = new FlacBitReader(data);
        Equal(0ul, r.ReadUtf8VariableLength(7));
        var r2 = new FlacBitReader(new byte[] { 0x7F });
        Equal(0x7Ful, r2.ReadUtf8VariableLength(7));
    }

    [TestMethod]
    public void FlacBitReader_ReadUtf8VarLen_TwoByte()
    {
        // Classic UTF-8 encoding of U+00A2 (¢): 0xC2 0xA2 => 0xA2 (162).
        byte[] data = { 0xC2, 0xA2 };
        var r = new FlacBitReader(data);
        Equal(0xA2ul, r.ReadUtf8VariableLength(7));
    }

    [TestMethod]
    public void FlacBitReader_ReadUtf8VarLen_ThreeByte()
    {
        // U+20AC (€): 0xE2 0x82 0xAC => 0x20AC (8364).
        byte[] data = { 0xE2, 0x82, 0xAC };
        var r = new FlacBitReader(data);
        Equal(0x20ACul, r.ReadUtf8VariableLength(7));
    }

    [TestMethod]
    public void FlacBitReader_ReadUtf8VarLen_FourByte()
    {
        // U+1F600 (😀): 0xF0 0x9F 0x98 0x80 => 0x1F600.
        byte[] data = { 0xF0, 0x9F, 0x98, 0x80 };
        var r = new FlacBitReader(data);
        Equal(0x1F600ul, r.ReadUtf8VariableLength(7));
    }

    [TestMethod]
    public void FlacBitReader_ReadUtf8VarLen_FiveByte()
    {
        // 5-byte form (0xF8-0xFB lead, 4 continuations, 26 bits). firstByte & 0x03 = 0.
        // 0xF8 0x82 0x83 0x84 0x85 -> value = (2<<18)|(3<<12)|(4<<6)|5
        byte[] data = { 0xF8, 0x82, 0x83, 0x84, 0x85 };
        var r = new FlacBitReader(data);
        ulong expected = (2UL << 18) | (3UL << 12) | (4UL << 6) | 5UL;
        Equal(expected, r.ReadUtf8VariableLength(7));
    }

    [TestMethod]
    public void FlacBitReader_ReadUtf8VarLen_SixByte_FrameNumber()
    {
        // 6-byte form (0xFC-0xFD lead, 5 continuations, 31 bits). firstByte & 0x01 = 0.
        // 0xFC 0x81 0x82 0x83 0x84 0x85 -> value = (1<<24)|(2<<18)|(3<<12)|(4<<6)|5
        byte[] data = { 0xFC, 0x81, 0x82, 0x83, 0x84, 0x85 };
        var r = new FlacBitReader(data);
        ulong expected = (1UL << 24) | (2UL << 18) | (3UL << 12) | (4UL << 6) | 5UL;
        Equal(expected, r.ReadUtf8VariableLength(6));
    }

    [TestMethod]
    public void FlacBitReader_ReadUtf8VarLen_SevenByte_SampleNumber()
    {
        // 7-byte form is the sample-number path; firstByte must be exactly 0xFE (36 bits).
        // 0xFE 0x82 0x83 0x84 0x85 0x86 0x87 -> value = (2<<30)|(3<<24)|(4<<18)|(5<<12)|(6<<6)|7
        byte[] data = { 0xFE, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87 };
        var r = new FlacBitReader(data);
        ulong expected = (2UL << 30) | (3UL << 24) | (4UL << 18) | (5UL << 12) | (6UL << 6) | 7UL;
        Equal(expected, r.ReadUtf8VariableLength(7));
    }

    [TestMethod]
    public void FlacBitReader_ReadUtf8VarLen_SevenByte_NotAllowed_Throws()
    {
        // maxBytes=6 (frame-number path) must reject a 7-byte 0xFE lead.
        byte[] data = { 0xFE, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87 };
        Throws<InvalidOperationException>(() =>
        {
            var r = new FlacBitReader(data);
            _ = r.ReadUtf8VariableLength(6);
        });
    }

    [TestMethod]
    public void FlacBitReader_ReadUtf8VarLen_InvalidContinuation_Throws()
    {
        // 2-byte start (0xC2) followed by a non-continuation byte (0x00).
        byte[] data = { 0xC2, 0x00 };
        Throws<InvalidOperationException>(() =>
        {
            var r = new FlacBitReader(data);
            _ = r.ReadUtf8VariableLength(7);
        });
    }

    [TestMethod]
    public void FlacBitReader_ReadUtf8VarLen_InvalidLead_Throws()
    {
        // 0x80 is a bare continuation byte, cannot be a lead.
        byte[] data = { 0x80 };
        Throws<InvalidOperationException>(() =>
        {
            var r = new FlacBitReader(data);
            _ = r.ReadUtf8VariableLength(7);
        });
    }

    [TestMethod]
    public void FlacBitReader_ReadUtf8VarLen_InvalidMaxBytes_Throws()
    {
        byte[] data = { 0x00 };
        Throws<ArgumentException>(() =>
        {
            var r = new FlacBitReader(data);
            _ = r.ReadUtf8VariableLength(5);
        });
        Throws<ArgumentException>(() =>
        {
            var r = new FlacBitReader(data);
            _ = r.ReadUtf8VariableLength(8);
        });
    }
}
