using SpawnDev.Codecs.Audio.Flac;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for <see cref="FlacSubframeHeaderParser"/>. The 1-2 byte subframe header
/// layout per RFC 9639 Section 10: 1 zero bit + 6-bit type code + 1-bit wasted flag,
/// followed by unary-coded wasted-bit count if the flag is set.
/// </summary>
public abstract partial class CodecsTestBase
{
    private static FlacSubframeHeader ParseSubframeHeader(params byte[] data)
    {
        var r = new FlacBitReader(data);
        return FlacSubframeHeaderParser.Parse(ref r);
    }

    [TestMethod]
    public void FlacSubframeHeader_Constant_NoWastedBits()
    {
        // 0b00000000: reserved 0, type 0b000000 (CONSTANT), wasted flag 0.
        var h = ParseSubframeHeader(0x00);
        Equal(FlacSubframeKind.Constant, h.Kind);
        Equal(0, h.Order);
        Equal(0, h.WastedBitsPerSample);
    }

    [TestMethod]
    public void FlacSubframeHeader_Verbatim_NoWastedBits()
    {
        // 0b00000010: reserved 0, type 0b000001 (VERBATIM), wasted flag 0.
        var h = ParseSubframeHeader(0x02);
        Equal(FlacSubframeKind.Verbatim, h.Kind);
        Equal(0, h.Order);
        Equal(0, h.WastedBitsPerSample);
    }

    [TestMethod]
    public void FlacSubframeHeader_FixedOrder0()
    {
        // 0b00010000: type 0b001000 = FIXED order 0.
        var h = ParseSubframeHeader(0x10);
        Equal(FlacSubframeKind.Fixed, h.Kind);
        Equal(0, h.Order);
    }

    [TestMethod]
    public void FlacSubframeHeader_FixedOrder4()
    {
        // 0b00011000: type 0b001100 = FIXED order 4 (max valid FIXED order).
        var h = ParseSubframeHeader(0x18);
        Equal(FlacSubframeKind.Fixed, h.Kind);
        Equal(4, h.Order);
    }

    [TestMethod]
    public void FlacSubframeHeader_FixedOrder5_Throws()
    {
        // 0b00011010: type 0b001101 = FIXED order 5 (invalid).
        bool threw = false;
        try { ParseSubframeHeader(0x1A); } catch (InvalidDataException) { threw = true; }
        True(threw, "FIXED order 5 should throw.");
    }

    [TestMethod]
    public void FlacSubframeHeader_LpcOrder1()
    {
        // 0b01000000: reserved 0, type 0b100000 (code minus 32 = 0 => order 1), wasted flag 0.
        var h = ParseSubframeHeader(0x40);
        Equal(FlacSubframeKind.Lpc, h.Kind);
        Equal(1, h.Order);
    }

    [TestMethod]
    public void FlacSubframeHeader_LpcOrder32()
    {
        // 0b01111110: reserved 0, type 0b111111 (code minus 32 = 31 => order 32), wasted flag 0.
        var h = ParseSubframeHeader(0x7E);
        Equal(FlacSubframeKind.Lpc, h.Kind);
        Equal(32, h.Order);
    }

    [TestMethod]
    public void FlacSubframeHeader_ReservedType_Throws()
    {
        // 0b00000100: type 0b000010 (reserved).
        bool threw = false;
        try { ParseSubframeHeader(0x04); } catch (InvalidDataException) { threw = true; }
        True(threw, "Reserved type 0b000010 should throw.");
    }

    [TestMethod]
    public void FlacSubframeHeader_ReservedBit_NonZero_Throws()
    {
        // 0b10000000: reserved bit set (invalid).
        bool threw = false;
        try { ParseSubframeHeader(0x80); } catch (InvalidDataException) { threw = true; }
        True(threw, "Non-zero reserved bit should throw.");
    }

    [TestMethod]
    public void FlacSubframeHeader_Constant_OneWastedBit()
    {
        // Bits: reserved=0, type=000000 (CONSTANT), flag=1, unary terminator "1".
        // byte 0: 0_000000_1 = 0b00000001 = 0x01
        // byte 1: 1_0000000 = 0b10000000 = 0x80 (ReadUnary returns 0 => 1 wasted bit)
        var h = ParseSubframeHeader(0x01, 0x80);
        Equal(FlacSubframeKind.Constant, h.Kind);
        Equal(1, h.WastedBitsPerSample);
    }

    [TestMethod]
    public void FlacSubframeHeader_Constant_ThreeWastedBits()
    {
        // Bits: reserved=0, type=000000 (CONSTANT), flag=1, unary "001" (2 zeros + 1).
        // byte 0: 0_000000_1 = 0b00000001 = 0x01
        // byte 1: 001_00000 = 0b00100000 = 0x20 (ReadUnary returns 2 => 3 wasted bits)
        var h = ParseSubframeHeader(0x01, 0x20);
        Equal(FlacSubframeKind.Constant, h.Kind);
        Equal(3, h.WastedBitsPerSample);
    }

    [TestMethod]
    public void FlacSubframeHeader_LpcOrder8_WithWastedBits()
    {
        // Type 0b100111 = LPC order 8. Code value = 32 + 7 = 39. 0b100111 = 39.
        // Pattern: 0_100111_1_1_0...
        // byte 0: 0b01001111 = 0x4F (reserved + type 6 bits + wasted flag)
        // byte 1: 0b10000000 = 0x80 (unary "1" for wasted=1)
        var h = ParseSubframeHeader(0x4F, 0x80);
        Equal(FlacSubframeKind.Lpc, h.Kind);
        Equal(8, h.Order);
        Equal(1, h.WastedBitsPerSample);
    }
}
