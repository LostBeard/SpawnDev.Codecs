using SpawnDev.Codecs.Audio.Vorbis;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for <see cref="VorbisBitReader"/>. Vorbis uses LSB-first bit packing
/// opposite to FLAC. Byte 0x81 decodes as 1 bit of 1 then 6 bits of 0 then a
/// final bit of 1 - NOT as 1 0 0 0 0 0 0 1 from the MSB.
/// </summary>
public abstract partial class CodecsTestBase
{
    private static uint VorbisRead(ReadOnlySpan<byte> data, int bits)
    {
        var r = new VorbisBitReader(data);
        return r.ReadBits(bits);
    }

    [TestMethod]
    public void VorbisBitReader_SingleBit_LSBFirst()
    {
        // byte 0x01: bit 0 = 1, bits 1..7 = 0.
        byte[] data = { 0x01 };
        var r = new VorbisBitReader(data);
        Equal(1u, r.ReadBit());
        Equal(0u, r.ReadBit());
        Equal(0u, r.ReadBit());
        Equal(0u, r.ReadBit());
        Equal(0u, r.ReadBit());
        Equal(0u, r.ReadBit());
        Equal(0u, r.ReadBit());
        Equal(0u, r.ReadBit());
        True(r.IsEnd);
    }

    [TestMethod]
    public void VorbisBitReader_BytePrimer_LsbFirstReverseOfFlac()
    {
        // byte 0x81 = 1000_0001. LSB-first reads: 1, 0, 0, 0, 0, 0, 0, 1.
        byte[] data = { 0x81 };
        var r = new VorbisBitReader(data);
        Equal(1u, r.ReadBit()); // bit 0
        Equal(0u, r.ReadBit()); // bit 1
        Equal(0u, r.ReadBit()); // bit 2
        Equal(0u, r.ReadBit()); // bit 3
        Equal(0u, r.ReadBit()); // bit 4
        Equal(0u, r.ReadBit()); // bit 5
        Equal(0u, r.ReadBit()); // bit 6
        Equal(1u, r.ReadBit()); // bit 7
    }

    [TestMethod]
    public void VorbisBitReader_FullByteAligned_ReturnsRawByte()
    {
        byte[] data = { 0xA5, 0x3C };
        var r = new VorbisBitReader(data);
        Equal(0xA5u, r.ReadBits(8));
        Equal(0x3Cu, r.ReadBits(8));
    }

    [TestMethod]
    public void VorbisBitReader_CrossByteBoundary_PacksBitsLsbFirst()
    {
        // Two bytes 0x0F 0xF0 = 0000_1111 1111_0000.
        // Read 4 bits then 8 bits: first 4 = 0x0F (low nibble of byte 0),
        // next 8 = (high nibble of byte 0) | (low nibble of byte 1 << 4) = 0 | (0 << 4)? wait
        //   high nibble of byte 0 = 0x0 (top 4 bits), low nibble of byte 1 = 0x0 (bottom 4 bits of 0xF0).
        //   Actually 0xF0 = 1111_0000, low 4 bits = 0x0, high 4 bits = 0xF.
        // So next 8 bits = (0 << 4) | 0 = 0x00.
        byte[] data = { 0x0F, 0xF0 };
        var r = new VorbisBitReader(data);
        Equal(0x0Fu, r.ReadBits(4)); // low 4 bits of byte 0 = 1111
        Equal(0x00u, r.ReadBits(8)); // high 4 bits of byte 0 (0000) + low 4 bits of byte 1 (0000)
        Equal(0x0Fu, r.ReadBits(4)); // high 4 bits of byte 1 (1111)
    }

    [TestMethod]
    public void VorbisBitReader_24BitIntegerLe()
    {
        // Vorbis codebook sync word is 0x564342 packed LSB-first over 24 bits.
        // LSB-first 24 bits of 0x564342 is stored as bytes 0x42 0x43 0x56.
        byte[] data = { 0x42, 0x43, 0x56 };
        var r = new VorbisBitReader(data);
        Equal(0x564342u, r.ReadBits(24));
    }

    [TestMethod]
    public void VorbisBitReader_ReadBits_Zero_NoAdvance()
    {
        byte[] data = { 0xFF };
        var r = new VorbisBitReader(data);
        Equal(0u, r.ReadBits(0));
        Equal(0, r.Position);
        Equal(8, r.BitsRemaining);
    }

    [TestMethod]
    public void VorbisBitReader_Signed_NegativeInterpretation()
    {
        // 5-bit value 0b11111 = -1 two's complement.
        byte[] data = { 0x1F }; // bottom 5 bits = 11111
        var r = new VorbisBitReader(data);
        Equal(-1, r.ReadBitsSigned(5));
    }

    [TestMethod]
    public void VorbisBitReader_Signed_PositiveInterpretation()
    {
        // 5-bit value 0b01010 = +10. Byte 0x0A.
        byte[] data = { 0x0A };
        var r = new VorbisBitReader(data);
        Equal(10, r.ReadBitsSigned(5));
    }

    [TestMethod]
    public void VorbisBitReader_AlignToByte_DiscardsPartialByte()
    {
        byte[] data = { 0x55, 0xAA };
        var r = new VorbisBitReader(data);
        _ = r.ReadBits(3);
        r.AlignToByte();
        Equal(8, r.Position);
        Equal(0xAAu, r.ReadBits(8));
    }

    [TestMethod]
    public void VorbisBitReader_InsufficientBits_Throws()
    {
        byte[] data = { 0xFF };
        bool threw = false;
        try
        {
            var r = new VorbisBitReader(data);
            _ = r.ReadBits(16);
        }
        catch (InvalidOperationException) { threw = true; }
        True(threw);
    }

    [TestMethod]
    public void VorbisBitReader_32BitMax_ReadsWithoutOverflow()
    {
        // 32-bit value 0xFFFFFFFF packed LSB-first = bytes FF FF FF FF.
        byte[] data = { 0xFF, 0xFF, 0xFF, 0xFF };
        Equal(0xFFFFFFFFu, VorbisRead(data, 32));
    }
}
