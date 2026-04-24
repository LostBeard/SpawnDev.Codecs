using SpawnDev.Codecs.Container.Ebml;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for <see cref="EbmlVint"/>. Covers the standard 1-8 byte widths and
/// both ID mode (marker preserved) and size mode (marker stripped + unknown-
/// size all-ones tail).
/// </summary>
public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void EbmlVint_OneByte_SizeReadsPayload()
    {
        // 0x82 = 10000010 -> width 1, stripped = 0000010 = 2.
        byte[] data = { 0x82 };
        ulong v = EbmlVint.ReadSize(data, 0, out int bytesRead);
        Equal(2UL, v);
        Equal(1, bytesRead);
    }

    [TestMethod]
    public void EbmlVint_OneByte_IdPreservesMarker()
    {
        // EBML root ID is 0x1A45DFA3 (4 bytes). Test a 1-byte ID: 0x80 = 10000000
        // -> value preserves marker = 0x80.
        byte[] data = { 0x80 };
        ulong v = EbmlVint.ReadId(data, 0, out int bytesRead);
        Equal(0x80UL, v);
        Equal(1, bytesRead);
    }

    [TestMethod]
    public void EbmlVint_FourByte_EbmlHeaderId()
    {
        // 0x1A45DFA3 is the Matroska/WebM root element ID.
        // First byte 0x1A = 00011010 -> width 4 (first set bit at position 4).
        byte[] data = { 0x1A, 0x45, 0xDF, 0xA3 };
        ulong v = EbmlVint.ReadId(data, 0, out int bytesRead);
        Equal(0x1A45DFA3UL, v);
        Equal(4, bytesRead);
    }

    [TestMethod]
    public void EbmlVint_TwoByte_Size()
    {
        // 0x40 0x7F = 01000000 01111111 -> width 2, stripped = 00000000 01111111 = 0x7F.
        byte[] data = { 0x40, 0x7F };
        ulong v = EbmlVint.ReadSize(data, 0, out int bytesRead);
        Equal(0x7FUL, v);
        Equal(2, bytesRead);
    }

    [TestMethod]
    public void EbmlVint_UnknownSize_AllOnesTail()
    {
        // 1-byte VINT with marker at bit 7 and all remaining bits set: 0xFF.
        // Stripped value = 0x7F, which equals the max 7-bit payload -> unknown size.
        byte[] data = { 0xFF };
        ulong v = EbmlVint.ReadSize(data, 0, out _);
        Equal(EbmlVint.UnknownSize, v);
    }

    [TestMethod]
    public void EbmlVint_EightByte_SizeReadsFull56BitPayload()
    {
        // First byte 0x01 = 00000001 -> width 8. Remaining 7 bytes = 0x01020304050607.
        // Stripped payload fills bits 0..55 from bytes 2..8.
        byte[] data = { 0x01, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07 };
        ulong v = EbmlVint.ReadSize(data, 0, out int bytesRead);
        Equal(0x01020304050607UL, v);
        Equal(8, bytesRead);
    }

    [TestMethod]
    public void EbmlVint_ReservedFirstByte_Throws()
    {
        // 0x00 = reserved (width >= 9, not supported by EBML).
        byte[] data = { 0x00, 0x00 };
        bool threw = false;
        try { _ = EbmlVint.ReadSize(data, 0, out _); }
        catch (InvalidDataException) { threw = true; }
        True(threw);
    }

    [TestMethod]
    public void EbmlVint_TruncatedBuffer_Throws()
    {
        // Width = 2 but only 1 byte available.
        byte[] data = { 0x40 };
        bool threw = false;
        try { _ = EbmlVint.ReadSize(data, 0, out _); }
        catch (InvalidDataException) { threw = true; }
        True(threw);
    }

    [TestMethod]
    public void EbmlVint_OffsetBeyondBuffer_Throws()
    {
        byte[] data = { 0x80 };
        bool threw = false;
        try { _ = EbmlVint.ReadSize(data, 5, out _); }
        catch (InvalidDataException) { threw = true; }
        True(threw);
    }
}
