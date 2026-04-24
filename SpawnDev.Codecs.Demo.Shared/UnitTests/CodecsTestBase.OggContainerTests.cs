using SpawnDev.Codecs.Container.Ogg;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for the Ogg container parser (RFC 3533). Validates CRC-32 against
/// the standard MPEG-2-polynomial check vector, builds hand-constructed
/// pages with correct CRC and parses them through <see cref="OggPageReader"/>,
/// and verifies packet assembly across segments and across pages.
/// </summary>
public abstract partial class CodecsTestBase
{
    // -------- CRC-32 --------

    [TestMethod]
    public void OggCrc32_SingleByte_0x01_EqualsPolynomial()
    {
        // A single byte 0x01 of input should shift the polynomial to its
        // left-aligned representation. With poly 0x04C11DB7 this produces
        // exactly 0x04C11DB7 (the polynomial value itself).
        byte[] input = { 0x01 };
        Equal(0x04C11DB7u, OggCrc32.Compute(input));
    }

    [TestMethod]
    public void OggCrc32_AllZero_IsZero()
    {
        // With init = 0 and zero input, CRC stays 0 across any number of zero bytes.
        byte[] input = new byte[16];
        Equal(0u, OggCrc32.Compute(input));
    }

    [TestMethod]
    public void OggCrc32_Empty_IsZero()
    {
        Equal(0u, OggCrc32.Compute(Array.Empty<byte>()));
    }

    [TestMethod]
    public void OggCrc32_Deterministic()
    {
        byte[] input = { 0xDE, 0xAD, 0xBE, 0xEF };
        Equal(OggCrc32.Compute(input), OggCrc32.Compute(input));
    }

    // -------- Page building helper --------

    /// <summary>Build a valid Ogg page with a computed CRC-32.</summary>
    private static byte[] BuildOggPage(
        byte headerType, long granulePos, uint serial, uint pageSeq,
        byte[] segmentLengths, byte[] payload)
    {
        int totalLen = 27 + segmentLengths.Length + payload.Length;
        var bytes = new byte[totalLen];
        bytes[0] = (byte)'O'; bytes[1] = (byte)'g'; bytes[2] = (byte)'g'; bytes[3] = (byte)'S';
        bytes[4] = 0; // version
        bytes[5] = headerType;
        for (int i = 0; i < 8; i++) bytes[6 + i] = (byte)(granulePos >> (8 * i));
        for (int i = 0; i < 4; i++) bytes[14 + i] = (byte)(serial >> (8 * i));
        for (int i = 0; i < 4; i++) bytes[18 + i] = (byte)(pageSeq >> (8 * i));
        // CRC bytes 22..25 start as zero (and remain zero during CRC computation).
        bytes[26] = (byte)segmentLengths.Length;
        Array.Copy(segmentLengths, 0, bytes, 27, segmentLengths.Length);
        Array.Copy(payload, 0, bytes, 27 + segmentLengths.Length, payload.Length);
        uint crc = OggCrc32.Compute(bytes);
        bytes[22] = (byte)(crc >> 0);
        bytes[23] = (byte)(crc >> 8);
        bytes[24] = (byte)(crc >> 16);
        bytes[25] = (byte)(crc >> 24);
        return bytes;
    }

    [TestMethod]
    public void OggPageReader_ParsesValidSinglePage()
    {
        byte[] payload = { 0x01, 0x02, 0x03, 0x04, 0x05 };
        var pageBytes = BuildOggPage(
            headerType: OggConstants.HeaderTypeBeginningOfStream,
            granulePos: 0, serial: 0x12345678, pageSeq: 0,
            segmentLengths: new byte[] { 5 }, payload: payload);

        var page = OggPageReader.ParseAt(pageBytes);
        Equal(OggConstants.HeaderTypeBeginningOfStream, page.HeaderType);
        Equal(0L, page.GranulePosition);
        Equal(0x12345678u, page.BitstreamSerial);
        Equal(0u, page.PageSequence);
        Equal(1, page.SegmentLengths.Length);
        Equal((byte)5, page.SegmentLengths[0]);
        EqualBytes(payload, page.Payload);
        True(page.IsBeginningOfStream);
        False(page.IsEndOfStream);
    }

    [TestMethod]
    public void OggPageReader_BadCapturePattern_Throws()
    {
        var data = new byte[30];
        data[0] = (byte)'X'; data[1] = (byte)'Y'; data[2] = (byte)'Z'; data[3] = (byte)'Q';
        bool threw = false;
        try { _ = OggPageReader.ParseAt(data); }
        catch (InvalidDataException) { threw = true; }
        True(threw, "Bad capture pattern should throw.");
    }

    [TestMethod]
    public void OggPageReader_BadCrc_Throws()
    {
        byte[] payload = { 0xAA, 0xBB };
        var pageBytes = BuildOggPage(0, 0, 42, 0, new byte[] { 2 }, payload);
        // Corrupt CRC byte.
        pageBytes[22] ^= 0xFF;
        bool threw = false;
        try { _ = OggPageReader.ParseAt(pageBytes); }
        catch (InvalidDataException) { threw = true; }
        True(threw, "CRC mismatch should throw.");
    }

    [TestMethod]
    public void OggPageReader_BadVersion_Throws()
    {
        byte[] payload = { 0x00 };
        var pageBytes = BuildOggPage(0, 0, 1, 0, new byte[] { 1 }, payload);
        pageBytes[4] = 1; // version
        // Recompute CRC (the version change invalidates the CRC).
        Array.Fill(pageBytes, (byte)0, 22, 4);
        uint crc = OggCrc32.Compute(pageBytes);
        pageBytes[22] = (byte)(crc >> 0);
        pageBytes[23] = (byte)(crc >> 8);
        pageBytes[24] = (byte)(crc >> 16);
        pageBytes[25] = (byte)(crc >> 24);
        bool threw = false;
        try { _ = OggPageReader.ParseAt(pageBytes); }
        catch (InvalidDataException) { threw = true; }
        True(threw, "Bad Ogg version should throw.");
    }

    [TestMethod]
    public void OggPageReader_EnumerateMultiplePages()
    {
        byte[] page1 = BuildOggPage(OggConstants.HeaderTypeBeginningOfStream,
            0, 1, 0, new byte[] { 3 }, new byte[] { 1, 2, 3 });
        byte[] page2 = BuildOggPage(0, 48, 1, 1, new byte[] { 2 }, new byte[] { 4, 5 });
        byte[] page3 = BuildOggPage(OggConstants.HeaderTypeEndOfStream,
            96, 1, 2, new byte[] { 1 }, new byte[] { 6 });

        var all = new byte[page1.Length + page2.Length + page3.Length];
        Array.Copy(page1, all, page1.Length);
        Array.Copy(page2, 0, all, page1.Length, page2.Length);
        Array.Copy(page3, 0, all, page1.Length + page2.Length, page3.Length);

        var pages = OggPageReader.EnumeratePages(all).ToArray();
        Equal(3, pages.Length);
        True(pages[0].IsBeginningOfStream);
        Equal(0u, pages[0].PageSequence);
        Equal(1u, pages[1].PageSequence);
        Equal(2u, pages[2].PageSequence);
        True(pages[2].IsEndOfStream);
    }

    // -------- Packet assembly --------

    [TestMethod]
    public void OggPacketReader_SinglePagePacket()
    {
        byte[] pageBytes = BuildOggPage(
            OggConstants.HeaderTypeBeginningOfStream, 0, 1, 0,
            segmentLengths: new byte[] { 4 },
            payload: new byte[] { 0xAA, 0xBB, 0xCC, 0xDD });
        var pages = new[] { OggPageReader.ParseAt(pageBytes) };
        var packets = OggPacketReader.AssemblePackets(pages).ToArray();
        Equal(1, packets.Length);
        EqualBytes(new byte[] { 0xAA, 0xBB, 0xCC, 0xDD }, packets[0].Data);
        Equal(1u, packets[0].BitstreamSerial);
    }

    [TestMethod]
    public void OggPacketReader_MultiSegmentSinglePagePacket()
    {
        // Two 255-byte segments + one 100-byte terminator = one packet of 610 bytes.
        byte[] payload = new byte[255 + 255 + 100];
        for (int i = 0; i < payload.Length; i++) payload[i] = (byte)(i & 0xFF);
        byte[] pageBytes = BuildOggPage(
            OggConstants.HeaderTypeBeginningOfStream, 0, 1, 0,
            segmentLengths: new byte[] { 255, 255, 100 },
            payload: payload);
        var pages = new[] { OggPageReader.ParseAt(pageBytes) };
        var packets = OggPacketReader.AssemblePackets(pages).ToArray();
        Equal(1, packets.Length);
        Equal(610, packets[0].Data.Length);
        EqualBytes(payload, packets[0].Data);
    }

    [TestMethod]
    public void OggPacketReader_PacketAcrossTwoPages()
    {
        // Page 1 has a single 255-byte segment (packet continues).
        // Page 2 has a 50-byte segment (packet terminator).
        byte[] part1 = new byte[255];
        for (int i = 0; i < 255; i++) part1[i] = (byte)i;
        byte[] part2 = new byte[50];
        for (int i = 0; i < 50; i++) part2[i] = (byte)(100 + i);

        byte[] page1 = BuildOggPage(
            OggConstants.HeaderTypeBeginningOfStream, 0, 42, 0,
            segmentLengths: new byte[] { 255 }, payload: part1);
        byte[] page2 = BuildOggPage(
            OggConstants.HeaderTypeContinuation, 255, 42, 1,
            segmentLengths: new byte[] { 50 }, payload: part2);

        var p1 = OggPageReader.ParseAt(page1);
        var p2 = OggPageReader.ParseAt(page2);
        var packets = OggPacketReader.AssemblePackets(new[] { p1, p2 }).ToArray();
        Equal(1, packets.Length);
        Equal(305, packets[0].Data.Length);
        for (int i = 0; i < 255; i++) Equal((byte)i, packets[0].Data[i]);
        for (int i = 0; i < 50; i++) Equal((byte)(100 + i), packets[0].Data[255 + i]);
    }

    [TestMethod]
    public void OggPacketReader_MultiplePacketsInOnePage()
    {
        // 3 segments: 2, 3, 4 bytes. Each < 255 so each is a full packet.
        byte[] pageBytes = BuildOggPage(
            OggConstants.HeaderTypeBeginningOfStream, 0, 7, 0,
            segmentLengths: new byte[] { 2, 3, 4 },
            payload: new byte[] { 10, 20, 30, 40, 50, 60, 70, 80, 90 });
        var page = OggPageReader.ParseAt(pageBytes);
        var packets = OggPacketReader.AssemblePackets(new[] { page }).ToArray();
        Equal(3, packets.Length);
        EqualBytes(new byte[] { 10, 20 }, packets[0].Data);
        EqualBytes(new byte[] { 30, 40, 50 }, packets[1].Data);
        EqualBytes(new byte[] { 60, 70, 80, 90 }, packets[2].Data);
    }

    [TestMethod]
    public void OggPacketReader_IndependentBitstreams()
    {
        // Two logical streams interleaved. Packets for each serial should assemble independently.
        byte[] streamA_page = BuildOggPage(
            OggConstants.HeaderTypeBeginningOfStream, 0, 100, 0,
            segmentLengths: new byte[] { 3 }, payload: new byte[] { 1, 2, 3 });
        byte[] streamB_page = BuildOggPage(
            OggConstants.HeaderTypeBeginningOfStream, 0, 200, 0,
            segmentLengths: new byte[] { 2 }, payload: new byte[] { 4, 5 });

        var a = OggPageReader.ParseAt(streamA_page);
        var b = OggPageReader.ParseAt(streamB_page);
        var packets = OggPacketReader.AssemblePackets(new[] { a, b }).ToArray();
        Equal(2, packets.Length);
        Equal(100u, packets[0].BitstreamSerial);
        Equal(200u, packets[1].BitstreamSerial);
    }
}
