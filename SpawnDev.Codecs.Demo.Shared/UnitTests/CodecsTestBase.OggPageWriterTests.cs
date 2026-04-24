using SpawnDev.Codecs.Audio.Opus;
using SpawnDev.Codecs.Container.Ogg;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for <see cref="OggPageWriter"/>. Each test writes a page, re-parses
/// it through <see cref="OggPageReader"/>, and asserts the round-trip matches.
/// This validates segment-table generation, CRC-32 computation, and the
/// end-to-end byte layout without depending on any external Ogg tooling.
/// </summary>
public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void OggWriter_SmallPacket_RoundtripsThroughReader()
    {
        byte[] payload = { 0xAA, 0xBB, 0xCC, 0xDD, 0xEE };
        byte[] pageBytes = OggPageWriter.WriteSinglePacketPage(
            headerType: OggConstants.HeaderTypeBeginningOfStream,
            granulePosition: 0,
            bitstreamSerial: 0xBEEFCAFE,
            pageSequence: 0,
            packet: payload);
        var page = OggPageReader.ParseAt(pageBytes);
        Equal(OggConstants.HeaderTypeBeginningOfStream, page.HeaderType);
        Equal(0xBEEFCAFEu, page.BitstreamSerial);
        Equal(0u, page.PageSequence);
        EqualBytes(payload, page.Payload);
    }

    [TestMethod]
    public void OggWriter_ExactMultipleOf255_NeedsTerminator()
    {
        // 255-byte packet: segmentize produces [255, 0].
        byte[] payload = new byte[255];
        for (int i = 0; i < 255; i++) payload[i] = (byte)i;
        byte[] pageBytes = OggPageWriter.WriteSinglePacketPage(0, 0, 1, 0, payload);
        var page = OggPageReader.ParseAt(pageBytes);
        Equal(2, page.SegmentLengths.Length);
        Equal((byte)255, page.SegmentLengths[0]);
        Equal((byte)0, page.SegmentLengths[1]);
        EqualBytes(payload, page.Payload);
    }

    [TestMethod]
    public void OggWriter_600BytePacket_SegmentTableHasThreeEntries()
    {
        // 600 = 255 + 255 + 90 -> 3 segments.
        byte[] payload = new byte[600];
        for (int i = 0; i < 600; i++) payload[i] = (byte)(i & 0xFF);
        byte[] pageBytes = OggPageWriter.WriteSinglePacketPage(0, 0, 7, 0, payload);
        var page = OggPageReader.ParseAt(pageBytes);
        Equal(3, page.SegmentLengths.Length);
        Equal((byte)255, page.SegmentLengths[0]);
        Equal((byte)255, page.SegmentLengths[1]);
        Equal((byte)90, page.SegmentLengths[2]);
        EqualBytes(payload, page.Payload);
    }

    [TestMethod]
    public void OggWriter_ZeroLengthPacket_SingleZeroSegment()
    {
        byte[] pageBytes = OggPageWriter.WriteSinglePacketPage(0, 0, 1, 0, ReadOnlySpan<byte>.Empty);
        var page = OggPageReader.ParseAt(pageBytes);
        Equal(1, page.SegmentLengths.Length);
        Equal((byte)0, page.SegmentLengths[0]);
        Equal(0, page.Payload.Length);
    }

    [TestMethod]
    public void OggWriter_WriteStream_PacketsAreAssembledBack()
    {
        var packets = new[]
        {
            new OggOutgoingPacket { Data = new byte[] { 0xA1, 0xA2, 0xA3 }, GranulePosition = 0 },
            new OggOutgoingPacket { Data = new byte[] { 0xB1, 0xB2 }, GranulePosition = 960 },
            new OggOutgoingPacket { Data = new byte[] { 0xC1 }, GranulePosition = 1920 },
        };
        byte[] streamBytes = OggPageWriter.WriteStream(bitstreamSerial: 42, packets);
        var pages = OggPageReader.EnumeratePages(streamBytes).ToArray();
        Equal(3, pages.Length);
        True(pages[0].IsBeginningOfStream);
        False(pages[1].IsBeginningOfStream);
        False(pages[1].IsEndOfStream);
        True(pages[2].IsEndOfStream);
        Equal(0u, pages[0].PageSequence);
        Equal(1u, pages[1].PageSequence);
        Equal(2u, pages[2].PageSequence);
        Equal(960L, pages[1].GranulePosition);
        Equal(1920L, pages[2].GranulePosition);

        var assembled = OggPacketReader.AssemblePackets(pages).ToArray();
        Equal(3, assembled.Length);
        EqualBytes(packets[0].Data, assembled[0].Data);
        EqualBytes(packets[1].Data, assembled[1].Data);
        EqualBytes(packets[2].Data, assembled[2].Data);
    }

    [TestMethod]
    public void OggWriter_ReaderRejectsCorruptedWriterOutput()
    {
        byte[] payload = { 1, 2, 3, 4 };
        byte[] pageBytes = OggPageWriter.WriteSinglePacketPage(0, 0, 1, 0, payload);
        // Flip one byte in the payload without updating CRC - reader should detect.
        pageBytes[^1] ^= 0x01;
        bool threw = false;
        try { _ = OggPageReader.ParseAt(pageBytes); } catch (InvalidDataException) { threw = true; }
        True(threw, "Writer output with a corrupted payload byte should fail CRC check.");
    }

    [TestMethod]
    public void OggWriter_EmptyPacketList_Throws()
    {
        bool threw = false;
        try { _ = OggPageWriter.WriteStream(1, Array.Empty<OggOutgoingPacket>()); }
        catch (ArgumentException) { threw = true; }
        True(threw);
    }

    [TestMethod]
    public void OggWriter_RoundtripOpusStream_WritesDecodesTheSameWay()
    {
        // Build a minimal Opus-in-Ogg stream via the writer and ensure OpusOggDecoder accepts it.
        // Not all test signals will be SILK; guard with UnsupportedTestException.
        int frameLen = 960;
        var pcm = ReferenceOracle.GenerateSineWave(440, 48000, 1, frameLen * 2);
        var frame = new float[frameLen];
        var opusPackets = new List<byte[]>();
        for (int f = 0; f < 2; f++)
        {
            Array.Copy(pcm, f * frameLen, frame, 0, frameLen);
            byte[] enc = ReferenceOracle.EncodeFrame(frame, 48000, 1, frameLen,
                Concentus.Enums.OpusApplication.OPUS_APPLICATION_VOIP);
            var toc = new OpusTocByte(enc[0]);
            if (toc.Mode != SpawnDev.Codecs.Audio.Opus.OpusMode.Silk)
                throw new UnsupportedTestException($"Concentus chose {toc.Mode}, need SILK.");
            opusPackets.Add(enc);
        }

        // Build OpusHead + OpusTags packets.
        byte[] head = BuildOpusHeadPacket(channels: 1, preSkip: 312, inputRate: 48000, gainQ78: 0, family: 0);
        byte[] tags = BuildOpusTagsPacket("SpawnDev.Codecs OggWriter test", Array.Empty<string>());

        var outgoing = new List<OggOutgoingPacket>
        {
            new OggOutgoingPacket { Data = head, GranulePosition = 0 },
            new OggOutgoingPacket { Data = tags, GranulePosition = 0 },
        };
        long running = 0;
        for (int i = 0; i < opusPackets.Count; i++)
        {
            running += frameLen;
            outgoing.Add(new OggOutgoingPacket { Data = opusPackets[i], GranulePosition = running });
        }
        byte[] streamBytes = OggPageWriter.WriteStream(bitstreamSerial: 0x1234_5678, outgoing);

        SpawnDev.Codecs.Audio.Opus.OpusOggDecodeResult decoded;
        try
        {
            decoded = SpawnDev.Codecs.Audio.Opus.OpusOggDecoder.DecodeAsync(streamBytes).Result;
        }
        catch (AggregateException ae) when (ae.InnerException is NotImplementedException)
        {
            throw new UnsupportedTestException($"OpusDecoder stub: {ae.InnerException.Message}");
        }
        Equal(1, decoded.Head.OutputChannels);
        Equal(312, decoded.Head.PreSkip);
        Equal("SpawnDev.Codecs OggWriter test", decoded.Tags.Vendor);
        Equal(2 * frameLen - 312, decoded.TotalSamplesPerChannel);
    }
}
