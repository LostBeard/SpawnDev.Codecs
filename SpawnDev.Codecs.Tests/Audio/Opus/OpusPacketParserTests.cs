using SpawnDev.Codecs.Audio.Opus;

namespace SpawnDev.Codecs.Tests.Audio.Opus;

/// <summary>
/// Tests for <see cref="OpusPacketParser"/>. Covers all four count-code paths from RFC 6716
/// section 3 (frame packing), error conditions, padding, and self-delimited framing.
/// </summary>
public class OpusPacketParserTests
{
    // Helper: build a packet with a given TOC and raw frame bytes appended.
    private static byte[] BuildSimplePacket(byte toc, params byte[] frameBytes)
    {
        var packet = new byte[1 + frameBytes.Length];
        packet[0] = toc;
        frameBytes.CopyTo(packet, 1);
        return packet;
    }

    // -------- Argument / length edge cases --------

    [Fact]
    public void Parse_EmptyPacket_Throws()
    {
        Assert.Throws<ArgumentException>(() => OpusPacketParser.Parse(ReadOnlyMemory<byte>.Empty));
    }

    [Fact]
    public void TryParse_EmptyPacket_ReturnsInvalid()
    {
        bool ok = OpusPacketParser.TryParse(ReadOnlyMemory<byte>.Empty, false, out _, out var err);
        Assert.False(ok);
        Assert.Equal(OpusPacketError.InvalidPacket, err);
    }

    // -------- Count code 0: single frame --------

    [Fact]
    public void Parse_CountCode0_SingleFrame()
    {
        // TOC config=0 (SILK NB 10ms mono), count=0 (1 frame)
        byte toc = 0x00;
        byte[] payload = { 0x11, 0x22, 0x33, 0x44, 0x55 };
        byte[] packet = BuildSimplePacket(toc, payload);

        var parsed = OpusPacketParser.Parse(packet);
        Assert.Equal(1, parsed.FrameCount);
        Assert.Equal(5, parsed.Frames[0].Length);
        Assert.Equal(payload, parsed.Frames[0].ToArray());
        Assert.Equal(OpusMode.Silk, parsed.Toc.Mode);
        Assert.Equal(OpusBandwidth.Narrowband, parsed.Toc.Bandwidth);
        Assert.Equal(1, parsed.PayloadOffset);
        Assert.Equal(6, parsed.PacketLength);
        Assert.True(parsed.Padding.IsEmpty);
    }

    // -------- Count code 1: two equal CBR frames --------

    [Fact]
    public void Parse_CountCode1_TwoCbrFrames()
    {
        // TOC config=1 SILK NB 20ms, count=1 (2 CBR)
        byte toc = 0x09; // (1 << 3) | 0x01
        byte[] payload = { 0xAA, 0xBB, 0xCC, 0xDD }; // 4 bytes = 2 frames of 2 bytes each
        byte[] packet = BuildSimplePacket(toc, payload);

        var parsed = OpusPacketParser.Parse(packet);
        Assert.Equal(2, parsed.FrameCount);
        Assert.Equal(new byte[] { 0xAA, 0xBB }, parsed.Frames[0].ToArray());
        Assert.Equal(new byte[] { 0xCC, 0xDD }, parsed.Frames[1].ToArray());
    }

    [Fact]
    public void Parse_CountCode1_OddPayload_Throws()
    {
        // count=1 requires even payload length for 2 equal frames
        byte toc = 0x09;
        byte[] packet = BuildSimplePacket(toc, 0xAA, 0xBB, 0xCC);
        Assert.Throws<ArgumentException>(() => OpusPacketParser.Parse(packet));
    }

    // -------- Count code 2: two VBR frames with explicit first-frame size --------

    [Fact]
    public void Parse_CountCode2_TwoVbrFrames_ShortSize()
    {
        // TOC config=1 SILK NB 20ms, count=2 (2 VBR)
        // First size byte = 3 (single-byte size, value < 252)
        // First frame: 3 bytes {0xAA,0xBB,0xCC}
        // Second frame: remaining {0xDD,0xEE}
        byte toc = 0x0A; // (1 << 3) | 0x02
        byte[] packet = { toc, 3, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE };

        var parsed = OpusPacketParser.Parse(packet);
        Assert.Equal(2, parsed.FrameCount);
        Assert.Equal(3, parsed.Frames[0].Length);
        Assert.Equal(2, parsed.Frames[1].Length);
        Assert.Equal(new byte[] { 0xAA, 0xBB, 0xCC }, parsed.Frames[0].ToArray());
        Assert.Equal(new byte[] { 0xDD, 0xEE }, parsed.Frames[1].ToArray());
    }

    [Fact]
    public void Parse_CountCode2_TwoVbrFrames_LongSize()
    {
        // Two-byte size encoding when value >= 252.
        // Let's encode size = 260: b0 = 252 + (260 & 3) = 252, b1 = (260 - 252) >> 2 = 2
        // size = 4 * 2 + 252 = 260
        byte toc = 0x0A;
        byte[] firstFrame = new byte[260];
        for (int i = 0; i < 260; i++) firstFrame[i] = (byte)(i & 0xFF);
        byte[] secondFrame = new byte[10];
        for (int i = 0; i < 10; i++) secondFrame[i] = (byte)(0xF0 + i);

        var packet = new byte[1 + 2 + 260 + 10];
        packet[0] = toc;
        packet[1] = 252;
        packet[2] = 2;
        firstFrame.CopyTo(packet, 3);
        secondFrame.CopyTo(packet, 3 + 260);

        var parsed = OpusPacketParser.Parse(packet);
        Assert.Equal(2, parsed.FrameCount);
        Assert.Equal(260, parsed.Frames[0].Length);
        Assert.Equal(10, parsed.Frames[1].Length);
        Assert.Equal(firstFrame[0], parsed.Frames[0].Span[0]);
        Assert.Equal(firstFrame[259], parsed.Frames[0].Span[259]);
        Assert.Equal(secondFrame[0], parsed.Frames[1].Span[0]);
    }

    [Fact]
    public void Parse_CountCode2_SizeBeyondBuffer_Throws()
    {
        byte toc = 0x0A;
        byte[] packet = { toc, 100, 0x01, 0x02 }; // says first frame = 100 bytes but only 2 available
        Assert.Throws<ArgumentException>(() => OpusPacketParser.Parse(packet));
    }

    // -------- Count code 3: multiple frames --------

    [Fact]
    public void Parse_CountCode3_CbrMultiple()
    {
        // TOC config=0 SILK NB 10ms mono, count=3 (arbitrary)
        // Count byte: ch = 4 (4 frames), VBR bit 7 = 0 means CBR, padding bit 6 = 0
        // Remaining must be divisible by count.
        byte toc = 0x03;
        byte ch = 0x04;
        byte[] payload = { 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88 }; // 8 bytes / 4 frames = 2 bytes each
        var packet = new byte[1 + 1 + payload.Length];
        packet[0] = toc;
        packet[1] = ch;
        payload.CopyTo(packet, 2);

        var parsed = OpusPacketParser.Parse(packet);
        Assert.Equal(4, parsed.FrameCount);
        for (int i = 0; i < 4; i++)
        {
            Assert.Equal(2, parsed.Frames[i].Length);
        }
        Assert.Equal(new byte[] { 0x11, 0x22 }, parsed.Frames[0].ToArray());
        Assert.Equal(new byte[] { 0x77, 0x88 }, parsed.Frames[3].ToArray());
    }

    [Fact]
    public void Parse_CountCode3_VbrMultiple()
    {
        // TOC config=0 SILK NB 10ms mono, count=3
        // ch = 0x83: bit 7 set (VBR), count = 3
        // Then 2 size bytes for first 2 frames; last frame size is implicit.
        byte toc = 0x03;
        byte ch = 0x83;
        byte size0 = 3;
        byte size1 = 2;
        byte[] frame0 = { 0xAA, 0xAA, 0xAA };
        byte[] frame1 = { 0xBB, 0xBB };
        byte[] frame2 = { 0xCC, 0xCC, 0xCC, 0xCC };

        var packet = new byte[1 + 1 + 2 + frame0.Length + frame1.Length + frame2.Length];
        int idx = 0;
        packet[idx++] = toc;
        packet[idx++] = ch;
        packet[idx++] = size0;
        packet[idx++] = size1;
        frame0.CopyTo(packet, idx); idx += frame0.Length;
        frame1.CopyTo(packet, idx); idx += frame1.Length;
        frame2.CopyTo(packet, idx); idx += frame2.Length;

        var parsed = OpusPacketParser.Parse(packet);
        Assert.Equal(3, parsed.FrameCount);
        Assert.Equal(frame0, parsed.Frames[0].ToArray());
        Assert.Equal(frame1, parsed.Frames[1].ToArray());
        Assert.Equal(frame2, parsed.Frames[2].ToArray());
    }

    [Fact]
    public void Parse_CountCode3_WithPadding()
    {
        // VBR with padding: ch = 0xC3 (VBR + padding bit 6 + count=3)
        byte toc = 0x03;
        byte ch = 0xC3;
        byte padBytes = 5; // single-byte padding value (< 255)
        byte size0 = 2;
        byte size1 = 2;
        byte[] frame0 = { 0x11, 0x22 };
        byte[] frame1 = { 0x33, 0x44 };
        byte[] frame2 = { 0x55, 0x66, 0x77 };
        byte[] padding = { 0xFF, 0xFF, 0xFF, 0xFF, 0xFF };

        var packet = new byte[1 + 1 + 1 + 2 + 2 + 2 + 3 + 5];
        int idx = 0;
        packet[idx++] = toc;
        packet[idx++] = ch;
        packet[idx++] = padBytes;
        packet[idx++] = size0;
        packet[idx++] = size1;
        frame0.CopyTo(packet, idx); idx += frame0.Length;
        frame1.CopyTo(packet, idx); idx += frame1.Length;
        frame2.CopyTo(packet, idx); idx += frame2.Length;
        padding.CopyTo(packet, idx);

        var parsed = OpusPacketParser.Parse(packet);
        Assert.Equal(3, parsed.FrameCount);
        Assert.Equal(frame2, parsed.Frames[2].ToArray());
        Assert.Equal(5, parsed.Padding.Length);
    }

    [Fact]
    public void Parse_CountCode3_ZeroFrames_Throws()
    {
        byte toc = 0x03;
        byte ch = 0x00;
        byte[] packet = { toc, ch };
        Assert.Throws<ArgumentException>(() => OpusPacketParser.Parse(packet));
    }

    [Fact]
    public void Parse_CountCode3_TooManyFrames_Throws()
    {
        // framesize for config 3 SILK NB 60ms = 2880 samples at 48k
        // 2880 * N > 5760 for N > 2, so count > 2 triggers invalid
        byte toc = 0x1B; // config 3 SILK NB 60ms
        byte ch = 0x05; // count = 5
        byte[] packet = { toc, ch, 0x00, 0x00, 0x00, 0x00, 0x00 };
        Assert.Throws<ArgumentException>(() => OpusPacketParser.Parse(packet));
    }

    // -------- Frame slicing: frames reference the original buffer --------

    [Fact]
    public void Parse_Frames_ReferenceOriginalBuffer_NoCopy()
    {
        byte[] packet = BuildSimplePacket(0x00, 0x11, 0x22, 0x33);
        var data = packet.AsMemory();
        var parsed = OpusPacketParser.Parse(data);

        // Mutating the original buffer should be reflected in the frame slice.
        packet[1] = 0xFF;
        Assert.Equal(0xFF, parsed.Frames[0].Span[0]);
    }

    // -------- Packet-level sample counting --------

    [Fact]
    public void GetTotalSamples_SingleFrame20ms_Matches()
    {
        // config 1 = SILK NB 20ms = 960 samples at 48k, 1 frame
        byte[] packet = BuildSimplePacket(0x08, 0x00, 0x00);
        var parsed = OpusPacketParser.Parse(packet);
        Assert.Equal(960, parsed.GetTotalSamples(48_000));
    }

    [Fact]
    public void GetTotalSamples_MultipleFrames_Matches()
    {
        // config 1 SILK NB 20ms (960 samples), count=1 CBR = 2 frames
        byte[] packet = BuildSimplePacket(0x09, 0x00, 0x00, 0x00, 0x00);
        var parsed = OpusPacketParser.Parse(packet);
        Assert.Equal(2, parsed.FrameCount);
        Assert.Equal(1920, parsed.GetTotalSamples(48_000));
    }

    // -------- Self-delimited framing --------

    [Fact]
    public void Parse_SelfDelimited_SingleFrame_AppendsSize()
    {
        // Self-delimited: ALL packets carry an explicit size for the LAST frame.
        // count=0 = 1 frame. Size byte then frame data.
        byte toc = 0x00;
        byte size = 4;
        byte[] frame = { 0x11, 0x22, 0x33, 0x44 };

        var packet = new byte[1 + 1 + frame.Length];
        packet[0] = toc;
        packet[1] = size;
        frame.CopyTo(packet, 2);

        var parsed = OpusPacketParser.Parse(packet, selfDelimited: true);
        Assert.Equal(1, parsed.FrameCount);
        Assert.Equal(frame, parsed.Frames[0].ToArray());
    }
}
