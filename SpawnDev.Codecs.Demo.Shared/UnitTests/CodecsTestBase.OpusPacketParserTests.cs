using SpawnDev.Codecs.Audio.Opus;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for <see cref="OpusPacketParser"/>. Covers all four RFC 6716 count-code paths,
/// padding, self-delimited framing, zero-copy frame slicing, and error conditions.
/// </summary>
public abstract partial class CodecsTestBase
{
    private static byte[] BuildSimplePacket(byte toc, params byte[] frameBytes)
    {
        var packet = new byte[1 + frameBytes.Length];
        packet[0] = toc;
        frameBytes.CopyTo(packet, 1);
        return packet;
    }

    [TestMethod]
    public void Parser_EmptyPacket_Throws()
    {
        Throws<ArgumentException>(() => OpusPacketParser.Parse(ReadOnlyMemory<byte>.Empty));
    }

    [TestMethod]
    public void Parser_TryParse_EmptyPacket_ReturnsInvalid()
    {
        bool ok = OpusPacketParser.TryParse(ReadOnlyMemory<byte>.Empty, false, out _, out var err);
        False(ok);
        Equal(OpusPacketError.InvalidPacket, err);
    }

    [TestMethod]
    public void Parser_CountCode0_SingleFrame()
    {
        byte toc = 0x00;
        byte[] payload = { 0x11, 0x22, 0x33, 0x44, 0x55 };
        byte[] packet = BuildSimplePacket(toc, payload);

        var parsed = OpusPacketParser.Parse(packet);
        Equal(1, parsed.FrameCount);
        Equal(5, parsed.Frames[0].Length);
        EqualBytes(payload, parsed.Frames[0].ToArray());
        Equal(OpusMode.Silk, parsed.Toc.Mode);
        Equal(OpusBandwidth.Narrowband, parsed.Toc.Bandwidth);
        Equal(1, parsed.PayloadOffset);
        Equal(6, parsed.PacketLength);
        True(parsed.Padding.IsEmpty);
    }

    [TestMethod]
    public void Parser_CountCode1_TwoCbrFrames()
    {
        byte toc = 0x09; // config 1 SILK NB 20ms, count=1
        byte[] payload = { 0xAA, 0xBB, 0xCC, 0xDD };
        byte[] packet = BuildSimplePacket(toc, payload);

        var parsed = OpusPacketParser.Parse(packet);
        Equal(2, parsed.FrameCount);
        EqualBytes(new byte[] { 0xAA, 0xBB }, parsed.Frames[0].ToArray());
        EqualBytes(new byte[] { 0xCC, 0xDD }, parsed.Frames[1].ToArray());
    }

    [TestMethod]
    public void Parser_CountCode1_OddPayload_Throws()
    {
        byte[] packet = BuildSimplePacket(0x09, 0xAA, 0xBB, 0xCC);
        Throws<ArgumentException>(() => OpusPacketParser.Parse(packet));
    }

    [TestMethod]
    public void Parser_CountCode2_TwoVbrFrames_ShortSize()
    {
        byte[] packet = { 0x0A, 3, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE };
        var parsed = OpusPacketParser.Parse(packet);
        Equal(2, parsed.FrameCount);
        Equal(3, parsed.Frames[0].Length);
        Equal(2, parsed.Frames[1].Length);
        EqualBytes(new byte[] { 0xAA, 0xBB, 0xCC }, parsed.Frames[0].ToArray());
        EqualBytes(new byte[] { 0xDD, 0xEE }, parsed.Frames[1].ToArray());
    }

    [TestMethod]
    public void Parser_CountCode2_TwoVbrFrames_LongSize()
    {
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
        Equal(2, parsed.FrameCount);
        Equal(260, parsed.Frames[0].Length);
        Equal(10, parsed.Frames[1].Length);
        Equal(firstFrame[0], parsed.Frames[0].Span[0]);
        Equal(firstFrame[259], parsed.Frames[0].Span[259]);
        Equal(secondFrame[0], parsed.Frames[1].Span[0]);
    }

    [TestMethod]
    public void Parser_CountCode2_SizeBeyondBuffer_Throws()
    {
        byte[] packet = { 0x0A, 100, 0x01, 0x02 };
        Throws<ArgumentException>(() => OpusPacketParser.Parse(packet));
    }

    [TestMethod]
    public void Parser_CountCode3_CbrMultiple()
    {
        byte toc = 0x03;
        byte ch = 0x04; // 4 frames, VBR=0 (CBR)
        byte[] payload = { 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88 };
        var packet = new byte[1 + 1 + payload.Length];
        packet[0] = toc;
        packet[1] = ch;
        payload.CopyTo(packet, 2);

        var parsed = OpusPacketParser.Parse(packet);
        Equal(4, parsed.FrameCount);
        for (int i = 0; i < 4; i++) Equal(2, parsed.Frames[i].Length, $"frame {i}");
        EqualBytes(new byte[] { 0x11, 0x22 }, parsed.Frames[0].ToArray());
        EqualBytes(new byte[] { 0x77, 0x88 }, parsed.Frames[3].ToArray());
    }

    [TestMethod]
    public void Parser_CountCode3_VbrMultiple()
    {
        byte toc = 0x03;
        byte ch = 0x83; // 3 frames, VBR=1
        byte[] frame0 = { 0xAA, 0xAA, 0xAA };
        byte[] frame1 = { 0xBB, 0xBB };
        byte[] frame2 = { 0xCC, 0xCC, 0xCC, 0xCC };

        var packet = new byte[1 + 1 + 2 + frame0.Length + frame1.Length + frame2.Length];
        int idx = 0;
        packet[idx++] = toc;
        packet[idx++] = ch;
        packet[idx++] = 3; // size0
        packet[idx++] = 2; // size1
        frame0.CopyTo(packet, idx); idx += frame0.Length;
        frame1.CopyTo(packet, idx); idx += frame1.Length;
        frame2.CopyTo(packet, idx);

        var parsed = OpusPacketParser.Parse(packet);
        Equal(3, parsed.FrameCount);
        EqualBytes(frame0, parsed.Frames[0].ToArray());
        EqualBytes(frame1, parsed.Frames[1].ToArray());
        EqualBytes(frame2, parsed.Frames[2].ToArray());
    }

    [TestMethod]
    public void Parser_CountCode3_WithPadding()
    {
        byte toc = 0x03;
        byte ch = 0xC3; // VBR + padding + count=3
        byte padBytes = 5;
        byte[] frame0 = { 0x11, 0x22 };
        byte[] frame1 = { 0x33, 0x44 };
        byte[] frame2 = { 0x55, 0x66, 0x77 };
        byte[] padding = { 0xFF, 0xFF, 0xFF, 0xFF, 0xFF };

        var packet = new byte[1 + 1 + 1 + 2 + 2 + 2 + 3 + 5];
        int idx = 0;
        packet[idx++] = toc;
        packet[idx++] = ch;
        packet[idx++] = padBytes;
        packet[idx++] = 2; // size0
        packet[idx++] = 2; // size1
        frame0.CopyTo(packet, idx); idx += frame0.Length;
        frame1.CopyTo(packet, idx); idx += frame1.Length;
        frame2.CopyTo(packet, idx); idx += frame2.Length;
        padding.CopyTo(packet, idx);

        var parsed = OpusPacketParser.Parse(packet);
        Equal(3, parsed.FrameCount);
        EqualBytes(frame2, parsed.Frames[2].ToArray());
        Equal(5, parsed.Padding.Length);
    }

    [TestMethod]
    public void Parser_CountCode3_ZeroFrames_Throws()
    {
        byte[] packet = { 0x03, 0x00 };
        Throws<ArgumentException>(() => OpusPacketParser.Parse(packet));
    }

    [TestMethod]
    public void Parser_CountCode3_TooManyFrames_Throws()
    {
        // config 3 SILK NB 60ms = 2880 samples @ 48k. 5 * 2880 = 14400 > 5760.
        byte[] packet = { 0x1B, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00 };
        Throws<ArgumentException>(() => OpusPacketParser.Parse(packet));
    }

    [TestMethod]
    public void Parser_Frames_ReferenceOriginalBuffer_NoCopy()
    {
        byte[] packet = BuildSimplePacket(0x00, 0x11, 0x22, 0x33);
        var data = packet.AsMemory();
        var parsed = OpusPacketParser.Parse(data);

        packet[1] = 0xFF;
        Equal((byte)0xFF, parsed.Frames[0].Span[0]);
    }

    [TestMethod]
    public void Parser_GetTotalSamples_SingleFrame20ms()
    {
        byte[] packet = BuildSimplePacket(0x08, 0x00, 0x00);
        var parsed = OpusPacketParser.Parse(packet);
        Equal(960, parsed.GetTotalSamples(48_000));
    }

    [TestMethod]
    public void Parser_GetTotalSamples_TwoFrames20ms()
    {
        byte[] packet = BuildSimplePacket(0x09, 0x00, 0x00, 0x00, 0x00);
        var parsed = OpusPacketParser.Parse(packet);
        Equal(2, parsed.FrameCount);
        Equal(1920, parsed.GetTotalSamples(48_000));
    }

    [TestMethod]
    public void Parser_SelfDelimited_SingleFrame()
    {
        byte[] frame = { 0x11, 0x22, 0x33, 0x44 };
        var packet = new byte[1 + 1 + frame.Length];
        packet[0] = 0x00;
        packet[1] = 4;
        frame.CopyTo(packet, 2);

        var parsed = OpusPacketParser.Parse(packet, selfDelimited: true);
        Equal(1, parsed.FrameCount);
        EqualBytes(frame, parsed.Frames[0].ToArray());
    }
}
