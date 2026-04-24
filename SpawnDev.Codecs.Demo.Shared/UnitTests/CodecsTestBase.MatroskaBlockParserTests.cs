// Tests for MatroskaBlockParser. Hand-builds SimpleBlock / Block bodies
// with each of the four lacing types and verifies the frame byte output
// round-trips. These are unit tests for the on-wire parser, independent
// of any file fixture.

using SpawnDev.Codecs.Container.Matroska;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    /// <summary>
    /// Build a block body: 1-byte VINT track + int16 BE timestamp + flags + payload.
    /// </summary>
    private static byte[] BuildSimpleBlockNoLace(
        int trackNum, short relTs, bool keyframe, byte[] payload)
    {
        // 1-byte VINT for track (0x80 | n) - valid for 0..0x7E.
        if (trackNum < 0 || trackNum > 0x7E) throw new ArgumentOutOfRangeException(nameof(trackNum));
        byte flags = 0;
        if (keyframe) flags |= 0x80;
        // lacing bits stay 0.
        var body = new byte[1 + 2 + 1 + payload.Length];
        body[0] = (byte)(0x80 | trackNum);
        body[1] = (byte)(relTs >> 8);
        body[2] = (byte)(relTs & 0xFF);
        body[3] = flags;
        Array.Copy(payload, 0, body, 4, payload.Length);
        return body;
    }

    [TestMethod]
    public void BlockParser_NoLacing_YieldsSingleFrame_PreservesPayload()
    {
        var payload = new byte[] { 0xDE, 0xAD, 0xBE, 0xEF, 0x55 };
        var body = BuildSimpleBlockNoLace(trackNum: 1, relTs: 0, keyframe: true, payload: payload);
        var frames = MatroskaBlockParser.Parse(body, clusterTimestamp: 100, isSimpleBlock: true);
        Equal(1, frames.Count);
        Equal(1UL, frames[0].TrackNumber);
        Equal(100L, frames[0].Timestamp);
        True(frames[0].IsKeyframe, "keyframe bit must be set");
        Equal(0, frames[0].LaceIndex);
        True(payload.SequenceEqual(frames[0].Data), "payload bytes must round-trip");
    }

    [TestMethod]
    public void BlockParser_Timestamp_IsSignedInt16_AddedToCluster()
    {
        var body = BuildSimpleBlockNoLace(1, relTs: -5, keyframe: false, payload: new byte[] { 0xAA });
        var frames = MatroskaBlockParser.Parse(body, clusterTimestamp: 1000, isSimpleBlock: true);
        // int16 BE relative timestamp -5 + cluster 1000 = 995.
        Equal(995L, frames[0].Timestamp);
    }

    [TestMethod]
    public void BlockParser_Block_NotSimpleBlock_IgnoresKeyframeBit()
    {
        // Same bit pattern that sets keyframe on a SimpleBlock; isSimpleBlock=false
        // must clear it.
        var body = BuildSimpleBlockNoLace(1, 0, keyframe: true, new byte[] { 0x01 });
        var frames = MatroskaBlockParser.Parse(body, 0, isSimpleBlock: false);
        False(frames[0].IsKeyframe, "Block (non-simple) must not surface the keyframe bit");
    }

    [TestMethod]
    public void BlockParser_XiphLacing_ThreeFrames_RoundTrips()
    {
        // Xiph lacing flags: bits 1-2 = 01 -> 0x02.
        // frame_count - 1 = 2 -> 0x02 in the lacing-count byte.
        // Frame sizes: 300, 100, remainder. Size 300 Xiph-encoded: 0xFF 0x2D.
        // Size 100 encoded: 0x64.
        byte[] f0 = new byte[300];
        byte[] f1 = new byte[100];
        byte[] f2 = new byte[50];
        for (int i = 0; i < f0.Length; i++) f0[i] = 0x11;
        for (int i = 0; i < f1.Length; i++) f1[i] = 0x22;
        for (int i = 0; i < f2.Length; i++) f2[i] = 0x33;
        var body = new List<byte>
        {
            0x81,          // track 1 (VINT)
            0x00, 0x00,    // relative timestamp 0
            0x02,          // flags: lacing bits = 01 (Xiph)
            0x02,          // frame_count - 1 = 2
            0xFF, 0x2D,    // size 300 Xiph-encoded
            0x64,          // size 100 Xiph-encoded
        };
        body.AddRange(f0);
        body.AddRange(f1);
        body.AddRange(f2);
        var frames = MatroskaBlockParser.Parse(body.ToArray(), 0, isSimpleBlock: true);
        Equal(3, frames.Count);
        True(f0.SequenceEqual(frames[0].Data));
        True(f1.SequenceEqual(frames[1].Data));
        True(f2.SequenceEqual(frames[2].Data));
        Equal(0, frames[0].LaceIndex);
        Equal(1, frames[1].LaceIndex);
        Equal(2, frames[2].LaceIndex);
    }

    [TestMethod]
    public void BlockParser_FixedLacing_FourEqualFrames()
    {
        // 4 frames of 32 bytes each = 128 bytes total payload.
        byte[] payload = new byte[128];
        for (int i = 0; i < 128; i++) payload[i] = (byte)(i / 32); // stripe by quarter
        var body = new List<byte>
        {
            0x81,         // track 1
            0x00, 0x00,   // timestamp 0
            0x04,         // flags: lacing bits = 10 (fixed)
            0x03,         // frame_count - 1 = 3
        };
        body.AddRange(payload);
        var frames = MatroskaBlockParser.Parse(body.ToArray(), 0, isSimpleBlock: true);
        Equal(4, frames.Count);
        for (int i = 0; i < 4; i++)
        {
            Equal(32, frames[i].Data.Length);
            Equal((byte)i, frames[i].Data[0]); // each stripe starts with its index
        }
    }

    [TestMethod]
    public void BlockParser_EbmlLacing_DecreasingSizes_RoundTrips()
    {
        // EBML lacing: first size = 100 as unsigned VINT. Second = 80, encoded
        // as signed VINT delta (-20 from 100). Third = remainder.
        byte[] f0 = new byte[100];
        byte[] f1 = new byte[80];
        byte[] f2 = new byte[40];
        for (int i = 0; i < f0.Length; i++) f0[i] = 0xAA;
        for (int i = 0; i < f1.Length; i++) f1[i] = 0xBB;
        for (int i = 0; i < f2.Length; i++) f2[i] = 0xCC;
        // 100 as unsigned VINT (2-byte VINT since 100 fits in 7 bits actually,
        // but use 2-byte for clarity): 0x40 0x64. Or 1-byte: 0x80 | 100 = 0xE4.
        // 100 < 0x7F so fits in 1-byte VINT: marker 0x80 | 100 = 0xE4.
        // Delta -20: 2-byte signed VINT, bias = 2^13 - 1 = 8191.
        //   raw = -20 + 8191 = 8171 = 0x1FEB. 2-byte VINT marker 0x40 = 0x40 | 0x1F = 0x5F, second byte 0xEB.
        var body = new List<byte>
        {
            0x81,       // track 1
            0x00, 0x00, // timestamp 0
            0x06,       // flags: lacing bits = 11 (EBML)
            0x02,       // frame_count - 1 = 2
            0xE4,       // frame-0 size = 100 (1-byte unsigned VINT)
            0x5F, 0xEB, // frame-1 size = 100 + (-20) = 80 (2-byte signed VINT)
        };
        body.AddRange(f0);
        body.AddRange(f1);
        body.AddRange(f2);
        var frames = MatroskaBlockParser.Parse(body.ToArray(), 0, isSimpleBlock: true);
        Equal(3, frames.Count);
        Equal(100, frames[0].Data.Length);
        Equal(80, frames[1].Data.Length);
        Equal(40, frames[2].Data.Length);
        True(f0.SequenceEqual(frames[0].Data));
        True(f1.SequenceEqual(frames[1].Data));
        True(f2.SequenceEqual(frames[2].Data));
    }

    [TestMethod]
    public void BlockParser_TruncatedBody_Throws()
    {
        // Just enough for the VINT + 1 byte of timestamp - truncated before
        // completing the 2-byte timestamp read.
        byte[] body = new byte[] { 0x81, 0x00 };
        bool threw = false;
        try { _ = MatroskaBlockParser.Parse(body, 0, isSimpleBlock: true); }
        catch (InvalidDataException) { threw = true; }
        True(threw, "truncated body must throw InvalidDataException");
    }
}
