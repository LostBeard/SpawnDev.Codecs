// Tests for Vp8FrameTagParser - VP8 uncompressed prefix (3-byte frame
// tag + 7-byte key-frame extension). RFC 6386 sec 9.1 / 19.1.

using SpawnDev.Codecs.Video.Vp8;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp8FrameTag_KeyFrame320x240_ParsesAllFields()
    {
        // Build a synthetic key frame tag:
        //   bit 0 = 0 (key)
        //   bits 1..3 = 0 (Bicubic version)
        //   bit 4 = 1 (show_frame)
        //   bits 5..23 = 100 (first_partition_size)
        // tag value: (100 << 5) | (1 << 4) | (0 << 1) | 0 = 0xC10
        // Little-endian bytes: 0x10 0x0C 0x00
        var frame = new byte[]
        {
            0x10, 0x0C, 0x00,             // frame tag
            0x9D, 0x01, 0x2A,             // start code
            0x40, 0x01,                   // horiz size: width=320 (0x140), scale=0
            0xF0, 0x00,                   // vert size: height=240 (0xF0), scale=0
        };
        var tag = Vp8FrameTagParser.Parse(frame);

        True(tag.IsKeyFrame, "should be key frame");
        Equal(Vp8Version.Bicubic, tag.Version);
        True(tag.ShowFrame, "show_frame");
        Equal(100, tag.FirstPartitionSize);
        Equal(320, tag.Width!.Value);
        Equal(240, tag.Height!.Value);
        Equal(0, tag.HorizontalScale!.Value);
        Equal(0, tag.VerticalScale!.Value);
    }

    [TestMethod]
    public void Vp8FrameTag_InterFrameNoSync_ParsesPrefixOnly()
    {
        // Inter frame: bit 0 = 1
        //   bits 1..3 = 1 (BilinearSimpleLoopFilter)
        //   bit 4 = 1 (show_frame)
        //   bits 5..23 = 50 (first_partition_size)
        // tag value: (50 << 5) | (1 << 4) | (1 << 1) | 1 = 0x653
        // bytes: 0x53 0x06 0x00
        var frame = new byte[] { 0x53, 0x06, 0x00 };
        var tag = Vp8FrameTagParser.Parse(frame);

        False(tag.IsKeyFrame, "should be inter frame");
        Equal(Vp8Version.BilinearSimpleLoopFilter, tag.Version);
        True(tag.ShowFrame, "show_frame");
        Equal(50, tag.FirstPartitionSize);
        True(tag.Width is null, "Width should be null for inter frame");
        True(tag.Height is null, "Height should be null for inter frame");
    }

    [TestMethod]
    public void Vp8FrameTag_KeyFrameInvalidStartCode_Throws()
    {
        var frame = new byte[]
        {
            0x10, 0x0C, 0x00,
            0x00, 0x00, 0x00,             // wrong start code
            0x40, 0x01, 0xF0, 0x00,
        };
        Throws<InvalidDataException>(() => Vp8FrameTagParser.Parse(frame));
    }

    [TestMethod]
    public void Vp8FrameTag_TruncatedFrame_Throws()
    {
        Throws<ArgumentException>(() => Vp8FrameTagParser.Parse(new byte[2]));
    }

    [TestMethod]
    public void Vp8FrameTag_TruncatedKeyFrame_Throws()
    {
        // Key frame tag (3 bytes) but missing the 7-byte key extension.
        var frame = new byte[] { 0x10, 0x0C, 0x00, 0x9D, 0x01 };
        Throws<ArgumentException>(() => Vp8FrameTagParser.Parse(frame));
    }

    [TestMethod]
    public void Vp8FrameTag_KeyFrameMaxDimensions_ParsesCorrectly()
    {
        // 14-bit width + 14-bit height max = 0x3FFF = 16383 each.
        var frame = new byte[]
        {
            0x00, 0x00, 0x00,
            0x9D, 0x01, 0x2A,
            0xFF, 0x3F,                   // width = 0x3FFF (14 bits), scale = 0
            0xFF, 0x3F,                   // height = 0x3FFF (14 bits), scale = 0
        };
        var tag = Vp8FrameTagParser.Parse(frame);

        Equal(0x3FFF, tag.Width!.Value);
        Equal(0x3FFF, tag.Height!.Value);
    }
}
