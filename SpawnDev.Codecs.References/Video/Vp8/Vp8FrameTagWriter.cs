// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP8 frame tag writer - encoder-side counterpart of Vp8FrameTagParser.
// Emits the 3-byte uncompressed frame tag (and 7-byte key extension if
// applicable) per RFC 6386 sec 9.1 / 19.1.

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>VP8 frame tag writer.</summary>
public static class Vp8FrameTagWriter
{
    /// <summary>
    /// Emit the frame tag for <paramref name="tag"/> into a byte array.
    /// For key frames the result is 10 bytes (3 tag + 3 start code + 4 size).
    /// For inter frames it's 3 bytes.
    /// </summary>
    public static byte[] WriteTag(Vp8FrameTag tag)
    {
        ArgumentNullException.ThrowIfNull(tag);

        // 24-bit frame tag layout (LE byte order):
        //   bit 0       frame_type (0 = key, 1 = inter)
        //   bits 1..3   version
        //   bit 4       show_frame
        //   bits 5..23  first_partition_size
        if ((uint)(int)tag.Version > 7)
            throw new ArgumentOutOfRangeException(nameof(tag), "Version must fit in 3 bits");
        if (tag.FirstPartitionSize < 0 || tag.FirstPartitionSize > 0x7FFFF)
            throw new ArgumentOutOfRangeException(nameof(tag), "FirstPartitionSize must fit in 19 bits");

        uint tagBits = 0;
        if (!tag.IsKeyFrame) tagBits |= 0x1u;
        tagBits |= ((uint)tag.Version & 0x7u) << 1;
        if (tag.ShowFrame) tagBits |= 0x10u;
        tagBits |= ((uint)tag.FirstPartitionSize & 0x7FFFFu) << 5;

        if (!tag.IsKeyFrame)
        {
            return new byte[]
            {
                (byte)(tagBits & 0xFF),
                (byte)((tagBits >> 8) & 0xFF),
                (byte)((tagBits >> 16) & 0xFF),
            };
        }

        if (tag.Width is null || tag.Height is null
            || tag.HorizontalScale is null || tag.VerticalScale is null)
            throw new ArgumentException("Key frame requires Width / Height / HorizontalScale / VerticalScale", nameof(tag));
        int w = tag.Width.Value;
        int h = tag.Height.Value;
        int hs = tag.HorizontalScale.Value;
        int vs = tag.VerticalScale.Value;
        if ((uint)w > 0x3FFF || (uint)h > 0x3FFF)
            throw new ArgumentOutOfRangeException(nameof(tag), "Width/Height must fit in 14 bits");
        if ((uint)hs > 0x3 || (uint)vs > 0x3)
            throw new ArgumentOutOfRangeException(nameof(tag), "HorizontalScale/VerticalScale must fit in 2 bits");

        int horizSizeCode = w | (hs << 14);
        int vertSizeCode = h | (vs << 14);

        return new byte[]
        {
            (byte)(tagBits & 0xFF),
            (byte)((tagBits >> 8) & 0xFF),
            (byte)((tagBits >> 16) & 0xFF),
            Vp8FrameTagParser.StartCode[0],
            Vp8FrameTagParser.StartCode[1],
            Vp8FrameTagParser.StartCode[2],
            (byte)(horizSizeCode & 0xFF),
            (byte)((horizSizeCode >> 8) & 0xFF),
            (byte)(vertSizeCode & 0xFF),
            (byte)((vertSizeCode >> 8) & 0xFF),
        };
    }
}
