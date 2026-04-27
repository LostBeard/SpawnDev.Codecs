// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP8 frame tag (3 bytes) + key-frame extension (7 bytes). RFC 6386 sec
// 9.1 / 19.1 + libvpx vp8/decoder/decodeframe.c.
//
// Layout (RFC 6386 sec 9.1, 24-bit little-endian frame_tag):
//   bit 0       frame_type    (0 = key, 1 = inter)
//   bits 1..3   version       (0..3 reconstruction-filter / loop-filter profile)
//   bit 4       show_frame    (0 = hidden, 1 = displayed)
//   bits 5..23  first_part_size (size of first encoded partition in bytes)
//
// Key frames immediately follow the 3-byte tag with a 7-byte sync block:
//   bytes 3..5   start_code (must be 0x9D 0x01 0x2A)
//   bytes 6..7   horiz_size_code (BE16): scale<<14 | width(14)
//   bytes 8..9   vert_size_code  (BE16): scale<<14 | height(14)

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>
/// VP8 reconstruction-filter / loop-filter profile per the version field.
/// RFC 6386 sec 9.1 table.
/// </summary>
public enum Vp8Version : byte
{
    /// <summary>Bicubic reconstruction, normal loop filter.</summary>
    Bicubic = 0,
    /// <summary>Bilinear reconstruction, simple loop filter.</summary>
    BilinearSimpleLoopFilter = 1,
    /// <summary>Bilinear reconstruction, no loop filter.</summary>
    BilinearNoLoopFilter = 2,
    /// <summary>No reconstruction, no loop filter.</summary>
    NoReconNoLoopFilter = 3,
}

/// <summary>VP8 frame tag (decoded 3-byte prefix + optional key-frame extension).</summary>
public sealed record Vp8FrameTag
{
    /// <summary>True if this is a key frame (intra-only).</summary>
    public required bool IsKeyFrame { get; init; }
    /// <summary>3-bit version (reconstruction + loop filter profile).</summary>
    public required Vp8Version Version { get; init; }
    /// <summary>True if the frame is for display; false for hidden alt-ref-style frames.</summary>
    public required bool ShowFrame { get; init; }
    /// <summary>Size of the first compressed partition in bytes (uncompressed header + mode info).</summary>
    public required int FirstPartitionSize { get; init; }
    /// <summary>Frame width in pixels (key frames only; null for interframes).</summary>
    public int? Width { get; init; }
    /// <summary>Frame height in pixels (key frames only; null for interframes).</summary>
    public int? Height { get; init; }
    /// <summary>Horizontal scale factor (key frames only; null for interframes).</summary>
    public int? HorizontalScale { get; init; }
    /// <summary>Vertical scale factor (key frames only; null for interframes).</summary>
    public int? VerticalScale { get; init; }
}

/// <summary>VP8 frame tag parser. RFC 6386 sec 9.1 / 19.1.</summary>
public static class Vp8FrameTagParser
{
    /// <summary>Required start-code bytes after the frame tag for key frames.</summary>
    public static readonly byte[] StartCode = new byte[] { 0x9D, 0x01, 0x2A };

    /// <summary>
    /// Parse the VP8 frame tag (and key-frame extension if present) from
    /// <paramref name="frame"/>. Throws on truncated data or invalid start
    /// code on a key frame.
    /// </summary>
    public static Vp8FrameTag Parse(ReadOnlySpan<byte> frame)
    {
        if (frame.Length < 3)
            throw new ArgumentException("VP8 frame tag requires at least 3 bytes", nameof(frame));

        // Read 24-bit little-endian frame tag.
        uint tag = (uint)frame[0] | ((uint)frame[1] << 8) | ((uint)frame[2] << 16);
        bool isKey = (tag & 0x1u) == 0;
        int version = (int)((tag >> 1) & 0x7u);
        bool showFrame = ((tag >> 4) & 0x1u) != 0;
        int firstPartSize = (int)((tag >> 5) & 0x7FFFFu);

        if (!isKey)
        {
            return new Vp8FrameTag
            {
                IsKeyFrame = false,
                Version = (Vp8Version)version,
                ShowFrame = showFrame,
                FirstPartitionSize = firstPartSize,
            };
        }

        if (frame.Length < 10)
            throw new ArgumentException("VP8 key frame requires at least 10 bytes (3 tag + 3 start code + 4 size)", nameof(frame));
        if (frame[3] != StartCode[0] || frame[4] != StartCode[1] || frame[5] != StartCode[2])
            throw new InvalidDataException(
                $"VP8 key-frame start code mismatch: got {frame[3]:X2} {frame[4]:X2} {frame[5]:X2}, expected 9D 01 2A");

        int horizSizeCode = frame[6] | (frame[7] << 8);
        int vertSizeCode = frame[8] | (frame[9] << 8);
        int width = horizSizeCode & 0x3FFF;
        int horizScale = (horizSizeCode >> 14) & 0x3;
        int height = vertSizeCode & 0x3FFF;
        int vertScale = (vertSizeCode >> 14) & 0x3;

        return new Vp8FrameTag
        {
            IsKeyFrame = true,
            Version = (Vp8Version)version,
            ShowFrame = showFrame,
            FirstPartitionSize = firstPartSize,
            Width = width,
            Height = height,
            HorizontalScale = horizScale,
            VerticalScale = vertScale,
        };
    }
}
