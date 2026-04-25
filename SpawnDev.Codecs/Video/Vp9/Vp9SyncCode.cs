// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 frame sync code constants. Mirror of libvpx
// vp9/common/vp9_blockd.h VP9_SYNC_CODE_0..2.
//
// Three f(8) bytes that follow the profile / show_existing_frame
// header bits in keyframes and intra-only frames. The decoder
// verifies all three bytes against the expected magic to confirm
// the frame is well-formed before parsing the rest of the
// uncompressed header (VP9 spec sec 6.2.1).

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 frame sync code byte triple (0x49, 0x83, 0x42).</summary>
public static class Vp9SyncCode
{
    /// <summary>libvpx <c>VP9_SYNC_CODE_0</c> = 0x49 ('I').</summary>
    public const byte Byte0 = 0x49;

    /// <summary>libvpx <c>VP9_SYNC_CODE_1</c> = 0x83.</summary>
    public const byte Byte1 = 0x83;

    /// <summary>libvpx <c>VP9_SYNC_CODE_2</c> = 0x42 ('B').</summary>
    public const byte Byte2 = 0x42;

    /// <summary>Length of the sync code in bytes.</summary>
    public const int Length = 3;

    /// <summary>
    /// Verify that the next 3 bytes of <paramref name="data"/> from
    /// <paramref name="offset"/> match the VP9 sync code.
    /// </summary>
    public static bool Matches(ReadOnlySpan<byte> data, int offset)
    {
        if (offset < 0)
            throw new ArgumentOutOfRangeException(nameof(offset), offset, "offset must be >= 0.");
        if (offset + Length > data.Length) return false;
        return data[offset] == Byte0
            && data[offset + 1] == Byte1
            && data[offset + 2] == Byte2;
    }

    /// <summary>
    /// Returns the sync code as a 3-byte array. Useful for tests +
    /// constructing test bitstreams.
    /// </summary>
    public static byte[] AsArray() => new byte[] { Byte0, Byte1, Byte2 };
}
