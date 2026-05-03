// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Vorbis comment header (packet 1). Defined in Vorbis I spec section 5. Also
// used verbatim by Opus and FLAC for metadata storage ("VorbisComment").

using System.Text;

namespace SpawnDev.Codecs.Audio.Vorbis;

/// <summary>
/// Parsed Vorbis comment header: vendor string plus a list of "FIELD=value"
/// user comments. UTF-8 encoded throughout.
/// </summary>
public sealed record VorbisCommentHeader
{
    /// <summary>Vendor string set by the encoder, e.g. "Xiph.Org libVorbis I 20150105".</summary>
    public required string Vendor { get; init; }

    /// <summary>User comments, each in "TAG=value" form (TAG may not contain '=' or control chars).</summary>
    public required IReadOnlyList<string> UserComments { get; init; }
}

/// <summary>Parses the Vorbis comment header packet (also used by Opus/FLAC metadata).</summary>
public static class VorbisCommentHeaderParser
{
    /// <summary>
    /// Parse the comment header bytes. The packet layout is:
    /// [1 byte type=0x03][6 bytes "vorbis"][uint32 vendor length][vendor bytes]
    /// [uint32 comment count][repeated: uint32 comment length + comment bytes]
    /// [1 byte framing flag (LSB must be 1)].
    /// </summary>
    public static VorbisCommentHeader Parse(ReadOnlySpan<byte> packet)
    {
        int pos = 0;
        if (packet.Length < 8)
            throw new InvalidDataException("Vorbis comment header is too short.");
        VorbisIdentificationHeaderParser.ValidatePacketType(packet[pos++], expected: 0x03);
        VorbisIdentificationHeaderParser.ValidateMagic(packet.Slice(pos, 6));
        pos += 6;

        uint vendorLen = ReadUInt32Le(packet.Slice(pos, 4));
        pos += 4;
        if (packet.Length < pos + vendorLen)
            throw new InvalidDataException("Vendor string truncated.");
        string vendor = Encoding.UTF8.GetString(packet.Slice(pos, (int)vendorLen));
        pos += (int)vendorLen;

        if (packet.Length < pos + 4)
            throw new InvalidDataException("Comment count truncated.");
        uint commentCount = ReadUInt32Le(packet.Slice(pos, 4));
        pos += 4;

        var comments = new List<string>((int)Math.Min(commentCount, 1024));
        for (uint i = 0; i < commentCount; i++)
        {
            if (packet.Length < pos + 4)
                throw new InvalidDataException($"Comment #{i} length truncated.");
            uint cmtLen = ReadUInt32Le(packet.Slice(pos, 4));
            pos += 4;
            if (packet.Length < pos + cmtLen)
                throw new InvalidDataException($"Comment #{i} body truncated.");
            comments.Add(Encoding.UTF8.GetString(packet.Slice(pos, (int)cmtLen)));
            pos += (int)cmtLen;
        }

        // Framing flag: LSB must be 1. Some streams (Opus, FLAC) omit the byte.
        if (packet.Length > pos)
        {
            byte framing = packet[pos++];
            if ((framing & 1) == 0)
                throw new InvalidDataException("Vorbis comment framing flag not set.");
        }

        return new VorbisCommentHeader { Vendor = vendor, UserComments = comments };
    }

    private static uint ReadUInt32Le(ReadOnlySpan<byte> s)
    {
        uint v = 0;
        for (int i = 0; i < 4; i++) v |= (uint)s[i] << (8 * i);
        return v;
    }
}
