// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Opus-in-Ogg comment header per RFC 7845 Section 5.2. Binary layout is
// identical to the Vorbis comment header body (vendor string + user comment
// list), preceded by the "OpusTags" magic and omitting the framing flag.

using System.Text;

namespace SpawnDev.Codecs.Audio.Opus;

/// <summary>Parsed Opus-in-Ogg comment header ("OpusTags").</summary>
public sealed record OpusTags
{
    /// <summary>Vendor string set by the encoder (e.g. "libopus 1.3.1").</summary>
    public required string Vendor { get; init; }

    /// <summary>"TAG=value" user comments (UTF-8).</summary>
    public required IReadOnlyList<string> UserComments { get; init; }
}

/// <summary>Parses the "OpusTags" comment header.</summary>
public static class OpusTagsParser
{
    private static readonly byte[] Magic = { (byte)'O', (byte)'p', (byte)'u', (byte)'s', (byte)'T', (byte)'a', (byte)'g', (byte)'s' };

    /// <summary>Parse the OpusTags packet bytes.</summary>
    public static OpusTags Parse(ReadOnlySpan<byte> packet)
    {
        if (packet.Length < 16)
            throw new InvalidDataException($"OpusTags packet too short: {packet.Length}.");
        for (int i = 0; i < 8; i++)
        {
            if (packet[i] != Magic[i])
                throw new InvalidDataException($"OpusTags magic mismatch at byte {i}: 0x{packet[i]:X2}.");
        }
        int pos = 8;
        uint vendorLen = ReadUInt32Le(packet.Slice(pos, 4));
        pos += 4;
        if (packet.Length < pos + vendorLen)
            throw new InvalidDataException("OpusTags vendor truncated.");
        string vendor = Encoding.UTF8.GetString(packet.Slice(pos, (int)vendorLen));
        pos += (int)vendorLen;
        if (packet.Length < pos + 4)
            throw new InvalidDataException("OpusTags comment count truncated.");
        uint commentCount = ReadUInt32Le(packet.Slice(pos, 4));
        pos += 4;
        var comments = new List<string>((int)Math.Min(commentCount, 1024));
        for (uint i = 0; i < commentCount; i++)
        {
            if (packet.Length < pos + 4)
                throw new InvalidDataException($"OpusTags comment #{i} length truncated.");
            uint cmtLen = ReadUInt32Le(packet.Slice(pos, 4));
            pos += 4;
            if (packet.Length < pos + cmtLen)
                throw new InvalidDataException($"OpusTags comment #{i} body truncated.");
            comments.Add(Encoding.UTF8.GetString(packet.Slice(pos, (int)cmtLen)));
            pos += (int)cmtLen;
        }
        return new OpusTags { Vendor = vendor, UserComments = comments };
    }

    private static uint ReadUInt32Le(ReadOnlySpan<byte> s)
    {
        uint v = 0;
        for (int i = 0; i < 4; i++) v |= (uint)s[i] << (8 * i);
        return v;
    }
}
