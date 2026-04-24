// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// 'ftyp' box parser per ISO/IEC 14496-12. The first box in every valid MP4
// file identifies the file's major brand and compatible brands. Tells
// downstream decoders whether to expect fragmented MP4, HEIF, AV1-in-MP4,
// etc.

namespace SpawnDev.Codecs.Container.Mp4;

/// <summary>Parsed 'ftyp' box.</summary>
public sealed record Mp4FileTypeBox
{
    /// <summary>4-byte FourCC major brand (e.g. "isom", "mp42", "avc1", "av01").</summary>
    public required string MajorBrand { get; init; }

    /// <summary>Minor version uint32 BE.</summary>
    public uint MinorVersion { get; init; }

    /// <summary>Compatible brands list (zero or more FourCC codes).</summary>
    public required IReadOnlyList<string> CompatibleBrands { get; init; }
}

/// <summary>Parses the 'ftyp' box payload.</summary>
public static class Mp4FileTypeBoxParser
{
    /// <summary>Parse a 'ftyp' <see cref="Mp4Box"/> against its backing buffer.</summary>
    public static Mp4FileTypeBox Parse(Mp4Box box, ReadOnlySpan<byte> data)
    {
        if (box.Type != "ftyp")
            throw new InvalidDataException($"Expected 'ftyp' box, got '{box.Type}'.");
        int payloadOffset = box.Offset + box.HeaderSize;
        int payloadLength = (int)(box.Size - box.HeaderSize);
        if (payloadLength < 8)
            throw new InvalidDataException($"'ftyp' payload too small: {payloadLength} bytes.");
        if (data.Length < box.Offset + box.Size)
            throw new InvalidDataException("'ftyp' extends past provided buffer.");

        var payload = data.Slice(payloadOffset, payloadLength);
        string majorBrand = System.Text.Encoding.ASCII.GetString(payload.Slice(0, 4));
        uint minorVersion = ReadUInt32Be(payload.Slice(4, 4));

        var compat = new List<string>();
        int pos = 8;
        while (pos + 4 <= payloadLength)
        {
            compat.Add(System.Text.Encoding.ASCII.GetString(payload.Slice(pos, 4)));
            pos += 4;
        }

        return new Mp4FileTypeBox
        {
            MajorBrand = majorBrand,
            MinorVersion = minorVersion,
            CompatibleBrands = compat,
        };
    }

    private static uint ReadUInt32Be(ReadOnlySpan<byte> s)
    {
        uint v = 0;
        for (int i = 0; i < 4; i++) v = (v << 8) | s[i];
        return v;
    }
}
