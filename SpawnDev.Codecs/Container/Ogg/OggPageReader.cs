// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Parse Ogg pages from a byte span. Each page is validated against its
// stored CRC-32 (the CRC field itself is zeroed during computation).

namespace SpawnDev.Codecs.Container.Ogg;

/// <summary>
/// Parses Ogg pages. Supports both single-page parsing at a given offset
/// (<see cref="ParseAt"/>) and iterative parsing across a multi-page buffer
/// (<see cref="EnumeratePages"/>).
/// </summary>
public static class OggPageReader
{
    /// <summary>
    /// Parse the Ogg page starting at byte offset 0 of <paramref name="data"/>.
    /// Validates the 4-byte "OggS" capture pattern, the version byte, the
    /// segment table, and the CRC-32 over the page contents.
    /// </summary>
    public static OggPage ParseAt(ReadOnlySpan<byte> data)
    {
        if (data.Length < OggConstants.FixedHeaderLength)
            throw new InvalidDataException("Ogg page header truncated (<27 bytes).");
        for (int i = 0; i < 4; i++)
            if (data[i] != OggConstants.CapturePattern[i])
                throw new InvalidDataException(
                    $"Ogg capture pattern mismatch at offset {i}: expected '{(char)OggConstants.CapturePattern[i]}' (0x{OggConstants.CapturePattern[i]:X2}), got 0x{data[i]:X2}.");
        byte version = data[4];
        if (version != OggConstants.Version)
            throw new InvalidDataException($"Ogg page version must be 0, got {version}.");
        byte headerType = data[5];
        long granulePos = ReadInt64Le(data.Slice(6, 8));
        uint serial = ReadUInt32Le(data.Slice(14, 4));
        uint pageSeq = ReadUInt32Le(data.Slice(18, 4));
        uint storedCrc = ReadUInt32Le(data.Slice(22, 4));
        int segmentCount = data[26];

        if (data.Length < OggConstants.FixedHeaderLength + segmentCount)
            throw new InvalidDataException("Ogg page segment table truncated.");
        var segLens = new byte[segmentCount];
        int payloadLength = 0;
        for (int i = 0; i < segmentCount; i++)
        {
            segLens[i] = data[OggConstants.FixedHeaderLength + i];
            payloadLength += segLens[i];
        }

        int headerBytes = OggConstants.FixedHeaderLength + segmentCount;
        if (data.Length < headerBytes + payloadLength)
            throw new InvalidDataException(
                $"Ogg page payload truncated: expected {payloadLength} bytes, have {data.Length - headerBytes}.");

        var payload = data.Slice(headerBytes, payloadLength).ToArray();

        // CRC-32: compute over the whole page with the 4-byte CRC field zeroed.
        int totalBytes = headerBytes + payloadLength;
        byte[] forCrc = data.Slice(0, totalBytes).ToArray();
        forCrc[22] = 0;
        forCrc[23] = 0;
        forCrc[24] = 0;
        forCrc[25] = 0;
        uint actualCrc = OggCrc32.Compute(forCrc);
        if (actualCrc != storedCrc)
            throw new InvalidDataException(
                $"Ogg page CRC-32 mismatch: stored 0x{storedCrc:X8}, computed 0x{actualCrc:X8}.");

        return new OggPage
        {
            HeaderType = headerType,
            GranulePosition = granulePos,
            BitstreamSerial = serial,
            PageSequence = pageSeq,
            Crc = storedCrc,
            SegmentLengths = segLens,
            Payload = payload,
            TotalPageBytes = totalBytes,
        };
    }

    /// <summary>Iterate over every Ogg page in <paramref name="data"/>, in order.</summary>
    public static IEnumerable<OggPage> EnumeratePages(byte[] data)
    {
        int offset = 0;
        while (offset < data.Length)
        {
            var page = ParseAt(data.AsSpan(offset));
            yield return page;
            offset += page.TotalPageBytes;
        }
    }

    private static long ReadInt64Le(ReadOnlySpan<byte> s)
    {
        long v = 0;
        for (int i = 0; i < 8; i++) v |= (long)s[i] << (8 * i);
        return v;
    }

    private static uint ReadUInt32Le(ReadOnlySpan<byte> s)
    {
        uint v = 0;
        for (int i = 0; i < 4; i++) v |= (uint)s[i] << (8 * i);
        return v;
    }
}
