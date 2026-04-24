// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Writer that packs packets into Ogg pages per RFC 3533 Section 6. Handles
// splitting packets across multiple pages when a packet exceeds the 255-byte
// segment maximum (libogg's conventional page-size cap is 255 segments of up
// to 255 bytes = 65025 bytes per page, so most real-world packets fit in one
// page).

namespace SpawnDev.Codecs.Container.Ogg;

/// <summary>
/// Single packet to be written to one or more Ogg pages.
/// </summary>
public sealed record OggOutgoingPacket
{
    /// <summary>Packet bytes.</summary>
    public required byte[] Data { get; init; }

    /// <summary>
    /// Granule position reported by the terminating page for this packet.
    /// Codec-specific (e.g. total decoded samples produced up to and including this packet).
    /// Set to <c>-1</c> for "not yet known" (no granule update).
    /// </summary>
    public long GranulePosition { get; init; }
}

/// <summary>
/// Writes a sequence of packets into a sequence of Ogg pages under a single
/// logical bitstream serial. Each packet can either terminate on its own page
/// or pack with neighbors; this simple writer emits one page per packet, which
/// is the common layout for Opus-in-Ogg (and trivially correct).
/// </summary>
public static class OggPageWriter
{
    /// <summary>
    /// Segmentize a packet into the byte values stored in an Ogg page's segment
    /// table. 255-byte runs signal "packet continues"; a final &lt;255 byte entry
    /// terminates. A packet of exactly N*255 bytes needs one extra 0-length
    /// terminator segment.
    /// </summary>
    internal static byte[] Segmentize(int packetLength)
    {
        if (packetLength == 0) return new byte[] { 0 };
        var list = new List<byte>();
        while (packetLength >= 255)
        {
            list.Add(255);
            packetLength -= 255;
        }
        list.Add((byte)packetLength);
        return list.ToArray();
    }

    /// <summary>
    /// Write one page containing a single packet. Computes the CRC-32 and
    /// returns the full on-wire byte sequence. The packet may require multiple
    /// Ogg pages if it exceeds the 255-segment maximum; this overload assumes
    /// the packet fits in one page (which holds up to 65025 bytes).
    /// </summary>
    public static byte[] WriteSinglePacketPage(
        byte headerType, long granulePosition, uint bitstreamSerial, uint pageSequence,
        ReadOnlySpan<byte> packet)
    {
        byte[] segTable = Segmentize(packet.Length);
        if (segTable.Length > 255)
            throw new ArgumentException("Packet too large for a single page (> 65025 bytes).", nameof(packet));
        int pageSize = OggConstants.FixedHeaderLength + segTable.Length + packet.Length;
        var bytes = new byte[pageSize];
        // Capture + version + header type.
        bytes[0] = (byte)'O'; bytes[1] = (byte)'g'; bytes[2] = (byte)'g'; bytes[3] = (byte)'S';
        bytes[4] = OggConstants.Version;
        bytes[5] = headerType;
        // Granule position (int64 LE).
        for (int i = 0; i < 8; i++) bytes[6 + i] = (byte)(granulePosition >> (8 * i));
        // Bitstream serial (uint32 LE).
        for (int i = 0; i < 4; i++) bytes[14 + i] = (byte)(bitstreamSerial >> (8 * i));
        // Page sequence (uint32 LE).
        for (int i = 0; i < 4; i++) bytes[18 + i] = (byte)(pageSequence >> (8 * i));
        // CRC (22..25) left zero during computation.
        bytes[26] = (byte)segTable.Length;
        // Segment table.
        Array.Copy(segTable, 0, bytes, 27, segTable.Length);
        // Payload.
        packet.CopyTo(bytes.AsSpan(27 + segTable.Length));
        // CRC-32 over the whole page (with CRC field zeroed).
        uint crc = OggCrc32.Compute(bytes);
        bytes[22] = (byte)crc;
        bytes[23] = (byte)(crc >> 8);
        bytes[24] = (byte)(crc >> 16);
        bytes[25] = (byte)(crc >> 24);
        return bytes;
    }

    /// <summary>
    /// Write a whole logical Ogg bitstream from a sequence of packets. One page
    /// per packet. The first page is BOS, the last is EOS. Granule positions
    /// are taken from each packet's <see cref="OggOutgoingPacket.GranulePosition"/>.
    /// </summary>
    public static byte[] WriteStream(uint bitstreamSerial, IReadOnlyList<OggOutgoingPacket> packets)
    {
        if (packets.Count == 0)
            throw new ArgumentException("Cannot write an Ogg stream with zero packets.", nameof(packets));
        var all = new List<byte>();
        for (int i = 0; i < packets.Count; i++)
        {
            byte type = 0;
            if (i == 0) type |= OggConstants.HeaderTypeBeginningOfStream;
            if (i == packets.Count - 1) type |= OggConstants.HeaderTypeEndOfStream;
            all.AddRange(WriteSinglePacketPage(
                type, packets[i].GranulePosition, bitstreamSerial, (uint)i, packets[i].Data));
        }
        return all.ToArray();
    }
}
