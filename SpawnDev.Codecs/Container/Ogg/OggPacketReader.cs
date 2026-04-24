// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Assemble Ogg packets from a sequence of pages. Per RFC 3533: each page
// contains a segment table; consecutive 255-byte segments belong to the same
// packet, and a segment of length &lt; 255 terminates a packet. A packet may
// span multiple pages when the last segment of a page is 255 bytes long and
// the next page has the Continuation flag set.

namespace SpawnDev.Codecs.Container.Ogg;

/// <summary>
/// Assembles packets from a sequence of Ogg pages. Each packet is associated
/// with the page that terminated it (i.e., the page containing the final
/// non-255 segment).
/// </summary>
public sealed record OggPacket
{
    /// <summary>Packet data bytes.</summary>
    public required byte[] Data { get; init; }

    /// <summary>Bitstream serial number of the logical stream this packet belongs to.</summary>
    public uint BitstreamSerial { get; init; }

    /// <summary>
    /// Granule position from the page that terminated this packet. Codec-specific
    /// interpretation: for Opus, samples produced since the BOS page.
    /// </summary>
    public long TerminatingPageGranule { get; init; }

    /// <summary>True if the terminating page was marked EOS.</summary>
    public bool IsLastPacket { get; init; }
}

/// <summary>Assembles Ogg packets from a sequence of pages.</summary>
public static class OggPacketReader
{
    /// <summary>
    /// Read all packets from a pre-parsed list of Ogg pages. Pages must be in
    /// serial order as written on the wire (but may interleave multiple
    /// logical bitstream serials; each serial's packets are assembled
    /// independently).
    /// </summary>
    public static IEnumerable<OggPacket> AssemblePackets(IEnumerable<OggPage> pages)
    {
        // Per-bitstream packet accumulator.
        var buffers = new Dictionary<uint, List<byte>>();

        foreach (var page in pages)
        {
            if (!buffers.TryGetValue(page.BitstreamSerial, out var buf))
            {
                buf = new List<byte>();
                buffers[page.BitstreamSerial] = buf;
            }

            int payloadIndex = 0;
            for (int s = 0; s < page.SegmentLengths.Length; s++)
            {
                int segLen = page.SegmentLengths[s];
                if (segLen > 0)
                {
                    buf.AddRange(new ArraySegment<byte>(page.Payload, payloadIndex, segLen));
                    payloadIndex += segLen;
                }

                if (segLen < 255)
                {
                    // Terminating segment: emit packet.
                    bool isLast = page.IsEndOfStream && s == page.SegmentLengths.Length - 1;
                    yield return new OggPacket
                    {
                        Data = buf.ToArray(),
                        BitstreamSerial = page.BitstreamSerial,
                        TerminatingPageGranule = page.GranulePosition,
                        IsLastPacket = isLast,
                    };
                    buf.Clear();
                }
                // else segment exactly 255 bytes -> packet continues into next segment/page.
            }
        }
        // Dangling partial packets (no terminator) are discarded, per most Ogg demuxers.
    }
}
