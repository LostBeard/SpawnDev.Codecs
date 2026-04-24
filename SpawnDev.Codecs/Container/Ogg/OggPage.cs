// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Parsed Ogg page. Matches the on-wire layout of RFC 3533 Section 6.

namespace SpawnDev.Codecs.Container.Ogg;

/// <summary>
/// A single parsed Ogg page: header fields + the raw payload bytes.
/// </summary>
public sealed record OggPage
{
    /// <summary>
    /// Header flags. Typically 0 (fresh packet) or OR of <c>Continuation(0x01)</c>,
    /// <c>BeginningOfStream(0x02)</c>, <c>EndOfStream(0x04)</c>.
    /// </summary>
    public byte HeaderType { get; init; }

    /// <summary>Granule position (codec-specific running position). 8 bytes on the wire, signed.</summary>
    public long GranulePosition { get; init; }

    /// <summary>Identifies the logical bitstream this page belongs to (multiple streams may be multiplexed).</summary>
    public uint BitstreamSerial { get; init; }

    /// <summary>Monotonically increasing per-stream page counter.</summary>
    public uint PageSequence { get; init; }

    /// <summary>Page CRC-32 as read from the wire.</summary>
    public uint Crc { get; init; }

    /// <summary>
    /// Segment table (lengths of each segment in the payload). Segments pack together
    /// to form packets; a segment length &lt; 255 terminates a packet.
    /// </summary>
    public required byte[] SegmentLengths { get; init; }

    /// <summary>Raw page payload bytes, concatenation of all segments.</summary>
    public required byte[] Payload { get; init; }

    /// <summary>Total bytes this page occupied on the wire (header + segment table + payload).</summary>
    public int TotalPageBytes { get; init; }

    /// <summary>True if the continuation-of-packet flag is set.</summary>
    public bool IsContinuation => (HeaderType & OggConstants.HeaderTypeContinuation) != 0;

    /// <summary>True if beginning-of-stream flag is set.</summary>
    public bool IsBeginningOfStream => (HeaderType & OggConstants.HeaderTypeBeginningOfStream) != 0;

    /// <summary>True if end-of-stream flag is set.</summary>
    public bool IsEndOfStream => (HeaderType & OggConstants.HeaderTypeEndOfStream) != 0;
}
