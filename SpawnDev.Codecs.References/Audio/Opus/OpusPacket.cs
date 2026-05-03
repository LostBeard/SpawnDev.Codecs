// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
// See NOTICE.md for upstream attributions.

namespace SpawnDev.Codecs.Audio.Opus;

/// <summary>
/// A parsed Opus packet: its TOC header, the list of individual compressed frames,
/// and any framing metadata needed to re-serialize or process it. Produced by
/// <see cref="OpusPacketParser.Parse(ReadOnlyMemory{byte}, bool)"/>.
/// </summary>
public sealed record OpusPacket
{
    /// <summary>The decoded TOC byte (first byte of the packet).</summary>
    public required OpusTocByte Toc { get; init; }

    /// <summary>
    /// The individual compressed frames carried by this packet.
    /// For mode <see cref="OpusMode.Silk"/> and <see cref="OpusMode.Hybrid"/> these feed
    /// the SILK + CELT decoders in sequence; for <see cref="OpusMode.Celt"/> they feed CELT only.
    /// Each frame is a slice of the original packet buffer - no copying occurs during parsing.
    /// </summary>
    public required IReadOnlyList<ReadOnlyMemory<byte>> Frames { get; init; }

    /// <summary>
    /// Offset within the original packet buffer at which the first frame's data begins.
    /// This is the byte immediately after the TOC + frame count + frame size headers.
    /// </summary>
    public required int PayloadOffset { get; init; }

    /// <summary>
    /// Total number of bytes consumed by this packet including any padding. For a
    /// non-self-delimited packet this is always the length of the input buffer. For
    /// self-delimited packets it may be less (allowing further packets to be parsed from
    /// the remaining bytes).
    /// </summary>
    public required int PacketLength { get; init; }

    /// <summary>Padding bytes at the end of the packet (mode-3 packets may include padding per RFC 6716 section 3.2).</summary>
    public required ReadOnlyMemory<byte> Padding { get; init; }

    /// <summary>Number of frames carried by the packet. Equivalent to <see cref="Frames"/>.Count.</summary>
    public int FrameCount => Frames.Count;

    /// <summary>Samples per frame at the given output sample rate.</summary>
    public int GetSamplesPerFrame(int sampleRateHz) => Toc.GetSamplesPerFrame(sampleRateHz);

    /// <summary>Total audio samples carried by this packet at the given output sample rate.</summary>
    public int GetTotalSamples(int sampleRateHz) => GetSamplesPerFrame(sampleRateHz) * FrameCount;
}
