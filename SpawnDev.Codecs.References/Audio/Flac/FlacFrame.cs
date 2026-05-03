// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).

namespace SpawnDev.Codecs.Audio.Flac;

/// <summary>
/// Fully decoded FLAC frame: the parsed header plus a channel-major sample
/// buffer. Samples are arranged as <c>samples[channel * BlockSize + n]</c>.
/// </summary>
public sealed record FlacFrame
{
    /// <summary>Frame-level metadata parsed from the 4+ byte header.</summary>
    public required FlacFrameHeader Header { get; init; }

    /// <summary>
    /// Channel-major decoded samples at the frame's bit depth. Length =
    /// <see cref="FlacFrameHeader.Channels"/> * <see cref="FlacFrameHeader.BlockSize"/>.
    /// Channel decorrelation (L/R/M-side) has already been applied.
    /// </summary>
    public required int[] Samples { get; init; }

    /// <summary>
    /// Total bytes consumed by this frame (header + subframes + alignment padding
    /// + CRC-16 footer).
    /// </summary>
    public int FrameBytesConsumed { get; init; }
}
