// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).

namespace SpawnDev.Codecs.Audio.Flac;

/// <summary>
/// Parsed FLAC frame header (RFC 9639 Section 9.1). All fields are fully
/// resolved: block size, sample rate, and bits-per-sample are dereferenced
/// against STREAMINFO when the header uses the "get from STREAMINFO" codes.
/// </summary>
public sealed record FlacFrameHeader
{
    /// <summary>Blocking strategy: Fixed (frame-numbered) or Variable (sample-numbered).</summary>
    public FlacBlockingStrategy BlockingStrategy { get; init; }

    /// <summary>Block size in samples per channel (resolved from code or side-byte).</summary>
    public int BlockSize { get; init; }

    /// <summary>Sample rate in Hz (resolved from code, side-byte, or STREAMINFO).</summary>
    public int SampleRateHz { get; init; }

    /// <summary>Channel assignment: independent 1-8 channels or L/R/M-side stereo.</summary>
    public FlacChannelAssignment ChannelAssignment { get; init; }

    /// <summary>Channel count (always 2 for stereo decorrelation modes).</summary>
    public int Channels { get; init; }

    /// <summary>Bits per sample (resolved from code or STREAMINFO).</summary>
    public int BitsPerSample { get; init; }

    /// <summary>
    /// Frame number (for fixed-block-size streams) or starting sample number
    /// (for variable-block-size streams), depending on <see cref="BlockingStrategy"/>.
    /// </summary>
    public ulong SampleOrFrameNumber { get; init; }

    /// <summary>Total size of the parsed frame header in bytes (including CRC-8).</summary>
    public int HeaderBytesConsumed { get; init; }
}
