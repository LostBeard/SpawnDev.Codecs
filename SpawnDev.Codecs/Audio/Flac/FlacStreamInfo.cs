// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// STREAMINFO is the first metadata block in every FLAC stream. It carries the
// geometry information a decoder needs before it can parse any frames.
// See RFC 9639 Section 8.1 or libFLAC format.h FLAC__StreamMetadata_StreamInfo.

namespace SpawnDev.Codecs.Audio.Flac;

/// <summary>
/// FLAC STREAMINFO metadata. Carries sample rate, channel count, bit depth,
/// block-size bounds, frame-size bounds, total sample count, and the MD5
/// signature of the decoded (unencoded) PCM. 34 bytes on the wire.
/// </summary>
public sealed record FlacStreamInfo
{
    /// <summary>Minimum block size (samples per block) across all frames. 16-bit field.</summary>
    public int MinBlockSize { get; init; }

    /// <summary>Maximum block size (samples per block) across all frames. 16-bit field.</summary>
    public int MaxBlockSize { get; init; }

    /// <summary>Minimum encoded frame size in bytes, or 0 if unknown. 24-bit field.</summary>
    public int MinFrameSize { get; init; }

    /// <summary>Maximum encoded frame size in bytes, or 0 if unknown. 24-bit field.</summary>
    public int MaxFrameSize { get; init; }

    /// <summary>Sample rate in Hz (1 to 655350). 20-bit field.</summary>
    public int SampleRateHz { get; init; }

    /// <summary>Channel count (1 to 8). Encoded on wire as <c>channels - 1</c> in 3 bits.</summary>
    public int Channels { get; init; }

    /// <summary>Bits per sample (4 to 32). Encoded on wire as <c>bps - 1</c> in 5 bits.</summary>
    public int BitsPerSample { get; init; }

    /// <summary>
    /// Total samples per channel in the stream, or 0 if unknown. 36-bit field.
    /// </summary>
    public ulong TotalSamples { get; init; }

    /// <summary>
    /// MD5 signature of the decoded PCM, in big-endian byte order. All zeros if not computed.
    /// 128-bit (16-byte) field.
    /// </summary>
    public required byte[] Md5Signature { get; init; }
}
