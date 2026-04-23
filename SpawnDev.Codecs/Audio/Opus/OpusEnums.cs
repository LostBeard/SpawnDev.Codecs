// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
// See NOTICE.md for upstream attributions.

namespace SpawnDev.Codecs.Audio.Opus;

/// <summary>
/// Which internal coding mode an Opus packet uses. Selected per-packet by the encoder;
/// the decoder reads this from the TOC byte (RFC 6716 section 3.1, configs 0-31).
/// </summary>
public enum OpusMode
{
    /// <summary>SILK-only (speech). Configs 0-11. Up to 16 kHz.</summary>
    Silk,

    /// <summary>SILK for low frequencies + CELT for high frequencies. Configs 12-15. 24 or 48 kHz.</summary>
    Hybrid,

    /// <summary>CELT-only (music, low-latency). Configs 16-31. Any supported bandwidth.</summary>
    Celt
}

/// <summary>
/// Audio bandwidth encoded in an Opus packet. Determines the effective sample rate
/// the decoder outputs for that packet (RFC 6716 section 2).
/// </summary>
public enum OpusBandwidth
{
    /// <summary>4 kHz bandwidth, 8 kHz sample rate.</summary>
    Narrowband,

    /// <summary>6 kHz bandwidth, 12 kHz sample rate.</summary>
    Mediumband,

    /// <summary>8 kHz bandwidth, 16 kHz sample rate.</summary>
    Wideband,

    /// <summary>12 kHz bandwidth, 24 kHz sample rate.</summary>
    Superwideband,

    /// <summary>20 kHz bandwidth, 48 kHz sample rate.</summary>
    Fullband
}

/// <summary>
/// Error codes returned by packet-parsing operations. Negative values match libopus
/// OPUS_BAD_ARG / OPUS_INVALID_PACKET for easy porting of downstream code.
/// </summary>
public enum OpusPacketError
{
    /// <summary>The packet parsed successfully. Only used as a sentinel.</summary>
    None = 0,

    /// <summary>A required argument was invalid (e.g. null or negative length). Matches libopus OPUS_BAD_ARG.</summary>
    BadArgument = -1,

    /// <summary>The packet bytes were malformed or internally inconsistent. Matches libopus OPUS_INVALID_PACKET.</summary>
    InvalidPacket = -4
}
