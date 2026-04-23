// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).

namespace SpawnDev.Codecs.Audio;

/// <summary>
/// Identifies which audio codec an <see cref="IAudioDecoder"/> (or future
/// <c>IAudioEncoder</c>, once Phase 1b lands) implements. New codecs extend this enum as they ship.
/// </summary>
public enum AudioCodec
{
    /// <summary>Opus (RFC 6716). Phase 1.</summary>
    Opus,

    /// <summary>FLAC (lossless). Later phase.</summary>
    Flac,

    /// <summary>Vorbis (open alternative to AAC). Later phase.</summary>
    Vorbis
}
