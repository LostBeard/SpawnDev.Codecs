// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).

namespace SpawnDev.Codecs.Video;

/// <summary>
/// Patent-clean video codecs targeted by SpawnDev.Codecs. Each codec in this
/// enum ships as an <see cref="IVideoDecoder"/> implementation (decoder first,
/// encoder to follow in a later phase).
/// </summary>
public enum VideoCodec
{
    /// <summary>VP8 (RFC 6386). Patent-clean via Google's WebM pledge.</summary>
    Vp8,

    /// <summary>VP9. Patent-clean via Google's WebM pledge.</summary>
    Vp9,

    /// <summary>AV1. Patent-clean via the AOMedia patent pledge.</summary>
    Av1,
}
