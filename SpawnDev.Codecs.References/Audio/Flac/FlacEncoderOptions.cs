// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).

namespace SpawnDev.Codecs.Audio.Flac;

/// <summary>
/// Optional encoder settings. Pass an instance to the metadata-aware
/// <see cref="FlacEncoder"/> overload to include VORBIS_COMMENT tags in the
/// encoded stream.
/// </summary>
public sealed record FlacEncoderOptions
{
    /// <summary>Samples per channel per frame. Defaults to 4096.</summary>
    public int BlockSize { get; init; } = 4096;

    /// <summary>Encoder vendor string written into an embedded VORBIS_COMMENT block.</summary>
    public string Vendor { get; init; } = "SpawnDev.Codecs";

    /// <summary>
    /// <c>TAG=value</c> metadata entries to write into an embedded VORBIS_COMMENT
    /// block. If null or empty, no VORBIS_COMMENT block is emitted.
    /// </summary>
    public IReadOnlyList<string>? VorbisComments { get; init; }
}
