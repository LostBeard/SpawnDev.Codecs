// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// A single /Segment/Tracks/TrackEntry parsed out of a Matroska or WebM
// document. Track metadata here is the minimum required to route the
// track's frames to the right codec decoder in later slices (VP8, VP9,
// AV1 for video; Opus, Vorbis, FLAC for audio).

namespace SpawnDev.Codecs.Container.Matroska;

/// <summary>
/// Summary of a single Matroska track (aka a /Segment/Tracks/TrackEntry).
/// </summary>
public sealed record MatroskaTrack
{
    /// <summary>
    /// 1-based track number the Block / SimpleBlock elements reference.
    /// </summary>
    public required ulong TrackNumber { get; init; }

    /// <summary>
    /// Raw Matroska track type code (see <see cref="MatroskaTrackType"/>
    /// for the named enum).
    /// </summary>
    public required ulong TrackType { get; init; }

    /// <summary>
    /// Matroska codec ID string - "V_VP8", "V_VP9", "V_AV1", "A_OPUS",
    /// "A_VORBIS", "A_FLAC", etc. See the Matroska codec registry for the
    /// full list.
    /// </summary>
    public required string CodecId { get; init; }

    /// <summary>CodecPrivate (e.g. the OpusHead or the Vorbis headers), or null.</summary>
    public byte[]? CodecPrivate { get; init; }

    /// <summary>DefaultDuration in nanoseconds (one frame / one audio block), or null.</summary>
    public ulong? DefaultDurationNs { get; init; }

    /// <summary>CodecDelay in nanoseconds (Opus pre-skip), or null.</summary>
    public ulong? CodecDelayNs { get; init; }

    /// <summary>SeekPreRoll in nanoseconds, or null.</summary>
    public ulong? SeekPreRollNs { get; init; }

    /// <summary>Video/PixelWidth, or null.</summary>
    public ulong? PixelWidth { get; init; }

    /// <summary>Video/PixelHeight, or null.</summary>
    public ulong? PixelHeight { get; init; }

    /// <summary>Video/AlphaMode (1 = the BlockAdditional of each block carries an alpha picture), or null.</summary>
    public ulong? AlphaMode { get; init; }

    /// <summary>Audio/SamplingFrequency in Hz, or null.</summary>
    public double? SamplingFrequency { get; init; }

    /// <summary>Audio/Channels, or null.</summary>
    public ulong? Channels { get; init; }

    /// <summary>Audio/BitDepth, or null.</summary>
    public ulong? BitDepth { get; init; }

    /// <summary>True when <see cref="TrackType"/> codes a video track (1).</summary>
    public bool IsVideo => TrackType == (ulong)MatroskaTrackType.Video;

    /// <summary>True when <see cref="TrackType"/> codes an audio track (2).</summary>
    public bool IsAudio => TrackType == (ulong)MatroskaTrackType.Audio;
}

/// <summary>Matroska track-type enum per the Matroska spec.</summary>
public enum MatroskaTrackType : ulong
{
    /// <summary>Video track.</summary>
    Video = 1,
    /// <summary>Audio track.</summary>
    Audio = 2,
    /// <summary>Complex track (both video and audio, legacy).</summary>
    Complex = 3,
    /// <summary>Logo track (Matroska-only, not used in WebM).</summary>
    Logo = 16,
    /// <summary>Subtitle track.</summary>
    Subtitle = 17,
    /// <summary>Buttons track (Matroska menu overlay, not used in WebM).</summary>
    Buttons = 18,
    /// <summary>Control track (DVD chapter control, not used in WebM).</summary>
    Control = 32,
    /// <summary>Metadata-only track.</summary>
    Metadata = 33,
}
