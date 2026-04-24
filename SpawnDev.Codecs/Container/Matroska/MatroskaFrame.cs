// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// One Matroska frame produced by walking Segment/Cluster/SimpleBlock or
// Segment/Cluster/BlockGroup/Block. The byte payload is exactly the codec
// packet that a downstream decoder (VP8/VP9/AV1/Opus/Vorbis/FLAC) consumes.

namespace SpawnDev.Codecs.Container.Matroska;

/// <summary>
/// A single codec frame (or laced sub-frame) extracted from a Matroska /
/// WebM Cluster.
/// </summary>
public sealed record MatroskaFrame
{
    /// <summary>
    /// Track number this frame belongs to. Matches
    /// <see cref="MatroskaTrack.TrackNumber"/> from the container's
    /// Tracks enumeration.
    /// </summary>
    public required ulong TrackNumber { get; init; }

    /// <summary>
    /// Absolute timestamp = Cluster.Timestamp + SimpleBlock.RelativeTimestamp.
    /// Units are Matroska ticks (multiply by <c>TimestampScale</c> from
    /// /Segment/Info to get nanoseconds). For most WebM files the scale is
    /// 1,000,000 so ticks are already milliseconds.
    /// </summary>
    public required long Timestamp { get; init; }

    /// <summary>
    /// Raw codec packet bytes. For VP9 / AV1 / Opus / etc. this is the
    /// decoder input exactly as it appears on the wire.
    /// </summary>
    public required byte[] Data { get; init; }

    /// <summary>
    /// True when this frame came from a SimpleBlock with the keyframe bit
    /// set. Always false for Block (non-simple) frames because Matroska
    /// encodes keyframe info on the enclosing BlockGroup's ReferenceBlock
    /// children instead (not parsed in this slice).
    /// </summary>
    public required bool IsKeyframe { get; init; }

    /// <summary>
    /// Which of the block's potentially multiple laced frames this record
    /// represents (0-based). Always 0 when the block had no lacing.
    /// </summary>
    public required int LaceIndex { get; init; }
}
