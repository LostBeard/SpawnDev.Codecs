// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Strongly-typed view of the /Segment/Info master element. Exposes the
// fields downstream codec pipelines and UIs actually consume - most
// importantly TimestampScale, which is the multiplier that turns the
// raw MatroskaFrame.Timestamp ticks into nanoseconds.

namespace SpawnDev.Codecs.Container.Matroska;

/// <summary>
/// Snapshot of the /Segment/Info master element.
/// </summary>
public sealed record MatroskaSegmentInfo
{
    /// <summary>
    /// Nanoseconds per timestamp tick. Default per Matroska spec is
    /// 1,000,000 (so one tick = one millisecond), which is what most
    /// WebM files ship with. Multiply any MatroskaFrame.Timestamp by
    /// this to get nanoseconds.
    /// </summary>
    public required ulong TimestampScaleNs { get; init; }

    /// <summary>
    /// Total segment duration in timestamp ticks (same unit as
    /// MatroskaFrame.Timestamp). May be null - the Duration element is
    /// optional and Chrome's MediaRecorder famously omits it on live
    /// WebM recordings (the exact pain point SpawnDev.EBML was built to
    /// fix). Use <see cref="DurationTimeSpan"/> for a friendly form.
    /// </summary>
    public double? DurationTicks { get; init; }

    /// <summary>
    /// Optional user-provided title from /Segment/Info/Title.
    /// </summary>
    public string? Title { get; init; }

    /// <summary>
    /// Muxer that wrote this file (e.g. "libebml2 v0.10.0+libmatroska2
    /// v0.10.1"). Mandatory per spec for Matroska; WebM files usually
    /// populate it too.
    /// </summary>
    public string? MuxingApp { get; init; }

    /// <summary>
    /// Application that requested the muxing (e.g. "Chrome", "ffmpeg").
    /// </summary>
    public string? WritingApp { get; init; }

    /// <summary>
    /// Convenience: <see cref="DurationTicks"/> scaled by <see cref="TimestampScaleNs"/>
    /// into a CLR <see cref="TimeSpan"/>. Null when Duration was absent.
    /// </summary>
    public TimeSpan? DurationTimeSpan
    {
        get
        {
            if (DurationTicks is null) return null;
            double ns = DurationTicks.Value * TimestampScaleNs;
            return TimeSpan.FromTicks((long)(ns / 100.0)); // 1 CLR tick = 100 ns
        }
    }
}
