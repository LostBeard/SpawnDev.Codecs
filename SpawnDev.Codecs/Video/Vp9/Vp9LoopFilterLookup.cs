// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 loop filter level resolution. Combines the frame-level
// filter_level with per-segment ALT_LF deltas to produce the
// effective level for a given block. Mode/ref delta application
// (per-frame mode_ref_delta_enabled) is handled in a downstream
// slice once block context (mode, ref_frame) is available.
//
// libvpx reference: vp9/common/vp9_loopfilter.c vp9_get_filter_level
// (the segmentation portion).

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 loop filter level lookup helpers.</summary>
public static class Vp9LoopFilterLookup
{
    /// <summary>libvpx <c>MAX_LOOP_FILTER</c>.</summary>
    public const int MaxLoopFilter = 63;

    /// <summary>
    /// Resolve the effective loop filter level for a block in segment
    /// <paramref name="segmentId"/>, given the frame-level
    /// <paramref name="frameFilterLevel"/>. Applies ALT_LF feature
    /// (when active) using the abs/delta interpretation. Result is
    /// clamped to [0, <see cref="MaxLoopFilter"/>].
    ///
    /// Mirror of the ALT_LF branch of libvpx
    /// <c>vp9_get_filter_level</c>. Mode/ref delta application is
    /// the caller's job once block context (mode + ref_frame[0]) is
    /// available; this helper just resolves the segment-level layer.
    /// </summary>
    public static int ResolveSegmentLevel(
        Vp9SegmentationParams segmentation,
        int segmentId,
        int frameFilterLevel)
    {
        ArgumentNullException.ThrowIfNull(segmentation);

        if (!Vp9SegmentationLookup.IsFeatureActive(segmentation, segmentId, Vp9SegFeature.AltLf))
            return Math.Clamp(frameFilterLevel, 0, MaxLoopFilter);

        int payload = Vp9SegmentationLookup.GetFeatureData(segmentation, segmentId, Vp9SegFeature.AltLf);
        int level = segmentation.AbsDelta ? payload : frameFilterLevel + payload;
        return Math.Clamp(level, 0, MaxLoopFilter);
    }
}
