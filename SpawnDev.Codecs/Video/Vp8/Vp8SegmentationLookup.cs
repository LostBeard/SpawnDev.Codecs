// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP8 segmentation lookup. Resolves a segment_id + the per-segment
// feature data carried in Vp8SegmentationParams into the actual per-MB
// quantizer index and loop filter level the walker should use.
//
// Two segmentation features per MB (libvpx MB_LVL_MAX = 2):
//   index 0 = MB_LVL_ALT_Q   - alternate quantizer (delta or absolute)
//   index 1 = MB_LVL_ALT_LF  - alternate loop filter level (delta or absolute)
//
// AbsDelta selector (Vp8SegmentationParams.AbsDelta):
//   true  = SEGMENT_ABSDATA   - feature_data is the absolute value
//   false = SEGMENT_DELTADATA - feature_data is added to the frame default
//
// Reference: libvpx vp8/decoder/decodeframe.c vp8_mb_init_dequantizer
// (the QIndex resolution path) + vp8_loop_filter_frame_init (the LF level
// resolution path).

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>VP8 segmentation feature index (libvpx MB_LVL enum).</summary>
public enum Vp8SegmentationFeature : byte
{
    /// <summary>Alternate quantizer index (libvpx MB_LVL_ALT_Q).</summary>
    AltQuant = 0,
    /// <summary>Alternate loop filter level (libvpx MB_LVL_ALT_LF).</summary>
    AltLoopFilter = 1,
}

/// <summary>VP8 per-MB Q index + LF level resolution from segmentation params.</summary>
public static class Vp8SegmentationLookup
{
    /// <summary>VP8 maximum Q index (libvpx MAXQ).</summary>
    public const int MaxQIndex = 127;
    /// <summary>VP8 maximum loop filter level.</summary>
    public const int MaxLoopFilterLevel = 63;

    /// <summary>
    /// Resolve the per-MB quantizer index for <paramref name="segmentId"/>.
    /// When segmentation is disabled or its alt-Q feature isn't in the
    /// frame's feature data, returns the frame default.
    /// </summary>
    public static int ResolveQIndex(
        int segmentId,
        int frameDefaultQIndex,
        Vp8SegmentationParams segmentation)
    {
        if (!segmentation.Enabled) return ClampQ(frameDefaultQIndex);
        if (segmentId < 0 || segmentId >= 4)
            throw new ArgumentOutOfRangeException(nameof(segmentId), "must be in [0, 3]");

        int featureValue = segmentation.FeatureData[(int)Vp8SegmentationFeature.AltQuant, segmentId];
        int q = segmentation.AbsDelta ? featureValue : frameDefaultQIndex + featureValue;
        return ClampQ(q);
    }

    /// <summary>
    /// Resolve the per-MB loop filter level for <paramref name="segmentId"/>.
    /// When segmentation is disabled or its alt-LF feature isn't in the
    /// frame's feature data, returns the frame default.
    /// </summary>
    public static int ResolveLoopFilterLevel(
        int segmentId,
        int frameDefaultLfLevel,
        Vp8SegmentationParams segmentation)
    {
        if (!segmentation.Enabled) return ClampLf(frameDefaultLfLevel);
        if (segmentId < 0 || segmentId >= 4)
            throw new ArgumentOutOfRangeException(nameof(segmentId), "must be in [0, 3]");

        int featureValue = segmentation.FeatureData[(int)Vp8SegmentationFeature.AltLoopFilter, segmentId];
        int lf = segmentation.AbsDelta ? featureValue : frameDefaultLfLevel + featureValue;
        return ClampLf(lf);
    }

    private static int ClampQ(int v) => v < 0 ? 0 : v > MaxQIndex ? MaxQIndex : v;
    private static int ClampLf(int v) => v < 0 ? 0 : v > MaxLoopFilterLevel ? MaxLoopFilterLevel : v;
}
