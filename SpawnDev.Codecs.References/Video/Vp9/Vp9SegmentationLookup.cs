// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 segmentation feature lookup helpers. Pure functions that
// resolve per-block segment feature state from a parsed
// <see cref="Vp9SegmentationParams"/>. Mirror of libvpx
// vp9/common/vp9_seg_common.c.
//
// Three helpers:
//   IsFeatureActive : segmentation is enabled AND the segment has
//                     the feature enabled.
//   GetFeatureData  : raw feature payload (must be active to be
//                     meaningful).
//   ResolveQIndex   : block's effective base_qindex factoring in
//                     ALT_Q feature with abs/delta interpretation.
//                     Clamps to [0, MaxQuantizerIndex].

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 segmentation feature lookup helpers.</summary>
public static class Vp9SegmentationLookup
{
    /// <summary>
    /// libvpx <c>MAXQ</c> for VP9 Profile 0 (8-bit). 10/12-bit
    /// profiles raise this to 1023 / 4095 respectively (future slice).
    /// </summary>
    public const int MaxQuantizerIndex = 255;

    /// <summary>
    /// libvpx <c>vp9_segfeature_active</c>. Returns true when
    /// segmentation is in effect AND the requested feature is
    /// enabled for the given segment id.
    /// </summary>
    public static bool IsFeatureActive(
        Vp9SegmentationParams segmentation,
        int segmentId,
        Vp9SegFeature feature)
    {
        ArgumentNullException.ThrowIfNull(segmentation);
        if (!segmentation.Enabled) return false;
        if ((uint)segmentId >= (uint)Vp9SegmentationParams.MaxSegments)
            throw new ArgumentOutOfRangeException(nameof(segmentId), segmentId,
                $"segmentId must be in [0, {Vp9SegmentationParams.MaxSegments}).");
        int featureIdx = (int)feature;
        if ((uint)featureIdx >= (uint)Vp9SegmentationParams.FeaturesPerSegment)
            throw new ArgumentOutOfRangeException(nameof(feature), feature,
                "feature index out of range.");
        if (segmentation.FeatureEnabled.GetLength(0) == 0) return false;
        return segmentation.FeatureEnabled[segmentId, featureIdx];
    }

    /// <summary>
    /// libvpx <c>vp9_get_segdata</c>. Returns the raw feature payload;
    /// caller is responsible for checking <see cref="IsFeatureActive"/>
    /// first - if the feature is inactive the returned value has no
    /// meaning.
    /// </summary>
    public static int GetFeatureData(
        Vp9SegmentationParams segmentation,
        int segmentId,
        Vp9SegFeature feature)
    {
        ArgumentNullException.ThrowIfNull(segmentation);
        if ((uint)segmentId >= (uint)Vp9SegmentationParams.MaxSegments)
            throw new ArgumentOutOfRangeException(nameof(segmentId));
        int featureIdx = (int)feature;
        if ((uint)featureIdx >= (uint)Vp9SegmentationParams.FeaturesPerSegment)
            throw new ArgumentOutOfRangeException(nameof(feature));
        if (segmentation.FeatureData.GetLength(0) == 0) return 0;
        return segmentation.FeatureData[segmentId, featureIdx];
    }

    /// <summary>
    /// libvpx <c>vp9_get_qindex</c>. Returns the effective base
    /// quantizer index for a block in segment <paramref name="segmentId"/>
    /// given the frame-level <paramref name="baseQIndex"/>. If the
    /// segment has ALT_Q active, applies the segmentation feature
    /// payload as either an absolute value (when
    /// <c>seg.AbsDelta = true</c>) or a delta from the base. Result
    /// is clamped to [0, <see cref="MaxQuantizerIndex"/>].
    /// </summary>
    public static int ResolveQIndex(
        Vp9SegmentationParams segmentation,
        int segmentId,
        int baseQIndex)
    {
        if (!IsFeatureActive(segmentation, segmentId, Vp9SegFeature.AltQ))
            return Math.Clamp(baseQIndex, 0, MaxQuantizerIndex);
        int payload = GetFeatureData(segmentation, segmentId, Vp9SegFeature.AltQ);
        int qindex = segmentation.AbsDelta ? payload : baseQIndex + payload;
        return Math.Clamp(qindex, 0, MaxQuantizerIndex);
    }
}
