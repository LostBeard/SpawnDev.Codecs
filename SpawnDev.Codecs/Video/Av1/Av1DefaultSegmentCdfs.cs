// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 default segmentation CDF tables.
//
// Upstream Copyright (c) 2016, Alliance for Open Media. All rights reserved.
// Upstream license: BSD 2-Clause + AV1 Patent License 1.0. See NOTICE.md.
// Upstream commit: 136511836e54093f24d23f02cf93943ff5fc97a2 (libaom main).
// Upstream source: aomedia.googlesource.com/aom av1/common/entropymode.c
//   default_segment_pred_cdf            (lines 853-856)
//   default_spatial_pred_seg_tree_cdf   (lines 858-870)
//
// Constants (av1/common/seg_common.h):
//   MAX_SEGMENTS              = 8
//   SEG_TREE_PROBS            = MAX_SEGMENTS - 1 = 7
//   SEG_TEMPORAL_PRED_CTXS    = 3
//   SPATIAL_PREDICTION_PROBS  = 3
//
// Stored as inverse CDF (ICDF: CDF_PROB_TOP - cumprob), padded to libaom's
// CDF_SIZE() row width per AOM_CDFn macro semantics. Compatible directly
// with Av1RangeDecoder.DecodeCdfQ15.

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>
/// AV1 default CDFs for segmentation entropy decode (segment id and
/// temporal-prediction flag).
/// </summary>
internal static class Av1DefaultSegmentCdfs
{
    /// <summary>
    /// <c>default_segment_pred_cdf[SEG_TEMPORAL_PRED_CTXS][CDF_SIZE(2)]</c>.
    /// Temporal segment-id prediction flag (whether to copy the segment id
    /// from the collocated block in the previous frame). Indexed by the
    /// 3-context derived from above + left segment-pred neighbors.
    /// </summary>
    public static readonly ushort[][] DefaultSegmentPredCdf = new ushort[][]
    {
        new ushort[] { 16384, 0, 0 },
        new ushort[] { 16384, 0, 0 },
        new ushort[] { 16384, 0, 0 },
    };

    /// <summary>
    /// <c>default_spatial_pred_seg_tree_cdf[SPATIAL_PREDICTION_PROBS][CDF_SIZE(MAX_SEGMENTS)]</c>.
    /// Spatial-prediction segment id tree CDF. Indexed by the 3-context derived
    /// from above + left segment ids. 8 active symbols (one per segment id 0..7).
    /// </summary>
    public static readonly ushort[][] DefaultSpatialPredSegTreeCdf = new ushort[][]
    {
        new ushort[] { 27146, 24875, 16675, 14535, 4959, 4395, 235, 0, 0 },
        new ushort[] { 18494, 14538, 10211, 7833, 2788, 1917, 424, 0, 0 },
        new ushort[] { 5241, 4281, 4045, 3878, 371, 121, 89, 0, 0 },
    };
}
