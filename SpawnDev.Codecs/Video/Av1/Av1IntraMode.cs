// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 intra-prediction mode enum (spec sec 6.4.1).
// 13 modes, indexed in entropy CDFs by this exact order.

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>AV1 luma intra prediction mode. Same numeric ordering as libaom <c>PREDICTION_MODE</c>.</summary>
public enum Av1IntraMode : byte
{
    /// <summary>DC_PRED: average of available top + left edge pixels.</summary>
    Dc = 0,
    /// <summary>V_PRED: copy the top edge into every row.</summary>
    Vertical = 1,
    /// <summary>H_PRED: copy the left edge into every column.</summary>
    Horizontal = 2,
    /// <summary>D45_PRED: directional 45 degree prediction (NE, north-east).</summary>
    D45 = 3,
    /// <summary>D135_PRED: directional 135 degrees (NW, north-west).</summary>
    D135 = 4,
    /// <summary>D113_PRED: directional 113 degrees.</summary>
    D113 = 5,
    /// <summary>D157_PRED: directional 157 degrees.</summary>
    D157 = 6,
    /// <summary>D203_PRED: directional 203 degrees.</summary>
    D203 = 7,
    /// <summary>D67_PRED: directional 67 degrees.</summary>
    D67 = 8,
    /// <summary>SMOOTH_PRED: bilinear weighted blend of top + left + bottom-left + top-right.</summary>
    Smooth = 9,
    /// <summary>SMOOTH_V_PRED: vertical-axis smooth blend (top + bottom-left).</summary>
    SmoothV = 10,
    /// <summary>SMOOTH_H_PRED: horizontal-axis smooth blend (left + top-right).</summary>
    SmoothH = 11,
    /// <summary>PAETH_PRED: per-pixel "minimum gradient" prediction.</summary>
    Paeth = 12,
}
