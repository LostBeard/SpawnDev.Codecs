// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>VP8 4x4 intra prediction mode (RFC 6386 sec 12.3).</summary>
public enum Vp8IntraMode4x4 : byte
{
    /// <summary>DC predictor - average of 4 above + 4 left samples.</summary>
    BDcPred = 0,
    /// <summary>TrueMotion predictor.</summary>
    BTmPred = 1,
    /// <summary>Vertical with 3-tap edge filter.</summary>
    BVePred = 2,
    /// <summary>Horizontal with 3-tap edge filter.</summary>
    BHePred = 3,
    /// <summary>Down-left diagonal (D45 variant).</summary>
    BLdPred = 4,
    /// <summary>Right-down diagonal (D135).</summary>
    BRdPred = 5,
    /// <summary>Vertical-right (D117).</summary>
    BVrPred = 6,
    /// <summary>Vertical-left (D63 variant).</summary>
    BVlPred = 7,
    /// <summary>Horizontal-down (D153).</summary>
    BHdPred = 8,
    /// <summary>Horizontal-up (D207).</summary>
    BHuPred = 9,
}
