// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP8 per-macroblock dequantizer setup. Combines a segmentation-resolved
// Q index with the frame's per-plane delta values (Y1_DC, Y2_DC, Y2_AC,
// UV_DC, UV_AC) to produce the 6 per-MB dequantization values the
// inverse transform multiplies coefficients by.
//
// Mirrors libvpx <c>vp8_mb_init_dequantizer</c> (vp8/decoder/decodeframe.c).

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>VP8 per-macroblock dequantization values.</summary>
public sealed record Vp8MbDequant
{
    /// <summary>Y1 DC dequantizer.</summary>
    public required int Y1Dc { get; init; }
    /// <summary>Y1 AC dequantizer.</summary>
    public required int Y1Ac { get; init; }
    /// <summary>Y2 (second-order) DC dequantizer.</summary>
    public required int Y2Dc { get; init; }
    /// <summary>Y2 (second-order) AC dequantizer.</summary>
    public required int Y2Ac { get; init; }
    /// <summary>UV DC dequantizer.</summary>
    public required int UvDc { get; init; }
    /// <summary>UV AC dequantizer.</summary>
    public required int UvAc { get; init; }
}

/// <summary>VP8 per-MB dequantizer setup from frame quantizer indices + segment.</summary>
public static class Vp8MbDequantizer
{
    /// <summary>
    /// Compute the 6 per-MB dequantization values for a given segment_id.
    /// Resolves the segmentation-adjusted Q index first, then applies the
    /// frame's per-plane deltas through the Vp8Quantizer lookup tables.
    /// </summary>
    public static Vp8MbDequant Compute(
        int segmentId,
        Vp8QuantizerIndices frameQuant,
        Vp8SegmentationParams segmentation)
    {
        int baseQ = Vp8SegmentationLookup.ResolveQIndex(
            segmentId, frameQuant.BaseQIndex, segmentation);

        return new Vp8MbDequant
        {
            Y1Dc = Vp8Quantizer.Y1Dc(baseQ, frameQuant.Y1DcDeltaQ),
            Y1Ac = Vp8Quantizer.Y1Ac(baseQ),
            Y2Dc = Vp8Quantizer.Y2Dc(baseQ, frameQuant.Y2DcDeltaQ),
            Y2Ac = Vp8Quantizer.Y2Ac(baseQ, frameQuant.Y2AcDeltaQ),
            UvDc = Vp8Quantizer.UvDc(baseQ, frameQuant.UvDcDeltaQ),
            UvAc = Vp8Quantizer.UvAc(baseQ, frameQuant.UvAcDeltaQ),
        };
    }
}
