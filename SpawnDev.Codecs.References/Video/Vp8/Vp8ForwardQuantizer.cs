// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP8 forward quantizer - encoder-side counterpart of Vp8Quantizer's
// dequantization lookups. Maps a 16-element coefficient block (output
// of Vp8ForwardTransform) into a quantized integer block that's about
// to be entropy-encoded.
//
// Naive truncation-toward-zero quantizer:
//   quantized[i] = signed_round(coef[i] / Q[i])
// with Q[0] = DC dequant, Q[1..15] = AC dequant (per-block-type).
//
// Production encoders implement rate-distortion-optimized quantization
// (vp8/encoder/quantize.c vp8cx_quantize_b_c with zbin / round-up / dead-zone
// adjustments) for compression efficiency. This naive version is
// correct but produces larger files than libvpx; replace with RD-opt
// later when encoder tuning matters.

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>VP8 forward quantizer (naive truncation).</summary>
public static class Vp8ForwardQuantizer
{
    /// <summary>
    /// Quantize a 16-element coefficient block in-place.
    /// </summary>
    /// <param name="coefs">Input/output coefficients (raster 4x4).</param>
    /// <param name="dcQ">DC dequantizer value (used for coefs[0]).</param>
    /// <param name="acQ">AC dequantizer value (used for coefs[1..15]).</param>
    public static void QuantizeBlock(Span<short> coefs, int dcQ, int acQ)
    {
        if (coefs.Length < 16) throw new ArgumentException("coefs must have 16 entries", nameof(coefs));
        if (dcQ <= 0) throw new ArgumentOutOfRangeException(nameof(dcQ), "must be > 0");
        if (acQ <= 0) throw new ArgumentOutOfRangeException(nameof(acQ), "must be > 0");

        coefs[0] = (short)RoundedDivide(coefs[0], dcQ);
        for (int i = 1; i < 16; i++)
            coefs[i] = (short)RoundedDivide(coefs[i], acQ);
    }

    /// <summary>
    /// Quantize using the Vp8MbDequant Y1 dequantizer values
    /// (DC=Y1Dc, AC=Y1Ac).
    /// </summary>
    public static void QuantizeY1Block(Span<short> coefs, Vp8MbDequant dequant)
        => QuantizeBlock(coefs, dequant.Y1Dc, dequant.Y1Ac);

    /// <summary>
    /// Quantize using the Y2 dequantizer values (DC=Y2Dc, AC=Y2Ac).
    /// Y2 holds the 16 Y4 DC values transformed through Walsh-Hadamard.
    /// </summary>
    public static void QuantizeY2Block(Span<short> coefs, Vp8MbDequant dequant)
        => QuantizeBlock(coefs, dequant.Y2Dc, dequant.Y2Ac);

    /// <summary>Quantize using the UV dequantizer values (DC=UvDc, AC=UvAc).</summary>
    public static void QuantizeUvBlock(Span<short> coefs, Vp8MbDequant dequant)
        => QuantizeBlock(coefs, dequant.UvDc, dequant.UvAc);

    /// <summary>
    /// Round-half-to-even integer division. Matches the property that
    /// dequantizing the result via multiply gives the closest representable
    /// value to the original.
    /// </summary>
    private static int RoundedDivide(int value, int divisor)
    {
        // Symmetric rounding for negative values.
        if (value >= 0)
            return (value + divisor / 2) / divisor;
        return -(((-value) + divisor / 2) / divisor);
    }
}
