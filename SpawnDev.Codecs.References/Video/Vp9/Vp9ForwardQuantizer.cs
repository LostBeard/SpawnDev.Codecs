// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 forward quantizer - encoder-side counterpart of Vp9Dequantizer.
// Naive truncation-toward-zero quantizer mirroring the VP8 forward
// quantizer pattern. Production VP9 encoders use rate-distortion-
// optimized quantization for better compression; this is the floor.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 forward quantizer (naive truncation).</summary>
public static class Vp9ForwardQuantizer
{
    /// <summary>
    /// Quantize a coefficient block in-place.
    /// </summary>
    /// <param name="coefs">Input/output coefficients (raster order).</param>
    /// <param name="dcQ">DC dequantizer value (used for coefs[0]).</param>
    /// <param name="acQ">AC dequantizer value (used for coefs[1..]).</param>
    public static void QuantizeBlock(Span<int> coefs, int dcQ, int acQ)
    {
        if (coefs.Length < 1) throw new ArgumentException("coefs must have at least 1 entry", nameof(coefs));
        if (dcQ <= 0) throw new ArgumentOutOfRangeException(nameof(dcQ), "must be > 0");
        if (acQ <= 0) throw new ArgumentOutOfRangeException(nameof(acQ), "must be > 0");

        coefs[0] = RoundedDivide(coefs[0], dcQ);
        for (int i = 1; i < coefs.Length; i++)
            coefs[i] = RoundedDivide(coefs[i], acQ);
    }

    private static int RoundedDivide(int value, int divisor)
    {
        if (value >= 0) return (value + divisor / 2) / divisor;
        return -(((-value) + divisor / 2) / divisor);
    }
}
