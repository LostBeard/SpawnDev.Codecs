// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 forward quantizer - encoder-side counterpart of the AV1
// dequantization tables. Naive truncation-toward-zero quantizer.
//
// Production AV1 encoders use libaom's vp9_quantize-derived RD-optimized
// quantizer with zbin / round / dead-zone adjustments per (qIndex, plane,
// is_inter) tuple. This naive version is correct but compresses worse;
// upgrade path: port libaom av1/encoder/av1_quantize.c quantize_b_helper.

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>AV1 forward quantizer (naive truncation).</summary>
public static class Av1ForwardQuantizer
{
    /// <summary>
    /// Quantize a coefficient block in-place. <paramref name="dcQ"/> applies
    /// to coefs[0]; <paramref name="acQ"/> applies to coefs[1..].
    /// </summary>
    public static void QuantizeBlock(Span<int> coefs, int dcQ, int acQ)
    {
        if (coefs.Length < 1) throw new ArgumentException("coefs must hold >= 1 entry", nameof(coefs));
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
