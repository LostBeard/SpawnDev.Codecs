// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 forward quantizer, GPU-callable form for in-kernel reuse.
// Bit-exact mirror of Vp9ForwardQuantizer.QuantizeBlock.
//
// Naive truncation-toward-zero quantizer with rounding-half-up:
//   coefs[0] uses the DC dequantizer value
//   coefs[1..N-1] uses the AC dequantizer value
//   value /= q (positive) or -((-value) /= q) (negative)
//
// Production VP9 encoders use rate-distortion-optimized quantization;
// this is the v1 floor and exists so the upcoming
// Vp9FrameSequentialEncodeKernel can quantize a per-block coefficient
// vector inline.
//
// The integer division here uses positive divisors only (dcQ / acQ
// are derived from the 256-entry DC/AC quantizer tables which start
// at 4, so always > 0). The numerator can be negative; the explicit
// sign extraction keeps the result truncate-toward-zero on every
// ILGPU backend (see feedback_ilgpu_int_div_two_breaks_on_negatives
// for why we never trust the implicit signed-int division
// semantics).

using ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// GPU-callable VP9 forward quantizer helper. Bit-exact mirror of
/// <see cref="Vp9ForwardQuantizer"/> for in-kernel use.
/// </summary>
public static class Vp9ForwardQuantizerGpu
{
    /// <summary>
    /// Quantize a coefficient block in place. Reads
    /// <paramref name="count"/> ints starting at
    /// <paramref name="coefBase"/> in <paramref name="coefs"/>;
    /// scan position 0 uses <paramref name="dcQ"/>; positions 1..N-1
    /// use <paramref name="acQ"/>.
    /// </summary>
    public static void QuantizeBlock(
        ArrayView<int> coefs, long coefBase, int count,
        int dcQ, int acQ)
    {
        coefs[coefBase] = RoundedDivide(coefs[coefBase], dcQ);
        for (int i = 1; i < count; i++)
            coefs[coefBase + i] = RoundedDivide(coefs[coefBase + i], acQ);
    }

    /// <summary>
    /// Round-half-up signed integer division. Mirrors C#
    /// <c>(value + divisor/2) / divisor</c> for positive values and
    /// <c>-(((-value) + divisor/2) / divisor)</c> for negatives.
    /// Caller guarantees <paramref name="divisor"/> &gt; 0.
    /// </summary>
    private static int RoundedDivide(int value, int divisor)
    {
        int half = divisor >> 1; // divisor > 0 so >> 1 == /2.
        if (value >= 0) return (value + half) / divisor;
        return -(((-value) + half) / divisor);
    }
}
