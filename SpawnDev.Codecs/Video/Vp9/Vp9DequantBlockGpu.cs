// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 per-block dequantizer, GPU-callable form for in-kernel reuse.
// Bit-exact mirror of Vp9Dequantizer.DequantizeInPlace.
//
// Vp9DequantKernel is the existing per-coefficient parallel dispatch
// (one thread per coef, one quantizer pair per dispatch). This single-
// block helper is the in-kernel companion for the v3 sequential
// encoder/decoder path: the per-frame kernel iterates blocks
// sequentially, calling DequantizeBlock once per block.
//
// Dequantization is `coef * dequant`, saturated to int16 range.
// VP9 quantized coefficients are int16 and dequantizer values are
// int16 too; the int32 product fits without overflow.

using ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// GPU-callable VP9 per-block dequantizer helper. Bit-exact mirror of
/// <see cref="Vp9Dequantizer.DequantizeInPlace"/> for in-kernel use.
/// </summary>
public static class Vp9DequantBlockGpu
{
    /// <summary>
    /// Dequantize <paramref name="count"/> coefficients in place.
    /// Position 0 (the DC slot) uses <paramref name="dcQ"/>; positions
    /// 1..N-1 use <paramref name="acQ"/>. Coefficients are clamped to
    /// [<see cref="short.MinValue"/>, <see cref="short.MaxValue"/>] -
    /// matches Vp9Dequantizer.DequantizeInPlace's int16 saturation
    /// semantics so the dequantized block feeds the inverse transform
    /// in the same domain it would in the CPU encoder.
    /// </summary>
    public static void DequantizeBlock(
        ArrayView<short> coefs, long coefBase, int count,
        int dcQ, int acQ)
    {
        coefs[coefBase] = SaturatingMul(coefs[coefBase], (short)dcQ);
        for (int i = 1; i < count; i++)
            coefs[coefBase + i] = SaturatingMul(coefs[coefBase + i], (short)acQ);
    }

    private static short SaturatingMul(short coeff, short dequant)
    {
        int product = coeff * dequant;
        if (product > short.MaxValue) return short.MaxValue;
        if (product < short.MinValue) return short.MinValue;
        return (short)product;
    }
}
