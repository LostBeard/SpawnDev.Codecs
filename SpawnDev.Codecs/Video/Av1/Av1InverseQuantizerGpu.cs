// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable AV1 inverse quantizer (dequantizer). Mirror of the
// dequantization step performed by the AV1 decoder before the inverse
// transform: out[i] = quantized[i] * dequant[i] where dequant is the
// per-band step size from Av1DequantTables.
//
// Per-coefficient parallel: each thread reads one quantized coef and
// multiplies by its dequant value (DC for index 0, AC for indices > 0).
// True parallel-per-element across all 6 ILGPU backends.
//
// Pairs with Av1ForwardQuantizerGpu (encoder-side). Together they
// complete the AV1 quant + dequant primitives needed for the
// transform-coefficient round trip.

using ILGPU;

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>
/// GPU-callable AV1 inverse quantizer. Per-coefficient dequantization:
/// out[0] = quantized[0] * dcQ; out[i] = quantized[i] * acQ for i &gt; 0.
/// </summary>
public static class Av1InverseQuantizerGpu
{
    /// <summary>
    /// Dequantize one coefficient at index <paramref name="i"/>:
    /// <c>output[i] = quantized[i] * (i == 0 ? dcQ : acQ)</c>. Caller
    /// dispatches blockSize threads.
    /// </summary>
    /// <param name="quantized">Input quantized coefficients (length blockSize).</param>
    /// <param name="quantBase">Base offset.</param>
    /// <param name="output">Output dequantized coefficients (length blockSize).</param>
    /// <param name="outBase">Base offset.</param>
    /// <param name="dcQ">DC-band dequant step size.</param>
    /// <param name="acQ">AC-band dequant step size.</param>
    /// <param name="i">Coefficient index in [0, blockSize).</param>
    public static void DequantizeAt(
        ArrayView<int> quantized, long quantBase,
        ArrayView<int> output, long outBase,
        int dcQ, int acQ, int i)
    {
        int q = i == 0 ? dcQ : acQ;
        output[outBase + i] = quantized[quantBase + i] * q;
    }
}
