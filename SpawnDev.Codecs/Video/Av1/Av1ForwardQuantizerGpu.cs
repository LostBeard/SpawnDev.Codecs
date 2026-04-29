// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 forward quantizer, GPU-callable form for in-kernel reuse.
// Bit-exact mirror of Av1ForwardQuantizer.QuantizeBlock (and
// Av1ForwardQuantizerKernel's per-block body).
//
// The Av1FrameSequentialEncodeKernel walker calls this once per
// coefficient block to produce quantized indices for the entropy
// stage. Coef[0] uses dcQ; coefs[1..N-1] use acQ. Naive truncation
// quantizer: matches the CPU reference exactly. Production AV1
// encoders use libaom's RD-optimized quantize_b_helper - upgrade
// path documented in Av1ForwardQuantizer.cs.

using ILGPU;

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>
/// GPU-callable AV1 forward quantizer helper. Bit-exact mirror of
/// <see cref="Av1ForwardQuantizer"/> for in-kernel use.
/// </summary>
public static class Av1ForwardQuantizerGpu
{
    /// <summary>
    /// Quantize a coefficient block in place starting at
    /// <paramref name="coefBase"/>. <paramref name="coefsPerBlock"/>
    /// must match the transform-block coef count (16 for 4x4, 64 for
    /// 8x8, 256 for 16x16, 1024 for 32x32, etc.). Coef[0] uses dcQ;
    /// coefs[1..N-1] use acQ.
    /// </summary>
    public static void QuantizeBlock(
        ArrayView<int> coefs, long coefBase, int coefsPerBlock,
        int dcQ, int acQ)
    {
        coefs[coefBase + 0] = RoundedDivide(coefs[coefBase + 0], dcQ);
        for (int i = 1; i < coefsPerBlock; i++)
            coefs[coefBase + i] = RoundedDivide(coefs[coefBase + i], acQ);
    }

    /// <summary>
    /// Symmetric rounded division matching libaom's quantizer rounding.
    /// Rounds toward +infinity on positive inputs and toward
    /// -infinity on negative inputs (so the magnitude rounds away
    /// from zero by &lt;divisor/2&gt;). Mirrors
    /// <see cref="Av1ForwardQuantizer"/>.RoundedDivide bit-for-bit.
    /// </summary>
    private static int RoundedDivide(int value, int divisor)
    {
        if (value >= 0) return (value + divisor / 2) / divisor;
        return -(((-value) + divisor / 2) / divisor);
    }
}
