// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// ILGPU kernel for the VP8 forward quantizer. Bit-exact mirror of
// Vp8ForwardQuantizer.QuantizeBlock - one thread per coef block.
//
// Quantizer values are passed as 4 ArrayViews indexed by block (so
// each block can have its own DC/AC quantizer; the encoder's per-MB
// dequant pre-computes these). Caller responsibility to lay out the
// quant arrays correctly.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>
/// Batched ILGPU forward quantizer for VP8. One thread per 4x4 coef
/// block. Per-block quantizer values (DC + AC) supplied as parallel
/// ArrayViews of length blockCount.
/// </summary>
public sealed class Vp8ForwardQuantizerKernel : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<short>, ArrayView<short>, ArrayView<short>, int> _kernel;

    /// <summary>Compile the kernel onto <paramref name="accelerator"/>.</summary>
    public Vp8ForwardQuantizerKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<short>, ArrayView<short>, ArrayView<short>, int>(QuantizeKernel);
    }

    /// <summary>
    /// Quantize <paramref name="blockCount"/> coef blocks in place.
    /// </summary>
    /// <param name="coefs">block-major coefs (input + output), 16 shorts per block.</param>
    /// <param name="dcQ">per-block DC quantizer (length = blockCount).</param>
    /// <param name="acQ">per-block AC quantizer (length = blockCount).</param>
    public void Run(ArrayView<short> coefs, ArrayView<short> dcQ, ArrayView<short> acQ, int blockCount)
    {
        if (blockCount < 0) throw new ArgumentOutOfRangeException(nameof(blockCount));
        if (blockCount == 0) return;
        if (coefs.Length < blockCount * 16L)
            throw new ArgumentException("coefs must hold blockCount*16 shorts.", nameof(coefs));
        if (dcQ.Length < blockCount)
            throw new ArgumentException("dcQ must hold blockCount shorts.", nameof(dcQ));
        if (acQ.Length < blockCount)
            throw new ArgumentException("acQ must hold blockCount shorts.", nameof(acQ));
        _kernel(blockCount, coefs, dcQ, acQ, blockCount);
    }

    /// <summary>Kernel body. One thread per 4x4 coef block.</summary>
    private static void QuantizeKernel(
        Index1D blockIdx,
        ArrayView<short> coefs,
        ArrayView<short> dcQ,
        ArrayView<short> acQ,
        int blockCount)
    {
        int idx = blockIdx;
        if (idx >= blockCount) return;
        long cBase = (long)idx * 16;
        int dq = dcQ[idx];
        int aq = acQ[idx];
        if (dq <= 0) dq = 1;
        if (aq <= 0) aq = 1;

        // Coef 0 uses DC quantizer; 1..15 use AC.
        coefs[cBase + 0] = RoundedDivide(coefs[cBase + 0], dq);
        for (int i = 1; i < 16; i++)
            coefs[cBase + i] = RoundedDivide(coefs[cBase + i], aq);
    }

    /// <summary>
    /// Round-half-toward-zero division matching Vp8ForwardQuantizer.RoundedDivide.
    /// Returns short to match libvpx's quantized output type.
    /// </summary>
    private static short RoundedDivide(short value, int divisor)
    {
        if (value >= 0)
            return (short)((value + divisor / 2) / divisor);
        return (short)(-(((-value) + divisor / 2) / divisor));
    }

    /// <summary>Release kernel resources. Does NOT dispose the accelerator.</summary>
    public void Dispose() { /* auto-grouped */ }
}
