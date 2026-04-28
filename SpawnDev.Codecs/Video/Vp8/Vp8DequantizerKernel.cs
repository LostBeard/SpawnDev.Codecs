// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// ILGPU kernel for the VP8 inverse (de)quantizer. Bit-exact mirror of
// the per-block dequantization libvpx applies before the inverse
// transform: coef[0] *= dcQ, coef[i] *= acQ for i in 1..15.
//
// Quantizer values are passed as 2 ArrayViews indexed by block (so
// each block can have its own DC/AC dequantizer). The decoder's
// per-MB dequant pre-computes these per plane (Y1, Y2, UV).

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>
/// Batched ILGPU dequantizer for VP8. One thread per 4x4 coef block.
/// Per-block DC + AC dequantizer values supplied as parallel
/// ArrayViews of length blockCount.
/// </summary>
public sealed class Vp8DequantizerKernel : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<short>, ArrayView<short>, ArrayView<short>, int> _kernel;

    /// <summary>Compile the kernel onto <paramref name="accelerator"/>.</summary>
    public Vp8DequantizerKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<short>, ArrayView<short>, ArrayView<short>, int>(DequantizeKernel);
    }

    /// <summary>
    /// Dequantize <paramref name="blockCount"/> coef blocks in place.
    /// </summary>
    /// <param name="coefs">block-major coefs (input + output), 16 shorts per block.</param>
    /// <param name="dcQ">per-block DC dequantizer (length = blockCount).</param>
    /// <param name="acQ">per-block AC dequantizer (length = blockCount).</param>
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

    private static void DequantizeKernel(
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

        // Coef 0 uses DC dequant; 1..15 use AC. Multiply happens at int
        // and is wrapped back to short - matches libvpx behavior on
        // coefficient overflow (rare in practice; bitstream-bounded).
        coefs[cBase + 0] = (short)(coefs[cBase + 0] * dq);
        for (int i = 1; i < 16; i++)
            coefs[cBase + i] = (short)(coefs[cBase + i] * aq);
    }

    /// <summary>Release kernel resources. Does NOT dispose the accelerator.</summary>
    public void Dispose() { /* auto-grouped */ }
}
