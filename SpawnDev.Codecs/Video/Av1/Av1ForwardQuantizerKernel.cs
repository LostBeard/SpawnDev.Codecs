// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// ILGPU kernel for the AV1 forward quantizer. Bit-exact mirror of
// Av1ForwardQuantizer.QuantizeBlock - one thread per coefficient block.
//
// Each block has an arbitrary coefficient count (16 for 4x4, 64 for
// 8x8, 256 for 16x16, 1024 for 32x32, etc.). Per-block quantizer values
// (DC + AC) are supplied as parallel ArrayViews of length blockCount.
// Coef[0] of each block uses dcQ; the rest use acQ. Naive truncation
// quantizer matches the CPU reference exactly (see
// Av1ForwardQuantizer.cs for the future RD-optimized upgrade path).

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>
/// Batched ILGPU forward quantizer for AV1. One thread per coefficient
/// block. Per-block quantizer values (DC + AC) supplied as parallel
/// ArrayViews of length blockCount. Naive truncation quantizer.
/// </summary>
public sealed class Av1ForwardQuantizerKernel : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, int, int> _kernel;

    /// <summary>Compile the kernel onto <paramref name="accelerator"/>.</summary>
    public Av1ForwardQuantizerKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, int, int>(QuantizeKernel);
    }

    /// <summary>
    /// Quantize <paramref name="blockCount"/> coefficient blocks in
    /// place. Each block holds <paramref name="coefsPerBlock"/> ints.
    /// </summary>
    /// <param name="coefs">Block-major coefficient buffer (input + output).
    /// Length must be at least blockCount * coefsPerBlock.</param>
    /// <param name="dcQ">Per-block DC quantizer (length = blockCount). Must be > 0.</param>
    /// <param name="acQ">Per-block AC quantizer (length = blockCount). Must be > 0.</param>
    /// <param name="blockCount">Number of coefficient blocks to quantize.</param>
    /// <param name="coefsPerBlock">Coefficients per block (16 for 4x4, 64 for 8x8, 256 for 16x16, etc.).</param>
    public void Run(ArrayView<int> coefs, ArrayView<int> dcQ, ArrayView<int> acQ, int blockCount, int coefsPerBlock)
    {
        if (blockCount < 0) throw new ArgumentOutOfRangeException(nameof(blockCount));
        if (blockCount == 0) return;
        if (coefsPerBlock <= 0) throw new ArgumentOutOfRangeException(nameof(coefsPerBlock), "must be > 0");
        if (coefs.Length < blockCount * (long)coefsPerBlock)
            throw new ArgumentException($"coefs must hold at least blockCount*coefsPerBlock ints (got {coefs.Length}).", nameof(coefs));
        if (dcQ.Length < blockCount)
            throw new ArgumentException("dcQ must hold at least blockCount ints.", nameof(dcQ));
        if (acQ.Length < blockCount)
            throw new ArgumentException("acQ must hold at least blockCount ints.", nameof(acQ));
        _kernel(blockCount, coefs, dcQ, acQ, blockCount, coefsPerBlock);
    }

    /// <summary>Kernel body. One thread per coefficient block.</summary>
    private static void QuantizeKernel(
        Index1D blockIdx,
        ArrayView<int> coefs,
        ArrayView<int> dcQ,
        ArrayView<int> acQ,
        int blockCount,
        int coefsPerBlock)
    {
        int idx = blockIdx;
        if (idx >= blockCount) return;
        long cBase = (long)idx * coefsPerBlock;

        int dq = dcQ[idx];
        int aq = acQ[idx];

        // Coef 0 uses DC quantizer; 1..N-1 use AC. Matches CPU reference.
        coefs[cBase + 0] = RoundedDivide(coefs[cBase + 0], dq);
        for (int i = 1; i < coefsPerBlock; i++)
            coefs[cBase + i] = RoundedDivide(coefs[cBase + i], aq);
    }

    /// <summary>
    /// Symmetric rounded division matching <see cref="Av1ForwardQuantizer"/>.RoundedDivide.
    /// Rounds toward zero on negative inputs, toward +infinity on positive.
    /// </summary>
    private static int RoundedDivide(int value, int divisor)
    {
        if (value >= 0) return (value + divisor / 2) / divisor;
        return -(((-value) + divisor / 2) / divisor);
    }

    /// <summary>Release kernel resources. Does NOT dispose the accelerator.</summary>
    public void Dispose() { /* auto-grouped */ }
}
