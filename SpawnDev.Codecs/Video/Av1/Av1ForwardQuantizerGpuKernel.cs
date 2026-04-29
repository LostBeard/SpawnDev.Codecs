// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Test/integration kernel that drives the
// Av1ForwardQuantizerGpu.QuantizeBlock static helper through ILGPU.
// Verifies the helper composes inside a kernel.
//
// One thread per coefficient block. Per-block dcQ + acQ supplied as
// parallel arrays of length blockCount.

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>
/// Batched ILGPU kernel that drives
/// <see cref="Av1ForwardQuantizerGpu.QuantizeBlock"/>.
/// </summary>
public sealed class Av1ForwardQuantizerGpuKernel : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, int, int> _kernel;

    /// <summary>Compile.</summary>
    public Av1ForwardQuantizerGpuKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, int, int>(QuantizeKernel);
    }

    /// <summary>Run on <paramref name="blockCount"/> coefficient blocks of <paramref name="coefsPerBlock"/> ints each.</summary>
    public void Run(ArrayView<int> coefs, ArrayView<int> dcQ, ArrayView<int> acQ, int blockCount, int coefsPerBlock)
    {
        if (blockCount < 0) throw new ArgumentOutOfRangeException(nameof(blockCount));
        if (blockCount == 0) return;
        if (coefsPerBlock <= 0) throw new ArgumentOutOfRangeException(nameof(coefsPerBlock));
        if (coefs.Length < blockCount * (long)coefsPerBlock)
            throw new ArgumentException($"coefs must hold at least blockCount*coefsPerBlock ints (got {coefs.Length}).", nameof(coefs));
        if (dcQ.Length < blockCount)
            throw new ArgumentException("dcQ must hold at least blockCount ints.", nameof(dcQ));
        if (acQ.Length < blockCount)
            throw new ArgumentException("acQ must hold at least blockCount ints.", nameof(acQ));
        _kernel(blockCount, coefs, dcQ, acQ, blockCount, coefsPerBlock);
    }

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
        Av1ForwardQuantizerGpu.QuantizeBlock(coefs, cBase, coefsPerBlock, dcQ[idx], acQ[idx]);
    }

    /// <summary>Release kernel resources.</summary>
    public void Dispose() { /* auto-grouped */ }
}
