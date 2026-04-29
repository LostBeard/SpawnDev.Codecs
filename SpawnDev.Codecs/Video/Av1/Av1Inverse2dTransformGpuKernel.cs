// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Test/integration kernel that drives Av1Inverse2dTransformGpu through
// ILGPU.

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>
/// Batched ILGPU kernel that drives
/// <see cref="Av1Inverse2dTransformGpu"/>.Inverse8x8DctDct or
/// .Inverse16x16DctDct.
/// </summary>
public sealed class Av1Inverse2dTransformGpuKernel : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, int, int> _kernel;

    /// <summary>Compile.</summary>
    public Av1Inverse2dTransformGpuKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, int, int>(Inverse2dKernel);
    }

    /// <summary>
    /// Scratch ints per block. Tx8x8 needs 64; Tx16x16 needs 272 (256
    /// row-pass output + 16 column-gather buffer to avoid overwriting
    /// prior column's residual scatter).
    /// </summary>
    public static int ScratchPerBlock(int txSize) => txSize == 1 ? 64 : 272;

    /// <summary>Run on <paramref name="blockCount"/> blocks; txSize 1 = Tx8x8, 2 = Tx16x16.</summary>
    public void Run(ArrayView<int> coefs, ArrayView<int> residual, ArrayView<int> scratch, int blockCount, int txSize)
    {
        if (blockCount < 0) throw new ArgumentOutOfRangeException(nameof(blockCount));
        if (blockCount == 0) return;
        if (txSize != 1 && txSize != 2)
            throw new ArgumentOutOfRangeException(nameof(txSize));
        int n = txSize == 1 ? 64 : 256;
        int scratchN = ScratchPerBlock(txSize);
        if (coefs.Length < blockCount * (long)n)
            throw new ArgumentException("coefs too short.", nameof(coefs));
        if (residual.Length < blockCount * (long)n)
            throw new ArgumentException("residual too short.", nameof(residual));
        if (scratch.Length < blockCount * (long)scratchN)
            throw new ArgumentException($"scratch must hold blockCount*{scratchN} ints (got {scratch.Length}).", nameof(scratch));
        _kernel(blockCount, coefs, residual, scratch, blockCount, txSize);
    }

    private static void Inverse2dKernel(
        Index1D blockIdx,
        ArrayView<int> coefs,
        ArrayView<int> residual,
        ArrayView<int> scratch,
        int blockCount,
        int txSize)
    {
        int idx = blockIdx;
        if (idx >= blockCount) return;
        int n = txSize == 1 ? 64 : 256;
        int scratchN = txSize == 1 ? 64 : 272;
        long coefBase = (long)idx * n;
        long resBase = (long)idx * n;
        long scratchBase = (long)idx * scratchN;

        if (txSize == 1)
        {
            Av1Inverse2dTransformGpu.Inverse8x8DctDct(coefs, coefBase, residual, resBase, scratch, scratchBase);
        }
        else
        {
            Av1Inverse2dTransformGpu.Inverse16x16DctDct(coefs, coefBase, residual, resBase, scratch, scratchBase);
        }
    }

    /// <summary>Release kernel resources.</summary>
    public void Dispose() { /* auto-grouped */ }
}
