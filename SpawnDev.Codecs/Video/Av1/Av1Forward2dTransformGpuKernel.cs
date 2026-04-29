// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Test/integration kernel that drives Av1Forward2dTransformGpu through
// ILGPU. One thread per 2D transform block; supports Tx8x8 and
// Tx16x16 DCT_DCT.

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>
/// Batched ILGPU kernel that drives
/// <see cref="Av1Forward2dTransformGpu"/>.Forward8x8DctDct or
/// .Forward16x16DctDct based on the txSize parameter.
/// </summary>
public sealed class Av1Forward2dTransformGpuKernel : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<short>, ArrayView<int>, ArrayView<int>, int, int> _kernel;

    /// <summary>Compile.</summary>
    public Av1Forward2dTransformGpuKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<short>, ArrayView<int>, ArrayView<int>, int, int>(Forward2dKernel);
    }

    /// <summary>
    /// Run the 2D forward DCT_DCT on <paramref name="blockCount"/>
    /// independent blocks. Each block is <paramref name="txSize"/>=1
    /// (Tx8x8, 64 elements) or 2 (Tx16x16, 256 elements).
    /// <paramref name="scratch"/> must hold blockCount * coefsPerBlock
    /// ints (one scratch region per block).
    /// </summary>
    public void Run(ArrayView<short> input, ArrayView<int> output, ArrayView<int> scratch, int blockCount, int txSize)
    {
        if (blockCount < 0) throw new ArgumentOutOfRangeException(nameof(blockCount));
        if (blockCount == 0) return;
        if (txSize != 1 && txSize != 2)
            throw new ArgumentOutOfRangeException(nameof(txSize), "Only Tx8x8 (1) and Tx16x16 (2) supported in v1.");
        int n = txSize == 1 ? 64 : 256;
        if (input.Length < blockCount * (long)n)
            throw new ArgumentException("input too short.", nameof(input));
        if (output.Length < blockCount * (long)n)
            throw new ArgumentException("output too short.", nameof(output));
        if (scratch.Length < blockCount * (long)n)
            throw new ArgumentException("scratch too short.", nameof(scratch));
        _kernel(blockCount, input, output, scratch, blockCount, txSize);
    }

    private static void Forward2dKernel(
        Index1D blockIdx,
        ArrayView<short> input,
        ArrayView<int> output,
        ArrayView<int> scratch,
        int blockCount,
        int txSize)
    {
        int idx = blockIdx;
        if (idx >= blockCount) return;
        int n = txSize == 1 ? 64 : 256;
        long inBase = (long)idx * n;
        long outBase = (long)idx * n;
        long scratchBase = (long)idx * n;

        if (txSize == 1)
        {
            Av1Forward2dTransformGpu.Forward8x8DctDct(input, inBase, output, outBase, scratch, scratchBase);
        }
        else
        {
            Av1Forward2dTransformGpu.Forward16x16DctDct(input, inBase, output, outBase, scratch, scratchBase);
        }
    }

    /// <summary>Release kernel resources.</summary>
    public void Dispose() { /* auto-grouped */ }
}
