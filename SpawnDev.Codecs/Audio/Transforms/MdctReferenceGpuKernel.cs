// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Wrapper kernel that drives MdctReferenceGpu.Coefficient through
// ILGPU. One thread per output coefficient = parallel forward MDCT.
// Cross-backend compatible.

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs.Audio.Transforms;

/// <summary>
/// Batched ILGPU kernel for the O(N^2) forward MDCT. Threads per
/// dispatch = blockCount * N (one thread per output coefficient).
/// </summary>
public sealed class MdctReferenceGpuKernel : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, int, int> _kernel;

    /// <summary>Compile.</summary>
    public MdctReferenceGpuKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, int, int>(MdctKernel);
    }

    /// <summary>
    /// Run forward MDCT on <paramref name="blockCount"/> independent blocks.
    /// Each block has 2*<paramref name="n"/> input samples and produces
    /// <paramref name="n"/> output coefficients.
    /// </summary>
    public void Run(ArrayView<float> input, ArrayView<float> output, int blockCount, int n)
    {
        if (blockCount < 0) throw new ArgumentOutOfRangeException(nameof(blockCount));
        if (blockCount == 0) return;
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));
        if (input.Length < blockCount * 2L * n)
            throw new ArgumentException("input too short.", nameof(input));
        if (output.Length < blockCount * (long)n)
            throw new ArgumentException("output too short.", nameof(output));
        _kernel(blockCount * n, input, output, blockCount, n);
    }

    private static void MdctKernel(
        Index1D threadIdx,
        ArrayView<float> input,
        ArrayView<float> output,
        int blockCount,
        int n)
    {
        int tid = threadIdx;
        int total = blockCount * n;
        if (tid >= total) return;
        int blockIdx = tid / n;
        int k = tid - blockIdx * n;
        long inBase = (long)blockIdx * 2 * n;
        long outBase = (long)blockIdx * n;
        output[outBase + k] = MdctReferenceGpu.Coefficient(input, inBase, n, k);
    }

    /// <summary>Release kernel resources.</summary>
    public void Dispose() { /* auto-grouped */ }
}
