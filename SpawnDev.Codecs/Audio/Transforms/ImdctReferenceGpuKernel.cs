// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Wrapper kernel for ImdctReferenceGpu. Threads per dispatch =
// blockCount * 2N (one thread per output time-domain sample).

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs.Audio.Transforms;

/// <summary>
/// Batched ILGPU kernel for the O(N^2) inverse MDCT.
/// </summary>
public sealed class ImdctReferenceGpuKernel : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, int, int> _kernel;

    /// <summary>Compile.</summary>
    public ImdctReferenceGpuKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, int, int>(ImdctKernel);
    }

    /// <summary>
    /// Run inverse MDCT on <paramref name="blockCount"/> independent blocks.
    /// Each block has <paramref name="n"/> input coefficients and produces
    /// 2*<paramref name="n"/> output samples.
    /// </summary>
    public void Run(ArrayView<float> input, ArrayView<float> output, int blockCount, int n)
    {
        if (blockCount < 0) throw new ArgumentOutOfRangeException(nameof(blockCount));
        if (blockCount == 0) return;
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));
        if (input.Length < blockCount * (long)n)
            throw new ArgumentException("input too short.", nameof(input));
        if (output.Length < blockCount * 2L * n)
            throw new ArgumentException("output too short.", nameof(output));
        _kernel(blockCount * 2 * n, input, output, blockCount, n);
    }

    private static void ImdctKernel(
        Index1D threadIdx,
        ArrayView<float> input,
        ArrayView<float> output,
        int blockCount,
        int n)
    {
        int tid = threadIdx;
        int total = blockCount * 2 * n;
        if (tid >= total) return;
        int twoN = 2 * n;
        int blockIdx = tid / twoN;
        int idx = tid - blockIdx * twoN;
        long inBase = (long)blockIdx * n;
        long outBase = (long)blockIdx * twoN;
        output[outBase + idx] = ImdctReferenceGpu.Sample(input, inBase, n, idx);
    }

    /// <summary>Release kernel resources.</summary>
    public void Dispose() { /* auto-grouped */ }
}
