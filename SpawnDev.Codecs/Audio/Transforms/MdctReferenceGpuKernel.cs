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
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, int, int, int> _kernel;

    /// <summary>
    /// Cosine terms of one launch: a block costs N * 2N terms, so blocks go in groups of
    /// max(1, TermsPerLaunch / (2 N^2)) per launch and no launch grows with the block count.
    /// </summary>
    public const long TermsPerLaunch = 1L << 26;

    /// <summary>Blocks of one launch for block size <paramref name="n"/>.</summary>
    public static int BlocksPerLaunch(int n) => (int)Math.Max(1, TermsPerLaunch / (2L * n * n));


    /// <summary>Compile.</summary>
    public MdctReferenceGpuKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, int, int, int>(MdctReferenceKernelBody);
    }

    /// <summary>
    /// Run forward MDCT on <paramref name="blockCount"/> independent blocks.
    /// Each block has 2*<paramref name="n"/> input samples and produces
    /// <paramref name="n"/> output coefficients.
    /// </summary>
    public async Task RunAsync(ArrayView<float> input, ArrayView<float> output, int blockCount, int n)
    {
        if (blockCount < 0) throw new ArgumentOutOfRangeException(nameof(blockCount));
        if (blockCount == 0) return;
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));
        if (input.Length < blockCount * 2L * n)
            throw new ArgumentException("input too short.", nameof(input));
        if (output.Length < blockCount * (long)n)
            throw new ArgumentException("output too short.", nameof(output));
        // Groups of BlocksPerLaunch(n) blocks, one thread per output value, paced; completes when done.
        int group = BlocksPerLaunch(n);
        var pacer = new LaunchPacer(_accelerator);
        for (int first = 0; first < blockCount; first += group)
        {
            int blocks = Math.Min(group, blockCount - first);
            _kernel(checked(blocks * n), input, output, blocks, n, first);
            await pacer.LaunchedAsync();
        }
        await _accelerator.SynchronizeAsync();
    }

    private static void MdctReferenceKernelBody(
        Index1D threadIdx,
        ArrayView<float> input,
        ArrayView<float> output,
        int blockCount,
        int n,
        int firstBlock)
    {
        int tid = threadIdx;
        int total = blockCount * n;
        if (tid >= total) return;
        int local = tid / n;
        int k = tid - local * n;
        int blockIdx = firstBlock + local;
        long inBase = (long)blockIdx * 2 * n;
        long outBase = (long)blockIdx * n;
        output[outBase + k] = MdctReferenceGpu.Coefficient(input, inBase, n, k);
    }

    /// <summary>Release kernel resources.</summary>
    public void Dispose() { /* auto-grouped */ }
}
