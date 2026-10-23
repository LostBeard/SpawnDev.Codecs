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
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, int, int, int> _kernel;

    /// <summary>
    /// Cosine terms of one launch: a block costs N * 2N terms, so blocks go in groups of
    /// max(1, TermsPerLaunch / (2 N^2)) per launch and no launch grows with the block count.
    /// </summary>
    public const long TermsPerLaunch = 1L << 26;

    /// <summary>Blocks of one launch for block size <paramref name="n"/>.</summary>
    public static int BlocksPerLaunch(int n) => (int)Math.Max(1, TermsPerLaunch / (2L * n * n));


    /// <summary>Compile.</summary>
    public ImdctReferenceGpuKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, int, int, int>(ImdctReferenceKernelBody);
    }

    /// <summary>
    /// Run inverse MDCT on <paramref name="blockCount"/> independent blocks.
    /// Each block has <paramref name="n"/> input coefficients and produces
    /// 2*<paramref name="n"/> output samples.
    /// </summary>
    public async Task RunAsync(ArrayView<float> input, ArrayView<float> output, int blockCount, int n)
    {
        if (blockCount < 0) throw new ArgumentOutOfRangeException(nameof(blockCount));
        if (blockCount == 0) return;
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));
        if (input.Length < blockCount * (long)n)
            throw new ArgumentException("input too short.", nameof(input));
        if (output.Length < blockCount * 2L * n)
            throw new ArgumentException("output too short.", nameof(output));
        // Groups of BlocksPerLaunch(n) blocks, one thread per output value, paced; completes when done.
        int group = BlocksPerLaunch(n);
        var pacer = new LaunchPacer(_accelerator);
        for (int first = 0; first < blockCount; first += group)
        {
            int blocks = Math.Min(group, blockCount - first);
            _kernel(checked(blocks * 2 * n), input, output, blocks, n, first);
            await pacer.LaunchedAsync();
        }
        await _accelerator.SynchronizeAsync();
    }

    private static void ImdctReferenceKernelBody(
        Index1D threadIdx,
        ArrayView<float> input,
        ArrayView<float> output,
        int blockCount,
        int n,
        int firstBlock)
    {
        int tid = threadIdx;
        int total = blockCount * 2 * n;
        if (tid >= total) return;
        int twoN = 2 * n;
        int local = tid / twoN;
        int idx = tid - local * twoN;
        int blockIdx = firstBlock + local;
        long inBase = (long)blockIdx * n;
        long outBase = (long)blockIdx * twoN;
        output[outBase + idx] = ImdctReferenceGpu.Sample(input, inBase, n, idx);
    }

    /// <summary>Release kernel resources.</summary>
    public void Dispose() { /* auto-grouped */ }
}
