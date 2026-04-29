// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Wrapper kernel for SilkSigmoidGpu. One thread per input value -
// computes silk_sigm_Q15 in parallel.

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>Batched ILGPU kernel for SILK sigmoid.</summary>
public sealed class SilkSigmoidGpuKernel : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, int> _kernel;

    /// <summary>Compile.</summary>
    public SilkSigmoidGpuKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, ArrayView<int>, int>(SigmoidKernel);
    }

    /// <summary>Compute sigmoid(in[i]) for i in [0, count).</summary>
    public void Run(ArrayView<int> input, ArrayView<int> output, int count)
    {
        if (count < 0) throw new ArgumentOutOfRangeException(nameof(count));
        if (count == 0) return;
        _kernel(count, input, output, count);
    }

    private static void SigmoidKernel(
        Index1D threadIdx,
        ArrayView<int> input,
        ArrayView<int> output,
        int count)
    {
        int i = threadIdx;
        if (i >= count) return;
        output[i] = SilkSigmoidGpu.SigmQ15(input[i]);
    }

    /// <summary>Release kernel resources.</summary>
    public void Dispose() { /* auto-grouped */ }
}
