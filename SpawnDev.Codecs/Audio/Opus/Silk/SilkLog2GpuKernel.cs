// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Wrapper kernel for SilkLog2Gpu. One thread per input value -
// computes silk_log2lin or silk_lin2log in parallel.

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>Batched ILGPU kernel for SILK log2/lin conversions.</summary>
public sealed class SilkLog2GpuKernel : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, int, int> _kernel;

    /// <summary>Compile.</summary>
    public SilkLog2GpuKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, ArrayView<int>, int, int>(LogKernel);
    }

    /// <summary>
    /// Compute log2lin or lin2log for input[i] in [0, count).
    /// <paramref name="mode"/>: 0 = log2lin (log Q7 -> linear);
    /// 1 = lin2log (linear -> log Q7).
    /// </summary>
    public void Run(ArrayView<int> input, ArrayView<int> output, int count, int mode)
    {
        if (count < 0) throw new ArgumentOutOfRangeException(nameof(count));
        if (count == 0) return;
        if (mode != 0 && mode != 1) throw new ArgumentOutOfRangeException(nameof(mode));
        _kernel(count, input, output, count, mode);
    }

    private static void LogKernel(
        Index1D threadIdx,
        ArrayView<int> input,
        ArrayView<int> output,
        int count,
        int mode)
    {
        int i = threadIdx;
        if (i >= count) return;
        output[i] = mode == 0
            ? SilkLog2Gpu.Log2LinQ7(input[i])
            : SilkLog2Gpu.Lin2LogQ7(input[i]);
    }

    /// <summary>Release kernel resources.</summary>
    public void Dispose() { /* auto-grouped */ }
}
