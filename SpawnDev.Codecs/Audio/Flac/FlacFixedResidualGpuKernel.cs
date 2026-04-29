// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Wrapper kernel that drives FlacFixedResidualGpu. Threads per
// dispatch = residualCount (one thread per residual sample). For a
// subframe with sampleCount samples at order k: residualCount =
// sampleCount - k.

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs.Audio.Flac;

/// <summary>
/// Batched ILGPU kernel for FLAC FIXED residual computation.
/// </summary>
public sealed class FlacFixedResidualGpuKernel : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, int, int> _kernel;

    /// <summary>Compile.</summary>
    public FlacFixedResidualGpuKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, ArrayView<int>, int, int>(ResidualKernel);
    }

    /// <summary>
    /// Compute residual for <paramref name="sampleCount"/> input samples
    /// at FIXED order <paramref name="order"/>. <paramref name="residual"/>
    /// receives <paramref name="sampleCount"/> - <paramref name="order"/>
    /// values.
    /// </summary>
    public void Run(ArrayView<int> samples, ArrayView<int> residual, int sampleCount, int order)
    {
        if (order < 1 || order > 4) throw new ArgumentOutOfRangeException(nameof(order));
        if (sampleCount <= order) throw new ArgumentException("sampleCount must be > order.", nameof(sampleCount));
        int residualCount = sampleCount - order;
        if (samples.Length < sampleCount) throw new ArgumentException("samples too short.", nameof(samples));
        if (residual.Length < residualCount) throw new ArgumentException("residual too short.", nameof(residual));
        _kernel(residualCount, samples, residual, residualCount, order);
    }

    private static void ResidualKernel(
        Index1D threadIdx,
        ArrayView<int> samples,
        ArrayView<int> residual,
        int residualCount,
        int order)
    {
        int ri = threadIdx;
        if (ri >= residualCount) return;
        residual[ri] = FlacFixedResidualGpu.Residual(samples, 0, order, ri);
    }

    /// <summary>Release kernel resources.</summary>
    public void Dispose() { /* auto-grouped */ }
}
