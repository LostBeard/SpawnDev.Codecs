// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Test/integration kernel that exercises Vp9ForwardQuantizerGpu by
// quantizing a single block on the accelerator. Single-thread
// dispatch; the entire block-quantize runs in one GPU thread.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// Drives <see cref="Vp9ForwardQuantizerGpu.QuantizeBlock"/> on the
/// accelerator for one block per dispatch.
/// </summary>
public sealed class Vp9ForwardQuantizerGpuTestKernel : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<int>, int, int, int> _kernel;

    /// <summary>Compile.</summary>
    public Vp9ForwardQuantizerGpuTestKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, int, int, int>(QuantizeKernel);
    }

    /// <summary>Quantize a single block in place.</summary>
    public void Run(ArrayView<int> coefs, int count, int dcQ, int acQ)
    {
        if (count <= 0) throw new ArgumentOutOfRangeException(nameof(count));
        if (coefs.Length < count) throw new ArgumentException("coefs too short.", nameof(coefs));
        if (dcQ <= 0) throw new ArgumentOutOfRangeException(nameof(dcQ));
        if (acQ <= 0) throw new ArgumentOutOfRangeException(nameof(acQ));
        _kernel(1, coefs, count, dcQ, acQ);
    }

    private static void QuantizeKernel(
        Index1D _,
        ArrayView<int> coefs,
        int count,
        int dcQ,
        int acQ)
    {
        Vp9ForwardQuantizerGpu.QuantizeBlock(coefs, 0, count, dcQ, acQ);
    }

    /// <summary>Release kernel resources.</summary>
    public void Dispose() { /* auto-grouped */ }
}
