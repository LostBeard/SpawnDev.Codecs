// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Test/integration kernel that exercises Vp9DequantBlockGpu by
// running a single-block dequantize on the accelerator. Single-thread
// dispatch.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// Drives <see cref="Vp9DequantBlockGpu.DequantizeBlock"/> on the
/// accelerator for one block per dispatch.
/// </summary>
public sealed class Vp9DequantBlockGpuTestKernel : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<short>, int, int, int> _kernel;

    /// <summary>Compile.</summary>
    public Vp9DequantBlockGpuTestKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<short>, int, int, int>(DequantizeKernel);
    }

    /// <summary>Dequantize a single block in place.</summary>
    public void Run(ArrayView<short> coefs, int count, int dcQ, int acQ)
    {
        if (count <= 0) throw new ArgumentOutOfRangeException(nameof(count));
        if (coefs.Length < count) throw new ArgumentException("coefs too short.", nameof(coefs));
        _kernel(1, coefs, count, dcQ, acQ);
    }

    private static void DequantizeKernel(
        Index1D _,
        ArrayView<short> coefs,
        int count,
        int dcQ,
        int acQ)
    {
        Vp9DequantBlockGpu.DequantizeBlock(coefs, 0, count, dcQ, acQ);
    }

    /// <summary>Release kernel resources.</summary>
    public void Dispose() { /* auto-grouped */ }
}
