// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Test/integration kernel that exercises Vp9Idct16x16Gpu by running
// a single-block iDCT 16x16 + residual add on the accelerator.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// Drives <see cref="Vp9Idct16x16Gpu.Idct16x16"/> on the accelerator
/// for one 16x16 block per dispatch.
/// </summary>
public sealed class Vp9Idct16x16GpuTestKernel : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<short>, ArrayView<byte>, ArrayView<int>, int> _kernel;

    /// <summary>Compile.</summary>
    public Vp9Idct16x16GpuTestKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<short>, ArrayView<byte>, ArrayView<int>, int>(IdctKernel);
    }

    /// <summary>iDCT-add a single 16x16 block in place.</summary>
    public void Run(
        ArrayView<short> coefs,
        ArrayView<byte> dest,
        ArrayView<int> scratch,
        int destStride)
    {
        if (coefs.Length < 256) throw new ArgumentException("coefs must hold 256 shorts.", nameof(coefs));
        if (dest.Length < 16L * destStride) throw new ArgumentException("dest too short.", nameof(dest));
        if (scratch.Length < 256) throw new ArgumentException("scratch must hold 256 ints.", nameof(scratch));
        _kernel(1, coefs, dest, scratch, destStride);
    }

    private static void IdctKernel(
        Index1D _,
        ArrayView<short> coefs,
        ArrayView<byte> dest,
        ArrayView<int> scratch,
        int destStride)
    {
        Vp9Idct16x16Gpu.Idct16x16(coefs, 0, dest, 0, destStride, scratch);
    }

    /// <summary>Release kernel resources.</summary>
    public void Dispose() { /* auto-grouped */ }
}
