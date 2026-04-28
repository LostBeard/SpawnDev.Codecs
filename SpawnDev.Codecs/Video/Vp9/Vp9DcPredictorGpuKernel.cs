// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Test/integration kernel that exercises Vp9DcPredictorGpu by
// running a single-block DC predict on the accelerator. Single-
// thread dispatch; the full primitive runs in one GPU thread.
//
// The existing Vp9DcPredict4x4Kernel / 8x8Kernel / 16x16Kernel
// dispatch one thread per block across an array of blocks - that's
// the right shape when DC predict is the entire workload. This
// kernel exists for the v3 sequential-encode path where DC predict
// is one step of a per-block pipeline that runs in a single GPU
// thread, so we want a single-block dispatch shape that the
// future Vp9FrameSequentialEncodeKernel can exercise without
// imposing a separate kernel boundary.

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// Drives <see cref="Vp9DcPredictorGpu.Predict"/> on the accelerator
/// for one block per dispatch. Used to verify bit-exact agreement
/// with <see cref="Vp9DcPredictor"/>.
/// </summary>
public sealed class Vp9DcPredictorGpuTestKernel : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, int, int> _kernel;

    /// <summary>Compile.</summary>
    public Vp9DcPredictorGpuTestKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, int, int>(PredictKernel);
    }

    /// <summary>
    /// Run DC predict on a single NxN block. <paramref name="dst"/>
    /// receives the block in row-major layout with stride = N.
    /// </summary>
    public void Run(
        ArrayView<byte> above,
        ArrayView<byte> left,
        ArrayView<byte> dst,
        int n,
        Vp9DcVariant variant)
    {
        if (n != 4 && n != 8 && n != 16 && n != 32)
            throw new ArgumentOutOfRangeException(nameof(n), "n must be 4, 8, 16, or 32");
        if (above.Length < n) throw new ArgumentException("above too short.", nameof(above));
        if (left.Length < n) throw new ArgumentException("left too short.", nameof(left));
        if (dst.Length < (long)n * n) throw new ArgumentException("dst too short.", nameof(dst));
        _kernel(1, above, left, dst, n, (int)variant);
    }

    private static void PredictKernel(
        Index1D _,
        ArrayView<byte> above,
        ArrayView<byte> left,
        ArrayView<byte> dst,
        int n,
        int variant)
    {
        Vp9DcPredictorGpu.Predict(
            above, 0,
            left, 0,
            dst, 0, n,
            n, variant);
    }

    /// <summary>Release kernel resources.</summary>
    public void Dispose() { /* auto-grouped */ }
}
