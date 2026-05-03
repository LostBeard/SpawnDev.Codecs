// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Launchable kernel wrapper around Vp8StridedPlanePackGpu.PackAt. Used
// by Vp8KeyframeEncoderGpu.UploadPlane to GPU-resident-ize the strided
// -> packed plane copy that previously ran as a per-row CPU loop.
//
// One thread per output byte. Caller dispatches w * h threads.

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>
/// Launchable kernel for the strided -> packed plane copy. Wraps the
/// GPU-callable <see cref="Vp8StridedPlanePackGpu.PackAt"/>.
/// </summary>
public sealed class Vp8StridedPlanePackKernel : IDisposable
{
    private readonly Action<
        Index1D,
        ArrayView<byte>, long, int,
        ArrayView<byte>, long, int> _kernel;

    /// <summary>Compile.</summary>
    public Vp8StridedPlanePackKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<byte>, long, int,
            ArrayView<byte>, long, int>(PackKernel);
    }

    /// <summary>
    /// Dispatch <c>w * h</c> threads. Each thread copies one byte from
    /// the strided source plane at <c>(r * stride + c)</c> to the packed
    /// destination plane at <c>(r * w + c)</c>.
    /// </summary>
    public void Run(
        ArrayView<byte> strided, long stridedBase, int stride,
        ArrayView<byte> packed, long packedBase, int w, int h)
    {
        if (w <= 0) throw new ArgumentOutOfRangeException(nameof(w));
        if (h <= 0) throw new ArgumentOutOfRangeException(nameof(h));
        if (stride < w) throw new ArgumentException("stride must be >= w", nameof(stride));

        long total = (long)w * h;
        _kernel((Index1D)total, strided, stridedBase, stride, packed, packedBase, w);
    }

    private static void PackKernel(
        Index1D idx,
        ArrayView<byte> strided, long stridedBase, int stride,
        ArrayView<byte> packed, long packedBase, int w)
    {
        Vp8StridedPlanePackGpu.PackAt(
            strided, stridedBase, stride,
            packed, packedBase, w,
            idx.X);
    }

    /// <summary>No-op (kernels are owned by the accelerator).</summary>
    public void Dispose() { }
}
