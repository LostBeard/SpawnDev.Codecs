// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Strided source plane -> macroblock-aligned working plane, entirely on the accelerator.
// One thread per DESTINATION byte: copies the source byte when (row, col) is inside the
// source picture, writes 0 for the right/bottom pad. This replaces the host per-row copy loops
// (UploadPaddedPlane, the batch hostY/hostU/hostV builders) and the MemSetToZero pre-clear, so
// the source can be uploaded once, as-is, from managed memory, a JS TypedArray or an
// ArrayView1D, and the host does no per-pixel work.

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs;

/// <summary>
/// Pads a strided plane (<c>w x h</c>, row stride <c>srcStride</c>) into a dense
/// <c>destStride x destRows</c> working plane, zero-filling the pad.
/// </summary>
public sealed class PlanePadKernel : IDisposable
{
    private readonly Action<
        Index1D,
        ArrayView<byte>, long, int, int, int,
        ArrayView<byte>, int> _kernel;

    /// <summary>Compile.</summary>
    public PlanePadKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<byte>, long, int, int, int,
            ArrayView<byte>, int>(PadKernel);
    }

    /// <summary>
    /// Dispatch <c>destStride * destRows</c> threads. Destination byte <c>(r, c)</c> becomes
    /// <c>src[srcBase + r * srcStride + c]</c> when <c>r &lt; h</c> and <c>c &lt; w</c>, else 0.
    /// </summary>
    public void Run(
        ArrayView<byte> src, long srcBase, int srcStride, int w, int h,
        ArrayView<byte> dest, int destStride, int destRows)
    {
        if (w <= 0) throw new ArgumentOutOfRangeException(nameof(w));
        if (h <= 0) throw new ArgumentOutOfRangeException(nameof(h));
        if (srcStride < w) throw new ArgumentException("srcStride must be >= w", nameof(srcStride));
        if (destStride < w) throw new ArgumentException("destStride must be >= w", nameof(destStride));
        if (destRows < h) throw new ArgumentException("destRows must be >= h", nameof(destRows));
        long total = (long)destStride * destRows;
        if (total > int.MaxValue) throw new ArgumentOutOfRangeException(nameof(destRows));
        if (dest.Length < total) throw new ArgumentException("dest is smaller than destStride * destRows", nameof(dest));
        if (src.Length < srcBase + (long)(h - 1) * srcStride + w)
            throw new ArgumentException("src is smaller than the declared plane", nameof(src));

        _kernel((Index1D)(int)total, src, srcBase, srcStride, w, h, dest, destStride);
    }

    private static void PadKernel(
        Index1D idx,
        ArrayView<byte> src, long srcBase, int srcStride, int w, int h,
        ArrayView<byte> dest, int destStride)
    {
        int i = idx.X;
        int r = i / destStride;
        int c = i - r * destStride;
        // Non-short-circuit & into a bool local: the WGSL/GLSL codegen has dropped threads on
        // compound predicates written inline in the branch condition.
        bool inside = (r < h) & (c < w);
        byte value = 0;
        if (inside)
            value = src[srcBase + (long)r * srcStride + c];
        dest[i] = value;
    }

    /// <summary>No-op (kernels are owned by the accelerator).</summary>
    public void Dispose() { }
}
