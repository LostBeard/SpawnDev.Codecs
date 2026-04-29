// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable per-pixel strided -> packed plane copy. Mirror of the
// per-row Span.CopyTo CPU loop that previously ran inside
// Vp8KeyframeEncoderGpu.UploadPlane to strip source-side stride padding
// from a Y/U/V plane before the kernel chain consumes it.
//
// Per-(row, col) parallel: each thread reads one source byte from the
// strided buffer at (r * stride + c) and writes it to the corresponding
// position in the packed buffer at (r * w + c). True parallel-per-element
// across all 6 ILGPU backends.
//
// Caller dispatches w * h threads.

using ILGPU;

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>
/// GPU-callable strided -> packed plane copy. Mirror of the per-row
/// stride-strip CPU loop inside <see cref="Vp8KeyframeEncoderGpu"/>.
/// </summary>
public static class Vp8StridedPlanePackGpu
{
    /// <summary>
    /// Compute one packed output byte at thread index <paramref name="threadIdx"/>
    /// in [0, w * h). Maps r = threadIdx / w and c = threadIdx % w.
    /// </summary>
    /// <param name="strided">Source strided plane (length stride * h).</param>
    /// <param name="stridedBase">Base offset.</param>
    /// <param name="stride">Source row stride in bytes.</param>
    /// <param name="packed">Output packed plane (length w * h).</param>
    /// <param name="packedBase">Base offset.</param>
    /// <param name="w">Plane width in bytes.</param>
    /// <param name="threadIdx">Linear thread index in [0, w * h).</param>
    public static void PackAt(
        ArrayView<byte> strided, long stridedBase, int stride,
        ArrayView<byte> packed, long packedBase, int w,
        int threadIdx)
    {
        int r = threadIdx / w;
        int c = threadIdx - r * w;
        long src = stridedBase + (long)r * stride + c;
        long dst = packedBase + (long)r * w + c;
        packed[dst] = strided[src];
    }
}
