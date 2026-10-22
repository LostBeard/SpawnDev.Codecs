// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Small pixel kernels shared by the 8-bit image codecs (GIF, BMP): RGBA8 -> RGBA16 widening for decoders
// whose format has no 16-bit samples but whose caller asked for PixelFormat.Rgba16.

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs.Images;

/// <summary>Launches the shared 8-bit image pixel kernels.</summary>
internal sealed class ImagePixelKernels : IDisposable
{
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>> _widen;

    public ImagePixelKernels(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _widen = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<byte>>(WidenKernel);
    }

    /// <summary>
    /// Widen <paramref name="byteCount"/> RGBA8 bytes to RGBA16: each sample v becomes the little-endian
    /// 16-bit value v * 257 (bytes v, v), the exact 8 to 16 bit expansion PNG uses.
    /// </summary>
    public void Widen(ArrayView<byte> rgba8, ArrayView<byte> rgba16, int byteCount)
    {
        if (byteCount > 0) _widen(byteCount, rgba8, rgba16);
    }

    private static void WidenKernel(Index1D index, ArrayView<byte> src, ArrayView<byte> dst)
    {
        int i = index.X;
        byte v = src[i];
        dst[i * 2] = v;
        dst[i * 2 + 1] = v;
    }

    /// <summary>No-op (kernels are owned by the accelerator).</summary>
    public void Dispose() { }
}
