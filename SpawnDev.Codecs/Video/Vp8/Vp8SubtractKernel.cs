// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// ILGPU kernel for the encoder-side subtract step:
// residual[i] = (short)(src[i] - pred[i]) over per-block packed
// buffers. One thread per pixel. Trivially parallel; the kernel
// exists so the residual stays GPU-resident on the way into the
// FDCT.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>
/// Batched ILGPU kernel for VP8 encoder residual: per-pixel
/// <c>residual = src - pred</c>. One thread per pixel.
/// </summary>
public sealed class Vp8SubtractKernel : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<short>, int> _kernel;

    /// <summary>Compile the kernel onto <paramref name="accelerator"/>.</summary>
    public Vp8SubtractKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<short>, int>(SubtractKernel);
    }

    /// <summary>Run on N pixels. residual = (short)(src - pred).</summary>
    public void Run(ArrayView<byte> src, ArrayView<byte> pred, ArrayView<short> residual, int pixelCount)
    {
        if (pixelCount < 0) throw new ArgumentOutOfRangeException(nameof(pixelCount));
        if (pixelCount == 0) return;
        if (src.Length < pixelCount) throw new ArgumentException("src too short.", nameof(src));
        if (pred.Length < pixelCount) throw new ArgumentException("pred too short.", nameof(pred));
        if (residual.Length < pixelCount) throw new ArgumentException("residual too short.", nameof(residual));
        _kernel(pixelCount, src, pred, residual, pixelCount);
    }

    private static void SubtractKernel(
        Index1D idx,
        ArrayView<byte> src,
        ArrayView<byte> pred,
        ArrayView<short> residual,
        int pixelCount)
    {
        int i = idx;
        if (i >= pixelCount) return;
        residual[i] = (short)(src[i] - pred[i]);
    }

    /// <summary>Release kernel resources.</summary>
    public void Dispose() { /* auto-grouped */ }
}
