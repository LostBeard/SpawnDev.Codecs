// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// ILGPU kernel for VP8 4x4 DC-only IDCT fast path (libvpx
// vp8_dc_only_idct_add_c). When the AC slots of a 4x4 block are zero,
// the IDCT collapses to a constant a1 = (inputDc + 4) >> 3 added to
// every pixel of the predictor.
//
// One thread per 4x4 block. Operates on packed buffers (predStride=4,
// dstStride=4 per block). Caller scatters into the frame buffer after
// the kernel completes.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>
/// Batched ILGPU kernel for VP8 DC-only IDCT 4x4 add. One thread per
/// 4x4 block. Per-block input DC supplied as a parallel ArrayView of
/// length blockCount.
/// </summary>
public sealed class Vp8DcOnlyIdctAddKernel : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<short>, ArrayView<byte>, ArrayView<byte>, int> _kernel;

    /// <summary>Compile the kernel onto <paramref name="accelerator"/>.</summary>
    public Vp8DcOnlyIdctAddKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<short>, ArrayView<byte>, ArrayView<byte>, int>(DcOnlyKernel);
    }

    /// <summary>
    /// Run the DC-only IDCT add on <paramref name="blockCount"/> blocks.
    /// </summary>
    /// <param name="inputDc">per-block input DC (length = blockCount).</param>
    /// <param name="pred">predictor bytes (16 bytes/block, packed 4x4).</param>
    /// <param name="dst">output bytes (16 bytes/block).</param>
    public void Run(ArrayView<short> inputDc, ArrayView<byte> pred, ArrayView<byte> dst, int blockCount)
    {
        if (blockCount < 0) throw new ArgumentOutOfRangeException(nameof(blockCount));
        if (blockCount == 0) return;
        if (inputDc.Length < blockCount)
            throw new ArgumentException("inputDc must hold blockCount shorts.", nameof(inputDc));
        if (pred.Length < blockCount * 16L)
            throw new ArgumentException("pred must hold blockCount*16 bytes.", nameof(pred));
        if (dst.Length < blockCount * 16L)
            throw new ArgumentException("dst must hold blockCount*16 bytes.", nameof(dst));
        _kernel(blockCount, inputDc, pred, dst, blockCount);
    }

    private static void DcOnlyKernel(
        Index1D blockIdx,
        ArrayView<short> inputDc,
        ArrayView<byte> pred,
        ArrayView<byte> dst,
        int blockCount)
    {
        int idx = blockIdx;
        if (idx >= blockCount) return;
        long pBase = (long)idx * 16;
        long dBase = (long)idx * 16;

        int a1 = (inputDc[idx] + 4) >> 3;
        for (int r = 0; r < 4; r++)
        {
            long row = r * 4;
            for (int c = 0; c < 4; c++)
            {
                int a = a1 + pred[pBase + row + c];
                if (a < 0) a = 0;
                else if (a > 255) a = 255;
                dst[dBase + row + c] = (byte)a;
            }
        }
    }

    /// <summary>Release kernel resources. Does NOT dispose the accelerator.</summary>
    public void Dispose() { /* auto-grouped */ }
}
