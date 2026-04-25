// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// ILGPU kernel for the VP9 D45_PRED intra predictor at 4x4. First
// directional predictor GPU port. Per-block thread, batched. Bit-
// exact against Vp9DirectionalPredictor.D45Predict.
//
// D45 is the 45-degree up-right diagonal mode. It needs 2N above
// samples (the right-half extension feeds the diagonal as it
// extends past column N-1). The kernel:
//
//   1. Builds row 0 by AVG3-filtering the diagonal and writing
//      above[n-1] directly into the bottom-right cell.
//   2. Builds rows 1..n-1 as left-shifted copies of row 0 with the
//      right edge padded by above[n-1].
//
// libvpx reference: vpx_dsp/intrapred.c vpx_d45_predictor_4x4_c.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// Batched ILGPU kernel that runs VP9 D45_PRED across N independent
/// 4x4 blocks in parallel.
/// </summary>
public sealed class Vp9D45Predict4x4Kernel : IDisposable
{
    private const int N = 4;

    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, int, int> _kernel;

    /// <summary>Compile the kernel onto <paramref name="accelerator"/>.</summary>
    public Vp9D45Predict4x4Kernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<byte>, int, int>(D45Kernel);
    }

    /// <summary>
    /// Run D45 prediction on <paramref name="blockCount"/> blocks.
    /// </summary>
    /// <param name="aboveFlat">Block-major flat above samples (2*N=8 bytes per block).</param>
    /// <param name="dstFlat">Block-major destination (blockStrideBytes per block).</param>
    /// <param name="blockCount">Number of 4x4 blocks to predict.</param>
    /// <param name="blockStrideBytes">Bytes per destination block (default N*N=16).</param>
    public void Run(
        ArrayView<byte> aboveFlat,
        ArrayView<byte> dstFlat,
        int blockCount,
        int blockStrideBytes = N * N)
    {
        if (blockCount < 0) throw new ArgumentOutOfRangeException(nameof(blockCount));
        if (blockCount == 0) return;
        if (aboveFlat.Length < blockCount * (long)(2 * N))
            throw new ArgumentException(
                $"aboveFlat must hold at least blockCount*2N bytes (got {aboveFlat.Length}).",
                nameof(aboveFlat));
        if (dstFlat.Length < blockCount * (long)blockStrideBytes)
            throw new ArgumentException(
                $"dstFlat must hold at least blockCount*blockStrideBytes bytes.",
                nameof(dstFlat));
        _kernel(blockCount, aboveFlat, dstFlat, blockCount, blockStrideBytes);
    }

    /// <summary>Convenience: allocate, run, read back.</summary>
    public async Task RunAsync(
        ReadOnlyMemory<byte> aboveFlat,
        Memory<byte> dstFlat,
        int blockCount,
        int blockStrideBytes = N * N)
    {
        if (blockCount <= 0) return;
        using var dAbove = _accelerator.Allocate1D<byte>(blockCount * (long)(2 * N));
        using var dDst = _accelerator.Allocate1D<byte>(blockCount * (long)blockStrideBytes);
        dAbove.View.CopyFromCPU(aboveFlat.Span.ToArray());
        dDst.View.CopyFromCPU(dstFlat.Span.ToArray());
        _kernel(blockCount, dAbove.View, dDst.View, blockCount, blockStrideBytes);
        await _accelerator.SynchronizeAsync();
        var readBack = await dDst.CopyToHostAsync();
        readBack.AsSpan(0, dstFlat.Length).CopyTo(dstFlat.Span);
    }

    /// <summary>
    /// Kernel body. One thread per block. Row 0 is held in registers
    /// so subsequent rows derive from those registers rather than
    /// re-reading dst (which lowers to atomic RMW on byte writes and
    /// hits a read-after-write hazard on WebGPU).
    /// </summary>
    private static void D45Kernel(
        Index1D blockIdx,
        ArrayView<byte> aboveFlat,
        ArrayView<byte> dstFlat,
        int blockCount,
        int blockStrideBytes)
    {
        int idx = blockIdx;
        if (idx >= blockCount) return;

        long aBase = (long)idx * (2 * N);
        long dBase = (long)idx * blockStrideBytes;

        // libvpx convention: above_right = above[n-1] (the right-most
        // in-block above sample, NOT one of the extension samples).
        byte aboveRight = aboveFlat[aBase + (N - 1)];

        // Compute row 0 cells in registers.
        // r0[x] = AVG3(above[x], above[x+1], above[x+2]) for x=0..N-2;
        // r0[N-1] = above_right.
        int a0 = aboveFlat[aBase + 0];
        int a1 = aboveFlat[aBase + 1];
        int a2 = aboveFlat[aBase + 2];
        int a3 = aboveFlat[aBase + 3];
        int a4 = aboveFlat[aBase + 4];
        int a5 = aboveFlat[aBase + 5];

        byte r0c0 = (byte)((a0 + 2 * a1 + a2 + 2) >> 2);
        byte r0c1 = (byte)((a1 + 2 * a2 + a3 + 2) >> 2);
        byte r0c2 = (byte)((a2 + 2 * a3 + a4 + 2) >> 2);
        byte r0c3 = aboveRight;

        // Row 0: write the 4 register cells.
        dstFlat[dBase + 0] = r0c0;
        dstFlat[dBase + 1] = r0c1;
        dstFlat[dBase + 2] = r0c2;
        dstFlat[dBase + 3] = r0c3;

        // Row 1: shift row 0 left by 1, pad right with above_right.
        dstFlat[dBase + N + 0] = r0c1;
        dstFlat[dBase + N + 1] = r0c2;
        dstFlat[dBase + N + 2] = r0c3;
        dstFlat[dBase + N + 3] = aboveRight;

        // Row 2: shift left by 2.
        dstFlat[dBase + 2 * N + 0] = r0c2;
        dstFlat[dBase + 2 * N + 1] = r0c3;
        dstFlat[dBase + 2 * N + 2] = aboveRight;
        dstFlat[dBase + 2 * N + 3] = aboveRight;

        // Row 3: all above_right.
        dstFlat[dBase + 3 * N + 0] = aboveRight;
        dstFlat[dBase + 3 * N + 1] = aboveRight;
        dstFlat[dBase + 3 * N + 2] = aboveRight;
        dstFlat[dBase + 3 * N + 3] = aboveRight;
    }

    /// <inheritdoc/>
    public void Dispose() { }
}
