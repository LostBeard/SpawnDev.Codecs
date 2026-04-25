// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// ILGPU kernel for the VP9 D117_PRED intra predictor at 4x4. Three-
// edge mode (above + left + topLeft).
//
// Layout (mirror of vpx_dsp/intrapred.c vpx_d117_predictor_4x4_c):
//   Row 0 (AVG2 with above-offset):
//     dst[0][0] = AVG2(topLeft, above[0])
//     dst[0][c] = AVG2(above[c-1], above[c]) for c=1..3
//   Row 1 (AVG3 with above-offset):
//     dst[1][0] = AVG3(left[0], topLeft, above[0])
//     dst[1][1] = AVG3(topLeft, above[0], above[1])
//     dst[1][c] = AVG3(above[c-2], above[c-1], above[c]) for c=2..3
//   First column rows 2..3:
//     dst[2][0] = AVG3(topLeft, left[0], left[1])
//     dst[3][0] = AVG3(left[0], left[1], left[2])
//   Propagation: dst[r][c] = dst[r-2][c-1] for r=2..3, c=1..3
//
// Hold all seed cells in registers (rows 0+1 plus first column for
// rows 2-3) so the propagation never reads dstFlat. Same WebGPU
// read-after-write avoidance as the slice 189 D45 kernel.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// Batched ILGPU kernel that runs VP9 D117_PRED across N independent
/// 4x4 blocks in parallel.
/// </summary>
public sealed class Vp9D117Predict4x4Kernel : IDisposable
{
    private const int N = 4;

    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, int, int> _kernel;

    /// <summary>Compile the kernel onto <paramref name="accelerator"/>.</summary>
    public Vp9D117Predict4x4Kernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, int, int>(D117Kernel);
    }

    /// <summary>Run D117 prediction on <paramref name="blockCount"/> blocks.</summary>
    public void Run(
        ArrayView<byte> topLeftFlat,
        ArrayView<byte> aboveFlat,
        ArrayView<byte> leftFlat,
        ArrayView<byte> dstFlat,
        int blockCount,
        int blockStrideBytes = N * N)
    {
        if (blockCount < 0) throw new ArgumentOutOfRangeException(nameof(blockCount));
        if (blockCount == 0) return;
        if (topLeftFlat.Length < blockCount)
            throw new ArgumentException(
                $"topLeftFlat must hold blockCount bytes (got {topLeftFlat.Length}).",
                nameof(topLeftFlat));
        if (aboveFlat.Length < blockCount * (long)N)
            throw new ArgumentException(
                $"aboveFlat must hold at least blockCount*N bytes (got {aboveFlat.Length}).",
                nameof(aboveFlat));
        if (leftFlat.Length < blockCount * (long)N)
            throw new ArgumentException(
                $"leftFlat must hold at least blockCount*N bytes (got {leftFlat.Length}).",
                nameof(leftFlat));
        if (dstFlat.Length < blockCount * (long)blockStrideBytes)
            throw new ArgumentException(
                $"dstFlat must hold at least blockCount*blockStrideBytes bytes.",
                nameof(dstFlat));
        _kernel(blockCount, topLeftFlat, aboveFlat, leftFlat, dstFlat, blockCount, blockStrideBytes);
    }

    /// <summary>Convenience: allocate, run, read back.</summary>
    public async Task RunAsync(
        ReadOnlyMemory<byte> topLeftFlat,
        ReadOnlyMemory<byte> aboveFlat,
        ReadOnlyMemory<byte> leftFlat,
        Memory<byte> dstFlat,
        int blockCount,
        int blockStrideBytes = N * N)
    {
        if (blockCount <= 0) return;
        long topLeftAllocSize = Math.Max(blockCount, 4L);
        var topLeftPadded = new byte[topLeftAllocSize];
        topLeftFlat.Span.CopyTo(topLeftPadded);

        using var dTopLeft = _accelerator.Allocate1D<byte>(topLeftAllocSize);
        using var dAbove = _accelerator.Allocate1D<byte>(blockCount * (long)N);
        using var dLeft = _accelerator.Allocate1D<byte>(blockCount * (long)N);
        using var dDst = _accelerator.Allocate1D<byte>(blockCount * (long)blockStrideBytes);
        dTopLeft.View.CopyFromCPU(topLeftPadded);
        dAbove.View.CopyFromCPU(aboveFlat.Span.ToArray());
        dLeft.View.CopyFromCPU(leftFlat.Span.ToArray());
        dDst.View.CopyFromCPU(dstFlat.Span.ToArray());
        _kernel(blockCount, dTopLeft.View, dAbove.View, dLeft.View, dDst.View, blockCount, blockStrideBytes);
        await _accelerator.SynchronizeAsync();
        var readBack = await dDst.CopyToHostAsync();
        readBack.AsSpan(0, dstFlat.Length).CopyTo(dstFlat.Span);
    }

    /// <summary>Kernel body. Seed cells in registers, propagate from registers.</summary>
    private static void D117Kernel(
        Index1D blockIdx,
        ArrayView<byte> topLeftFlat,
        ArrayView<byte> aboveFlat,
        ArrayView<byte> leftFlat,
        ArrayView<byte> dstFlat,
        int blockCount,
        int blockStrideBytes)
    {
        int idx = blockIdx;
        if (idx >= blockCount) return;

        long aBase = (long)idx * N;
        long lBase = (long)idx * N;
        long dBase = (long)idx * blockStrideBytes;

        int tl = topLeftFlat[idx];
        int a0 = aboveFlat[aBase + 0];
        int a1 = aboveFlat[aBase + 1];
        int a2 = aboveFlat[aBase + 2];
        int a3 = aboveFlat[aBase + 3];
        int l0 = leftFlat[lBase + 0];
        int l1 = leftFlat[lBase + 1];
        int l2 = leftFlat[lBase + 2];

        // Row 0 (AVG2 above-offset).
        byte r0c0 = (byte)((tl + a0 + 1) >> 1);
        byte r0c1 = (byte)((a0 + a1 + 1) >> 1);
        byte r0c2 = (byte)((a1 + a2 + 1) >> 1);
        byte r0c3 = (byte)((a2 + a3 + 1) >> 1);

        // Row 1 (AVG3 above-offset).
        byte r1c0 = (byte)((l0 + 2 * tl + a0 + 2) >> 2);
        byte r1c1 = (byte)((tl + 2 * a0 + a1 + 2) >> 2);
        byte r1c2 = (byte)((a0 + 2 * a1 + a2 + 2) >> 2);
        byte r1c3 = (byte)((a1 + 2 * a2 + a3 + 2) >> 2);

        // First column r=2,3.
        byte r2c0 = (byte)((tl + 2 * l0 + l1 + 2) >> 2);
        byte r3c0 = (byte)((l0 + 2 * l1 + l2 + 2) >> 2);

        // Row 0.
        dstFlat[dBase + 0] = r0c0;
        dstFlat[dBase + 1] = r0c1;
        dstFlat[dBase + 2] = r0c2;
        dstFlat[dBase + 3] = r0c3;

        // Row 1.
        dstFlat[dBase + N + 0] = r1c0;
        dstFlat[dBase + N + 1] = r1c1;
        dstFlat[dBase + N + 2] = r1c2;
        dstFlat[dBase + N + 3] = r1c3;

        // Row 2: col 0 = r2c0; cols 1..3 = row 0 cols 0..2.
        dstFlat[dBase + 2 * N + 0] = r2c0;
        dstFlat[dBase + 2 * N + 1] = r0c0;
        dstFlat[dBase + 2 * N + 2] = r0c1;
        dstFlat[dBase + 2 * N + 3] = r0c2;

        // Row 3: col 0 = r3c0; cols 1..3 = row 1 cols 0..2.
        dstFlat[dBase + 3 * N + 0] = r3c0;
        dstFlat[dBase + 3 * N + 1] = r1c0;
        dstFlat[dBase + 3 * N + 2] = r1c1;
        dstFlat[dBase + 3 * N + 3] = r1c2;
    }

    /// <inheritdoc/>
    public void Dispose() { }
}
