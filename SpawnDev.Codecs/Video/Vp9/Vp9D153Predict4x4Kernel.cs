// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// ILGPU kernel for the VP9 D153_PRED intra predictor at 4x4. Mirror
// image of D117 around the 135 deg axis. Three-edge mode.
//
// Layout (mirror of vpx_dsp/intrapred.c vpx_d153_predictor_4x4_c):
//   Column 0 (AVG2 with left-offset):
//     dst[0][0] = AVG2(topLeft, left[0])
//     dst[r][0] = AVG2(left[r-1], left[r]) for r=1..3
//   Column 1 (AVG3 with left-offset):
//     dst[0][1] = AVG3(left[0], topLeft, above[0])
//     dst[1][1] = AVG3(topLeft, left[0], left[1])
//     dst[r][1] = AVG3(left[r-2], left[r-1], left[r]) for r=2..3
//   First row cols 2..3:
//     dst[0][2] = AVG3(topLeft, above[0], above[1])
//     dst[0][3] = AVG3(above[0], above[1], above[2])
//   Propagation: dst[r][c] = dst[r-1][c-2] for r=1..3, c=2..3
//
// All seed cells held in registers; rows 1..3 cols 2..3 derive
// from those registers (not from re-reading dst). Same WebGPU
// hazard avoidance as the other directional kernels.
//
// Closes the 10-mode VP9 intra prediction GPU port at 4x4
// (DC, V, H, TM, D45, D63, D135, D117, D153, D207).

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// Batched ILGPU kernel that runs VP9 D153_PRED across N independent
/// 4x4 blocks in parallel.
/// </summary>
public sealed class Vp9D153Predict4x4Kernel : IDisposable
{
    private const int N = 4;

    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, int, int> _kernel;

    /// <summary>Compile the kernel onto <paramref name="accelerator"/>.</summary>
    public Vp9D153Predict4x4Kernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, int, int>(D153Kernel);
    }

    /// <summary>Run D153 prediction on <paramref name="blockCount"/> blocks.</summary>
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

    /// <summary>Kernel body. Two-column seed in registers, row 0 cols 2-3 too.</summary>
    private static void D153Kernel(
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
        int l0 = leftFlat[lBase + 0];
        int l1 = leftFlat[lBase + 1];
        int l2 = leftFlat[lBase + 2];
        int l3 = leftFlat[lBase + 3];

        // Column 0 (AVG2 left-offset).
        byte c0r0 = (byte)((tl + l0 + 1) >> 1);
        byte c0r1 = (byte)((l0 + l1 + 1) >> 1);
        byte c0r2 = (byte)((l1 + l2 + 1) >> 1);
        byte c0r3 = (byte)((l2 + l3 + 1) >> 1);

        // Column 1 (AVG3 left-offset).
        byte c1r0 = (byte)((l0 + 2 * tl + a0 + 2) >> 2);
        byte c1r1 = (byte)((tl + 2 * l0 + l1 + 2) >> 2);
        byte c1r2 = (byte)((l0 + 2 * l1 + l2 + 2) >> 2);
        byte c1r3 = (byte)((l1 + 2 * l2 + l3 + 2) >> 2);

        // Row 0 cols 2,3.
        byte r0c2 = (byte)((tl + 2 * a0 + a1 + 2) >> 2);
        byte r0c3 = (byte)((a0 + 2 * a1 + a2 + 2) >> 2);

        // Row 0.
        dstFlat[dBase + 0] = c0r0;
        dstFlat[dBase + 1] = c1r0;
        dstFlat[dBase + 2] = r0c2;
        dstFlat[dBase + 3] = r0c3;

        // Row 1: col 0,1 from registers; cols 2,3 = previous row's cols 0,1.
        dstFlat[dBase + N + 0] = c0r1;
        dstFlat[dBase + N + 1] = c1r1;
        dstFlat[dBase + N + 2] = c0r0;
        dstFlat[dBase + N + 3] = c1r0;

        // Row 2.
        dstFlat[dBase + 2 * N + 0] = c0r2;
        dstFlat[dBase + 2 * N + 1] = c1r2;
        dstFlat[dBase + 2 * N + 2] = c0r1;
        dstFlat[dBase + 2 * N + 3] = c1r1;

        // Row 3.
        dstFlat[dBase + 3 * N + 0] = c0r3;
        dstFlat[dBase + 3 * N + 1] = c1r3;
        dstFlat[dBase + 3 * N + 2] = c0r2;
        dstFlat[dBase + 3 * N + 3] = c1r2;
    }

    /// <inheritdoc/>
    public void Dispose() { }
}
