// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// 8x8 sibling of Vp9D153Predict4x4Kernel. Three-edge directional,
// mirror of D117. Cols 0+1 + first row past col 1 seeded (22
// register cells); rows 1-7 cols 2-7 derive via
// dst[r][c] = dst[r-1][c-2] from those registers.
//
// Closes the 10-mode VP9 intra prediction GPU port at 8x8 once
// this slice lands (DC, V/H, TM, D45, D63, D135, D117, D153, D207
// all done).

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// Batched ILGPU kernel that runs VP9 D153_PRED across N independent
/// 8x8 blocks in parallel.
/// </summary>
public sealed class Vp9D153Predict8x8Kernel : IDisposable
{
    private const int N = 8;

    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, int, int> _kernel;

    /// <summary>Compile the kernel onto <paramref name="accelerator"/>.</summary>
    public Vp9D153Predict8x8Kernel(Accelerator accelerator)
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

    /// <summary>Kernel body. 22 register cells; rows 1-7 derive from those.</summary>
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
        int a3 = aboveFlat[aBase + 3];
        int a4 = aboveFlat[aBase + 4];
        int a5 = aboveFlat[aBase + 5];
        int a6 = aboveFlat[aBase + 6];
        int l0 = leftFlat[lBase + 0];
        int l1 = leftFlat[lBase + 1];
        int l2 = leftFlat[lBase + 2];
        int l3 = leftFlat[lBase + 3];
        int l4 = leftFlat[lBase + 4];
        int l5 = leftFlat[lBase + 5];
        int l6 = leftFlat[lBase + 6];
        int l7 = leftFlat[lBase + 7];

        // Column 0 (AVG2 left-offset).
        byte c0r0 = (byte)((tl + l0 + 1) >> 1);
        byte c0r1 = (byte)((l0 + l1 + 1) >> 1);
        byte c0r2 = (byte)((l1 + l2 + 1) >> 1);
        byte c0r3 = (byte)((l2 + l3 + 1) >> 1);
        byte c0r4 = (byte)((l3 + l4 + 1) >> 1);
        byte c0r5 = (byte)((l4 + l5 + 1) >> 1);
        byte c0r6 = (byte)((l5 + l6 + 1) >> 1);
        byte c0r7 = (byte)((l6 + l7 + 1) >> 1);

        // Column 1 (AVG3 left-offset).
        byte c1r0 = (byte)((l0 + 2 * tl + a0 + 2) >> 2);
        byte c1r1 = (byte)((tl + 2 * l0 + l1 + 2) >> 2);
        byte c1r2 = (byte)((l0 + 2 * l1 + l2 + 2) >> 2);
        byte c1r3 = (byte)((l1 + 2 * l2 + l3 + 2) >> 2);
        byte c1r4 = (byte)((l2 + 2 * l3 + l4 + 2) >> 2);
        byte c1r5 = (byte)((l3 + 2 * l4 + l5 + 2) >> 2);
        byte c1r6 = (byte)((l4 + 2 * l5 + l6 + 2) >> 2);
        byte c1r7 = (byte)((l5 + 2 * l6 + l7 + 2) >> 2);

        // Row 0 cols 2..7.
        byte r0c2 = (byte)((tl + 2 * a0 + a1 + 2) >> 2);
        byte r0c3 = (byte)((a0 + 2 * a1 + a2 + 2) >> 2);
        byte r0c4 = (byte)((a1 + 2 * a2 + a3 + 2) >> 2);
        byte r0c5 = (byte)((a2 + 2 * a3 + a4 + 2) >> 2);
        byte r0c6 = (byte)((a3 + 2 * a4 + a5 + 2) >> 2);
        byte r0c7 = (byte)((a4 + 2 * a5 + a6 + 2) >> 2);

        // Row 0.
        dstFlat[dBase + 0] = c0r0;
        dstFlat[dBase + 1] = c1r0;
        dstFlat[dBase + 2] = r0c2;
        dstFlat[dBase + 3] = r0c3;
        dstFlat[dBase + 4] = r0c4;
        dstFlat[dBase + 5] = r0c5;
        dstFlat[dBase + 6] = r0c6;
        dstFlat[dBase + 7] = r0c7;

        // Row 1: c0r1, c1r1, c0r0, c1r0, r0c2, r0c3, r0c4, r0c5
        long row1 = dBase + N;
        dstFlat[row1 + 0] = c0r1;
        dstFlat[row1 + 1] = c1r1;
        dstFlat[row1 + 2] = c0r0;
        dstFlat[row1 + 3] = c1r0;
        dstFlat[row1 + 4] = r0c2;
        dstFlat[row1 + 5] = r0c3;
        dstFlat[row1 + 6] = r0c4;
        dstFlat[row1 + 7] = r0c5;

        // Row 2: c0r2, c1r2, c0r1, c1r1, c0r0, c1r0, r0c2, r0c3
        long row2 = dBase + 2 * N;
        dstFlat[row2 + 0] = c0r2;
        dstFlat[row2 + 1] = c1r2;
        dstFlat[row2 + 2] = c0r1;
        dstFlat[row2 + 3] = c1r1;
        dstFlat[row2 + 4] = c0r0;
        dstFlat[row2 + 5] = c1r0;
        dstFlat[row2 + 6] = r0c2;
        dstFlat[row2 + 7] = r0c3;

        // Row 3: c0r3, c1r3, c0r2, c1r2, c0r1, c1r1, c0r0, c1r0
        long row3 = dBase + 3 * N;
        dstFlat[row3 + 0] = c0r3;
        dstFlat[row3 + 1] = c1r3;
        dstFlat[row3 + 2] = c0r2;
        dstFlat[row3 + 3] = c1r2;
        dstFlat[row3 + 4] = c0r1;
        dstFlat[row3 + 5] = c1r1;
        dstFlat[row3 + 6] = c0r0;
        dstFlat[row3 + 7] = c1r0;

        // Row 4: c0r4, c1r4, c0r3, c1r3, c0r2, c1r2, c0r1, c1r1
        long row4 = dBase + 4 * N;
        dstFlat[row4 + 0] = c0r4;
        dstFlat[row4 + 1] = c1r4;
        dstFlat[row4 + 2] = c0r3;
        dstFlat[row4 + 3] = c1r3;
        dstFlat[row4 + 4] = c0r2;
        dstFlat[row4 + 5] = c1r2;
        dstFlat[row4 + 6] = c0r1;
        dstFlat[row4 + 7] = c1r1;

        // Row 5: c0r5, c1r5, c0r4, c1r4, c0r3, c1r3, c0r2, c1r2
        long row5 = dBase + 5 * N;
        dstFlat[row5 + 0] = c0r5;
        dstFlat[row5 + 1] = c1r5;
        dstFlat[row5 + 2] = c0r4;
        dstFlat[row5 + 3] = c1r4;
        dstFlat[row5 + 4] = c0r3;
        dstFlat[row5 + 5] = c1r3;
        dstFlat[row5 + 6] = c0r2;
        dstFlat[row5 + 7] = c1r2;

        // Row 6: c0r6, c1r6, c0r5, c1r5, c0r4, c1r4, c0r3, c1r3
        long row6 = dBase + 6 * N;
        dstFlat[row6 + 0] = c0r6;
        dstFlat[row6 + 1] = c1r6;
        dstFlat[row6 + 2] = c0r5;
        dstFlat[row6 + 3] = c1r5;
        dstFlat[row6 + 4] = c0r4;
        dstFlat[row6 + 5] = c1r4;
        dstFlat[row6 + 6] = c0r3;
        dstFlat[row6 + 7] = c1r3;

        // Row 7: c0r7, c1r7, c0r6, c1r6, c0r5, c1r5, c0r4, c1r4
        long row7 = dBase + 7 * N;
        dstFlat[row7 + 0] = c0r7;
        dstFlat[row7 + 1] = c1r7;
        dstFlat[row7 + 2] = c0r6;
        dstFlat[row7 + 3] = c1r6;
        dstFlat[row7 + 4] = c0r5;
        dstFlat[row7 + 5] = c1r5;
        dstFlat[row7 + 6] = c0r4;
        dstFlat[row7 + 7] = c1r4;
    }

    /// <inheritdoc/>
    public void Dispose() { }
}
