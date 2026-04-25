// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// 8x8 sibling of Vp9D117Predict4x4Kernel. Three-edge directional.
// Rows 0+1 + first column rows 2-7 seeded (22 register cells);
// remaining cells fill via dst[r][c] = dst[r-2][c-1] propagation,
// reading from registers only.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// Batched ILGPU kernel that runs VP9 D117_PRED across N independent
/// 8x8 blocks in parallel.
/// </summary>
public sealed class Vp9D117Predict8x8Kernel : IDisposable
{
    private const int N = 8;

    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, int, int> _kernel;

    /// <summary>Compile the kernel onto <paramref name="accelerator"/>.</summary>
    public Vp9D117Predict8x8Kernel(Accelerator accelerator)
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

    /// <summary>Kernel body. 22 register cells; rows 2-7 derive from those.</summary>
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
        int a4 = aboveFlat[aBase + 4];
        int a5 = aboveFlat[aBase + 5];
        int a6 = aboveFlat[aBase + 6];
        int a7 = aboveFlat[aBase + 7];
        int l0 = leftFlat[lBase + 0];
        int l1 = leftFlat[lBase + 1];
        int l2 = leftFlat[lBase + 2];
        int l3 = leftFlat[lBase + 3];
        int l4 = leftFlat[lBase + 4];
        int l5 = leftFlat[lBase + 5];
        int l6 = leftFlat[lBase + 6];

        // Row 0 (AVG2 above-offset).
        byte r0c0 = (byte)((tl + a0 + 1) >> 1);
        byte r0c1 = (byte)((a0 + a1 + 1) >> 1);
        byte r0c2 = (byte)((a1 + a2 + 1) >> 1);
        byte r0c3 = (byte)((a2 + a3 + 1) >> 1);
        byte r0c4 = (byte)((a3 + a4 + 1) >> 1);
        byte r0c5 = (byte)((a4 + a5 + 1) >> 1);
        byte r0c6 = (byte)((a5 + a6 + 1) >> 1);
        byte r0c7 = (byte)((a6 + a7 + 1) >> 1);

        // Row 1 (AVG3 above-offset).
        byte r1c0 = (byte)((l0 + 2 * tl + a0 + 2) >> 2);
        byte r1c1 = (byte)((tl + 2 * a0 + a1 + 2) >> 2);
        byte r1c2 = (byte)((a0 + 2 * a1 + a2 + 2) >> 2);
        byte r1c3 = (byte)((a1 + 2 * a2 + a3 + 2) >> 2);
        byte r1c4 = (byte)((a2 + 2 * a3 + a4 + 2) >> 2);
        byte r1c5 = (byte)((a3 + 2 * a4 + a5 + 2) >> 2);
        byte r1c6 = (byte)((a4 + 2 * a5 + a6 + 2) >> 2);
        byte r1c7 = (byte)((a5 + 2 * a6 + a7 + 2) >> 2);

        // First column rows 2-7.
        byte r2c0 = (byte)((tl + 2 * l0 + l1 + 2) >> 2);
        byte r3c0 = (byte)((l0 + 2 * l1 + l2 + 2) >> 2);
        byte r4c0 = (byte)((l1 + 2 * l2 + l3 + 2) >> 2);
        byte r5c0 = (byte)((l2 + 2 * l3 + l4 + 2) >> 2);
        byte r6c0 = (byte)((l3 + 2 * l4 + l5 + 2) >> 2);
        byte r7c0 = (byte)((l4 + 2 * l5 + l6 + 2) >> 2);

        // Row 0.
        dstFlat[dBase + 0] = r0c0;
        dstFlat[dBase + 1] = r0c1;
        dstFlat[dBase + 2] = r0c2;
        dstFlat[dBase + 3] = r0c3;
        dstFlat[dBase + 4] = r0c4;
        dstFlat[dBase + 5] = r0c5;
        dstFlat[dBase + 6] = r0c6;
        dstFlat[dBase + 7] = r0c7;

        // Row 1.
        long row1 = dBase + N;
        dstFlat[row1 + 0] = r1c0;
        dstFlat[row1 + 1] = r1c1;
        dstFlat[row1 + 2] = r1c2;
        dstFlat[row1 + 3] = r1c3;
        dstFlat[row1 + 4] = r1c4;
        dstFlat[row1 + 5] = r1c5;
        dstFlat[row1 + 6] = r1c6;
        dstFlat[row1 + 7] = r1c7;

        // Row 2: col 0 = r2c0; cols 1..7 = row 0 cols 0..6.
        long row2 = dBase + 2 * N;
        dstFlat[row2 + 0] = r2c0;
        dstFlat[row2 + 1] = r0c0;
        dstFlat[row2 + 2] = r0c1;
        dstFlat[row2 + 3] = r0c2;
        dstFlat[row2 + 4] = r0c3;
        dstFlat[row2 + 5] = r0c4;
        dstFlat[row2 + 6] = r0c5;
        dstFlat[row2 + 7] = r0c6;

        // Row 3: col 0 = r3c0; cols 1..7 = row 1 cols 0..6.
        long row3 = dBase + 3 * N;
        dstFlat[row3 + 0] = r3c0;
        dstFlat[row3 + 1] = r1c0;
        dstFlat[row3 + 2] = r1c1;
        dstFlat[row3 + 3] = r1c2;
        dstFlat[row3 + 4] = r1c3;
        dstFlat[row3 + 5] = r1c4;
        dstFlat[row3 + 6] = r1c5;
        dstFlat[row3 + 7] = r1c6;

        // Row 4: col 0 = r4c0; col 1 = r2c0; cols 2..7 = row 0 cols 0..5.
        long row4 = dBase + 4 * N;
        dstFlat[row4 + 0] = r4c0;
        dstFlat[row4 + 1] = r2c0;
        dstFlat[row4 + 2] = r0c0;
        dstFlat[row4 + 3] = r0c1;
        dstFlat[row4 + 4] = r0c2;
        dstFlat[row4 + 5] = r0c3;
        dstFlat[row4 + 6] = r0c4;
        dstFlat[row4 + 7] = r0c5;

        // Row 5: col 0 = r5c0; col 1 = r3c0; cols 2..7 = row 1 cols 0..5.
        long row5 = dBase + 5 * N;
        dstFlat[row5 + 0] = r5c0;
        dstFlat[row5 + 1] = r3c0;
        dstFlat[row5 + 2] = r1c0;
        dstFlat[row5 + 3] = r1c1;
        dstFlat[row5 + 4] = r1c2;
        dstFlat[row5 + 5] = r1c3;
        dstFlat[row5 + 6] = r1c4;
        dstFlat[row5 + 7] = r1c5;

        // Row 6: col 0 = r6c0; col 1 = r4c0; col 2 = r2c0; cols 3..7 = row 0 cols 0..4.
        long row6 = dBase + 6 * N;
        dstFlat[row6 + 0] = r6c0;
        dstFlat[row6 + 1] = r4c0;
        dstFlat[row6 + 2] = r2c0;
        dstFlat[row6 + 3] = r0c0;
        dstFlat[row6 + 4] = r0c1;
        dstFlat[row6 + 5] = r0c2;
        dstFlat[row6 + 6] = r0c3;
        dstFlat[row6 + 7] = r0c4;

        // Row 7: col 0 = r7c0; col 1 = r5c0; col 2 = r3c0; cols 3..7 = row 1 cols 0..4.
        long row7 = dBase + 7 * N;
        dstFlat[row7 + 0] = r7c0;
        dstFlat[row7 + 1] = r5c0;
        dstFlat[row7 + 2] = r3c0;
        dstFlat[row7 + 3] = r1c0;
        dstFlat[row7 + 4] = r1c1;
        dstFlat[row7 + 5] = r1c2;
        dstFlat[row7 + 6] = r1c3;
        dstFlat[row7 + 7] = r1c4;
    }

    /// <inheritdoc/>
    public void Dispose() { }
}
