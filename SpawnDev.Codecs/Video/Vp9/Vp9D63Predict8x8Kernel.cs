// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// 8x8 sibling of Vp9D63Predict4x4Kernel. Same two-filter seed; the
// shift+pad propagation goes 3 levels deep at this size (rows 2-3
// shift by 1 col, 4-5 shift by 2, 6-7 shift by 3). All seed cells
// (rows 0 and 1, 16 bytes total) live in registers.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// Batched ILGPU kernel that runs VP9 D63_PRED across N independent
/// 8x8 blocks in parallel.
/// </summary>
public sealed class Vp9D63Predict8x8Kernel : IDisposable
{
    private const int N = 8;

    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, int, int> _kernel;

    /// <summary>Compile the kernel onto <paramref name="accelerator"/>.</summary>
    public Vp9D63Predict8x8Kernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<byte>, int, int>(D63Kernel);
    }

    /// <summary>Run D63 prediction on <paramref name="blockCount"/> blocks.</summary>
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

    /// <summary>Kernel body. Rows 0+1 in 16 registers, rows 2-7 derived.</summary>
    private static void D63Kernel(
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

        int a0 = aboveFlat[aBase + 0];
        int a1 = aboveFlat[aBase + 1];
        int a2 = aboveFlat[aBase + 2];
        int a3 = aboveFlat[aBase + 3];
        int a4 = aboveFlat[aBase + 4];
        int a5 = aboveFlat[aBase + 5];
        int a6 = aboveFlat[aBase + 6];
        int a7 = aboveFlat[aBase + 7];
        int a8 = aboveFlat[aBase + 8];
        int a9 = aboveFlat[aBase + 9];

        byte fillVal = (byte)a7;  // above[N-1]

        // Row 0 (AVG2 of consecutive above pairs).
        byte r0c0 = (byte)((a0 + a1 + 1) >> 1);
        byte r0c1 = (byte)((a1 + a2 + 1) >> 1);
        byte r0c2 = (byte)((a2 + a3 + 1) >> 1);
        byte r0c3 = (byte)((a3 + a4 + 1) >> 1);
        byte r0c4 = (byte)((a4 + a5 + 1) >> 1);
        byte r0c5 = (byte)((a5 + a6 + 1) >> 1);
        byte r0c6 = (byte)((a6 + a7 + 1) >> 1);
        byte r0c7 = (byte)((a7 + a8 + 1) >> 1);

        // Row 1 (AVG3 of consecutive above triples).
        byte r1c0 = (byte)((a0 + 2 * a1 + a2 + 2) >> 2);
        byte r1c1 = (byte)((a1 + 2 * a2 + a3 + 2) >> 2);
        byte r1c2 = (byte)((a2 + 2 * a3 + a4 + 2) >> 2);
        byte r1c3 = (byte)((a3 + 2 * a4 + a5 + 2) >> 2);
        byte r1c4 = (byte)((a4 + 2 * a5 + a6 + 2) >> 2);
        byte r1c5 = (byte)((a5 + 2 * a6 + a7 + 2) >> 2);
        byte r1c6 = (byte)((a6 + 2 * a7 + a8 + 2) >> 2);
        byte r1c7 = (byte)((a7 + 2 * a8 + a9 + 2) >> 2);

        // Row 0
        dstFlat[dBase + 0] = r0c0;
        dstFlat[dBase + 1] = r0c1;
        dstFlat[dBase + 2] = r0c2;
        dstFlat[dBase + 3] = r0c3;
        dstFlat[dBase + 4] = r0c4;
        dstFlat[dBase + 5] = r0c5;
        dstFlat[dBase + 6] = r0c6;
        dstFlat[dBase + 7] = r0c7;

        // Row 1
        long row1 = dBase + N;
        dstFlat[row1 + 0] = r1c0;
        dstFlat[row1 + 1] = r1c1;
        dstFlat[row1 + 2] = r1c2;
        dstFlat[row1 + 3] = r1c3;
        dstFlat[row1 + 4] = r1c4;
        dstFlat[row1 + 5] = r1c5;
        dstFlat[row1 + 6] = r1c6;
        dstFlat[row1 + 7] = r1c7;

        // Row 2: row 0 shifted by 1, padded with fillVal at cols 6-7.
        long row2 = dBase + 2 * N;
        dstFlat[row2 + 0] = r0c1;
        dstFlat[row2 + 1] = r0c2;
        dstFlat[row2 + 2] = r0c3;
        dstFlat[row2 + 3] = r0c4;
        dstFlat[row2 + 4] = r0c5;
        dstFlat[row2 + 5] = r0c6;
        dstFlat[row2 + 6] = fillVal;
        dstFlat[row2 + 7] = fillVal;

        // Row 3: row 1 shifted by 1.
        long row3 = dBase + 3 * N;
        dstFlat[row3 + 0] = r1c1;
        dstFlat[row3 + 1] = r1c2;
        dstFlat[row3 + 2] = r1c3;
        dstFlat[row3 + 3] = r1c4;
        dstFlat[row3 + 4] = r1c5;
        dstFlat[row3 + 5] = r1c6;
        dstFlat[row3 + 6] = fillVal;
        dstFlat[row3 + 7] = fillVal;

        // Row 4: row 0 shifted by 2.
        long row4 = dBase + 4 * N;
        dstFlat[row4 + 0] = r0c2;
        dstFlat[row4 + 1] = r0c3;
        dstFlat[row4 + 2] = r0c4;
        dstFlat[row4 + 3] = r0c5;
        dstFlat[row4 + 4] = r0c6;
        dstFlat[row4 + 5] = fillVal;
        dstFlat[row4 + 6] = fillVal;
        dstFlat[row4 + 7] = fillVal;

        // Row 5: row 1 shifted by 2.
        long row5 = dBase + 5 * N;
        dstFlat[row5 + 0] = r1c2;
        dstFlat[row5 + 1] = r1c3;
        dstFlat[row5 + 2] = r1c4;
        dstFlat[row5 + 3] = r1c5;
        dstFlat[row5 + 4] = r1c6;
        dstFlat[row5 + 5] = fillVal;
        dstFlat[row5 + 6] = fillVal;
        dstFlat[row5 + 7] = fillVal;

        // Row 6: row 0 shifted by 3.
        long row6 = dBase + 6 * N;
        dstFlat[row6 + 0] = r0c3;
        dstFlat[row6 + 1] = r0c4;
        dstFlat[row6 + 2] = r0c5;
        dstFlat[row6 + 3] = r0c6;
        dstFlat[row6 + 4] = fillVal;
        dstFlat[row6 + 5] = fillVal;
        dstFlat[row6 + 6] = fillVal;
        dstFlat[row6 + 7] = fillVal;

        // Row 7: row 1 shifted by 3.
        long row7 = dBase + 7 * N;
        dstFlat[row7 + 0] = r1c3;
        dstFlat[row7 + 1] = r1c4;
        dstFlat[row7 + 2] = r1c5;
        dstFlat[row7 + 3] = r1c6;
        dstFlat[row7 + 4] = fillVal;
        dstFlat[row7 + 5] = fillVal;
        dstFlat[row7 + 6] = fillVal;
        dstFlat[row7 + 7] = fillVal;
    }

    /// <inheritdoc/>
    public void Dispose() { }
}
