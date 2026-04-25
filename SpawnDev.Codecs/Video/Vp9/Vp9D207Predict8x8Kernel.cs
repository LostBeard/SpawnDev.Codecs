// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// 8x8 sibling of Vp9D207Predict4x4Kernel. Left-only directional.
// Two seed columns (cols 0 + 1) in 16 byte registers; subsequent
// rows shift up by 2 cols. Last row's right half fills with
// left[N-1].
//
// Propagation rule: dst[r][c+2] = dst[r+1][c] for r=N-2..0, c=0..N-3.
// Equivalently dst[r][c2] reads cols (c2-2) of the row below.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// Batched ILGPU kernel that runs VP9 D207_PRED across N independent
/// 8x8 blocks in parallel.
/// </summary>
public sealed class Vp9D207Predict8x8Kernel : IDisposable
{
    private const int N = 8;

    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, int, int> _kernel;

    /// <summary>Compile the kernel onto <paramref name="accelerator"/>.</summary>
    public Vp9D207Predict8x8Kernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<byte>, int, int>(D207Kernel);
    }

    /// <summary>Run D207 prediction on <paramref name="blockCount"/> blocks.</summary>
    public void Run(
        ArrayView<byte> leftFlat,
        ArrayView<byte> dstFlat,
        int blockCount,
        int blockStrideBytes = N * N)
    {
        if (blockCount < 0) throw new ArgumentOutOfRangeException(nameof(blockCount));
        if (blockCount == 0) return;
        if (leftFlat.Length < blockCount * (long)N)
            throw new ArgumentException(
                $"leftFlat must hold at least blockCount*N bytes (got {leftFlat.Length}).",
                nameof(leftFlat));
        if (dstFlat.Length < blockCount * (long)blockStrideBytes)
            throw new ArgumentException(
                $"dstFlat must hold at least blockCount*blockStrideBytes bytes.",
                nameof(dstFlat));
        _kernel(blockCount, leftFlat, dstFlat, blockCount, blockStrideBytes);
    }

    /// <summary>Convenience: allocate, run, read back.</summary>
    public async Task RunAsync(
        ReadOnlyMemory<byte> leftFlat,
        Memory<byte> dstFlat,
        int blockCount,
        int blockStrideBytes = N * N)
    {
        if (blockCount <= 0) return;
        using var dLeft = _accelerator.Allocate1D<byte>(blockCount * (long)N);
        using var dDst = _accelerator.Allocate1D<byte>(blockCount * (long)blockStrideBytes);
        dLeft.View.CopyFromCPU(leftFlat.Span.ToArray());
        dDst.View.CopyFromCPU(dstFlat.Span.ToArray());
        _kernel(blockCount, dLeft.View, dDst.View, blockCount, blockStrideBytes);
        await _accelerator.SynchronizeAsync();
        var readBack = await dDst.CopyToHostAsync();
        readBack.AsSpan(0, dstFlat.Length).CopyTo(dstFlat.Span);
    }

    /// <summary>Kernel body. Cols 0+1 in 16 registers; rows 0-6 derive from those.</summary>
    private static void D207Kernel(
        Index1D blockIdx,
        ArrayView<byte> leftFlat,
        ArrayView<byte> dstFlat,
        int blockCount,
        int blockStrideBytes)
    {
        int idx = blockIdx;
        if (idx >= blockCount) return;

        long lBase = (long)idx * N;
        long dBase = (long)idx * blockStrideBytes;

        int l0 = leftFlat[lBase + 0];
        int l1 = leftFlat[lBase + 1];
        int l2 = leftFlat[lBase + 2];
        int l3 = leftFlat[lBase + 3];
        int l4 = leftFlat[lBase + 4];
        int l5 = leftFlat[lBase + 5];
        int l6 = leftFlat[lBase + 6];
        int l7 = leftFlat[lBase + 7];

        byte fillVal = (byte)l7;  // left[N-1]

        // Column 0: AVG2 pairs; bottom = left[N-1].
        byte c0r0 = (byte)((l0 + l1 + 1) >> 1);
        byte c0r1 = (byte)((l1 + l2 + 1) >> 1);
        byte c0r2 = (byte)((l2 + l3 + 1) >> 1);
        byte c0r3 = (byte)((l3 + l4 + 1) >> 1);
        byte c0r4 = (byte)((l4 + l5 + 1) >> 1);
        byte c0r5 = (byte)((l5 + l6 + 1) >> 1);
        byte c0r6 = (byte)((l6 + l7 + 1) >> 1);
        byte c0r7 = (byte)l7;

        // Column 1: AVG3 triples; second-to-last replicates left[N-1]; bottom = left[N-1].
        byte c1r0 = (byte)((l0 + 2 * l1 + l2 + 2) >> 2);
        byte c1r1 = (byte)((l1 + 2 * l2 + l3 + 2) >> 2);
        byte c1r2 = (byte)((l2 + 2 * l3 + l4 + 2) >> 2);
        byte c1r3 = (byte)((l3 + 2 * l4 + l5 + 2) >> 2);
        byte c1r4 = (byte)((l4 + 2 * l5 + l6 + 2) >> 2);
        byte c1r5 = (byte)((l5 + 2 * l6 + l7 + 2) >> 2);
        byte c1r6 = (byte)((l6 + 2 * l7 + l7 + 2) >> 2);
        byte c1r7 = (byte)l7;

        // Row 0: cols 0,1 + cols 2,3 (= row 1 cols 0,1) + cols 4,5 (= row 2 cols 0,1) + cols 6,7 (= row 3 cols 0,1).
        dstFlat[dBase + 0] = c0r0;
        dstFlat[dBase + 1] = c1r0;
        dstFlat[dBase + 2] = c0r1;
        dstFlat[dBase + 3] = c1r1;
        dstFlat[dBase + 4] = c0r2;
        dstFlat[dBase + 5] = c1r2;
        dstFlat[dBase + 6] = c0r3;
        dstFlat[dBase + 7] = c1r3;

        // Row 1.
        long row1 = dBase + N;
        dstFlat[row1 + 0] = c0r1;
        dstFlat[row1 + 1] = c1r1;
        dstFlat[row1 + 2] = c0r2;
        dstFlat[row1 + 3] = c1r2;
        dstFlat[row1 + 4] = c0r3;
        dstFlat[row1 + 5] = c1r3;
        dstFlat[row1 + 6] = c0r4;
        dstFlat[row1 + 7] = c1r4;

        // Row 2.
        long row2 = dBase + 2 * N;
        dstFlat[row2 + 0] = c0r2;
        dstFlat[row2 + 1] = c1r2;
        dstFlat[row2 + 2] = c0r3;
        dstFlat[row2 + 3] = c1r3;
        dstFlat[row2 + 4] = c0r4;
        dstFlat[row2 + 5] = c1r4;
        dstFlat[row2 + 6] = c0r5;
        dstFlat[row2 + 7] = c1r5;

        // Row 3.
        long row3 = dBase + 3 * N;
        dstFlat[row3 + 0] = c0r3;
        dstFlat[row3 + 1] = c1r3;
        dstFlat[row3 + 2] = c0r4;
        dstFlat[row3 + 3] = c1r4;
        dstFlat[row3 + 4] = c0r5;
        dstFlat[row3 + 5] = c1r5;
        dstFlat[row3 + 6] = c0r6;
        dstFlat[row3 + 7] = c1r6;

        // Row 4.
        long row4 = dBase + 4 * N;
        dstFlat[row4 + 0] = c0r4;
        dstFlat[row4 + 1] = c1r4;
        dstFlat[row4 + 2] = c0r5;
        dstFlat[row4 + 3] = c1r5;
        dstFlat[row4 + 4] = c0r6;
        dstFlat[row4 + 5] = c1r6;
        dstFlat[row4 + 6] = c0r7;
        dstFlat[row4 + 7] = c1r7;

        // Row 5.
        long row5 = dBase + 5 * N;
        dstFlat[row5 + 0] = c0r5;
        dstFlat[row5 + 1] = c1r5;
        dstFlat[row5 + 2] = c0r6;
        dstFlat[row5 + 3] = c1r6;
        dstFlat[row5 + 4] = c0r7;
        dstFlat[row5 + 5] = c1r7;
        dstFlat[row5 + 6] = fillVal;
        dstFlat[row5 + 7] = fillVal;

        // Row 6.
        long row6 = dBase + 6 * N;
        dstFlat[row6 + 0] = c0r6;
        dstFlat[row6 + 1] = c1r6;
        dstFlat[row6 + 2] = c0r7;
        dstFlat[row6 + 3] = c1r7;
        dstFlat[row6 + 4] = fillVal;
        dstFlat[row6 + 5] = fillVal;
        dstFlat[row6 + 6] = fillVal;
        dstFlat[row6 + 7] = fillVal;

        // Row 7.
        long row7 = dBase + 7 * N;
        dstFlat[row7 + 0] = c0r7;
        dstFlat[row7 + 1] = c1r7;
        dstFlat[row7 + 2] = fillVal;
        dstFlat[row7 + 3] = fillVal;
        dstFlat[row7 + 4] = fillVal;
        dstFlat[row7 + 5] = fillVal;
        dstFlat[row7 + 6] = fillVal;
        dstFlat[row7 + 7] = fillVal;
    }

    /// <inheritdoc/>
    public void Dispose() { }
}
