// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// 16x16 sibling of Vp9D63Predict4x4 / 8x8 kernels. N=16, two-filter
// seed (rows 0+1 = 32 register cells); shift-and-pad propagation
// for rows 2-15 with switch dispatch on src index. fillVal=above[15].

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// Batched ILGPU kernel that runs VP9 D63_PRED across N independent
/// 16x16 blocks in parallel.
/// </summary>
public sealed class Vp9D63Predict16x16Kernel : IDisposable
{
    private const int N = 16;

    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, int, int> _kernel;

    /// <summary>Compile the kernel onto <paramref name="accelerator"/>.</summary>
    public Vp9D63Predict16x16Kernel(Accelerator accelerator)
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

    /// <summary>Kernel body. 32 register cells; dispatch on (row, srcIdx).</summary>
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

        // Load 18 above samples.
        int v0 = aboveFlat[aBase + 0];
        int v1 = aboveFlat[aBase + 1];
        int v2 = aboveFlat[aBase + 2];
        int v3 = aboveFlat[aBase + 3];
        int v4 = aboveFlat[aBase + 4];
        int v5 = aboveFlat[aBase + 5];
        int v6 = aboveFlat[aBase + 6];
        int v7 = aboveFlat[aBase + 7];
        int v8 = aboveFlat[aBase + 8];
        int v9 = aboveFlat[aBase + 9];
        int v10 = aboveFlat[aBase + 10];
        int v11 = aboveFlat[aBase + 11];
        int v12 = aboveFlat[aBase + 12];
        int v13 = aboveFlat[aBase + 13];
        int v14 = aboveFlat[aBase + 14];
        int v15 = aboveFlat[aBase + 15];
        int v16 = aboveFlat[aBase + 16];
        int v17 = aboveFlat[aBase + 17];

        byte fillVal = (byte)v15;  // above[N-1]

        // Row 0 (AVG2 of consecutive above pairs).
        byte r0c0 = (byte)((v0 + v1 + 1) >> 1);
        byte r0c1 = (byte)((v1 + v2 + 1) >> 1);
        byte r0c2 = (byte)((v2 + v3 + 1) >> 1);
        byte r0c3 = (byte)((v3 + v4 + 1) >> 1);
        byte r0c4 = (byte)((v4 + v5 + 1) >> 1);
        byte r0c5 = (byte)((v5 + v6 + 1) >> 1);
        byte r0c6 = (byte)((v6 + v7 + 1) >> 1);
        byte r0c7 = (byte)((v7 + v8 + 1) >> 1);
        byte r0c8 = (byte)((v8 + v9 + 1) >> 1);
        byte r0c9 = (byte)((v9 + v10 + 1) >> 1);
        byte r0c10 = (byte)((v10 + v11 + 1) >> 1);
        byte r0c11 = (byte)((v11 + v12 + 1) >> 1);
        byte r0c12 = (byte)((v12 + v13 + 1) >> 1);
        byte r0c13 = (byte)((v13 + v14 + 1) >> 1);
        byte r0c14 = (byte)((v14 + v15 + 1) >> 1);
        byte r0c15 = (byte)((v15 + v16 + 1) >> 1);

        // Row 1 (AVG3 of consecutive above triples).
        byte r1c0 = (byte)((v0 + 2 * v1 + v2 + 2) >> 2);
        byte r1c1 = (byte)((v1 + 2 * v2 + v3 + 2) >> 2);
        byte r1c2 = (byte)((v2 + 2 * v3 + v4 + 2) >> 2);
        byte r1c3 = (byte)((v3 + 2 * v4 + v5 + 2) >> 2);
        byte r1c4 = (byte)((v4 + 2 * v5 + v6 + 2) >> 2);
        byte r1c5 = (byte)((v5 + 2 * v6 + v7 + 2) >> 2);
        byte r1c6 = (byte)((v6 + 2 * v7 + v8 + 2) >> 2);
        byte r1c7 = (byte)((v7 + 2 * v8 + v9 + 2) >> 2);
        byte r1c8 = (byte)((v8 + 2 * v9 + v10 + 2) >> 2);
        byte r1c9 = (byte)((v9 + 2 * v10 + v11 + 2) >> 2);
        byte r1c10 = (byte)((v10 + 2 * v11 + v12 + 2) >> 2);
        byte r1c11 = (byte)((v11 + 2 * v12 + v13 + 2) >> 2);
        byte r1c12 = (byte)((v12 + 2 * v13 + v14 + 2) >> 2);
        byte r1c13 = (byte)((v13 + 2 * v14 + v15 + 2) >> 2);
        byte r1c14 = (byte)((v14 + 2 * v15 + v16 + 2) >> 2);
        byte r1c15 = (byte)((v15 + 2 * v16 + v17 + 2) >> 2);

        // Fully unrolled 256 byte writes - the switch-dispatch
        // approach that worked for D45 16x16 (single 16-case switch
        // on r0) failed here on WebGPU when extended to 32 cases.
        // Mechanical unroll matches the verified D63 8x8 pattern.

        // Row 0
        dstFlat[dBase + 0] = r0c0;  dstFlat[dBase + 1] = r0c1;
        dstFlat[dBase + 2] = r0c2;  dstFlat[dBase + 3] = r0c3;
        dstFlat[dBase + 4] = r0c4;  dstFlat[dBase + 5] = r0c5;
        dstFlat[dBase + 6] = r0c6;  dstFlat[dBase + 7] = r0c7;
        dstFlat[dBase + 8] = r0c8;  dstFlat[dBase + 9] = r0c9;
        dstFlat[dBase + 10] = r0c10; dstFlat[dBase + 11] = r0c11;
        dstFlat[dBase + 12] = r0c12; dstFlat[dBase + 13] = r0c13;
        dstFlat[dBase + 14] = r0c14; dstFlat[dBase + 15] = r0c15;

        // Row 1
        long row1 = dBase + N;
        dstFlat[row1 + 0] = r1c0;  dstFlat[row1 + 1] = r1c1;
        dstFlat[row1 + 2] = r1c2;  dstFlat[row1 + 3] = r1c3;
        dstFlat[row1 + 4] = r1c4;  dstFlat[row1 + 5] = r1c5;
        dstFlat[row1 + 6] = r1c6;  dstFlat[row1 + 7] = r1c7;
        dstFlat[row1 + 8] = r1c8;  dstFlat[row1 + 9] = r1c9;
        dstFlat[row1 + 10] = r1c10; dstFlat[row1 + 11] = r1c11;
        dstFlat[row1 + 12] = r1c12; dstFlat[row1 + 13] = r1c13;
        dstFlat[row1 + 14] = r1c14; dstFlat[row1 + 15] = r1c15;

        // Even rows shift row 0 by row/2; odd rows shift row 1 by row/2.
        // Valid cells: c + row/2 <= N-2 = 14. Past that = fillVal.
        // For each even row r (>=2): c=0..(14-r/2) reads r0c[r/2..14]; rest fillVal.

        // Row 2 (offset 1, valid cols 0..13 from row 0[1..14]).
        long row2 = dBase + 2 * N;
        dstFlat[row2 + 0] = r0c1;  dstFlat[row2 + 1] = r0c2;
        dstFlat[row2 + 2] = r0c3;  dstFlat[row2 + 3] = r0c4;
        dstFlat[row2 + 4] = r0c5;  dstFlat[row2 + 5] = r0c6;
        dstFlat[row2 + 6] = r0c7;  dstFlat[row2 + 7] = r0c8;
        dstFlat[row2 + 8] = r0c9;  dstFlat[row2 + 9] = r0c10;
        dstFlat[row2 + 10] = r0c11; dstFlat[row2 + 11] = r0c12;
        dstFlat[row2 + 12] = r0c13; dstFlat[row2 + 13] = r0c14;
        dstFlat[row2 + 14] = fillVal; dstFlat[row2 + 15] = fillVal;

        // Row 3
        long row3 = dBase + 3 * N;
        dstFlat[row3 + 0] = r1c1;  dstFlat[row3 + 1] = r1c2;
        dstFlat[row3 + 2] = r1c3;  dstFlat[row3 + 3] = r1c4;
        dstFlat[row3 + 4] = r1c5;  dstFlat[row3 + 5] = r1c6;
        dstFlat[row3 + 6] = r1c7;  dstFlat[row3 + 7] = r1c8;
        dstFlat[row3 + 8] = r1c9;  dstFlat[row3 + 9] = r1c10;
        dstFlat[row3 + 10] = r1c11; dstFlat[row3 + 11] = r1c12;
        dstFlat[row3 + 12] = r1c13; dstFlat[row3 + 13] = r1c14;
        dstFlat[row3 + 14] = fillVal; dstFlat[row3 + 15] = fillVal;

        // Row 4 (offset 2, valid cols 0..12 from row 0[2..14]).
        long row4 = dBase + 4 * N;
        dstFlat[row4 + 0] = r0c2;  dstFlat[row4 + 1] = r0c3;
        dstFlat[row4 + 2] = r0c4;  dstFlat[row4 + 3] = r0c5;
        dstFlat[row4 + 4] = r0c6;  dstFlat[row4 + 5] = r0c7;
        dstFlat[row4 + 6] = r0c8;  dstFlat[row4 + 7] = r0c9;
        dstFlat[row4 + 8] = r0c10; dstFlat[row4 + 9] = r0c11;
        dstFlat[row4 + 10] = r0c12; dstFlat[row4 + 11] = r0c13;
        dstFlat[row4 + 12] = r0c14;
        dstFlat[row4 + 13] = fillVal; dstFlat[row4 + 14] = fillVal; dstFlat[row4 + 15] = fillVal;

        // Row 5
        long row5 = dBase + 5 * N;
        dstFlat[row5 + 0] = r1c2;  dstFlat[row5 + 1] = r1c3;
        dstFlat[row5 + 2] = r1c4;  dstFlat[row5 + 3] = r1c5;
        dstFlat[row5 + 4] = r1c6;  dstFlat[row5 + 5] = r1c7;
        dstFlat[row5 + 6] = r1c8;  dstFlat[row5 + 7] = r1c9;
        dstFlat[row5 + 8] = r1c10; dstFlat[row5 + 9] = r1c11;
        dstFlat[row5 + 10] = r1c12; dstFlat[row5 + 11] = r1c13;
        dstFlat[row5 + 12] = r1c14;
        dstFlat[row5 + 13] = fillVal; dstFlat[row5 + 14] = fillVal; dstFlat[row5 + 15] = fillVal;

        // Row 6 (offset 3, valid cols 0..11).
        long row6 = dBase + 6 * N;
        dstFlat[row6 + 0] = r0c3;  dstFlat[row6 + 1] = r0c4;
        dstFlat[row6 + 2] = r0c5;  dstFlat[row6 + 3] = r0c6;
        dstFlat[row6 + 4] = r0c7;  dstFlat[row6 + 5] = r0c8;
        dstFlat[row6 + 6] = r0c9;  dstFlat[row6 + 7] = r0c10;
        dstFlat[row6 + 8] = r0c11; dstFlat[row6 + 9] = r0c12;
        dstFlat[row6 + 10] = r0c13; dstFlat[row6 + 11] = r0c14;
        dstFlat[row6 + 12] = fillVal; dstFlat[row6 + 13] = fillVal;
        dstFlat[row6 + 14] = fillVal; dstFlat[row6 + 15] = fillVal;

        // Row 7
        long row7 = dBase + 7 * N;
        dstFlat[row7 + 0] = r1c3;  dstFlat[row7 + 1] = r1c4;
        dstFlat[row7 + 2] = r1c5;  dstFlat[row7 + 3] = r1c6;
        dstFlat[row7 + 4] = r1c7;  dstFlat[row7 + 5] = r1c8;
        dstFlat[row7 + 6] = r1c9;  dstFlat[row7 + 7] = r1c10;
        dstFlat[row7 + 8] = r1c11; dstFlat[row7 + 9] = r1c12;
        dstFlat[row7 + 10] = r1c13; dstFlat[row7 + 11] = r1c14;
        dstFlat[row7 + 12] = fillVal; dstFlat[row7 + 13] = fillVal;
        dstFlat[row7 + 14] = fillVal; dstFlat[row7 + 15] = fillVal;

        // Row 8 (offset 4, valid cols 0..10).
        long row8 = dBase + 8 * N;
        dstFlat[row8 + 0] = r0c4;  dstFlat[row8 + 1] = r0c5;
        dstFlat[row8 + 2] = r0c6;  dstFlat[row8 + 3] = r0c7;
        dstFlat[row8 + 4] = r0c8;  dstFlat[row8 + 5] = r0c9;
        dstFlat[row8 + 6] = r0c10; dstFlat[row8 + 7] = r0c11;
        dstFlat[row8 + 8] = r0c12; dstFlat[row8 + 9] = r0c13;
        dstFlat[row8 + 10] = r0c14;
        dstFlat[row8 + 11] = fillVal; dstFlat[row8 + 12] = fillVal;
        dstFlat[row8 + 13] = fillVal; dstFlat[row8 + 14] = fillVal; dstFlat[row8 + 15] = fillVal;

        // Row 9
        long row9 = dBase + 9 * N;
        dstFlat[row9 + 0] = r1c4;  dstFlat[row9 + 1] = r1c5;
        dstFlat[row9 + 2] = r1c6;  dstFlat[row9 + 3] = r1c7;
        dstFlat[row9 + 4] = r1c8;  dstFlat[row9 + 5] = r1c9;
        dstFlat[row9 + 6] = r1c10; dstFlat[row9 + 7] = r1c11;
        dstFlat[row9 + 8] = r1c12; dstFlat[row9 + 9] = r1c13;
        dstFlat[row9 + 10] = r1c14;
        dstFlat[row9 + 11] = fillVal; dstFlat[row9 + 12] = fillVal;
        dstFlat[row9 + 13] = fillVal; dstFlat[row9 + 14] = fillVal; dstFlat[row9 + 15] = fillVal;

        // Row 10 (offset 5, valid cols 0..9).
        long row10 = dBase + 10 * N;
        dstFlat[row10 + 0] = r0c5;  dstFlat[row10 + 1] = r0c6;
        dstFlat[row10 + 2] = r0c7;  dstFlat[row10 + 3] = r0c8;
        dstFlat[row10 + 4] = r0c9;  dstFlat[row10 + 5] = r0c10;
        dstFlat[row10 + 6] = r0c11; dstFlat[row10 + 7] = r0c12;
        dstFlat[row10 + 8] = r0c13; dstFlat[row10 + 9] = r0c14;
        dstFlat[row10 + 10] = fillVal; dstFlat[row10 + 11] = fillVal;
        dstFlat[row10 + 12] = fillVal; dstFlat[row10 + 13] = fillVal;
        dstFlat[row10 + 14] = fillVal; dstFlat[row10 + 15] = fillVal;

        // Row 11
        long row11 = dBase + 11 * N;
        dstFlat[row11 + 0] = r1c5;  dstFlat[row11 + 1] = r1c6;
        dstFlat[row11 + 2] = r1c7;  dstFlat[row11 + 3] = r1c8;
        dstFlat[row11 + 4] = r1c9;  dstFlat[row11 + 5] = r1c10;
        dstFlat[row11 + 6] = r1c11; dstFlat[row11 + 7] = r1c12;
        dstFlat[row11 + 8] = r1c13; dstFlat[row11 + 9] = r1c14;
        dstFlat[row11 + 10] = fillVal; dstFlat[row11 + 11] = fillVal;
        dstFlat[row11 + 12] = fillVal; dstFlat[row11 + 13] = fillVal;
        dstFlat[row11 + 14] = fillVal; dstFlat[row11 + 15] = fillVal;

        // Row 12 (offset 6, valid cols 0..8).
        long row12 = dBase + 12 * N;
        dstFlat[row12 + 0] = r0c6;  dstFlat[row12 + 1] = r0c7;
        dstFlat[row12 + 2] = r0c8;  dstFlat[row12 + 3] = r0c9;
        dstFlat[row12 + 4] = r0c10; dstFlat[row12 + 5] = r0c11;
        dstFlat[row12 + 6] = r0c12; dstFlat[row12 + 7] = r0c13;
        dstFlat[row12 + 8] = r0c14;
        dstFlat[row12 + 9] = fillVal; dstFlat[row12 + 10] = fillVal;
        dstFlat[row12 + 11] = fillVal; dstFlat[row12 + 12] = fillVal;
        dstFlat[row12 + 13] = fillVal; dstFlat[row12 + 14] = fillVal; dstFlat[row12 + 15] = fillVal;

        // Row 13
        long row13 = dBase + 13 * N;
        dstFlat[row13 + 0] = r1c6;  dstFlat[row13 + 1] = r1c7;
        dstFlat[row13 + 2] = r1c8;  dstFlat[row13 + 3] = r1c9;
        dstFlat[row13 + 4] = r1c10; dstFlat[row13 + 5] = r1c11;
        dstFlat[row13 + 6] = r1c12; dstFlat[row13 + 7] = r1c13;
        dstFlat[row13 + 8] = r1c14;
        dstFlat[row13 + 9] = fillVal; dstFlat[row13 + 10] = fillVal;
        dstFlat[row13 + 11] = fillVal; dstFlat[row13 + 12] = fillVal;
        dstFlat[row13 + 13] = fillVal; dstFlat[row13 + 14] = fillVal; dstFlat[row13 + 15] = fillVal;

        // Row 14 (offset 7, valid cols 0..7).
        long row14 = dBase + 14 * N;
        dstFlat[row14 + 0] = r0c7;  dstFlat[row14 + 1] = r0c8;
        dstFlat[row14 + 2] = r0c9;  dstFlat[row14 + 3] = r0c10;
        dstFlat[row14 + 4] = r0c11; dstFlat[row14 + 5] = r0c12;
        dstFlat[row14 + 6] = r0c13; dstFlat[row14 + 7] = r0c14;
        dstFlat[row14 + 8] = fillVal; dstFlat[row14 + 9] = fillVal;
        dstFlat[row14 + 10] = fillVal; dstFlat[row14 + 11] = fillVal;
        dstFlat[row14 + 12] = fillVal; dstFlat[row14 + 13] = fillVal;
        dstFlat[row14 + 14] = fillVal; dstFlat[row14 + 15] = fillVal;

        // Row 15
        long row15 = dBase + 15 * N;
        dstFlat[row15 + 0] = r1c7;  dstFlat[row15 + 1] = r1c8;
        dstFlat[row15 + 2] = r1c9;  dstFlat[row15 + 3] = r1c10;
        dstFlat[row15 + 4] = r1c11; dstFlat[row15 + 5] = r1c12;
        dstFlat[row15 + 6] = r1c13; dstFlat[row15 + 7] = r1c14;
        dstFlat[row15 + 8] = fillVal; dstFlat[row15 + 9] = fillVal;
        dstFlat[row15 + 10] = fillVal; dstFlat[row15 + 11] = fillVal;
        dstFlat[row15 + 12] = fillVal; dstFlat[row15 + 13] = fillVal;
        dstFlat[row15 + 14] = fillVal; dstFlat[row15 + 15] = fillVal;
    }

    /// <inheritdoc/>
    public void Dispose() { }
}
