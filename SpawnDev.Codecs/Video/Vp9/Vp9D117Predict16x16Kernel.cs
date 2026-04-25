// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// 16x16 sibling of Vp9D117Predict4x4 / 8x8 kernels. Three-edge mode.
// Rows 0+1 + first column rows 2-15 = 46 register cells; 256 fully
// unrolled output writes. dst[r][c] = dst[r-2][c-1] propagation.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// Batched ILGPU kernel that runs VP9 D117_PRED across N independent
/// 16x16 blocks in parallel.
/// </summary>
public sealed class Vp9D117Predict16x16Kernel : IDisposable
{
    private const int N = 16;

    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, int, int> _kernel;

    /// <summary>Compile the kernel onto <paramref name="accelerator"/>.</summary>
    public Vp9D117Predict16x16Kernel(Accelerator accelerator)
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

    /// <summary>Kernel body. 46 registers; 256 unrolled writes.</summary>
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
        int a8 = aboveFlat[aBase + 8];
        int a9 = aboveFlat[aBase + 9];
        int a10 = aboveFlat[aBase + 10];
        int a11 = aboveFlat[aBase + 11];
        int a12 = aboveFlat[aBase + 12];
        int a13 = aboveFlat[aBase + 13];
        int a14 = aboveFlat[aBase + 14];
        int a15 = aboveFlat[aBase + 15];
        int l0 = leftFlat[lBase + 0];
        int l1 = leftFlat[lBase + 1];
        int l2 = leftFlat[lBase + 2];
        int l3 = leftFlat[lBase + 3];
        int l4 = leftFlat[lBase + 4];
        int l5 = leftFlat[lBase + 5];
        int l6 = leftFlat[lBase + 6];
        int l7 = leftFlat[lBase + 7];
        int l8 = leftFlat[lBase + 8];
        int l9 = leftFlat[lBase + 9];
        int l10 = leftFlat[lBase + 10];
        int l11 = leftFlat[lBase + 11];
        int l12 = leftFlat[lBase + 12];
        int l13 = leftFlat[lBase + 13];
        int l14 = leftFlat[lBase + 14];

        // Row 0 (AVG2 above-offset). 16 cells.
        byte r0c0 = (byte)((tl + a0 + 1) >> 1);
        byte r0c1 = (byte)((a0 + a1 + 1) >> 1);
        byte r0c2 = (byte)((a1 + a2 + 1) >> 1);
        byte r0c3 = (byte)((a2 + a3 + 1) >> 1);
        byte r0c4 = (byte)((a3 + a4 + 1) >> 1);
        byte r0c5 = (byte)((a4 + a5 + 1) >> 1);
        byte r0c6 = (byte)((a5 + a6 + 1) >> 1);
        byte r0c7 = (byte)((a6 + a7 + 1) >> 1);
        byte r0c8 = (byte)((a7 + a8 + 1) >> 1);
        byte r0c9 = (byte)((a8 + a9 + 1) >> 1);
        byte r0c10 = (byte)((a9 + a10 + 1) >> 1);
        byte r0c11 = (byte)((a10 + a11 + 1) >> 1);
        byte r0c12 = (byte)((a11 + a12 + 1) >> 1);
        byte r0c13 = (byte)((a12 + a13 + 1) >> 1);
        byte r0c14 = (byte)((a13 + a14 + 1) >> 1);
        byte r0c15 = (byte)((a14 + a15 + 1) >> 1);

        // Row 1 (AVG3 above-offset). 16 cells.
        byte r1c0 = (byte)((l0 + 2 * tl + a0 + 2) >> 2);
        byte r1c1 = (byte)((tl + 2 * a0 + a1 + 2) >> 2);
        byte r1c2 = (byte)((a0 + 2 * a1 + a2 + 2) >> 2);
        byte r1c3 = (byte)((a1 + 2 * a2 + a3 + 2) >> 2);
        byte r1c4 = (byte)((a2 + 2 * a3 + a4 + 2) >> 2);
        byte r1c5 = (byte)((a3 + 2 * a4 + a5 + 2) >> 2);
        byte r1c6 = (byte)((a4 + 2 * a5 + a6 + 2) >> 2);
        byte r1c7 = (byte)((a5 + 2 * a6 + a7 + 2) >> 2);
        byte r1c8 = (byte)((a6 + 2 * a7 + a8 + 2) >> 2);
        byte r1c9 = (byte)((a7 + 2 * a8 + a9 + 2) >> 2);
        byte r1c10 = (byte)((a8 + 2 * a9 + a10 + 2) >> 2);
        byte r1c11 = (byte)((a9 + 2 * a10 + a11 + 2) >> 2);
        byte r1c12 = (byte)((a10 + 2 * a11 + a12 + 2) >> 2);
        byte r1c13 = (byte)((a11 + 2 * a12 + a13 + 2) >> 2);
        byte r1c14 = (byte)((a12 + 2 * a13 + a14 + 2) >> 2);
        byte r1c15 = (byte)((a13 + 2 * a14 + a15 + 2) >> 2);

        // First column rows 2-15. 14 cells.
        byte r2c0 = (byte)((tl + 2 * l0 + l1 + 2) >> 2);
        byte r3c0 = (byte)((l0 + 2 * l1 + l2 + 2) >> 2);
        byte r4c0 = (byte)((l1 + 2 * l2 + l3 + 2) >> 2);
        byte r5c0 = (byte)((l2 + 2 * l3 + l4 + 2) >> 2);
        byte r6c0 = (byte)((l3 + 2 * l4 + l5 + 2) >> 2);
        byte r7c0 = (byte)((l4 + 2 * l5 + l6 + 2) >> 2);
        byte r8c0 = (byte)((l5 + 2 * l6 + l7 + 2) >> 2);
        byte r9c0 = (byte)((l6 + 2 * l7 + l8 + 2) >> 2);
        byte r10c0 = (byte)((l7 + 2 * l8 + l9 + 2) >> 2);
        byte r11c0 = (byte)((l8 + 2 * l9 + l10 + 2) >> 2);
        byte r12c0 = (byte)((l9 + 2 * l10 + l11 + 2) >> 2);
        byte r13c0 = (byte)((l10 + 2 * l11 + l12 + 2) >> 2);
        byte r14c0 = (byte)((l11 + 2 * l12 + l13 + 2) >> 2);
        byte r15c0 = (byte)((l12 + 2 * l13 + l14 + 2) >> 2);

        // Row 0
        dstFlat[dBase + 0] = r0c0;  dstFlat[dBase + 1] = r0c1;  dstFlat[dBase + 2] = r0c2;  dstFlat[dBase + 3] = r0c3;
        dstFlat[dBase + 4] = r0c4;  dstFlat[dBase + 5] = r0c5;  dstFlat[dBase + 6] = r0c6;  dstFlat[dBase + 7] = r0c7;
        dstFlat[dBase + 8] = r0c8;  dstFlat[dBase + 9] = r0c9;  dstFlat[dBase + 10] = r0c10; dstFlat[dBase + 11] = r0c11;
        dstFlat[dBase + 12] = r0c12; dstFlat[dBase + 13] = r0c13; dstFlat[dBase + 14] = r0c14; dstFlat[dBase + 15] = r0c15;

        // Row 1
        long row1 = dBase + N;
        dstFlat[row1 + 0] = r1c0;  dstFlat[row1 + 1] = r1c1;  dstFlat[row1 + 2] = r1c2;  dstFlat[row1 + 3] = r1c3;
        dstFlat[row1 + 4] = r1c4;  dstFlat[row1 + 5] = r1c5;  dstFlat[row1 + 6] = r1c6;  dstFlat[row1 + 7] = r1c7;
        dstFlat[row1 + 8] = r1c8;  dstFlat[row1 + 9] = r1c9;  dstFlat[row1 + 10] = r1c10; dstFlat[row1 + 11] = r1c11;
        dstFlat[row1 + 12] = r1c12; dstFlat[row1 + 13] = r1c13; dstFlat[row1 + 14] = r1c14; dstFlat[row1 + 15] = r1c15;

        // Row r (>=2) cols [0] = r{r}c0; col[1..15] = (r even) row 0[0..14] : row 1[0..14], shifted via dst[r][c] = dst[r-2][c-1].
        // For row 2: cols 1..15 = row 0 cols 0..14.
        long row2 = dBase + 2 * N;
        dstFlat[row2 + 0] = r2c0;
        dstFlat[row2 + 1] = r0c0; dstFlat[row2 + 2] = r0c1; dstFlat[row2 + 3] = r0c2;
        dstFlat[row2 + 4] = r0c3; dstFlat[row2 + 5] = r0c4; dstFlat[row2 + 6] = r0c5; dstFlat[row2 + 7] = r0c6;
        dstFlat[row2 + 8] = r0c7; dstFlat[row2 + 9] = r0c8; dstFlat[row2 + 10] = r0c9; dstFlat[row2 + 11] = r0c10;
        dstFlat[row2 + 12] = r0c11; dstFlat[row2 + 13] = r0c12; dstFlat[row2 + 14] = r0c13; dstFlat[row2 + 15] = r0c14;

        // Row 3: cols 1..15 = row 1 cols 0..14.
        long row3 = dBase + 3 * N;
        dstFlat[row3 + 0] = r3c0;
        dstFlat[row3 + 1] = r1c0; dstFlat[row3 + 2] = r1c1; dstFlat[row3 + 3] = r1c2;
        dstFlat[row3 + 4] = r1c3; dstFlat[row3 + 5] = r1c4; dstFlat[row3 + 6] = r1c5; dstFlat[row3 + 7] = r1c6;
        dstFlat[row3 + 8] = r1c7; dstFlat[row3 + 9] = r1c8; dstFlat[row3 + 10] = r1c9; dstFlat[row3 + 11] = r1c10;
        dstFlat[row3 + 12] = r1c11; dstFlat[row3 + 13] = r1c12; dstFlat[row3 + 14] = r1c13; dstFlat[row3 + 15] = r1c14;

        // Row 4: col 0 = r4c0; col 1 = r2c0; cols 2..15 = row 0 cols 0..13.
        long row4 = dBase + 4 * N;
        dstFlat[row4 + 0] = r4c0;
        dstFlat[row4 + 1] = r2c0;
        dstFlat[row4 + 2] = r0c0; dstFlat[row4 + 3] = r0c1; dstFlat[row4 + 4] = r0c2; dstFlat[row4 + 5] = r0c3;
        dstFlat[row4 + 6] = r0c4; dstFlat[row4 + 7] = r0c5; dstFlat[row4 + 8] = r0c6; dstFlat[row4 + 9] = r0c7;
        dstFlat[row4 + 10] = r0c8; dstFlat[row4 + 11] = r0c9; dstFlat[row4 + 12] = r0c10; dstFlat[row4 + 13] = r0c11;
        dstFlat[row4 + 14] = r0c12; dstFlat[row4 + 15] = r0c13;

        // Row 5: col 0 = r5c0; col 1 = r3c0; cols 2..15 = row 1 cols 0..13.
        long row5 = dBase + 5 * N;
        dstFlat[row5 + 0] = r5c0;
        dstFlat[row5 + 1] = r3c0;
        dstFlat[row5 + 2] = r1c0; dstFlat[row5 + 3] = r1c1; dstFlat[row5 + 4] = r1c2; dstFlat[row5 + 5] = r1c3;
        dstFlat[row5 + 6] = r1c4; dstFlat[row5 + 7] = r1c5; dstFlat[row5 + 8] = r1c6; dstFlat[row5 + 9] = r1c7;
        dstFlat[row5 + 10] = r1c8; dstFlat[row5 + 11] = r1c9; dstFlat[row5 + 12] = r1c10; dstFlat[row5 + 13] = r1c11;
        dstFlat[row5 + 14] = r1c12; dstFlat[row5 + 15] = r1c13;

        // Row 6: col 0 = r6c0; col 1 = r4c0; col 2 = r2c0; cols 3..15 = row 0 cols 0..12.
        long row6 = dBase + 6 * N;
        dstFlat[row6 + 0] = r6c0;
        dstFlat[row6 + 1] = r4c0;
        dstFlat[row6 + 2] = r2c0;
        dstFlat[row6 + 3] = r0c0; dstFlat[row6 + 4] = r0c1; dstFlat[row6 + 5] = r0c2; dstFlat[row6 + 6] = r0c3;
        dstFlat[row6 + 7] = r0c4; dstFlat[row6 + 8] = r0c5; dstFlat[row6 + 9] = r0c6; dstFlat[row6 + 10] = r0c7;
        dstFlat[row6 + 11] = r0c8; dstFlat[row6 + 12] = r0c9; dstFlat[row6 + 13] = r0c10; dstFlat[row6 + 14] = r0c11;
        dstFlat[row6 + 15] = r0c12;

        // Row 7
        long row7 = dBase + 7 * N;
        dstFlat[row7 + 0] = r7c0;
        dstFlat[row7 + 1] = r5c0;
        dstFlat[row7 + 2] = r3c0;
        dstFlat[row7 + 3] = r1c0; dstFlat[row7 + 4] = r1c1; dstFlat[row7 + 5] = r1c2; dstFlat[row7 + 6] = r1c3;
        dstFlat[row7 + 7] = r1c4; dstFlat[row7 + 8] = r1c5; dstFlat[row7 + 9] = r1c6; dstFlat[row7 + 10] = r1c7;
        dstFlat[row7 + 11] = r1c8; dstFlat[row7 + 12] = r1c9; dstFlat[row7 + 13] = r1c10; dstFlat[row7 + 14] = r1c11;
        dstFlat[row7 + 15] = r1c12;

        // Row 8: col 0..3 = r8/r6/r4/r2; cols 4..15 = row 0 cols 0..11.
        long row8 = dBase + 8 * N;
        dstFlat[row8 + 0] = r8c0;
        dstFlat[row8 + 1] = r6c0;
        dstFlat[row8 + 2] = r4c0;
        dstFlat[row8 + 3] = r2c0;
        dstFlat[row8 + 4] = r0c0; dstFlat[row8 + 5] = r0c1; dstFlat[row8 + 6] = r0c2; dstFlat[row8 + 7] = r0c3;
        dstFlat[row8 + 8] = r0c4; dstFlat[row8 + 9] = r0c5; dstFlat[row8 + 10] = r0c6; dstFlat[row8 + 11] = r0c7;
        dstFlat[row8 + 12] = r0c8; dstFlat[row8 + 13] = r0c9; dstFlat[row8 + 14] = r0c10; dstFlat[row8 + 15] = r0c11;

        // Row 9
        long row9 = dBase + 9 * N;
        dstFlat[row9 + 0] = r9c0;
        dstFlat[row9 + 1] = r7c0;
        dstFlat[row9 + 2] = r5c0;
        dstFlat[row9 + 3] = r3c0;
        dstFlat[row9 + 4] = r1c0; dstFlat[row9 + 5] = r1c1; dstFlat[row9 + 6] = r1c2; dstFlat[row9 + 7] = r1c3;
        dstFlat[row9 + 8] = r1c4; dstFlat[row9 + 9] = r1c5; dstFlat[row9 + 10] = r1c6; dstFlat[row9 + 11] = r1c7;
        dstFlat[row9 + 12] = r1c8; dstFlat[row9 + 13] = r1c9; dstFlat[row9 + 14] = r1c10; dstFlat[row9 + 15] = r1c11;

        // Row 10
        long row10 = dBase + 10 * N;
        dstFlat[row10 + 0] = r10c0;
        dstFlat[row10 + 1] = r8c0;
        dstFlat[row10 + 2] = r6c0;
        dstFlat[row10 + 3] = r4c0;
        dstFlat[row10 + 4] = r2c0;
        dstFlat[row10 + 5] = r0c0; dstFlat[row10 + 6] = r0c1; dstFlat[row10 + 7] = r0c2;
        dstFlat[row10 + 8] = r0c3; dstFlat[row10 + 9] = r0c4; dstFlat[row10 + 10] = r0c5; dstFlat[row10 + 11] = r0c6;
        dstFlat[row10 + 12] = r0c7; dstFlat[row10 + 13] = r0c8; dstFlat[row10 + 14] = r0c9; dstFlat[row10 + 15] = r0c10;

        // Row 11
        long row11 = dBase + 11 * N;
        dstFlat[row11 + 0] = r11c0;
        dstFlat[row11 + 1] = r9c0;
        dstFlat[row11 + 2] = r7c0;
        dstFlat[row11 + 3] = r5c0;
        dstFlat[row11 + 4] = r3c0;
        dstFlat[row11 + 5] = r1c0; dstFlat[row11 + 6] = r1c1; dstFlat[row11 + 7] = r1c2;
        dstFlat[row11 + 8] = r1c3; dstFlat[row11 + 9] = r1c4; dstFlat[row11 + 10] = r1c5; dstFlat[row11 + 11] = r1c6;
        dstFlat[row11 + 12] = r1c7; dstFlat[row11 + 13] = r1c8; dstFlat[row11 + 14] = r1c9; dstFlat[row11 + 15] = r1c10;

        // Row 12
        long row12 = dBase + 12 * N;
        dstFlat[row12 + 0] = r12c0;
        dstFlat[row12 + 1] = r10c0;
        dstFlat[row12 + 2] = r8c0;
        dstFlat[row12 + 3] = r6c0;
        dstFlat[row12 + 4] = r4c0;
        dstFlat[row12 + 5] = r2c0;
        dstFlat[row12 + 6] = r0c0; dstFlat[row12 + 7] = r0c1;
        dstFlat[row12 + 8] = r0c2; dstFlat[row12 + 9] = r0c3; dstFlat[row12 + 10] = r0c4; dstFlat[row12 + 11] = r0c5;
        dstFlat[row12 + 12] = r0c6; dstFlat[row12 + 13] = r0c7; dstFlat[row12 + 14] = r0c8; dstFlat[row12 + 15] = r0c9;

        // Row 13
        long row13 = dBase + 13 * N;
        dstFlat[row13 + 0] = r13c0;
        dstFlat[row13 + 1] = r11c0;
        dstFlat[row13 + 2] = r9c0;
        dstFlat[row13 + 3] = r7c0;
        dstFlat[row13 + 4] = r5c0;
        dstFlat[row13 + 5] = r3c0;
        dstFlat[row13 + 6] = r1c0; dstFlat[row13 + 7] = r1c1;
        dstFlat[row13 + 8] = r1c2; dstFlat[row13 + 9] = r1c3; dstFlat[row13 + 10] = r1c4; dstFlat[row13 + 11] = r1c5;
        dstFlat[row13 + 12] = r1c6; dstFlat[row13 + 13] = r1c7; dstFlat[row13 + 14] = r1c8; dstFlat[row13 + 15] = r1c9;

        // Row 14
        long row14 = dBase + 14 * N;
        dstFlat[row14 + 0] = r14c0;
        dstFlat[row14 + 1] = r12c0;
        dstFlat[row14 + 2] = r10c0;
        dstFlat[row14 + 3] = r8c0;
        dstFlat[row14 + 4] = r6c0;
        dstFlat[row14 + 5] = r4c0;
        dstFlat[row14 + 6] = r2c0;
        dstFlat[row14 + 7] = r0c0;
        dstFlat[row14 + 8] = r0c1; dstFlat[row14 + 9] = r0c2; dstFlat[row14 + 10] = r0c3; dstFlat[row14 + 11] = r0c4;
        dstFlat[row14 + 12] = r0c5; dstFlat[row14 + 13] = r0c6; dstFlat[row14 + 14] = r0c7; dstFlat[row14 + 15] = r0c8;

        // Row 15
        long row15 = dBase + 15 * N;
        dstFlat[row15 + 0] = r15c0;
        dstFlat[row15 + 1] = r13c0;
        dstFlat[row15 + 2] = r11c0;
        dstFlat[row15 + 3] = r9c0;
        dstFlat[row15 + 4] = r7c0;
        dstFlat[row15 + 5] = r5c0;
        dstFlat[row15 + 6] = r3c0;
        dstFlat[row15 + 7] = r1c0;
        dstFlat[row15 + 8] = r1c1; dstFlat[row15 + 9] = r1c2; dstFlat[row15 + 10] = r1c3; dstFlat[row15 + 11] = r1c4;
        dstFlat[row15 + 12] = r1c5; dstFlat[row15 + 13] = r1c6; dstFlat[row15 + 14] = r1c7; dstFlat[row15 + 15] = r1c8;
    }

    /// <inheritdoc/>
    public void Dispose() { }
}
