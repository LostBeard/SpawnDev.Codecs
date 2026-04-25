// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// 16x16 sibling of Vp9D153Predict4x4 / 8x8 kernels. Mirror of D117
// at 16x16. Three-edge mode. 46 register cells (cols 0+1 + first
// row past col 1) drive 256 unrolled output writes.
//
// Closes the 10-mode VP9 intra prediction GPU port at 16x16.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// Batched ILGPU kernel that runs VP9 D153_PRED across N independent
/// 16x16 blocks in parallel.
/// </summary>
public sealed class Vp9D153Predict16x16Kernel : IDisposable
{
    private const int N = 16;

    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, int, int> _kernel;

    /// <summary>Compile the kernel onto <paramref name="accelerator"/>.</summary>
    public Vp9D153Predict16x16Kernel(Accelerator accelerator)
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

    /// <summary>
    /// Kernel body. 46 register cells; 256 unrolled writes.
    /// For row r, col c:
    ///   c=0: c0r[r];  c=1: c1r[r]
    ///   c even, c/2 <= r: c0r[r - c/2]
    ///   c odd, (c-1)/2 <= r: c1r[r - (c-1)/2]
    ///   else: r0c[c - 2*r]
    /// </summary>
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
        int a7 = aboveFlat[aBase + 7];
        int a8 = aboveFlat[aBase + 8];
        int a9 = aboveFlat[aBase + 9];
        int a10 = aboveFlat[aBase + 10];
        int a11 = aboveFlat[aBase + 11];
        int a12 = aboveFlat[aBase + 12];
        int a13 = aboveFlat[aBase + 13];
        int a14 = aboveFlat[aBase + 14];
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
        int l15 = leftFlat[lBase + 15];

        // Column 0 (16 cells): c0r0 = AVG2(tl, l0); c0r[r] = AVG2(l[r-1], l[r]) for r=1..15.
        byte c0r0 = (byte)((tl + l0 + 1) >> 1);
        byte c0r1 = (byte)((l0 + l1 + 1) >> 1);
        byte c0r2 = (byte)((l1 + l2 + 1) >> 1);
        byte c0r3 = (byte)((l2 + l3 + 1) >> 1);
        byte c0r4 = (byte)((l3 + l4 + 1) >> 1);
        byte c0r5 = (byte)((l4 + l5 + 1) >> 1);
        byte c0r6 = (byte)((l5 + l6 + 1) >> 1);
        byte c0r7 = (byte)((l6 + l7 + 1) >> 1);
        byte c0r8 = (byte)((l7 + l8 + 1) >> 1);
        byte c0r9 = (byte)((l8 + l9 + 1) >> 1);
        byte c0r10 = (byte)((l9 + l10 + 1) >> 1);
        byte c0r11 = (byte)((l10 + l11 + 1) >> 1);
        byte c0r12 = (byte)((l11 + l12 + 1) >> 1);
        byte c0r13 = (byte)((l12 + l13 + 1) >> 1);
        byte c0r14 = (byte)((l13 + l14 + 1) >> 1);
        byte c0r15 = (byte)((l14 + l15 + 1) >> 1);

        // Column 1 (16 cells).
        byte c1r0 = (byte)((l0 + 2 * tl + a0 + 2) >> 2);
        byte c1r1 = (byte)((tl + 2 * l0 + l1 + 2) >> 2);
        byte c1r2 = (byte)((l0 + 2 * l1 + l2 + 2) >> 2);
        byte c1r3 = (byte)((l1 + 2 * l2 + l3 + 2) >> 2);
        byte c1r4 = (byte)((l2 + 2 * l3 + l4 + 2) >> 2);
        byte c1r5 = (byte)((l3 + 2 * l4 + l5 + 2) >> 2);
        byte c1r6 = (byte)((l4 + 2 * l5 + l6 + 2) >> 2);
        byte c1r7 = (byte)((l5 + 2 * l6 + l7 + 2) >> 2);
        byte c1r8 = (byte)((l6 + 2 * l7 + l8 + 2) >> 2);
        byte c1r9 = (byte)((l7 + 2 * l8 + l9 + 2) >> 2);
        byte c1r10 = (byte)((l8 + 2 * l9 + l10 + 2) >> 2);
        byte c1r11 = (byte)((l9 + 2 * l10 + l11 + 2) >> 2);
        byte c1r12 = (byte)((l10 + 2 * l11 + l12 + 2) >> 2);
        byte c1r13 = (byte)((l11 + 2 * l12 + l13 + 2) >> 2);
        byte c1r14 = (byte)((l12 + 2 * l13 + l14 + 2) >> 2);
        byte c1r15 = (byte)((l13 + 2 * l14 + l15 + 2) >> 2);

        // Row 0 cols 2..15 (14 cells).
        byte r0c2 = (byte)((tl + 2 * a0 + a1 + 2) >> 2);
        byte r0c3 = (byte)((a0 + 2 * a1 + a2 + 2) >> 2);
        byte r0c4 = (byte)((a1 + 2 * a2 + a3 + 2) >> 2);
        byte r0c5 = (byte)((a2 + 2 * a3 + a4 + 2) >> 2);
        byte r0c6 = (byte)((a3 + 2 * a4 + a5 + 2) >> 2);
        byte r0c7 = (byte)((a4 + 2 * a5 + a6 + 2) >> 2);
        byte r0c8 = (byte)((a5 + 2 * a6 + a7 + 2) >> 2);
        byte r0c9 = (byte)((a6 + 2 * a7 + a8 + 2) >> 2);
        byte r0c10 = (byte)((a7 + 2 * a8 + a9 + 2) >> 2);
        byte r0c11 = (byte)((a8 + 2 * a9 + a10 + 2) >> 2);
        byte r0c12 = (byte)((a9 + 2 * a10 + a11 + 2) >> 2);
        byte r0c13 = (byte)((a10 + 2 * a11 + a12 + 2) >> 2);
        byte r0c14 = (byte)((a11 + 2 * a12 + a13 + 2) >> 2);
        byte r0c15 = (byte)((a12 + 2 * a13 + a14 + 2) >> 2);

        // Row 0: [c0r0, c1r0, r0c2..r0c15]
        dstFlat[dBase + 0] = c0r0;  dstFlat[dBase + 1] = c1r0;
        dstFlat[dBase + 2] = r0c2;  dstFlat[dBase + 3] = r0c3;
        dstFlat[dBase + 4] = r0c4;  dstFlat[dBase + 5] = r0c5;
        dstFlat[dBase + 6] = r0c6;  dstFlat[dBase + 7] = r0c7;
        dstFlat[dBase + 8] = r0c8;  dstFlat[dBase + 9] = r0c9;
        dstFlat[dBase + 10] = r0c10; dstFlat[dBase + 11] = r0c11;
        dstFlat[dBase + 12] = r0c12; dstFlat[dBase + 13] = r0c13;
        dstFlat[dBase + 14] = r0c14; dstFlat[dBase + 15] = r0c15;

        // Row 1: [c0r1, c1r1, c0r0, c1r0, r0c2..r0c13]
        long row1 = dBase + N;
        dstFlat[row1 + 0] = c0r1; dstFlat[row1 + 1] = c1r1;
        dstFlat[row1 + 2] = c0r0; dstFlat[row1 + 3] = c1r0;
        dstFlat[row1 + 4] = r0c2; dstFlat[row1 + 5] = r0c3;
        dstFlat[row1 + 6] = r0c4; dstFlat[row1 + 7] = r0c5;
        dstFlat[row1 + 8] = r0c6; dstFlat[row1 + 9] = r0c7;
        dstFlat[row1 + 10] = r0c8; dstFlat[row1 + 11] = r0c9;
        dstFlat[row1 + 12] = r0c10; dstFlat[row1 + 13] = r0c11;
        dstFlat[row1 + 14] = r0c12; dstFlat[row1 + 15] = r0c13;

        // Row 2: [c0r2, c1r2, c0r1, c1r1, c0r0, c1r0, r0c2..r0c11]
        long row2 = dBase + 2 * N;
        dstFlat[row2 + 0] = c0r2; dstFlat[row2 + 1] = c1r2;
        dstFlat[row2 + 2] = c0r1; dstFlat[row2 + 3] = c1r1;
        dstFlat[row2 + 4] = c0r0; dstFlat[row2 + 5] = c1r0;
        dstFlat[row2 + 6] = r0c2; dstFlat[row2 + 7] = r0c3;
        dstFlat[row2 + 8] = r0c4; dstFlat[row2 + 9] = r0c5;
        dstFlat[row2 + 10] = r0c6; dstFlat[row2 + 11] = r0c7;
        dstFlat[row2 + 12] = r0c8; dstFlat[row2 + 13] = r0c9;
        dstFlat[row2 + 14] = r0c10; dstFlat[row2 + 15] = r0c11;

        // Row 3: c0r3, c1r3, c0r2, c1r2, c0r1, c1r1, c0r0, c1r0, r0c2..r0c9
        long row3 = dBase + 3 * N;
        dstFlat[row3 + 0] = c0r3; dstFlat[row3 + 1] = c1r3;
        dstFlat[row3 + 2] = c0r2; dstFlat[row3 + 3] = c1r2;
        dstFlat[row3 + 4] = c0r1; dstFlat[row3 + 5] = c1r1;
        dstFlat[row3 + 6] = c0r0; dstFlat[row3 + 7] = c1r0;
        dstFlat[row3 + 8] = r0c2; dstFlat[row3 + 9] = r0c3;
        dstFlat[row3 + 10] = r0c4; dstFlat[row3 + 11] = r0c5;
        dstFlat[row3 + 12] = r0c6; dstFlat[row3 + 13] = r0c7;
        dstFlat[row3 + 14] = r0c8; dstFlat[row3 + 15] = r0c9;

        // Row 4
        long row4 = dBase + 4 * N;
        dstFlat[row4 + 0] = c0r4; dstFlat[row4 + 1] = c1r4;
        dstFlat[row4 + 2] = c0r3; dstFlat[row4 + 3] = c1r3;
        dstFlat[row4 + 4] = c0r2; dstFlat[row4 + 5] = c1r2;
        dstFlat[row4 + 6] = c0r1; dstFlat[row4 + 7] = c1r1;
        dstFlat[row4 + 8] = c0r0; dstFlat[row4 + 9] = c1r0;
        dstFlat[row4 + 10] = r0c2; dstFlat[row4 + 11] = r0c3;
        dstFlat[row4 + 12] = r0c4; dstFlat[row4 + 13] = r0c5;
        dstFlat[row4 + 14] = r0c6; dstFlat[row4 + 15] = r0c7;

        // Row 5
        long row5 = dBase + 5 * N;
        dstFlat[row5 + 0] = c0r5; dstFlat[row5 + 1] = c1r5;
        dstFlat[row5 + 2] = c0r4; dstFlat[row5 + 3] = c1r4;
        dstFlat[row5 + 4] = c0r3; dstFlat[row5 + 5] = c1r3;
        dstFlat[row5 + 6] = c0r2; dstFlat[row5 + 7] = c1r2;
        dstFlat[row5 + 8] = c0r1; dstFlat[row5 + 9] = c1r1;
        dstFlat[row5 + 10] = c0r0; dstFlat[row5 + 11] = c1r0;
        dstFlat[row5 + 12] = r0c2; dstFlat[row5 + 13] = r0c3;
        dstFlat[row5 + 14] = r0c4; dstFlat[row5 + 15] = r0c5;

        // Row 6
        long row6 = dBase + 6 * N;
        dstFlat[row6 + 0] = c0r6; dstFlat[row6 + 1] = c1r6;
        dstFlat[row6 + 2] = c0r5; dstFlat[row6 + 3] = c1r5;
        dstFlat[row6 + 4] = c0r4; dstFlat[row6 + 5] = c1r4;
        dstFlat[row6 + 6] = c0r3; dstFlat[row6 + 7] = c1r3;
        dstFlat[row6 + 8] = c0r2; dstFlat[row6 + 9] = c1r2;
        dstFlat[row6 + 10] = c0r1; dstFlat[row6 + 11] = c1r1;
        dstFlat[row6 + 12] = c0r0; dstFlat[row6 + 13] = c1r0;
        dstFlat[row6 + 14] = r0c2; dstFlat[row6 + 15] = r0c3;

        // Row 7
        long row7 = dBase + 7 * N;
        dstFlat[row7 + 0] = c0r7; dstFlat[row7 + 1] = c1r7;
        dstFlat[row7 + 2] = c0r6; dstFlat[row7 + 3] = c1r6;
        dstFlat[row7 + 4] = c0r5; dstFlat[row7 + 5] = c1r5;
        dstFlat[row7 + 6] = c0r4; dstFlat[row7 + 7] = c1r4;
        dstFlat[row7 + 8] = c0r3; dstFlat[row7 + 9] = c1r3;
        dstFlat[row7 + 10] = c0r2; dstFlat[row7 + 11] = c1r2;
        dstFlat[row7 + 12] = c0r1; dstFlat[row7 + 13] = c1r1;
        dstFlat[row7 + 14] = c0r0; dstFlat[row7 + 15] = c1r0;

        // Row 8 (r=8, c=15 → c1r[8 - 7] = c1r1; c=14 → c0r1; c=13 → c1r2; etc.)
        long row8 = dBase + 8 * N;
        dstFlat[row8 + 0] = c0r8; dstFlat[row8 + 1] = c1r8;
        dstFlat[row8 + 2] = c0r7; dstFlat[row8 + 3] = c1r7;
        dstFlat[row8 + 4] = c0r6; dstFlat[row8 + 5] = c1r6;
        dstFlat[row8 + 6] = c0r5; dstFlat[row8 + 7] = c1r5;
        dstFlat[row8 + 8] = c0r4; dstFlat[row8 + 9] = c1r4;
        dstFlat[row8 + 10] = c0r3; dstFlat[row8 + 11] = c1r3;
        dstFlat[row8 + 12] = c0r2; dstFlat[row8 + 13] = c1r2;
        dstFlat[row8 + 14] = c0r1; dstFlat[row8 + 15] = c1r1;

        // Row 9
        long row9 = dBase + 9 * N;
        dstFlat[row9 + 0] = c0r9; dstFlat[row9 + 1] = c1r9;
        dstFlat[row9 + 2] = c0r8; dstFlat[row9 + 3] = c1r8;
        dstFlat[row9 + 4] = c0r7; dstFlat[row9 + 5] = c1r7;
        dstFlat[row9 + 6] = c0r6; dstFlat[row9 + 7] = c1r6;
        dstFlat[row9 + 8] = c0r5; dstFlat[row9 + 9] = c1r5;
        dstFlat[row9 + 10] = c0r4; dstFlat[row9 + 11] = c1r4;
        dstFlat[row9 + 12] = c0r3; dstFlat[row9 + 13] = c1r3;
        dstFlat[row9 + 14] = c0r2; dstFlat[row9 + 15] = c1r2;

        // Row 10
        long row10 = dBase + 10 * N;
        dstFlat[row10 + 0] = c0r10; dstFlat[row10 + 1] = c1r10;
        dstFlat[row10 + 2] = c0r9; dstFlat[row10 + 3] = c1r9;
        dstFlat[row10 + 4] = c0r8; dstFlat[row10 + 5] = c1r8;
        dstFlat[row10 + 6] = c0r7; dstFlat[row10 + 7] = c1r7;
        dstFlat[row10 + 8] = c0r6; dstFlat[row10 + 9] = c1r6;
        dstFlat[row10 + 10] = c0r5; dstFlat[row10 + 11] = c1r5;
        dstFlat[row10 + 12] = c0r4; dstFlat[row10 + 13] = c1r4;
        dstFlat[row10 + 14] = c0r3; dstFlat[row10 + 15] = c1r3;

        // Row 11
        long row11 = dBase + 11 * N;
        dstFlat[row11 + 0] = c0r11; dstFlat[row11 + 1] = c1r11;
        dstFlat[row11 + 2] = c0r10; dstFlat[row11 + 3] = c1r10;
        dstFlat[row11 + 4] = c0r9; dstFlat[row11 + 5] = c1r9;
        dstFlat[row11 + 6] = c0r8; dstFlat[row11 + 7] = c1r8;
        dstFlat[row11 + 8] = c0r7; dstFlat[row11 + 9] = c1r7;
        dstFlat[row11 + 10] = c0r6; dstFlat[row11 + 11] = c1r6;
        dstFlat[row11 + 12] = c0r5; dstFlat[row11 + 13] = c1r5;
        dstFlat[row11 + 14] = c0r4; dstFlat[row11 + 15] = c1r4;

        // Row 12
        long row12 = dBase + 12 * N;
        dstFlat[row12 + 0] = c0r12; dstFlat[row12 + 1] = c1r12;
        dstFlat[row12 + 2] = c0r11; dstFlat[row12 + 3] = c1r11;
        dstFlat[row12 + 4] = c0r10; dstFlat[row12 + 5] = c1r10;
        dstFlat[row12 + 6] = c0r9; dstFlat[row12 + 7] = c1r9;
        dstFlat[row12 + 8] = c0r8; dstFlat[row12 + 9] = c1r8;
        dstFlat[row12 + 10] = c0r7; dstFlat[row12 + 11] = c1r7;
        dstFlat[row12 + 12] = c0r6; dstFlat[row12 + 13] = c1r6;
        dstFlat[row12 + 14] = c0r5; dstFlat[row12 + 15] = c1r5;

        // Row 13
        long row13 = dBase + 13 * N;
        dstFlat[row13 + 0] = c0r13; dstFlat[row13 + 1] = c1r13;
        dstFlat[row13 + 2] = c0r12; dstFlat[row13 + 3] = c1r12;
        dstFlat[row13 + 4] = c0r11; dstFlat[row13 + 5] = c1r11;
        dstFlat[row13 + 6] = c0r10; dstFlat[row13 + 7] = c1r10;
        dstFlat[row13 + 8] = c0r9; dstFlat[row13 + 9] = c1r9;
        dstFlat[row13 + 10] = c0r8; dstFlat[row13 + 11] = c1r8;
        dstFlat[row13 + 12] = c0r7; dstFlat[row13 + 13] = c1r7;
        dstFlat[row13 + 14] = c0r6; dstFlat[row13 + 15] = c1r6;

        // Row 14
        long row14 = dBase + 14 * N;
        dstFlat[row14 + 0] = c0r14; dstFlat[row14 + 1] = c1r14;
        dstFlat[row14 + 2] = c0r13; dstFlat[row14 + 3] = c1r13;
        dstFlat[row14 + 4] = c0r12; dstFlat[row14 + 5] = c1r12;
        dstFlat[row14 + 6] = c0r11; dstFlat[row14 + 7] = c1r11;
        dstFlat[row14 + 8] = c0r10; dstFlat[row14 + 9] = c1r10;
        dstFlat[row14 + 10] = c0r9; dstFlat[row14 + 11] = c1r9;
        dstFlat[row14 + 12] = c0r8; dstFlat[row14 + 13] = c1r8;
        dstFlat[row14 + 14] = c0r7; dstFlat[row14 + 15] = c1r7;

        // Row 15
        long row15 = dBase + 15 * N;
        dstFlat[row15 + 0] = c0r15; dstFlat[row15 + 1] = c1r15;
        dstFlat[row15 + 2] = c0r14; dstFlat[row15 + 3] = c1r14;
        dstFlat[row15 + 4] = c0r13; dstFlat[row15 + 5] = c1r13;
        dstFlat[row15 + 6] = c0r12; dstFlat[row15 + 7] = c1r12;
        dstFlat[row15 + 8] = c0r11; dstFlat[row15 + 9] = c1r11;
        dstFlat[row15 + 10] = c0r10; dstFlat[row15 + 11] = c1r10;
        dstFlat[row15 + 12] = c0r9; dstFlat[row15 + 13] = c1r9;
        dstFlat[row15 + 14] = c0r8; dstFlat[row15 + 15] = c1r8;
    }

    /// <inheritdoc/>
    public void Dispose() { }
}
