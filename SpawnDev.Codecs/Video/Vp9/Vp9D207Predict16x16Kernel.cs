// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// 16x16 sibling of Vp9D207Predict4x4 / 8x8 kernels. N=16, left-only
// directional. Two seed columns (cols 0+1 = 32 register cells);
// fully unrolled output writes per the slice 202 lesson - 32+ cell
// register sets break the WGSL switch dispatch on WebGPU under
// rc.13.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// Batched ILGPU kernel that runs VP9 D207_PRED across N independent
/// 16x16 blocks in parallel.
/// </summary>
public sealed class Vp9D207Predict16x16Kernel : IDisposable
{
    private const int N = 16;

    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, int, int> _kernel;

    /// <summary>Compile the kernel onto <paramref name="accelerator"/>.</summary>
    public Vp9D207Predict16x16Kernel(Accelerator accelerator)
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

    /// <summary>Kernel body. 32 register cells; 256 unrolled output writes.</summary>
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
        int l8 = leftFlat[lBase + 8];
        int l9 = leftFlat[lBase + 9];
        int l10 = leftFlat[lBase + 10];
        int l11 = leftFlat[lBase + 11];
        int l12 = leftFlat[lBase + 12];
        int l13 = leftFlat[lBase + 13];
        int l14 = leftFlat[lBase + 14];
        int l15 = leftFlat[lBase + 15];

        byte fillVal = (byte)l15;

        // Column 0 (AVG2 pairs; bottom = left[N-1]).
        byte c0r0 = (byte)((l0 + l1 + 1) >> 1);
        byte c0r1 = (byte)((l1 + l2 + 1) >> 1);
        byte c0r2 = (byte)((l2 + l3 + 1) >> 1);
        byte c0r3 = (byte)((l3 + l4 + 1) >> 1);
        byte c0r4 = (byte)((l4 + l5 + 1) >> 1);
        byte c0r5 = (byte)((l5 + l6 + 1) >> 1);
        byte c0r6 = (byte)((l6 + l7 + 1) >> 1);
        byte c0r7 = (byte)((l7 + l8 + 1) >> 1);
        byte c0r8 = (byte)((l8 + l9 + 1) >> 1);
        byte c0r9 = (byte)((l9 + l10 + 1) >> 1);
        byte c0r10 = (byte)((l10 + l11 + 1) >> 1);
        byte c0r11 = (byte)((l11 + l12 + 1) >> 1);
        byte c0r12 = (byte)((l12 + l13 + 1) >> 1);
        byte c0r13 = (byte)((l13 + l14 + 1) >> 1);
        byte c0r14 = (byte)((l14 + l15 + 1) >> 1);
        byte c0r15 = (byte)l15;

        // Column 1 (AVG3 triples; second-to-last replicates left[N-1]; bottom = left[N-1]).
        byte c1r0 = (byte)((l0 + 2 * l1 + l2 + 2) >> 2);
        byte c1r1 = (byte)((l1 + 2 * l2 + l3 + 2) >> 2);
        byte c1r2 = (byte)((l2 + 2 * l3 + l4 + 2) >> 2);
        byte c1r3 = (byte)((l3 + 2 * l4 + l5 + 2) >> 2);
        byte c1r4 = (byte)((l4 + 2 * l5 + l6 + 2) >> 2);
        byte c1r5 = (byte)((l5 + 2 * l6 + l7 + 2) >> 2);
        byte c1r6 = (byte)((l6 + 2 * l7 + l8 + 2) >> 2);
        byte c1r7 = (byte)((l7 + 2 * l8 + l9 + 2) >> 2);
        byte c1r8 = (byte)((l8 + 2 * l9 + l10 + 2) >> 2);
        byte c1r9 = (byte)((l9 + 2 * l10 + l11 + 2) >> 2);
        byte c1r10 = (byte)((l10 + 2 * l11 + l12 + 2) >> 2);
        byte c1r11 = (byte)((l11 + 2 * l12 + l13 + 2) >> 2);
        byte c1r12 = (byte)((l12 + 2 * l13 + l14 + 2) >> 2);
        byte c1r13 = (byte)((l13 + 2 * l14 + l15 + 2) >> 2);
        byte c1r14 = (byte)((l14 + 2 * l15 + l15 + 2) >> 2);
        byte c1r15 = (byte)l15;

        // Row 0 (offset 0): cols pair (c0rk, c1rk) for k=0..7.
        dstFlat[dBase + 0] = c0r0;  dstFlat[dBase + 1] = c1r0;
        dstFlat[dBase + 2] = c0r1;  dstFlat[dBase + 3] = c1r1;
        dstFlat[dBase + 4] = c0r2;  dstFlat[dBase + 5] = c1r2;
        dstFlat[dBase + 6] = c0r3;  dstFlat[dBase + 7] = c1r3;
        dstFlat[dBase + 8] = c0r4;  dstFlat[dBase + 9] = c1r4;
        dstFlat[dBase + 10] = c0r5; dstFlat[dBase + 11] = c1r5;
        dstFlat[dBase + 12] = c0r6; dstFlat[dBase + 13] = c1r6;
        dstFlat[dBase + 14] = c0r7; dstFlat[dBase + 15] = c1r7;

        // Row 1 (offset 1).
        long row1 = dBase + N;
        dstFlat[row1 + 0] = c0r1;  dstFlat[row1 + 1] = c1r1;
        dstFlat[row1 + 2] = c0r2;  dstFlat[row1 + 3] = c1r2;
        dstFlat[row1 + 4] = c0r3;  dstFlat[row1 + 5] = c1r3;
        dstFlat[row1 + 6] = c0r4;  dstFlat[row1 + 7] = c1r4;
        dstFlat[row1 + 8] = c0r5;  dstFlat[row1 + 9] = c1r5;
        dstFlat[row1 + 10] = c0r6; dstFlat[row1 + 11] = c1r6;
        dstFlat[row1 + 12] = c0r7; dstFlat[row1 + 13] = c1r7;
        dstFlat[row1 + 14] = c0r8; dstFlat[row1 + 15] = c1r8;

        // Row 2 (offset 2).
        long row2 = dBase + 2 * N;
        dstFlat[row2 + 0] = c0r2;  dstFlat[row2 + 1] = c1r2;
        dstFlat[row2 + 2] = c0r3;  dstFlat[row2 + 3] = c1r3;
        dstFlat[row2 + 4] = c0r4;  dstFlat[row2 + 5] = c1r4;
        dstFlat[row2 + 6] = c0r5;  dstFlat[row2 + 7] = c1r5;
        dstFlat[row2 + 8] = c0r6;  dstFlat[row2 + 9] = c1r6;
        dstFlat[row2 + 10] = c0r7; dstFlat[row2 + 11] = c1r7;
        dstFlat[row2 + 12] = c0r8; dstFlat[row2 + 13] = c1r8;
        dstFlat[row2 + 14] = c0r9; dstFlat[row2 + 15] = c1r9;

        // Row 3.
        long row3 = dBase + 3 * N;
        dstFlat[row3 + 0] = c0r3;  dstFlat[row3 + 1] = c1r3;
        dstFlat[row3 + 2] = c0r4;  dstFlat[row3 + 3] = c1r4;
        dstFlat[row3 + 4] = c0r5;  dstFlat[row3 + 5] = c1r5;
        dstFlat[row3 + 6] = c0r6;  dstFlat[row3 + 7] = c1r6;
        dstFlat[row3 + 8] = c0r7;  dstFlat[row3 + 9] = c1r7;
        dstFlat[row3 + 10] = c0r8; dstFlat[row3 + 11] = c1r8;
        dstFlat[row3 + 12] = c0r9; dstFlat[row3 + 13] = c1r9;
        dstFlat[row3 + 14] = c0r10; dstFlat[row3 + 15] = c1r10;

        // Row 4.
        long row4 = dBase + 4 * N;
        dstFlat[row4 + 0] = c0r4;  dstFlat[row4 + 1] = c1r4;
        dstFlat[row4 + 2] = c0r5;  dstFlat[row4 + 3] = c1r5;
        dstFlat[row4 + 4] = c0r6;  dstFlat[row4 + 5] = c1r6;
        dstFlat[row4 + 6] = c0r7;  dstFlat[row4 + 7] = c1r7;
        dstFlat[row4 + 8] = c0r8;  dstFlat[row4 + 9] = c1r8;
        dstFlat[row4 + 10] = c0r9; dstFlat[row4 + 11] = c1r9;
        dstFlat[row4 + 12] = c0r10; dstFlat[row4 + 13] = c1r10;
        dstFlat[row4 + 14] = c0r11; dstFlat[row4 + 15] = c1r11;

        // Row 5.
        long row5 = dBase + 5 * N;
        dstFlat[row5 + 0] = c0r5;  dstFlat[row5 + 1] = c1r5;
        dstFlat[row5 + 2] = c0r6;  dstFlat[row5 + 3] = c1r6;
        dstFlat[row5 + 4] = c0r7;  dstFlat[row5 + 5] = c1r7;
        dstFlat[row5 + 6] = c0r8;  dstFlat[row5 + 7] = c1r8;
        dstFlat[row5 + 8] = c0r9;  dstFlat[row5 + 9] = c1r9;
        dstFlat[row5 + 10] = c0r10; dstFlat[row5 + 11] = c1r10;
        dstFlat[row5 + 12] = c0r11; dstFlat[row5 + 13] = c1r11;
        dstFlat[row5 + 14] = c0r12; dstFlat[row5 + 15] = c1r12;

        // Row 6.
        long row6 = dBase + 6 * N;
        dstFlat[row6 + 0] = c0r6;  dstFlat[row6 + 1] = c1r6;
        dstFlat[row6 + 2] = c0r7;  dstFlat[row6 + 3] = c1r7;
        dstFlat[row6 + 4] = c0r8;  dstFlat[row6 + 5] = c1r8;
        dstFlat[row6 + 6] = c0r9;  dstFlat[row6 + 7] = c1r9;
        dstFlat[row6 + 8] = c0r10; dstFlat[row6 + 9] = c1r10;
        dstFlat[row6 + 10] = c0r11; dstFlat[row6 + 11] = c1r11;
        dstFlat[row6 + 12] = c0r12; dstFlat[row6 + 13] = c1r12;
        dstFlat[row6 + 14] = c0r13; dstFlat[row6 + 15] = c1r13;

        // Row 7.
        long row7 = dBase + 7 * N;
        dstFlat[row7 + 0] = c0r7;  dstFlat[row7 + 1] = c1r7;
        dstFlat[row7 + 2] = c0r8;  dstFlat[row7 + 3] = c1r8;
        dstFlat[row7 + 4] = c0r9;  dstFlat[row7 + 5] = c1r9;
        dstFlat[row7 + 6] = c0r10; dstFlat[row7 + 7] = c1r10;
        dstFlat[row7 + 8] = c0r11; dstFlat[row7 + 9] = c1r11;
        dstFlat[row7 + 10] = c0r12; dstFlat[row7 + 11] = c1r12;
        dstFlat[row7 + 12] = c0r13; dstFlat[row7 + 13] = c1r13;
        dstFlat[row7 + 14] = c0r14; dstFlat[row7 + 15] = c1r14;

        // Row 8.
        long row8 = dBase + 8 * N;
        dstFlat[row8 + 0] = c0r8;  dstFlat[row8 + 1] = c1r8;
        dstFlat[row8 + 2] = c0r9;  dstFlat[row8 + 3] = c1r9;
        dstFlat[row8 + 4] = c0r10; dstFlat[row8 + 5] = c1r10;
        dstFlat[row8 + 6] = c0r11; dstFlat[row8 + 7] = c1r11;
        dstFlat[row8 + 8] = c0r12; dstFlat[row8 + 9] = c1r12;
        dstFlat[row8 + 10] = c0r13; dstFlat[row8 + 11] = c1r13;
        dstFlat[row8 + 12] = c0r14; dstFlat[row8 + 13] = c1r14;
        dstFlat[row8 + 14] = c0r15; dstFlat[row8 + 15] = c1r15;

        // Row 9 (offset 9): src indices 9..15 valid; src=16 → fillVal.
        long row9 = dBase + 9 * N;
        dstFlat[row9 + 0] = c0r9;  dstFlat[row9 + 1] = c1r9;
        dstFlat[row9 + 2] = c0r10; dstFlat[row9 + 3] = c1r10;
        dstFlat[row9 + 4] = c0r11; dstFlat[row9 + 5] = c1r11;
        dstFlat[row9 + 6] = c0r12; dstFlat[row9 + 7] = c1r12;
        dstFlat[row9 + 8] = c0r13; dstFlat[row9 + 9] = c1r13;
        dstFlat[row9 + 10] = c0r14; dstFlat[row9 + 11] = c1r14;
        dstFlat[row9 + 12] = c0r15; dstFlat[row9 + 13] = c1r15;
        dstFlat[row9 + 14] = fillVal; dstFlat[row9 + 15] = fillVal;

        // Row 10.
        long row10 = dBase + 10 * N;
        dstFlat[row10 + 0] = c0r10; dstFlat[row10 + 1] = c1r10;
        dstFlat[row10 + 2] = c0r11; dstFlat[row10 + 3] = c1r11;
        dstFlat[row10 + 4] = c0r12; dstFlat[row10 + 5] = c1r12;
        dstFlat[row10 + 6] = c0r13; dstFlat[row10 + 7] = c1r13;
        dstFlat[row10 + 8] = c0r14; dstFlat[row10 + 9] = c1r14;
        dstFlat[row10 + 10] = c0r15; dstFlat[row10 + 11] = c1r15;
        dstFlat[row10 + 12] = fillVal; dstFlat[row10 + 13] = fillVal;
        dstFlat[row10 + 14] = fillVal; dstFlat[row10 + 15] = fillVal;

        // Row 11.
        long row11 = dBase + 11 * N;
        dstFlat[row11 + 0] = c0r11; dstFlat[row11 + 1] = c1r11;
        dstFlat[row11 + 2] = c0r12; dstFlat[row11 + 3] = c1r12;
        dstFlat[row11 + 4] = c0r13; dstFlat[row11 + 5] = c1r13;
        dstFlat[row11 + 6] = c0r14; dstFlat[row11 + 7] = c1r14;
        dstFlat[row11 + 8] = c0r15; dstFlat[row11 + 9] = c1r15;
        dstFlat[row11 + 10] = fillVal; dstFlat[row11 + 11] = fillVal;
        dstFlat[row11 + 12] = fillVal; dstFlat[row11 + 13] = fillVal;
        dstFlat[row11 + 14] = fillVal; dstFlat[row11 + 15] = fillVal;

        // Row 12.
        long row12 = dBase + 12 * N;
        dstFlat[row12 + 0] = c0r12; dstFlat[row12 + 1] = c1r12;
        dstFlat[row12 + 2] = c0r13; dstFlat[row12 + 3] = c1r13;
        dstFlat[row12 + 4] = c0r14; dstFlat[row12 + 5] = c1r14;
        dstFlat[row12 + 6] = c0r15; dstFlat[row12 + 7] = c1r15;
        dstFlat[row12 + 8] = fillVal; dstFlat[row12 + 9] = fillVal;
        dstFlat[row12 + 10] = fillVal; dstFlat[row12 + 11] = fillVal;
        dstFlat[row12 + 12] = fillVal; dstFlat[row12 + 13] = fillVal;
        dstFlat[row12 + 14] = fillVal; dstFlat[row12 + 15] = fillVal;

        // Row 13.
        long row13 = dBase + 13 * N;
        dstFlat[row13 + 0] = c0r13; dstFlat[row13 + 1] = c1r13;
        dstFlat[row13 + 2] = c0r14; dstFlat[row13 + 3] = c1r14;
        dstFlat[row13 + 4] = c0r15; dstFlat[row13 + 5] = c1r15;
        dstFlat[row13 + 6] = fillVal; dstFlat[row13 + 7] = fillVal;
        dstFlat[row13 + 8] = fillVal; dstFlat[row13 + 9] = fillVal;
        dstFlat[row13 + 10] = fillVal; dstFlat[row13 + 11] = fillVal;
        dstFlat[row13 + 12] = fillVal; dstFlat[row13 + 13] = fillVal;
        dstFlat[row13 + 14] = fillVal; dstFlat[row13 + 15] = fillVal;

        // Row 14.
        long row14 = dBase + 14 * N;
        dstFlat[row14 + 0] = c0r14; dstFlat[row14 + 1] = c1r14;
        dstFlat[row14 + 2] = c0r15; dstFlat[row14 + 3] = c1r15;
        dstFlat[row14 + 4] = fillVal; dstFlat[row14 + 5] = fillVal;
        dstFlat[row14 + 6] = fillVal; dstFlat[row14 + 7] = fillVal;
        dstFlat[row14 + 8] = fillVal; dstFlat[row14 + 9] = fillVal;
        dstFlat[row14 + 10] = fillVal; dstFlat[row14 + 11] = fillVal;
        dstFlat[row14 + 12] = fillVal; dstFlat[row14 + 13] = fillVal;
        dstFlat[row14 + 14] = fillVal; dstFlat[row14 + 15] = fillVal;

        // Row 15.
        long row15 = dBase + 15 * N;
        dstFlat[row15 + 0] = c0r15; dstFlat[row15 + 1] = c1r15;
        dstFlat[row15 + 2] = fillVal; dstFlat[row15 + 3] = fillVal;
        dstFlat[row15 + 4] = fillVal; dstFlat[row15 + 5] = fillVal;
        dstFlat[row15 + 6] = fillVal; dstFlat[row15 + 7] = fillVal;
        dstFlat[row15 + 8] = fillVal; dstFlat[row15 + 9] = fillVal;
        dstFlat[row15 + 10] = fillVal; dstFlat[row15 + 11] = fillVal;
        dstFlat[row15 + 12] = fillVal; dstFlat[row15 + 13] = fillVal;
        dstFlat[row15 + 14] = fillVal; dstFlat[row15 + 15] = fillVal;
    }

    /// <inheritdoc/>
    public void Dispose() { }
}
