// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// 16x16 sibling of Vp9D135Predict4x4 / 8x8 kernels. 2N-1 = 31 byte
// border array in registers; each row writes a 16-byte slice at
// offset (N-1-row). Fully unrolled writes per the slice 202 lesson
// (32+ register cells break WGSL switch dispatch on WebGPU).

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// Batched ILGPU kernel that runs VP9 D135_PRED across N independent
/// 16x16 blocks in parallel.
/// </summary>
public sealed class Vp9D135Predict16x16Kernel : IDisposable
{
    private const int N = 16;

    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, int, int> _kernel;

    /// <summary>Compile the kernel onto <paramref name="accelerator"/>.</summary>
    public Vp9D135Predict16x16Kernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, int, int>(D135Kernel);
    }

    /// <summary>Run D135 prediction on <paramref name="blockCount"/> blocks.</summary>
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

    /// <summary>Kernel body. 31 border cells in registers; 16 row slices unrolled.</summary>
    private static void D135Kernel(
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
        int l15 = leftFlat[lBase + 15];

        // 31 border cells. b[0..13] descending-left; b[14] corner-left;
        // b[15] cross-corner; b[16] corner-above; b[17..30] ascending-above.
        byte b0 = (byte)((l13 + 2 * l14 + l15 + 2) >> 2);
        byte b1 = (byte)((l12 + 2 * l13 + l14 + 2) >> 2);
        byte b2 = (byte)((l11 + 2 * l12 + l13 + 2) >> 2);
        byte b3 = (byte)((l10 + 2 * l11 + l12 + 2) >> 2);
        byte b4 = (byte)((l9 + 2 * l10 + l11 + 2) >> 2);
        byte b5 = (byte)((l8 + 2 * l9 + l10 + 2) >> 2);
        byte b6 = (byte)((l7 + 2 * l8 + l9 + 2) >> 2);
        byte b7 = (byte)((l6 + 2 * l7 + l8 + 2) >> 2);
        byte b8 = (byte)((l5 + 2 * l6 + l7 + 2) >> 2);
        byte b9 = (byte)((l4 + 2 * l5 + l6 + 2) >> 2);
        byte b10 = (byte)((l3 + 2 * l4 + l5 + 2) >> 2);
        byte b11 = (byte)((l2 + 2 * l3 + l4 + 2) >> 2);
        byte b12 = (byte)((l1 + 2 * l2 + l3 + 2) >> 2);
        byte b13 = (byte)((l0 + 2 * l1 + l2 + 2) >> 2);
        byte b14 = (byte)((tl + 2 * l0 + l1 + 2) >> 2);
        byte b15 = (byte)((l0 + 2 * tl + a0 + 2) >> 2);
        byte b16 = (byte)((tl + 2 * a0 + a1 + 2) >> 2);
        byte b17 = (byte)((a0 + 2 * a1 + a2 + 2) >> 2);
        byte b18 = (byte)((a1 + 2 * a2 + a3 + 2) >> 2);
        byte b19 = (byte)((a2 + 2 * a3 + a4 + 2) >> 2);
        byte b20 = (byte)((a3 + 2 * a4 + a5 + 2) >> 2);
        byte b21 = (byte)((a4 + 2 * a5 + a6 + 2) >> 2);
        byte b22 = (byte)((a5 + 2 * a6 + a7 + 2) >> 2);
        byte b23 = (byte)((a6 + 2 * a7 + a8 + 2) >> 2);
        byte b24 = (byte)((a7 + 2 * a8 + a9 + 2) >> 2);
        byte b25 = (byte)((a8 + 2 * a9 + a10 + 2) >> 2);
        byte b26 = (byte)((a9 + 2 * a10 + a11 + 2) >> 2);
        byte b27 = (byte)((a10 + 2 * a11 + a12 + 2) >> 2);
        byte b28 = (byte)((a11 + 2 * a12 + a13 + 2) >> 2);
        byte b29 = (byte)((a12 + 2 * a13 + a14 + 2) >> 2);
        byte b30 = (byte)((a13 + 2 * a14 + a15 + 2) >> 2);

        // Row r writes border[15-r..30-r], 16 cells.
        // Row 0 (srcStart=15)
        dstFlat[dBase + 0] = b15;  dstFlat[dBase + 1] = b16;  dstFlat[dBase + 2] = b17;  dstFlat[dBase + 3] = b18;
        dstFlat[dBase + 4] = b19;  dstFlat[dBase + 5] = b20;  dstFlat[dBase + 6] = b21;  dstFlat[dBase + 7] = b22;
        dstFlat[dBase + 8] = b23;  dstFlat[dBase + 9] = b24;  dstFlat[dBase + 10] = b25; dstFlat[dBase + 11] = b26;
        dstFlat[dBase + 12] = b27; dstFlat[dBase + 13] = b28; dstFlat[dBase + 14] = b29; dstFlat[dBase + 15] = b30;

        // Row 1
        long row1 = dBase + N;
        dstFlat[row1 + 0] = b14; dstFlat[row1 + 1] = b15; dstFlat[row1 + 2] = b16; dstFlat[row1 + 3] = b17;
        dstFlat[row1 + 4] = b18; dstFlat[row1 + 5] = b19; dstFlat[row1 + 6] = b20; dstFlat[row1 + 7] = b21;
        dstFlat[row1 + 8] = b22; dstFlat[row1 + 9] = b23; dstFlat[row1 + 10] = b24; dstFlat[row1 + 11] = b25;
        dstFlat[row1 + 12] = b26; dstFlat[row1 + 13] = b27; dstFlat[row1 + 14] = b28; dstFlat[row1 + 15] = b29;

        // Row 2
        long row2 = dBase + 2 * N;
        dstFlat[row2 + 0] = b13; dstFlat[row2 + 1] = b14; dstFlat[row2 + 2] = b15; dstFlat[row2 + 3] = b16;
        dstFlat[row2 + 4] = b17; dstFlat[row2 + 5] = b18; dstFlat[row2 + 6] = b19; dstFlat[row2 + 7] = b20;
        dstFlat[row2 + 8] = b21; dstFlat[row2 + 9] = b22; dstFlat[row2 + 10] = b23; dstFlat[row2 + 11] = b24;
        dstFlat[row2 + 12] = b25; dstFlat[row2 + 13] = b26; dstFlat[row2 + 14] = b27; dstFlat[row2 + 15] = b28;

        // Row 3
        long row3 = dBase + 3 * N;
        dstFlat[row3 + 0] = b12; dstFlat[row3 + 1] = b13; dstFlat[row3 + 2] = b14; dstFlat[row3 + 3] = b15;
        dstFlat[row3 + 4] = b16; dstFlat[row3 + 5] = b17; dstFlat[row3 + 6] = b18; dstFlat[row3 + 7] = b19;
        dstFlat[row3 + 8] = b20; dstFlat[row3 + 9] = b21; dstFlat[row3 + 10] = b22; dstFlat[row3 + 11] = b23;
        dstFlat[row3 + 12] = b24; dstFlat[row3 + 13] = b25; dstFlat[row3 + 14] = b26; dstFlat[row3 + 15] = b27;

        // Row 4
        long row4 = dBase + 4 * N;
        dstFlat[row4 + 0] = b11; dstFlat[row4 + 1] = b12; dstFlat[row4 + 2] = b13; dstFlat[row4 + 3] = b14;
        dstFlat[row4 + 4] = b15; dstFlat[row4 + 5] = b16; dstFlat[row4 + 6] = b17; dstFlat[row4 + 7] = b18;
        dstFlat[row4 + 8] = b19; dstFlat[row4 + 9] = b20; dstFlat[row4 + 10] = b21; dstFlat[row4 + 11] = b22;
        dstFlat[row4 + 12] = b23; dstFlat[row4 + 13] = b24; dstFlat[row4 + 14] = b25; dstFlat[row4 + 15] = b26;

        // Row 5
        long row5 = dBase + 5 * N;
        dstFlat[row5 + 0] = b10; dstFlat[row5 + 1] = b11; dstFlat[row5 + 2] = b12; dstFlat[row5 + 3] = b13;
        dstFlat[row5 + 4] = b14; dstFlat[row5 + 5] = b15; dstFlat[row5 + 6] = b16; dstFlat[row5 + 7] = b17;
        dstFlat[row5 + 8] = b18; dstFlat[row5 + 9] = b19; dstFlat[row5 + 10] = b20; dstFlat[row5 + 11] = b21;
        dstFlat[row5 + 12] = b22; dstFlat[row5 + 13] = b23; dstFlat[row5 + 14] = b24; dstFlat[row5 + 15] = b25;

        // Row 6
        long row6 = dBase + 6 * N;
        dstFlat[row6 + 0] = b9; dstFlat[row6 + 1] = b10; dstFlat[row6 + 2] = b11; dstFlat[row6 + 3] = b12;
        dstFlat[row6 + 4] = b13; dstFlat[row6 + 5] = b14; dstFlat[row6 + 6] = b15; dstFlat[row6 + 7] = b16;
        dstFlat[row6 + 8] = b17; dstFlat[row6 + 9] = b18; dstFlat[row6 + 10] = b19; dstFlat[row6 + 11] = b20;
        dstFlat[row6 + 12] = b21; dstFlat[row6 + 13] = b22; dstFlat[row6 + 14] = b23; dstFlat[row6 + 15] = b24;

        // Row 7
        long row7 = dBase + 7 * N;
        dstFlat[row7 + 0] = b8; dstFlat[row7 + 1] = b9; dstFlat[row7 + 2] = b10; dstFlat[row7 + 3] = b11;
        dstFlat[row7 + 4] = b12; dstFlat[row7 + 5] = b13; dstFlat[row7 + 6] = b14; dstFlat[row7 + 7] = b15;
        dstFlat[row7 + 8] = b16; dstFlat[row7 + 9] = b17; dstFlat[row7 + 10] = b18; dstFlat[row7 + 11] = b19;
        dstFlat[row7 + 12] = b20; dstFlat[row7 + 13] = b21; dstFlat[row7 + 14] = b22; dstFlat[row7 + 15] = b23;

        // Row 8
        long row8 = dBase + 8 * N;
        dstFlat[row8 + 0] = b7; dstFlat[row8 + 1] = b8; dstFlat[row8 + 2] = b9; dstFlat[row8 + 3] = b10;
        dstFlat[row8 + 4] = b11; dstFlat[row8 + 5] = b12; dstFlat[row8 + 6] = b13; dstFlat[row8 + 7] = b14;
        dstFlat[row8 + 8] = b15; dstFlat[row8 + 9] = b16; dstFlat[row8 + 10] = b17; dstFlat[row8 + 11] = b18;
        dstFlat[row8 + 12] = b19; dstFlat[row8 + 13] = b20; dstFlat[row8 + 14] = b21; dstFlat[row8 + 15] = b22;

        // Row 9
        long row9 = dBase + 9 * N;
        dstFlat[row9 + 0] = b6; dstFlat[row9 + 1] = b7; dstFlat[row9 + 2] = b8; dstFlat[row9 + 3] = b9;
        dstFlat[row9 + 4] = b10; dstFlat[row9 + 5] = b11; dstFlat[row9 + 6] = b12; dstFlat[row9 + 7] = b13;
        dstFlat[row9 + 8] = b14; dstFlat[row9 + 9] = b15; dstFlat[row9 + 10] = b16; dstFlat[row9 + 11] = b17;
        dstFlat[row9 + 12] = b18; dstFlat[row9 + 13] = b19; dstFlat[row9 + 14] = b20; dstFlat[row9 + 15] = b21;

        // Row 10
        long row10 = dBase + 10 * N;
        dstFlat[row10 + 0] = b5; dstFlat[row10 + 1] = b6; dstFlat[row10 + 2] = b7; dstFlat[row10 + 3] = b8;
        dstFlat[row10 + 4] = b9; dstFlat[row10 + 5] = b10; dstFlat[row10 + 6] = b11; dstFlat[row10 + 7] = b12;
        dstFlat[row10 + 8] = b13; dstFlat[row10 + 9] = b14; dstFlat[row10 + 10] = b15; dstFlat[row10 + 11] = b16;
        dstFlat[row10 + 12] = b17; dstFlat[row10 + 13] = b18; dstFlat[row10 + 14] = b19; dstFlat[row10 + 15] = b20;

        // Row 11
        long row11 = dBase + 11 * N;
        dstFlat[row11 + 0] = b4; dstFlat[row11 + 1] = b5; dstFlat[row11 + 2] = b6; dstFlat[row11 + 3] = b7;
        dstFlat[row11 + 4] = b8; dstFlat[row11 + 5] = b9; dstFlat[row11 + 6] = b10; dstFlat[row11 + 7] = b11;
        dstFlat[row11 + 8] = b12; dstFlat[row11 + 9] = b13; dstFlat[row11 + 10] = b14; dstFlat[row11 + 11] = b15;
        dstFlat[row11 + 12] = b16; dstFlat[row11 + 13] = b17; dstFlat[row11 + 14] = b18; dstFlat[row11 + 15] = b19;

        // Row 12
        long row12 = dBase + 12 * N;
        dstFlat[row12 + 0] = b3; dstFlat[row12 + 1] = b4; dstFlat[row12 + 2] = b5; dstFlat[row12 + 3] = b6;
        dstFlat[row12 + 4] = b7; dstFlat[row12 + 5] = b8; dstFlat[row12 + 6] = b9; dstFlat[row12 + 7] = b10;
        dstFlat[row12 + 8] = b11; dstFlat[row12 + 9] = b12; dstFlat[row12 + 10] = b13; dstFlat[row12 + 11] = b14;
        dstFlat[row12 + 12] = b15; dstFlat[row12 + 13] = b16; dstFlat[row12 + 14] = b17; dstFlat[row12 + 15] = b18;

        // Row 13
        long row13 = dBase + 13 * N;
        dstFlat[row13 + 0] = b2; dstFlat[row13 + 1] = b3; dstFlat[row13 + 2] = b4; dstFlat[row13 + 3] = b5;
        dstFlat[row13 + 4] = b6; dstFlat[row13 + 5] = b7; dstFlat[row13 + 6] = b8; dstFlat[row13 + 7] = b9;
        dstFlat[row13 + 8] = b10; dstFlat[row13 + 9] = b11; dstFlat[row13 + 10] = b12; dstFlat[row13 + 11] = b13;
        dstFlat[row13 + 12] = b14; dstFlat[row13 + 13] = b15; dstFlat[row13 + 14] = b16; dstFlat[row13 + 15] = b17;

        // Row 14
        long row14 = dBase + 14 * N;
        dstFlat[row14 + 0] = b1; dstFlat[row14 + 1] = b2; dstFlat[row14 + 2] = b3; dstFlat[row14 + 3] = b4;
        dstFlat[row14 + 4] = b5; dstFlat[row14 + 5] = b6; dstFlat[row14 + 6] = b7; dstFlat[row14 + 7] = b8;
        dstFlat[row14 + 8] = b9; dstFlat[row14 + 9] = b10; dstFlat[row14 + 10] = b11; dstFlat[row14 + 11] = b12;
        dstFlat[row14 + 12] = b13; dstFlat[row14 + 13] = b14; dstFlat[row14 + 14] = b15; dstFlat[row14 + 15] = b16;

        // Row 15 (srcStart=0)
        long row15 = dBase + 15 * N;
        dstFlat[row15 + 0] = b0; dstFlat[row15 + 1] = b1; dstFlat[row15 + 2] = b2; dstFlat[row15 + 3] = b3;
        dstFlat[row15 + 4] = b4; dstFlat[row15 + 5] = b5; dstFlat[row15 + 6] = b6; dstFlat[row15 + 7] = b7;
        dstFlat[row15 + 8] = b8; dstFlat[row15 + 9] = b9; dstFlat[row15 + 10] = b10; dstFlat[row15 + 11] = b11;
        dstFlat[row15 + 12] = b12; dstFlat[row15 + 13] = b13; dstFlat[row15 + 14] = b14; dstFlat[row15 + 15] = b15;
    }

    /// <inheritdoc/>
    public void Dispose() { }
}
