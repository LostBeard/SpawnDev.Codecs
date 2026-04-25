// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// 8x8 sibling of Vp9D135Predict4x4Kernel. 2N-1 = 15 byte border
// array in registers, each row writes an N-byte slice at offset
// (N-1-row).
//
// Border layout (libvpx d135_predictor_8x8):
//   border[0..5]   AVG3 of bottom-left left-column samples (descending)
//   border[6]      AVG3(topLeft, left[0], left[1])
//   border[7]      AVG3(left[0], topLeft, above[0])
//   border[8]      AVG3(topLeft, above[0], above[1])
//   border[9..14]  AVG3 of remaining above samples (ascending)

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// Batched ILGPU kernel that runs VP9 D135_PRED across N independent
/// 8x8 blocks in parallel.
/// </summary>
public sealed class Vp9D135Predict8x8Kernel : IDisposable
{
    private const int N = 8;

    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, int, int> _kernel;

    /// <summary>Compile the kernel onto <paramref name="accelerator"/>.</summary>
    public Vp9D135Predict8x8Kernel(Accelerator accelerator)
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

    /// <summary>Kernel body. 15 border cells in registers; 8 row slices written from registers.</summary>
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
        int l0 = leftFlat[lBase + 0];
        int l1 = leftFlat[lBase + 1];
        int l2 = leftFlat[lBase + 2];
        int l3 = leftFlat[lBase + 3];
        int l4 = leftFlat[lBase + 4];
        int l5 = leftFlat[lBase + 5];
        int l6 = leftFlat[lBase + 6];
        int l7 = leftFlat[lBase + 7];

        // Build the 15 border cells.
        // border[i] = AVG3(left[N-3-i], left[N-2-i], left[N-1-i]) for i=0..N-3
        byte b0 = (byte)((l5 + 2 * l6 + l7 + 2) >> 2);
        byte b1 = (byte)((l4 + 2 * l5 + l6 + 2) >> 2);
        byte b2 = (byte)((l3 + 2 * l4 + l5 + 2) >> 2);
        byte b3 = (byte)((l2 + 2 * l3 + l4 + 2) >> 2);
        byte b4 = (byte)((l1 + 2 * l2 + l3 + 2) >> 2);
        byte b5 = (byte)((l0 + 2 * l1 + l2 + 2) >> 2);
        byte b6 = (byte)((tl + 2 * l0 + l1 + 2) >> 2);
        byte b7 = (byte)((l0 + 2 * tl + a0 + 2) >> 2);
        byte b8 = (byte)((tl + 2 * a0 + a1 + 2) >> 2);
        byte b9 = (byte)((a0 + 2 * a1 + a2 + 2) >> 2);
        byte b10 = (byte)((a1 + 2 * a2 + a3 + 2) >> 2);
        byte b11 = (byte)((a2 + 2 * a3 + a4 + 2) >> 2);
        byte b12 = (byte)((a3 + 2 * a4 + a5 + 2) >> 2);
        byte b13 = (byte)((a4 + 2 * a5 + a6 + 2) >> 2);
        byte b14 = (byte)((a5 + 2 * a6 + a7 + 2) >> 2);

        // Row r writes border[N-1-r..N-1-r+N-1] = border[7-r..14-r].

        // Row 0 (srcStart=7): border[7..14].
        dstFlat[dBase + 0] = b7;
        dstFlat[dBase + 1] = b8;
        dstFlat[dBase + 2] = b9;
        dstFlat[dBase + 3] = b10;
        dstFlat[dBase + 4] = b11;
        dstFlat[dBase + 5] = b12;
        dstFlat[dBase + 6] = b13;
        dstFlat[dBase + 7] = b14;

        // Row 1 (srcStart=6): border[6..13].
        long row1 = dBase + N;
        dstFlat[row1 + 0] = b6;
        dstFlat[row1 + 1] = b7;
        dstFlat[row1 + 2] = b8;
        dstFlat[row1 + 3] = b9;
        dstFlat[row1 + 4] = b10;
        dstFlat[row1 + 5] = b11;
        dstFlat[row1 + 6] = b12;
        dstFlat[row1 + 7] = b13;

        // Row 2 (srcStart=5): border[5..12].
        long row2 = dBase + 2 * N;
        dstFlat[row2 + 0] = b5;
        dstFlat[row2 + 1] = b6;
        dstFlat[row2 + 2] = b7;
        dstFlat[row2 + 3] = b8;
        dstFlat[row2 + 4] = b9;
        dstFlat[row2 + 5] = b10;
        dstFlat[row2 + 6] = b11;
        dstFlat[row2 + 7] = b12;

        // Row 3 (srcStart=4): border[4..11].
        long row3 = dBase + 3 * N;
        dstFlat[row3 + 0] = b4;
        dstFlat[row3 + 1] = b5;
        dstFlat[row3 + 2] = b6;
        dstFlat[row3 + 3] = b7;
        dstFlat[row3 + 4] = b8;
        dstFlat[row3 + 5] = b9;
        dstFlat[row3 + 6] = b10;
        dstFlat[row3 + 7] = b11;

        // Row 4 (srcStart=3): border[3..10].
        long row4 = dBase + 4 * N;
        dstFlat[row4 + 0] = b3;
        dstFlat[row4 + 1] = b4;
        dstFlat[row4 + 2] = b5;
        dstFlat[row4 + 3] = b6;
        dstFlat[row4 + 4] = b7;
        dstFlat[row4 + 5] = b8;
        dstFlat[row4 + 6] = b9;
        dstFlat[row4 + 7] = b10;

        // Row 5 (srcStart=2): border[2..9].
        long row5 = dBase + 5 * N;
        dstFlat[row5 + 0] = b2;
        dstFlat[row5 + 1] = b3;
        dstFlat[row5 + 2] = b4;
        dstFlat[row5 + 3] = b5;
        dstFlat[row5 + 4] = b6;
        dstFlat[row5 + 5] = b7;
        dstFlat[row5 + 6] = b8;
        dstFlat[row5 + 7] = b9;

        // Row 6 (srcStart=1): border[1..8].
        long row6 = dBase + 6 * N;
        dstFlat[row6 + 0] = b1;
        dstFlat[row6 + 1] = b2;
        dstFlat[row6 + 2] = b3;
        dstFlat[row6 + 3] = b4;
        dstFlat[row6 + 4] = b5;
        dstFlat[row6 + 5] = b6;
        dstFlat[row6 + 6] = b7;
        dstFlat[row6 + 7] = b8;

        // Row 7 (srcStart=0): border[0..7].
        long row7 = dBase + 7 * N;
        dstFlat[row7 + 0] = b0;
        dstFlat[row7 + 1] = b1;
        dstFlat[row7 + 2] = b2;
        dstFlat[row7 + 3] = b3;
        dstFlat[row7 + 4] = b4;
        dstFlat[row7 + 5] = b5;
        dstFlat[row7 + 6] = b6;
        dstFlat[row7 + 7] = b7;
    }

    /// <inheritdoc/>
    public void Dispose() { }
}
