// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// 8x8 sibling of Vp9D45Predict4x4Kernel. Same register-routing
// pattern, scaled to N=8: row 0 in 8 byte registers, rows 1-7
// derive from those registers + above_right. 2N=16 above samples
// per block.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// Batched ILGPU kernel that runs VP9 D45_PRED across N independent
/// 8x8 blocks in parallel.
/// </summary>
public sealed class Vp9D45Predict8x8Kernel : IDisposable
{
    private const int N = 8;

    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, int, int> _kernel;

    /// <summary>Compile the kernel onto <paramref name="accelerator"/>.</summary>
    public Vp9D45Predict8x8Kernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<byte>, int, int>(D45Kernel);
    }

    /// <summary>Run D45 prediction on <paramref name="blockCount"/> blocks.</summary>
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

    /// <summary>Kernel body. Row 0 in registers, rows 1-7 derived from those.</summary>
    private static void D45Kernel(
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

        byte aboveRight = aboveFlat[aBase + (N - 1)];

        // Compute row 0 cells in registers.
        // r0[x] = AVG3(above[x], above[x+1], above[x+2]) for x=0..N-2
        // r0[N-1] = above_right
        int a0 = aboveFlat[aBase + 0];
        int a1 = aboveFlat[aBase + 1];
        int a2 = aboveFlat[aBase + 2];
        int a3 = aboveFlat[aBase + 3];
        int a4 = aboveFlat[aBase + 4];
        int a5 = aboveFlat[aBase + 5];
        int a6 = aboveFlat[aBase + 6];
        int a7 = aboveFlat[aBase + 7];
        int a8 = aboveFlat[aBase + 8];

        byte r0c0 = (byte)((a0 + 2 * a1 + a2 + 2) >> 2);
        byte r0c1 = (byte)((a1 + 2 * a2 + a3 + 2) >> 2);
        byte r0c2 = (byte)((a2 + 2 * a3 + a4 + 2) >> 2);
        byte r0c3 = (byte)((a3 + 2 * a4 + a5 + 2) >> 2);
        byte r0c4 = (byte)((a4 + 2 * a5 + a6 + 2) >> 2);
        byte r0c5 = (byte)((a5 + 2 * a6 + a7 + 2) >> 2);
        byte r0c6 = (byte)((a6 + 2 * a7 + a8 + 2) >> 2);
        byte r0c7 = aboveRight;

        // Row 0
        dstFlat[dBase + 0] = r0c0;
        dstFlat[dBase + 1] = r0c1;
        dstFlat[dBase + 2] = r0c2;
        dstFlat[dBase + 3] = r0c3;
        dstFlat[dBase + 4] = r0c4;
        dstFlat[dBase + 5] = r0c5;
        dstFlat[dBase + 6] = r0c6;
        dstFlat[dBase + 7] = r0c7;

        // Row 1 (shift left by 1).
        long row1 = dBase + N;
        dstFlat[row1 + 0] = r0c1;
        dstFlat[row1 + 1] = r0c2;
        dstFlat[row1 + 2] = r0c3;
        dstFlat[row1 + 3] = r0c4;
        dstFlat[row1 + 4] = r0c5;
        dstFlat[row1 + 5] = r0c6;
        dstFlat[row1 + 6] = r0c7;
        dstFlat[row1 + 7] = aboveRight;

        // Row 2 (shift left by 2).
        long row2 = dBase + 2 * N;
        dstFlat[row2 + 0] = r0c2;
        dstFlat[row2 + 1] = r0c3;
        dstFlat[row2 + 2] = r0c4;
        dstFlat[row2 + 3] = r0c5;
        dstFlat[row2 + 4] = r0c6;
        dstFlat[row2 + 5] = r0c7;
        dstFlat[row2 + 6] = aboveRight;
        dstFlat[row2 + 7] = aboveRight;

        // Row 3
        long row3 = dBase + 3 * N;
        dstFlat[row3 + 0] = r0c3;
        dstFlat[row3 + 1] = r0c4;
        dstFlat[row3 + 2] = r0c5;
        dstFlat[row3 + 3] = r0c6;
        dstFlat[row3 + 4] = r0c7;
        dstFlat[row3 + 5] = aboveRight;
        dstFlat[row3 + 6] = aboveRight;
        dstFlat[row3 + 7] = aboveRight;

        // Row 4
        long row4 = dBase + 4 * N;
        dstFlat[row4 + 0] = r0c4;
        dstFlat[row4 + 1] = r0c5;
        dstFlat[row4 + 2] = r0c6;
        dstFlat[row4 + 3] = r0c7;
        dstFlat[row4 + 4] = aboveRight;
        dstFlat[row4 + 5] = aboveRight;
        dstFlat[row4 + 6] = aboveRight;
        dstFlat[row4 + 7] = aboveRight;

        // Row 5
        long row5 = dBase + 5 * N;
        dstFlat[row5 + 0] = r0c5;
        dstFlat[row5 + 1] = r0c6;
        dstFlat[row5 + 2] = r0c7;
        dstFlat[row5 + 3] = aboveRight;
        dstFlat[row5 + 4] = aboveRight;
        dstFlat[row5 + 5] = aboveRight;
        dstFlat[row5 + 6] = aboveRight;
        dstFlat[row5 + 7] = aboveRight;

        // Row 6
        long row6 = dBase + 6 * N;
        dstFlat[row6 + 0] = r0c6;
        dstFlat[row6 + 1] = r0c7;
        dstFlat[row6 + 2] = aboveRight;
        dstFlat[row6 + 3] = aboveRight;
        dstFlat[row6 + 4] = aboveRight;
        dstFlat[row6 + 5] = aboveRight;
        dstFlat[row6 + 6] = aboveRight;
        dstFlat[row6 + 7] = aboveRight;

        // Row 7
        long row7 = dBase + 7 * N;
        dstFlat[row7 + 0] = r0c7;
        dstFlat[row7 + 1] = aboveRight;
        dstFlat[row7 + 2] = aboveRight;
        dstFlat[row7 + 3] = aboveRight;
        dstFlat[row7 + 4] = aboveRight;
        dstFlat[row7 + 5] = aboveRight;
        dstFlat[row7 + 6] = aboveRight;
        dstFlat[row7 + 7] = aboveRight;
    }

    /// <inheritdoc/>
    public void Dispose() { }
}
