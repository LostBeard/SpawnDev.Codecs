// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// 16x16 sibling of Vp9D45Predict4x4 / 8x8 kernels. N=16, 32 above
// samples per block.
//
// Row 0 cells (16 registers) drive the propagation. dst[r][c] reads
// r0[r+c] when r+c < N-1, else aboveRight. Implemented as a nested
// loop with a switch dispatch on (r+c) into the named register set
// so the kernel body stays compact while still routing carries
// through registers (avoiding the WebGPU read-after-write byte-write
// hazard).

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// Batched ILGPU kernel that runs VP9 D45_PRED across N independent
/// 16x16 blocks in parallel.
/// </summary>
public sealed class Vp9D45Predict16x16Kernel : IDisposable
{
    private const int N = 16;

    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, int, int> _kernel;

    /// <summary>Compile the kernel onto <paramref name="accelerator"/>.</summary>
    public Vp9D45Predict16x16Kernel(Accelerator accelerator)
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

    /// <summary>
    /// Kernel body. One thread per block. Row 0 in 16 named byte
    /// registers; nested loop dispatches each output cell to the
    /// correct register via a switch on (row + col).
    /// </summary>
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

        // Load 17 above samples into ints; r0[c] = AVG3(above[c], above[c+1], above[c+2]) for c=0..14.
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

        byte aboveRight = (byte)v15;  // above[N-1]

        // 16 row-0 cells: r0c0..r0c14 are AVG3, r0c15 = aboveRight.
        byte r0c0 = (byte)((v0 + 2 * v1 + v2 + 2) >> 2);
        byte r0c1 = (byte)((v1 + 2 * v2 + v3 + 2) >> 2);
        byte r0c2 = (byte)((v2 + 2 * v3 + v4 + 2) >> 2);
        byte r0c3 = (byte)((v3 + 2 * v4 + v5 + 2) >> 2);
        byte r0c4 = (byte)((v4 + 2 * v5 + v6 + 2) >> 2);
        byte r0c5 = (byte)((v5 + 2 * v6 + v7 + 2) >> 2);
        byte r0c6 = (byte)((v6 + 2 * v7 + v8 + 2) >> 2);
        byte r0c7 = (byte)((v7 + 2 * v8 + v9 + 2) >> 2);
        byte r0c8 = (byte)((v8 + 2 * v9 + v10 + 2) >> 2);
        byte r0c9 = (byte)((v9 + 2 * v10 + v11 + 2) >> 2);
        byte r0c10 = (byte)((v10 + 2 * v11 + v12 + 2) >> 2);
        byte r0c11 = (byte)((v11 + 2 * v12 + v13 + 2) >> 2);
        byte r0c12 = (byte)((v12 + 2 * v13 + v14 + 2) >> 2);
        byte r0c13 = (byte)((v13 + 2 * v14 + v15 + 2) >> 2);
        byte r0c14 = (byte)((v14 + 2 * v15 + v16 + 2) >> 2);
        byte r0c15 = aboveRight;

        // 256 writes: dst[r][c] = (r+c < N) ? r0c[r+c] : aboveRight.
        for (int row = 0; row < N; row++)
        {
            long rowBase = dBase + row * N;
            for (int col = 0; col < N; col++)
            {
                int diag = row + col;
                byte v;
                if (diag >= N)
                {
                    v = aboveRight;
                }
                else
                {
                    switch (diag)
                    {
                        case 0: v = r0c0; break;
                        case 1: v = r0c1; break;
                        case 2: v = r0c2; break;
                        case 3: v = r0c3; break;
                        case 4: v = r0c4; break;
                        case 5: v = r0c5; break;
                        case 6: v = r0c6; break;
                        case 7: v = r0c7; break;
                        case 8: v = r0c8; break;
                        case 9: v = r0c9; break;
                        case 10: v = r0c10; break;
                        case 11: v = r0c11; break;
                        case 12: v = r0c12; break;
                        case 13: v = r0c13; break;
                        case 14: v = r0c14; break;
                        default: v = r0c15; break;
                    }
                }
                dstFlat[rowBase + col] = v;
            }
        }
    }

    /// <inheritdoc/>
    public void Dispose() { }
}
