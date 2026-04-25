// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// ILGPU kernel for the VP9 D63_PRED intra predictor at 4x4.
//
// D63 is the 63-degree variant - between vertical and the 45-degree
// diagonal. Two different per-row filters drive the seed rows:
//   row 0 (even): dst[0][c] = AVG2(above[c], above[c+1])
//   row 1 (odd):  dst[1][c] = AVG3(above[c], above[c+1], above[c+2])
// Subsequent rows are shifted-left copies of rows 0 and 1 with the
// right edge padded by above[n-1].
//
// Same register-routing pattern as D45 4x4: hold seed rows in
// registers so the propagation never reads dstFlat after writing
// (sidesteps the WebGPU atomic-RMW byte-write hazard).
//
// libvpx reference: vpx_dsp/intrapred.c vpx_d63_predictor_4x4_c.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// Batched ILGPU kernel that runs VP9 D63_PRED across N independent
/// 4x4 blocks in parallel.
/// </summary>
public sealed class Vp9D63Predict4x4Kernel : IDisposable
{
    private const int N = 4;

    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, int, int> _kernel;

    /// <summary>Compile the kernel onto <paramref name="accelerator"/>.</summary>
    public Vp9D63Predict4x4Kernel(Accelerator accelerator)
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

    /// <summary>Kernel body. One thread per block; rows 0+1 in registers.</summary>
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

        byte fillVal = (byte)a3;  // above[N-1] for N=4

        // Row 0 (AVG2 of consecutive above pairs).
        byte r0c0 = (byte)((a0 + a1 + 1) >> 1);
        byte r0c1 = (byte)((a1 + a2 + 1) >> 1);
        byte r0c2 = (byte)((a2 + a3 + 1) >> 1);
        byte r0c3 = (byte)((a3 + a4 + 1) >> 1);

        // Row 1 (AVG3 of consecutive above triples).
        byte r1c0 = (byte)((a0 + 2 * a1 + a2 + 2) >> 2);
        byte r1c1 = (byte)((a1 + 2 * a2 + a3 + 2) >> 2);
        byte r1c2 = (byte)((a2 + 2 * a3 + a4 + 2) >> 2);
        byte r1c3 = (byte)((a3 + 2 * a4 + a5 + 2) >> 2);

        // Row 0.
        dstFlat[dBase + 0] = r0c0;
        dstFlat[dBase + 1] = r0c1;
        dstFlat[dBase + 2] = r0c2;
        dstFlat[dBase + 3] = r0c3;

        // Row 1.
        dstFlat[dBase + N + 0] = r1c0;
        dstFlat[dBase + N + 1] = r1c1;
        dstFlat[dBase + N + 2] = r1c2;
        dstFlat[dBase + N + 3] = r1c3;

        // Row 2: row 0 shifted left by 1 (cols 1,2 -> 0,1; cols 2,3 -> fill).
        dstFlat[dBase + 2 * N + 0] = r0c1;
        dstFlat[dBase + 2 * N + 1] = r0c2;
        dstFlat[dBase + 2 * N + 2] = fillVal;
        dstFlat[dBase + 2 * N + 3] = fillVal;

        // Row 3: row 1 shifted left by 1.
        dstFlat[dBase + 3 * N + 0] = r1c1;
        dstFlat[dBase + 3 * N + 1] = r1c2;
        dstFlat[dBase + 3 * N + 2] = fillVal;
        dstFlat[dBase + 3 * N + 3] = fillVal;
    }

    /// <inheritdoc/>
    public void Dispose() { }
}
