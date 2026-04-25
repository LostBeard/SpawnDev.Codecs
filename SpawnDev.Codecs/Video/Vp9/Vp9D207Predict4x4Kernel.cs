// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// ILGPU kernel for the VP9 D207_PRED intra predictor at 4x4. Left-
// only directional - the libvpx C reference takes above as a
// parameter but ignores it.
//
// Algorithm (mirror of vpx_dsp/intrapred.c vpx_d207_predictor_4x4_c):
//   Column 0 (AVG2): dst[r][0] = AVG2(left[r], left[r+1])
//                    dst[N-1][0] = left[N-1]
//   Column 1 (AVG3): dst[r][1] = AVG3(left[r], left[r+1], left[r+2])
//                    dst[N-2][1] = AVG3(left[N-2], left[N-1], left[N-1])
//                    dst[N-1][1] = left[N-1]
//   Last row, cols 2..N-1: fill with left[N-1]
//   Remaining rows: dst[r][c] = dst[r+1][c-2] for c=2..N-1, r=N-2..0
//
// Same register-routing pattern as D45 / D63 4x4: hold cols 0 and 1
// in registers so the propagation never reads dstFlat. Avoids the
// WebGPU atomic-RMW byte-write read-after-write hazard.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// Batched ILGPU kernel that runs VP9 D207_PRED across N independent
/// 4x4 blocks in parallel. Reads only the left column.
/// </summary>
public sealed class Vp9D207Predict4x4Kernel : IDisposable
{
    private const int N = 4;

    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, int, int> _kernel;

    /// <summary>Compile the kernel onto <paramref name="accelerator"/>.</summary>
    public Vp9D207Predict4x4Kernel(Accelerator accelerator)
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

    /// <summary>Kernel body. One thread per block; cols 0+1 in registers.</summary>
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

        byte fillVal = (byte)l3;  // left[N-1]

        // Column 0: AVG2 pairs; bottom = left[N-1].
        byte c0r0 = (byte)((l0 + l1 + 1) >> 1);
        byte c0r1 = (byte)((l1 + l2 + 1) >> 1);
        byte c0r2 = (byte)((l2 + l3 + 1) >> 1);
        byte c0r3 = (byte)l3;

        // Column 1: AVG3 triples; second-to-last replicates left[N-1];
        // bottom = left[N-1].
        byte c1r0 = (byte)((l0 + 2 * l1 + l2 + 2) >> 2);
        byte c1r1 = (byte)((l1 + 2 * l2 + l3 + 2) >> 2);
        byte c1r2 = (byte)((l2 + 2 * l3 + l3 + 2) >> 2);
        byte c1r3 = (byte)l3;

        // Row 0: cols 0,1 from registers; cols 2,3 = next row's cols 0,1.
        dstFlat[dBase + 0] = c0r0;
        dstFlat[dBase + 1] = c1r0;
        dstFlat[dBase + 2] = c0r1;
        dstFlat[dBase + 3] = c1r1;

        // Row 1.
        dstFlat[dBase + N + 0] = c0r1;
        dstFlat[dBase + N + 1] = c1r1;
        dstFlat[dBase + N + 2] = c0r2;
        dstFlat[dBase + N + 3] = c1r2;

        // Row 2.
        dstFlat[dBase + 2 * N + 0] = c0r2;
        dstFlat[dBase + 2 * N + 1] = c1r2;
        dstFlat[dBase + 2 * N + 2] = c0r3;
        dstFlat[dBase + 2 * N + 3] = c1r3;

        // Row 3: cols 0,1 from registers (both = left[N-1]); cols 2,3 = fillVal.
        dstFlat[dBase + 3 * N + 0] = c0r3;
        dstFlat[dBase + 3 * N + 1] = c1r3;
        dstFlat[dBase + 3 * N + 2] = fillVal;
        dstFlat[dBase + 3 * N + 3] = fillVal;
    }

    /// <inheritdoc/>
    public void Dispose() { }
}
