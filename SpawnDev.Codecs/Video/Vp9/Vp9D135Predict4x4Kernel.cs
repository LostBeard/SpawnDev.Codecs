// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// ILGPU kernel for the VP9 D135_PRED intra predictor at 4x4. First
// three-edge directional mode (above + left + topLeft).
//
// libvpx algorithm: build a 2N-1 = 7 byte "border" array of AVG3-
// filtered edge samples reaching from the bottom-left of the left
// column up through the corner and across the top row. Each output
// row is an N-byte slice of that border at offset (N-1-row).
//
// Border layout (libvpx d135_predictor_4x4):
//   border[0]   AVG3(left[1], left[2], left[3])
//   border[1]   AVG3(left[0], left[1], left[2])
//   border[2]   AVG3(topLeft, left[0], left[1])
//   border[3]   AVG3(left[0], topLeft, above[0])
//   border[4]   AVG3(topLeft, above[0], above[1])
//   border[5]   AVG3(above[0], above[1], above[2])
//   border[6]   AVG3(above[1], above[2], above[3])
//
// Row r (r=0..3) writes border[3-r..6-r] across cols 0..3.
//
// Per-block thread; the 7 border cells live in registers so the
// output writes never read dst (the WebGPU atomic-RMW byte-write
// hazard pattern from slice 189). topLeftFlat is padded to 4 bytes
// for the rc.13 Wasm 1-byte allocation issue (slice 183).

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// Batched ILGPU kernel that runs VP9 D135_PRED across N independent
/// 4x4 blocks in parallel.
/// </summary>
public sealed class Vp9D135Predict4x4Kernel : IDisposable
{
    private const int N = 4;

    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, int, int> _kernel;

    /// <summary>Compile the kernel onto <paramref name="accelerator"/>.</summary>
    public Vp9D135Predict4x4Kernel(Accelerator accelerator)
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
        // Pad topLeftFlat to >= 4 bytes (rc.13 Wasm minimum allocation).
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

    /// <summary>Kernel body. One thread per block; 7 border cells in registers.</summary>
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
        int l0 = leftFlat[lBase + 0];
        int l1 = leftFlat[lBase + 1];
        int l2 = leftFlat[lBase + 2];
        int l3 = leftFlat[lBase + 3];

        // Build the 7 border cells.
        byte b0 = (byte)((l1 + 2 * l2 + l3 + 2) >> 2);
        byte b1 = (byte)((l0 + 2 * l1 + l2 + 2) >> 2);
        byte b2 = (byte)((tl + 2 * l0 + l1 + 2) >> 2);
        byte b3 = (byte)((l0 + 2 * tl + a0 + 2) >> 2);
        byte b4 = (byte)((tl + 2 * a0 + a1 + 2) >> 2);
        byte b5 = (byte)((a0 + 2 * a1 + a2 + 2) >> 2);
        byte b6 = (byte)((a1 + 2 * a2 + a3 + 2) >> 2);

        // Row 0 (srcStart=3): border[3..6] = b3, b4, b5, b6.
        dstFlat[dBase + 0] = b3;
        dstFlat[dBase + 1] = b4;
        dstFlat[dBase + 2] = b5;
        dstFlat[dBase + 3] = b6;

        // Row 1 (srcStart=2): border[2..5] = b2, b3, b4, b5.
        dstFlat[dBase + N + 0] = b2;
        dstFlat[dBase + N + 1] = b3;
        dstFlat[dBase + N + 2] = b4;
        dstFlat[dBase + N + 3] = b5;

        // Row 2 (srcStart=1): border[1..4] = b1, b2, b3, b4.
        dstFlat[dBase + 2 * N + 0] = b1;
        dstFlat[dBase + 2 * N + 1] = b2;
        dstFlat[dBase + 2 * N + 2] = b3;
        dstFlat[dBase + 2 * N + 3] = b4;

        // Row 3 (srcStart=0): border[0..3] = b0, b1, b2, b3.
        dstFlat[dBase + 3 * N + 0] = b0;
        dstFlat[dBase + 3 * N + 1] = b1;
        dstFlat[dBase + 3 * N + 2] = b2;
        dstFlat[dBase + 3 * N + 3] = b3;
    }

    /// <inheritdoc/>
    public void Dispose() { }
}
