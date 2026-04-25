// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// ILGPU kernel for the VP9 V_PRED and H_PRED intra predictors at 4x4.
// Both modes share the same per-block-thread structure so they live
// in one class with a Vp9VhMode parameter selecting which edge to
// copy:
//
//   V_PRED:  dst[r][c] = above[c]     (each output row is the above row)
//   H_PRED:  dst[r][c] = left[r]      (each output column is the left column)
//
// libvpx reference: vpx_dsp/intrapred.c vpx_v_predictor_NxN_c /
// vpx_h_predictor_NxN_c. Bit-exact against Vp9VHPredictor.VPredict /
// HPredict on every backend.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// V_PRED vs H_PRED selector for <see cref="Vp9VhPredict4x4Kernel"/>
/// and its larger-size siblings.
/// </summary>
public enum Vp9VhMode : byte
{
    /// <summary>Copy <c>above[col]</c> to every output row.</summary>
    V = 0,
    /// <summary>Copy <c>left[row]</c> across each output row.</summary>
    H = 1,
}

/// <summary>
/// Batched ILGPU kernel that runs V_PRED or H_PRED across N
/// independent 4x4 blocks in parallel.
/// </summary>
public sealed class Vp9VhPredict4x4Kernel : IDisposable
{
    private const int N = 4;

    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, int, int, int> _kernel;

    /// <summary>Compile the kernel onto <paramref name="accelerator"/>.</summary>
    public Vp9VhPredict4x4Kernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, int, int, int>(VhKernel);
    }

    /// <summary>Run the V/H predictor on <paramref name="blockCount"/> blocks.</summary>
    public void Run(
        ArrayView<byte> aboveFlat,
        ArrayView<byte> leftFlat,
        ArrayView<byte> dstFlat,
        Vp9VhMode mode,
        int blockCount,
        int blockStrideBytes = N * N)
    {
        if (blockCount < 0) throw new ArgumentOutOfRangeException(nameof(blockCount));
        if (blockCount == 0) return;
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
        _kernel(blockCount, aboveFlat, leftFlat, dstFlat, blockCount, blockStrideBytes, (int)mode);
    }

    /// <summary>Convenience: allocate, run, read back.</summary>
    public async Task RunAsync(
        ReadOnlyMemory<byte> aboveFlat,
        ReadOnlyMemory<byte> leftFlat,
        Memory<byte> dstFlat,
        Vp9VhMode mode,
        int blockCount,
        int blockStrideBytes = N * N)
    {
        if (blockCount <= 0) return;
        using var dAbove = _accelerator.Allocate1D<byte>(blockCount * (long)N);
        using var dLeft = _accelerator.Allocate1D<byte>(blockCount * (long)N);
        using var dDst = _accelerator.Allocate1D<byte>(blockCount * (long)blockStrideBytes);
        dAbove.View.CopyFromCPU(aboveFlat.Span.ToArray());
        dLeft.View.CopyFromCPU(leftFlat.Span.ToArray());
        dDst.View.CopyFromCPU(dstFlat.Span.ToArray());
        _kernel(blockCount, dAbove.View, dLeft.View, dDst.View, blockCount, blockStrideBytes, (int)mode);
        await _accelerator.SynchronizeAsync();
        var readBack = await dDst.CopyToHostAsync();
        readBack.AsSpan(0, dstFlat.Length).CopyTo(dstFlat.Span);
    }

    /// <summary>Kernel body. One thread per block.</summary>
    private static void VhKernel(
        Index1D blockIdx,
        ArrayView<byte> aboveFlat,
        ArrayView<byte> leftFlat,
        ArrayView<byte> dstFlat,
        int blockCount,
        int blockStrideBytes,
        int mode)
    {
        int idx = blockIdx;
        if (idx >= blockCount) return;

        long aBase = (long)idx * N;
        long lBase = (long)idx * N;
        long dBase = (long)idx * blockStrideBytes;

        if (mode == (int)Vp9VhMode.V)
        {
            // Each row is a copy of above[0..N-1].
            for (int row = 0; row < N; row++)
            {
                long rowBase = dBase + row * N;
                dstFlat[rowBase + 0] = aboveFlat[aBase + 0];
                dstFlat[rowBase + 1] = aboveFlat[aBase + 1];
                dstFlat[rowBase + 2] = aboveFlat[aBase + 2];
                dstFlat[rowBase + 3] = aboveFlat[aBase + 3];
            }
        }
        else
        {
            // Each row r is filled with left[r] across all columns.
            for (int row = 0; row < N; row++)
            {
                byte v = leftFlat[lBase + row];
                long rowBase = dBase + row * N;
                dstFlat[rowBase + 0] = v;
                dstFlat[rowBase + 1] = v;
                dstFlat[rowBase + 2] = v;
                dstFlat[rowBase + 3] = v;
            }
        }
    }

    /// <inheritdoc/>
    public void Dispose() { }
}
