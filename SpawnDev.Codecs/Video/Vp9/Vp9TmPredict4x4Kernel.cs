// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// ILGPU kernel for the VP9 TM_PRED intra predictor at 4x4. Per-block
// thread, batched. Bit-exact against Vp9TmPredictor.TmPredict.
//
// Per-pixel arithmetic:
//   dst[r][c] = clip_pixel(above[c] + left[r] - topLeft)
//
// libvpx reference: vpx_dsp/intrapred.c vpx_tm_predictor_4x4_c.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// Batched ILGPU kernel that runs VP9 TM_PRED across N independent
/// 4x4 blocks in parallel.
/// </summary>
public sealed class Vp9TmPredict4x4Kernel : IDisposable
{
    private const int N = 4;

    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, int, int> _kernel;

    /// <summary>Compile the kernel onto <paramref name="accelerator"/>.</summary>
    public Vp9TmPredict4x4Kernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, int, int>(TmKernel);
    }

    /// <summary>
    /// Run TM prediction on <paramref name="blockCount"/> blocks.
    /// </summary>
    /// <param name="topLeftFlat">Block-major flat top-left bytes (1 byte per block).</param>
    /// <param name="aboveFlat">Block-major flat above samples (N bytes per block).</param>
    /// <param name="leftFlat">Block-major flat left samples (N bytes per block).</param>
    /// <param name="dstFlat">Block-major destination (blockStrideBytes per block).</param>
    /// <param name="blockCount">Number of 4x4 blocks to predict.</param>
    /// <param name="blockStrideBytes">Bytes per destination block (default N*N=16).</param>
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
        // Pad topLeftFlat to >= 4 bytes - Wasm rejects sub-4-byte typed
        // array allocations under rc.13 ("Invalid typed array length: 4"
        // RangeError). The kernel only reads indices 0..blockCount-1 so
        // the trailing padding bytes are inert.
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

    /// <summary>Kernel body. One thread per block.</summary>
    private static void TmKernel(
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

        for (int row = 0; row < N; row++)
        {
            int leftR = leftFlat[lBase + row];
            long rowBase = dBase + row * N;
            for (int col = 0; col < N; col++)
            {
                int v = leftR + aboveFlat[aBase + col] - tl;
                if (v < 0) v = 0;
                else if (v > 255) v = 255;
                dstFlat[rowBase + col] = (byte)v;
            }
        }
    }

    /// <inheritdoc/>
    public void Dispose() { }
}
