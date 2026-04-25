// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// 8x8 sibling of Vp9DcPredict4x4Kernel. Same Vp9DcVariant routing
// and per-block-thread structure - only the per-edge sum extent
// (8 vs 4 samples) and shift counts (log2(8) = 3, log2(2*8) = 4)
// change. Bit-exact against Vp9DcPredictor at N=8.
//
// VP9 normative: 8x8 DC arithmetic must match libvpx
// vpx_dc_predictor_8x8_c (etc) byte-for-byte. The unit tests
// enforce this on every backend.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// Batched ILGPU kernel that runs the VP9 DC intra predictor across N
/// independent 8x8 blocks in parallel.
/// </summary>
public sealed class Vp9DcPredict8x8Kernel : IDisposable
{
    private const int N = 8;
    private const int Log2N = 3;

    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, int, int, int> _kernel;

    /// <summary>Compile the kernel onto <paramref name="accelerator"/>.</summary>
    public Vp9DcPredict8x8Kernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, int, int, int>(DcKernel);
    }

    /// <summary>
    /// Run the DC predictor on <paramref name="blockCount"/> blocks in
    /// parallel. Edge spans block-major flat with N=8 bytes per block;
    /// dst block-major flat with <paramref name="blockStrideBytes"/>
    /// per block (default 64 = 8*8).
    /// </summary>
    public void Run(
        ArrayView<byte> aboveFlat,
        ArrayView<byte> leftFlat,
        ArrayView<byte> dstFlat,
        Vp9DcVariant variant,
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
        _kernel(blockCount, aboveFlat, leftFlat, dstFlat, blockCount, blockStrideBytes, (int)variant);
    }

    /// <summary>
    /// Convenience: allocate temporary GPU buffers, run, and read back.
    /// Async because WebGPU forbids synchronous GPU-to-CPU copies.
    /// </summary>
    public async Task RunAsync(
        ReadOnlyMemory<byte> aboveFlat,
        ReadOnlyMemory<byte> leftFlat,
        Memory<byte> dstFlat,
        Vp9DcVariant variant,
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
        _kernel(blockCount, dAbove.View, dLeft.View, dDst.View, blockCount, blockStrideBytes, (int)variant);
        await _accelerator.SynchronizeAsync();
        var readBack = await dDst.CopyToHostAsync();
        readBack.AsSpan(0, dstFlat.Length).CopyTo(dstFlat.Span);
    }

    /// <summary>Kernel body. One thread per block.</summary>
    private static void DcKernel(
        Index1D blockIdx,
        ArrayView<byte> aboveFlat,
        ArrayView<byte> leftFlat,
        ArrayView<byte> dstFlat,
        int blockCount,
        int blockStrideBytes,
        int variant)
    {
        int idx = blockIdx;
        if (idx >= blockCount) return;

        long aBase = (long)idx * N;
        long lBase = (long)idx * N;
        long dBase = (long)idx * blockStrideBytes;

        byte dc;
        if (variant == (int)Vp9DcVariant.Both)
        {
            int sum = 0;
            for (int i = 0; i < N; i++) sum += aboveFlat[aBase + i];
            for (int i = 0; i < N; i++) sum += leftFlat[lBase + i];
            // (sum + N) >> (log2(N) + 1).
            dc = (byte)((sum + N) >> (Log2N + 1));
        }
        else if (variant == (int)Vp9DcVariant.TopOnly)
        {
            int sum = 0;
            for (int i = 0; i < N; i++) sum += aboveFlat[aBase + i];
            dc = (byte)((sum + (N >> 1)) >> Log2N);
        }
        else if (variant == (int)Vp9DcVariant.LeftOnly)
        {
            int sum = 0;
            for (int i = 0; i < N; i++) sum += leftFlat[lBase + i];
            dc = (byte)((sum + (N >> 1)) >> Log2N);
        }
        else
        {
            dc = 128;
        }

        // Fill 8x8 = 64 pixels.
        for (int row = 0; row < N; row++)
        {
            long rowBase = dBase + row * N;
            for (int col = 0; col < N; col++)
                dstFlat[rowBase + col] = dc;
        }
    }

    /// <inheritdoc/>
    public void Dispose() { }
}
