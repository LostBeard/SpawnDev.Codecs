// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// ILGPU kernel for the VP9 DC intra predictor at 4x4. Per-block
// thread, batched - one kernel dispatch covers N independent 4x4
// blocks. Bit-exact against Vp9DcPredictor.DcPredict /
// DcPredictTop / DcPredictLeft / DcPredict128 across all 6 backends
// (CPU emulator, CUDA, OpenCL, WebGPU, WebGL, Wasm).
//
// VP9 is a normative bitstream - the reference REQUIRES bit-for-bit
// output equality. The test suite asserts the kernel output matches
// the CPU oracle byte-for-byte on every backend. A one-byte
// divergence fails the test.
//
// Sizes 8x8 / 16x16 / 32x32 follow as separate kernel classes - the
// per-thread DC computation differs by N samples summed and the
// shift count, but the pattern is identical.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// VP9 DC variant code. Mirrors the four CPU oracle entry points
/// (<see cref="Vp9DcPredictor"/>) and selects which edges contribute
/// to the DC value.
/// </summary>
public enum Vp9DcVariant : byte
{
    /// <summary>Both above and left available: DC = (sum_a + sum_l + N) &gt;&gt; (log2(N) + 1).</summary>
    Both = 0,
    /// <summary>Above only: DC = (sum_a + N/2) &gt;&gt; log2(N).</summary>
    TopOnly = 1,
    /// <summary>Left only: DC = (sum_l + N/2) &gt;&gt; log2(N).</summary>
    LeftOnly = 2,
    /// <summary>Neither: DC = 128.</summary>
    None = 3,
}

/// <summary>
/// Batched ILGPU kernel that runs the VP9 DC intra predictor across N
/// independent 4x4 blocks in parallel. The kernel is stateless -
/// create once, dispatch many times.
/// </summary>
public sealed class Vp9DcPredict4x4Kernel : IDisposable
{
    private const int N = 4;
    private const int Log2N = 2;

    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, int, int, int> _kernel;

    /// <summary>Compile the kernel onto <paramref name="accelerator"/>.</summary>
    public Vp9DcPredict4x4Kernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, int, int, int>(DcKernel);
    }

    /// <summary>
    /// Run the DC predictor on <paramref name="blockCount"/> blocks in
    /// parallel. <paramref name="aboveFlat"/> / <paramref name="leftFlat"/>
    /// are block-major flat with N=4 bytes per block. <paramref name="dstFlat"/>
    /// is block-major flat with <paramref name="blockStrideBytes"/> per
    /// block (default 16 = 4*4).
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
    /// For tests and one-shot work. Async because WebGPU forbids
    /// synchronous GPU-to-CPU copies.
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
        // Pre-load dst with whatever the caller provided (the predictor
        // only writes block bytes; any padding past N*N stays as input).
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
            int sum = aboveFlat[aBase + 0] + aboveFlat[aBase + 1]
                    + aboveFlat[aBase + 2] + aboveFlat[aBase + 3]
                    + leftFlat[lBase + 0] + leftFlat[lBase + 1]
                    + leftFlat[lBase + 2] + leftFlat[lBase + 3];
            dc = (byte)((sum + N) >> (Log2N + 1));
        }
        else if (variant == (int)Vp9DcVariant.TopOnly)
        {
            int sum = aboveFlat[aBase + 0] + aboveFlat[aBase + 1]
                    + aboveFlat[aBase + 2] + aboveFlat[aBase + 3];
            dc = (byte)((sum + (N >> 1)) >> Log2N);
        }
        else if (variant == (int)Vp9DcVariant.LeftOnly)
        {
            int sum = leftFlat[lBase + 0] + leftFlat[lBase + 1]
                    + leftFlat[lBase + 2] + leftFlat[lBase + 3];
            dc = (byte)((sum + (N >> 1)) >> Log2N);
        }
        else
        {
            dc = 128;
        }

        // Fill 4x4 = 16 pixels, contiguous (no stride within the per-block
        // dest area; caller-supplied blockStrideBytes lets the buffer
        // include padding between blocks but the block itself is row-major
        // 4 bytes per row at offset row*4).
        for (int row = 0; row < N; row++)
        {
            long rowBase = dBase + row * N;
            dstFlat[rowBase + 0] = dc;
            dstFlat[rowBase + 1] = dc;
            dstFlat[rowBase + 2] = dc;
            dstFlat[rowBase + 3] = dc;
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        // Kernel handle is owned by the accelerator; nothing to release here.
    }
}
