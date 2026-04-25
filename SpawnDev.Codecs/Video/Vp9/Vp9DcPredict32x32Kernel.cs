// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// 32x32 sibling of Vp9DcPredict4x4 / 8x8 / 16x16 kernels. Same
// Vp9DcVariant routing and per-block-thread structure - only the
// per-edge sum extent (32 samples) and shift counts (Log2N = 5,
// Log2N + 1 = 6) change. Bit-exact against Vp9DcPredictor at N=32.
//
// rc.13 backend status: 4/6 green.
//   CPU + CUDA + OpenCL + Wasm: bit-exact, all six tests green.
//   WebGPU: 30+ second WGSL compile cliff on the inlined kernel
//           (same pattern as the pre-rc.14 Vp9Idct16x16Kernel issue).
//   WebGL: gated at the runner level (atomic-RMW for ArrayView<byte>
//          writes).
//
// rc.14 was supposed to fix WebGPU + Wasm via the function-definition
// codegen path + a wide-to-narrow narrowing patch. Verification on
// 2026-04-25 found new regressions: WebGPU emits invalid WGSL
// (`-> void` parse error from the function-definition path) and Wasm
// still silent-zero-writes when [MethodImpl(NoInlining)] is applied
// to the byte-write FillBlock helper. Filed back to Geordi for rc.15.
// Until rc.15 lands, keep this kernel inlined - 4/6 green is better
// than 3/6.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// Batched ILGPU kernel that runs the VP9 DC intra predictor across N
/// independent 32x32 blocks in parallel.
/// </summary>
public sealed class Vp9DcPredict32x32Kernel : IDisposable
{
    private const int N = 32;
    private const int Log2N = 5;

    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, int, int, int> _kernel;

    /// <summary>Compile the kernel onto <paramref name="accelerator"/>.</summary>
    public Vp9DcPredict32x32Kernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, int, int, int>(DcKernel);
    }

    /// <summary>Run the DC predictor on <paramref name="blockCount"/> blocks.</summary>
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

    /// <summary>Convenience: allocate, run, read back.</summary>
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

    /// <summary>Kernel body. One thread per block, fully inlined.</summary>
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
