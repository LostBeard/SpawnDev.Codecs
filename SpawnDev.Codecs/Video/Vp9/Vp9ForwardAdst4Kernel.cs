// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// ILGPU kernel for the VP9 forward 1D 4-point ADST. Bit-exact mirror of
// Vp9ForwardAdst4.Transform (the libvpx fadst4 port). Batched: one thread
// per 4-point ADST, N 4-point transforms in parallel.
//
// 1D primitive. The encoder composes it with itself across rows + columns
// to build the 2D ADST_ADST 4x4 transform, or with the 1D forward DCT4
// to build mixed DCT_ADST / ADST_DCT.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// Batched ILGPU kernel for the VP9 forward 1D 4-point ADST. Bit-exact
/// mirror of <see cref="Vp9ForwardAdst4.Transform"/>.
/// </summary>
public sealed class Vp9ForwardAdst4Kernel : IDisposable
{
    private const int Sinpi1_9 = 5283;
    private const int Sinpi2_9 = 9929;
    private const int Sinpi3_9 = 13377;
    private const int Sinpi4_9 = 15212;

    private const int DctConstBits = 14;
    private const int DctConstRounding = 1 << (DctConstBits - 1);

    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, int> _kernel;

    /// <summary>Compile the kernel onto <paramref name="accelerator"/>.</summary>
    public Vp9ForwardAdst4Kernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, ArrayView<int>, int>(FadstKernel);
    }

    /// <summary>
    /// Run the 4-point forward ADST on <paramref name="blockCount"/>
    /// independent 4-point vectors. Each transform: 4 contiguous ints
    /// in <paramref name="input"/> + 4 contiguous ints in <paramref name="output"/>.
    /// </summary>
    public void Run(ArrayView<int> input, ArrayView<int> output, int blockCount)
    {
        if (blockCount < 0) throw new ArgumentOutOfRangeException(nameof(blockCount));
        if (blockCount == 0) return;
        if (input.Length < blockCount * 4L)
            throw new ArgumentException($"input must hold at least blockCount*4 ints (got {input.Length}).", nameof(input));
        if (output.Length < blockCount * 4L)
            throw new ArgumentException($"output must hold at least blockCount*4 ints (got {output.Length}).", nameof(output));
        _kernel(blockCount, input, output, blockCount);
    }

    /// <summary>
    /// Convenience: allocate temporary GPU buffers, run, copy back.
    /// </summary>
    public async Task RunAsync(ReadOnlyMemory<int> input, Memory<int> output, int blockCount)
    {
        if (blockCount <= 0) return;
        using var dIn = _accelerator.Allocate1D<int>(blockCount * 4);
        using var dOut = _accelerator.Allocate1D<int>(blockCount * 4);
        dIn.View.CopyFromCPU(input.Span.ToArray());
        _kernel(blockCount, dIn.View, dOut.View, blockCount);
        await _accelerator.SynchronizeAsync();
        var readBack = await dOut.CopyToHostAsync();
        readBack.AsSpan(0, blockCount * 4).CopyTo(output.Span);
    }

    private static void FadstKernel(
        Index1D blockIdx,
        ArrayView<int> input,
        ArrayView<int> output,
        int blockCount)
    {
        int idx = blockIdx;
        if (idx >= blockCount) return;
        long inBase = (long)idx * 4;
        long outBase = (long)idx * 4;

        int x0 = input[inBase + 0];
        int x1 = input[inBase + 1];
        int x2 = input[inBase + 2];
        int x3 = input[inBase + 3];

        // libvpx fadst4 fast-path zero check: if all inputs are 0, output zeros.
        // Math identity: zero inputs naturally produce zero outputs in the
        // formulas below (every term has at least one x_i factor), so the
        // unconditional path is bit-exact for zeros - skip the branch for
        // GPU divergence cleanliness. Reference does the early-return as
        // an optimization; the kernel matches its outputs without it.

        long s0 = (long)Sinpi1_9 * x0;
        long s1 = (long)Sinpi4_9 * x0;
        long s2 = (long)Sinpi2_9 * x1;
        long s3 = (long)Sinpi1_9 * x1;
        long s4 = (long)Sinpi3_9 * x2;
        long s5 = (long)Sinpi4_9 * x3;
        long s6 = (long)Sinpi2_9 * x3;
        long s7 = x0 + x1 - x3;

        long y0 = s0 + s2 + s5;
        long y1 = (long)Sinpi3_9 * s7;
        long y2 = s1 - s3 + s6;
        long y3 = s4;

        long t0 = y0 + y3;
        long t1 = y1;
        long t2 = y2 - y3;
        long t3 = y2 - y0 + y3;

        output[outBase + 0] = (int)((t0 + DctConstRounding) >> DctConstBits);
        output[outBase + 1] = (int)((t1 + DctConstRounding) >> DctConstBits);
        output[outBase + 2] = (int)((t2 + DctConstRounding) >> DctConstBits);
        output[outBase + 3] = (int)((t3 + DctConstRounding) >> DctConstBits);
    }

    /// <summary>Release kernel resources. Does NOT dispose the accelerator.</summary>
    public void Dispose() { /* auto-grouped */ }
}
