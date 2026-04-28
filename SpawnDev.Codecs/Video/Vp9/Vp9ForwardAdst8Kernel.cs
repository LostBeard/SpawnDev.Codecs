// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// ILGPU kernel for the VP9 forward 1D 8-point ADST. Bit-exact mirror of
// Vp9ForwardAdst8.Transform (the libvpx fadst8 port). Batched: one thread
// per 8-point ADST, N 8-point transforms in parallel.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// Batched ILGPU kernel for the VP9 forward 1D 8-point ADST. Bit-exact
/// mirror of <see cref="Vp9ForwardAdst8.Transform"/>.
/// </summary>
public sealed class Vp9ForwardAdst8Kernel : IDisposable
{
    private const int CosPi2_64  = 16305;
    private const int CosPi6_64  = 15679;
    private const int CosPi8_64  = 15137;
    private const int CosPi10_64 = 14449;
    private const int CosPi14_64 = 12665;
    private const int CosPi16_64 = 11585;
    private const int CosPi18_64 = 10394;
    private const int CosPi22_64 = 7723;
    private const int CosPi24_64 = 6270;
    private const int CosPi26_64 = 4756;
    private const int CosPi30_64 = 1606;

    private const int DctConstBits = 14;
    private const int DctConstRounding = 1 << (DctConstBits - 1);

    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, int> _kernel;

    /// <summary>Compile the kernel onto <paramref name="accelerator"/>.</summary>
    public Vp9ForwardAdst8Kernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, ArrayView<int>, int>(FadstKernel);
    }

    /// <summary>
    /// Run the 8-point forward ADST on <paramref name="blockCount"/>
    /// independent 8-point vectors. Each transform: 8 contiguous ints
    /// in / 8 contiguous ints out.
    /// </summary>
    public void Run(ArrayView<int> input, ArrayView<int> output, int blockCount)
    {
        if (blockCount < 0) throw new ArgumentOutOfRangeException(nameof(blockCount));
        if (blockCount == 0) return;
        if (input.Length < blockCount * 8L)
            throw new ArgumentException($"input must hold at least blockCount*8 ints.", nameof(input));
        if (output.Length < blockCount * 8L)
            throw new ArgumentException($"output must hold at least blockCount*8 ints.", nameof(output));
        _kernel(blockCount, input, output, blockCount);
    }

    /// <summary>
    /// Convenience: allocate temporary GPU buffers, run, copy back.
    /// </summary>
    public async Task RunAsync(ReadOnlyMemory<int> input, Memory<int> output, int blockCount)
    {
        if (blockCount <= 0) return;
        using var dIn = _accelerator.Allocate1D<int>(blockCount * 8);
        using var dOut = _accelerator.Allocate1D<int>(blockCount * 8);
        dIn.View.CopyFromCPU(input.Span.ToArray());
        _kernel(blockCount, dIn.View, dOut.View, blockCount);
        await _accelerator.SynchronizeAsync();
        var readBack = await dOut.CopyToHostAsync();
        readBack.AsSpan(0, blockCount * 8).CopyTo(output.Span);
    }

    private static void FadstKernel(
        Index1D blockIdx,
        ArrayView<int> input,
        ArrayView<int> output,
        int blockCount)
    {
        int idx = blockIdx;
        if (idx >= blockCount) return;
        long inBase = (long)idx * 8;
        long outBase = (long)idx * 8;

        // libvpx fadst8 input remap: x_i indexes shuffled.
        long x0 = input[inBase + 7], x1 = input[inBase + 0], x2 = input[inBase + 5], x3 = input[inBase + 2];
        long x4 = input[inBase + 3], x5 = input[inBase + 4], x6 = input[inBase + 1], x7 = input[inBase + 6];

        // Stage 1
        long s0 = (long)CosPi2_64 * x0 + (long)CosPi30_64 * x1;
        long s1 = (long)CosPi30_64 * x0 - (long)CosPi2_64 * x1;
        long s2 = (long)CosPi10_64 * x2 + (long)CosPi22_64 * x3;
        long s3 = (long)CosPi22_64 * x2 - (long)CosPi10_64 * x3;
        long s4 = (long)CosPi18_64 * x4 + (long)CosPi14_64 * x5;
        long s5 = (long)CosPi14_64 * x4 - (long)CosPi18_64 * x5;
        long s6 = (long)CosPi26_64 * x6 + (long)CosPi6_64 * x7;
        long s7 = (long)CosPi6_64 * x6 - (long)CosPi26_64 * x7;

        x0 = RoundShift(s0 + s4);
        x1 = RoundShift(s1 + s5);
        x2 = RoundShift(s2 + s6);
        x3 = RoundShift(s3 + s7);
        x4 = RoundShift(s0 - s4);
        x5 = RoundShift(s1 - s5);
        x6 = RoundShift(s2 - s6);
        x7 = RoundShift(s3 - s7);

        // Stage 2
        long t0 = x0, t1 = x1, t2 = x2, t3 = x3;
        long t4 = (long)CosPi8_64 * x4 + (long)CosPi24_64 * x5;
        long t5 = (long)CosPi24_64 * x4 - (long)CosPi8_64 * x5;
        long t6 = -(long)CosPi24_64 * x6 + (long)CosPi8_64 * x7;
        long t7 = (long)CosPi8_64 * x6 + (long)CosPi24_64 * x7;

        x0 = t0 + t2;
        x1 = t1 + t3;
        x2 = t0 - t2;
        x3 = t1 - t3;
        x4 = RoundShift(t4 + t6);
        x5 = RoundShift(t5 + t7);
        x6 = RoundShift(t4 - t6);
        x7 = RoundShift(t5 - t7);

        // Stage 3
        long u2 = (long)CosPi16_64 * (x2 + x3);
        long u3 = (long)CosPi16_64 * (x2 - x3);
        long u6 = (long)CosPi16_64 * (x6 + x7);
        long u7 = (long)CosPi16_64 * (x6 - x7);

        x2 = RoundShift(u2);
        x3 = RoundShift(u3);
        x6 = RoundShift(u6);
        x7 = RoundShift(u7);

        // Output (libvpx negation pattern).
        output[outBase + 0] = (int)x0;
        output[outBase + 1] = (int)-x4;
        output[outBase + 2] = (int)x6;
        output[outBase + 3] = (int)-x2;
        output[outBase + 4] = (int)x3;
        output[outBase + 5] = (int)-x7;
        output[outBase + 6] = (int)x5;
        output[outBase + 7] = (int)-x1;
    }

    private static long RoundShift(long input) =>
        (input + DctConstRounding) >> DctConstBits;

    /// <summary>Release kernel resources. Does NOT dispose the accelerator.</summary>
    public void Dispose() { /* auto-grouped */ }
}
