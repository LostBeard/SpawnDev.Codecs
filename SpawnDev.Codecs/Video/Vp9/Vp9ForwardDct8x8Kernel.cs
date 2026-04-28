// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// ILGPU kernel for the VP9 forward DCT 8x8. Bit-exact mirror of
// Vp9ForwardDct8x8.Transform (the libvpx vpx_fdct8x8_c port). Batched:
// one thread per 8x8 block.
//
// VP9 is a normative bitstream so the kernel must produce bit-identical
// output to the reference function across every backend. Tests assert
// this directly via Vp9ForwardDct8x8KernelTests (cross-backend).
//
// LocalMemory layout
//   tmp[c*8 + r] - pass-1 column-c output-r (intermediate). Pass 2 reads
//   tmp[c + r*8] for column c, sub-row r (raster column of intermediate).
//
// Pass 1 multiplies inputs by 4 (input * 4); pass 2 reads intermediate
// directly. Final post-pass divides every output by 2 (libvpx
// `final_output[i] /= 2`, integer truncate-toward-zero - same on both
// .NET and C compilers for signed int /2).

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// Batched ILGPU kernel that runs the VP9 forward DCT 8x8 across N
/// independent 8x8 blocks. Bit-exact mirror of
/// <see cref="Vp9ForwardDct8x8.Transform"/>.
/// </summary>
public sealed class Vp9ForwardDct8x8Kernel : IDisposable
{
    private const int CosPi4_64  = 16069;
    private const int CosPi8_64  = 15137;
    private const int CosPi12_64 = 13623;
    private const int CosPi16_64 = 11585;
    private const int CosPi20_64 = 9102;
    private const int CosPi24_64 = 6270;
    private const int CosPi28_64 = 3196;

    private const int DctConstBits = 14;
    private const int DctConstRounding = 1 << (DctConstBits - 1);

    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<short>, ArrayView<int>, int> _kernel;

    /// <summary>Compile the kernel onto <paramref name="accelerator"/>.</summary>
    public Vp9ForwardDct8x8Kernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<short>, ArrayView<int>, int>(FdctKernel);
    }

    /// <summary>
    /// Run the FDCT on <paramref name="blockCount"/> 8x8 blocks. Each
    /// block occupies 64 contiguous shorts in <paramref name="input"/>
    /// and 64 contiguous ints in <paramref name="output"/>.
    /// </summary>
    public void Run(ArrayView<short> input, ArrayView<int> output, int blockCount)
    {
        if (blockCount < 0) throw new ArgumentOutOfRangeException(nameof(blockCount));
        if (blockCount == 0) return;
        if (input.Length < blockCount * 64L)
            throw new ArgumentException($"input must hold at least blockCount*64 shorts (got {input.Length}).", nameof(input));
        if (output.Length < blockCount * 64L)
            throw new ArgumentException($"output must hold at least blockCount*64 ints (got {output.Length}).", nameof(output));
        _kernel(blockCount, input, output, blockCount);
    }

    /// <summary>
    /// Convenience: allocate temporary GPU buffers, run, copy back.
    /// Async because WebGPU forbids synchronous GPU-to-CPU copies.
    /// </summary>
    public async Task RunAsync(
        ReadOnlyMemory<short> input, Memory<int> output, int blockCount)
    {
        if (blockCount <= 0) return;
        using var dIn = _accelerator.Allocate1D<short>(blockCount * 64);
        using var dOut = _accelerator.Allocate1D<int>(blockCount * 64);
        dIn.View.CopyFromCPU(input.Span.ToArray());
        _kernel(blockCount, dIn.View, dOut.View, blockCount);
        await _accelerator.SynchronizeAsync();
        var readBack = await dOut.CopyToHostAsync();
        readBack.AsSpan(0, blockCount * 64).CopyTo(output.Span);
    }

    /// <summary>Kernel body. One thread per 8x8 block.</summary>
    private static void FdctKernel(
        Index1D blockIdx,
        ArrayView<short> input,
        ArrayView<int> output,
        int blockCount)
    {
        int idx = blockIdx;
        if (idx >= blockCount) return;

        long inBase = (long)idx * 64;
        long outBase = (long)idx * 64;

        // tmp[c*8 + r] holds pass-1 column-c output-r.
        var tmp = LocalMemory.Allocate<int>(64);

        // Pass 1: 8 column DCTs (input *= 4 inline).
        for (int col = 0; col < 8; col++)
        {
            int s0 = (input[inBase + col + 0 * 8] + input[inBase + col + 7 * 8]) * 4;
            int s1 = (input[inBase + col + 1 * 8] + input[inBase + col + 6 * 8]) * 4;
            int s2 = (input[inBase + col + 2 * 8] + input[inBase + col + 5 * 8]) * 4;
            int s3 = (input[inBase + col + 3 * 8] + input[inBase + col + 4 * 8]) * 4;
            int s4 = (input[inBase + col + 3 * 8] - input[inBase + col + 4 * 8]) * 4;
            int s5 = (input[inBase + col + 2 * 8] - input[inBase + col + 5 * 8]) * 4;
            int s6 = (input[inBase + col + 1 * 8] - input[inBase + col + 6 * 8]) * 4;
            int s7 = (input[inBase + col + 0 * 8] - input[inBase + col + 7 * 8]) * 4;

            ButterflyAndStore(s0, s1, s2, s3, s4, s5, s6, s7,
                out int o0, out int o1, out int o2, out int o3,
                out int o4, out int o5, out int o6, out int o7);

            tmp[col * 8 + 0] = o0;
            tmp[col * 8 + 1] = o1;
            tmp[col * 8 + 2] = o2;
            tmp[col * 8 + 3] = o3;
            tmp[col * 8 + 4] = o4;
            tmp[col * 8 + 5] = o5;
            tmp[col * 8 + 6] = o6;
            tmp[col * 8 + 7] = o7;
        }

        // Pass 2: 8 row DCTs (input from intermediate, no scaling).
        // Pass 2 col c reads intermediate[c + r*8] for r=0..7 (column-major
        // read into the transposed buffer). Output goes to output[c*8 + ...]
        // which after the final post-pass divide by 2 is the kernel result.
        for (int col = 0; col < 8; col++)
        {
            int s0 = tmp[col + 0 * 8] + tmp[col + 7 * 8];
            int s1 = tmp[col + 1 * 8] + tmp[col + 6 * 8];
            int s2 = tmp[col + 2 * 8] + tmp[col + 5 * 8];
            int s3 = tmp[col + 3 * 8] + tmp[col + 4 * 8];
            int s4 = tmp[col + 3 * 8] - tmp[col + 4 * 8];
            int s5 = tmp[col + 2 * 8] - tmp[col + 5 * 8];
            int s6 = tmp[col + 1 * 8] - tmp[col + 6 * 8];
            int s7 = tmp[col + 0 * 8] - tmp[col + 7 * 8];

            ButterflyAndStore(s0, s1, s2, s3, s4, s5, s6, s7,
                out int o0, out int o1, out int o2, out int o3,
                out int o4, out int o5, out int o6, out int o7);

            // Final post-pass: divide by 2 (truncate-toward-zero).
            output[outBase + col * 8 + 0] = o0 / 2;
            output[outBase + col * 8 + 1] = o1 / 2;
            output[outBase + col * 8 + 2] = o2 / 2;
            output[outBase + col * 8 + 3] = o3 / 2;
            output[outBase + col * 8 + 4] = o4 / 2;
            output[outBase + col * 8 + 5] = o5 / 2;
            output[outBase + col * 8 + 6] = o6 / 2;
            output[outBase + col * 8 + 7] = o7 / 2;
        }
    }

    /// <summary>
    /// 8-point 1D forward DCT butterfly. Mirrors libvpx
    /// <c>vpx_fdct8x8_c</c>'s inner butterfly. Bit-exact against
    /// <see cref="Vp9ForwardDct8x8"/>.
    /// </summary>
    private static void ButterflyAndStore(
        int s0, int s1, int s2, int s3, int s4, int s5, int s6, int s7,
        out int o0, out int o1, out int o2, out int o3,
        out int o4, out int o5, out int o6, out int o7)
    {
        // fdct4(s0..s3)
        int x0 = s0 + s3;
        int x1 = s1 + s2;
        int x2 = s1 - s2;
        int x3 = s0 - s3;
        long t0 = (long)(x0 + x1) * CosPi16_64;
        long t1 = (long)(x0 - x1) * CosPi16_64;
        long t2 = (long)x2 * CosPi24_64 + (long)x3 * CosPi8_64;
        long t3 = (long)(-x2) * CosPi8_64 + (long)x3 * CosPi24_64;
        o0 = RoundShift(t0);
        o2 = RoundShift(t2);
        o4 = RoundShift(t1);
        o6 = RoundShift(t3);

        // s4..s7 stages
        long u0 = (long)(s6 - s5) * CosPi16_64;
        long u1 = (long)(s6 + s5) * CosPi16_64;
        int v2 = RoundShift(u0);
        int v3 = RoundShift(u1);

        int y0 = s4 + v2;
        int y1 = s4 - v2;
        int y2 = s7 - v3;
        int y3 = s7 + v3;

        long w0 = (long)y0 * CosPi28_64 + (long)y3 * CosPi4_64;
        long w1 = (long)y1 * CosPi12_64 + (long)y2 * CosPi20_64;
        long w2 = (long)y2 * CosPi12_64 + (long)y1 * (-CosPi20_64);
        long w3 = (long)y3 * CosPi28_64 + (long)y0 * (-CosPi4_64);
        o1 = RoundShift(w0);
        o3 = RoundShift(w2);
        o5 = RoundShift(w1);
        o7 = RoundShift(w3);
    }

    private static int RoundShift(long input) =>
        (int)((input + DctConstRounding) >> DctConstBits);

    /// <summary>Release kernel resources. Does NOT dispose the accelerator.</summary>
    public void Dispose() { /* auto-grouped */ }
}
