// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// ILGPU kernel for the VP8 forward DCT 4x4. Runs the same normative
// integer butterfly as Vp8ForwardTransform.ShortFdct4x4 on every ILGPU
// backend - CPU emulator, CUDA, OpenCL, WebGPU, WebGL, Wasm. Batched:
// one thread per 4x4 block, N blocks in parallel.
//
// VP8 is a normative bitstream so the kernel must produce bit-identical
// output to the reference function across every backend. Tests assert
// this directly via Vp8ForwardDct4x4KernelTests (cross-backend).
//
// One macroblock contains 25 4x4 transforms (16 Y4 + 1 Y2 Walsh + 4 U
// + 4 V). At FullHD that's 8160 macroblocks * 25 transforms = ~204k
// transforms per frame - embarrassingly parallel.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>
/// Batched ILGPU kernel that runs the VP8 forward DCT 4x4 across N
/// independent 4x4 blocks in parallel. Bit-exact mirror of
/// <see cref="Vp8ForwardTransform.ShortFdct4x4"/> (the reference C# CPU
/// implementation ported from libvpx <c>vp8_short_fdct4x4_c</c>).
/// </summary>
public sealed class Vp8ForwardDct4x4Kernel : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<short>, ArrayView<short>, int> _kernel;

    /// <summary>Compile the kernel onto <paramref name="accelerator"/>.</summary>
    public Vp8ForwardDct4x4Kernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<short>, ArrayView<short>, int>(FdctKernel);
    }

    /// <summary>
    /// Run the FDCT on <paramref name="blockCount"/> blocks. Each block
    /// occupies 16 contiguous shorts in <paramref name="input"/> and
    /// <paramref name="output"/>. Both views must hold at least
    /// <c>blockCount * 16</c> shorts.
    /// </summary>
    public void Run(ArrayView<short> input, ArrayView<short> output, int blockCount)
    {
        if (blockCount < 0) throw new ArgumentOutOfRangeException(nameof(blockCount));
        if (blockCount == 0) return;
        if (input.Length < blockCount * 16L)
            throw new ArgumentException($"input must hold at least blockCount*16 shorts (got {input.Length}).", nameof(input));
        if (output.Length < blockCount * 16L)
            throw new ArgumentException($"output must hold at least blockCount*16 shorts (got {output.Length}).", nameof(output));
        _kernel(blockCount, input, output, blockCount);
    }

    /// <summary>
    /// Convenience: allocate temporary GPU buffers, run, copy back.
    /// Async because WebGPU forbids synchronous GPU-to-CPU copies.
    /// </summary>
    public async Task RunAsync(
        ReadOnlyMemory<short> input, Memory<short> output, int blockCount)
    {
        if (blockCount <= 0) return;
        using var dIn = _accelerator.Allocate1D<short>(blockCount * 16);
        using var dOut = _accelerator.Allocate1D<short>(blockCount * 16);
        dIn.View.CopyFromCPU(input.Span.ToArray());
        _kernel(blockCount, dIn.View, dOut.View, blockCount);
        await _accelerator.SynchronizeAsync();
        var readBack = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dOut);
        readBack.AsSpan(0, blockCount * 16).CopyTo(output.Span);
    }

    /// <summary>Kernel body. One thread per 4x4 block.</summary>
    private static void FdctKernel(
        Index1D blockIdx,
        ArrayView<short> input,
        ArrayView<short> output,
        int blockCount)
    {
        int idx = blockIdx;
        if (idx >= blockCount) return;
        long inBase = (long)idx * 16;
        long outBase = (long)idx * 16;

        // Pull 16 input samples into registers.
        short i00 = input[inBase + 0],  i01 = input[inBase + 1],
              i02 = input[inBase + 2],  i03 = input[inBase + 3];
        short i10 = input[inBase + 4],  i11 = input[inBase + 5],
              i12 = input[inBase + 6],  i13 = input[inBase + 7];
        short i20 = input[inBase + 8],  i21 = input[inBase + 9],
              i22 = input[inBase + 10], i23 = input[inBase + 11];
        short i30 = input[inBase + 12], i31 = input[inBase + 13],
              i32 = input[inBase + 14], i33 = input[inBase + 15];

        // Pass 1: rows -> stage1 (16 vars).
        FdctRow(i00, i01, i02, i03, out short s00, out short s01, out short s02, out short s03);
        FdctRow(i10, i11, i12, i13, out short s10, out short s11, out short s12, out short s13);
        FdctRow(i20, i21, i22, i23, out short s20, out short s21, out short s22, out short s23);
        FdctRow(i30, i31, i32, i33, out short s30, out short s31, out short s32, out short s33);

        // Pass 2: columns -> output. Different rounding constants per
        // libvpx vp8_short_fdct4x4_c; see Vp8ForwardTransform.ShortFdct4x4.
        // Column 0
        FdctCol(s00, s10, s20, s30, out short o00, out short o10, out short o20, out short o30);
        output[outBase + 0]  = o00;
        output[outBase + 4]  = o10;
        output[outBase + 8]  = o20;
        output[outBase + 12] = o30;

        // Column 1
        FdctCol(s01, s11, s21, s31, out short o01, out short o11, out short o21, out short o31);
        output[outBase + 1]  = o01;
        output[outBase + 5]  = o11;
        output[outBase + 9]  = o21;
        output[outBase + 13] = o31;

        // Column 2
        FdctCol(s02, s12, s22, s32, out short o02, out short o12, out short o22, out short o32);
        output[outBase + 2]  = o02;
        output[outBase + 6]  = o12;
        output[outBase + 10] = o22;
        output[outBase + 14] = o32;

        // Column 3
        FdctCol(s03, s13, s23, s33, out short o03, out short o13, out short o23, out short o33);
        output[outBase + 3]  = o03;
        output[outBase + 7]  = o13;
        output[outBase + 11] = o23;
        output[outBase + 15] = o33;
    }

    /// <summary>4-point row pass of the VP8 forward DCT. Bit-exact to libvpx.</summary>
    private static void FdctRow(
        short s0, short s1, short s2, short s3,
        out short t0, out short t1, out short t2, out short t3)
    {
        int a1 = (s0 + s3) * 8;
        int b1 = (s1 + s2) * 8;
        int c1 = (s1 - s2) * 8;
        int d1 = (s0 - s3) * 8;
        t0 = (short)(a1 + b1);
        t2 = (short)(a1 - b1);
        t1 = (short)((c1 * 2217 + d1 * 5352 + 14500) >> 12);
        t3 = (short)((d1 * 2217 - c1 * 5352 + 7500) >> 12);
    }

    /// <summary>4-point column pass of the VP8 forward DCT. Bit-exact.</summary>
    private static void FdctCol(
        short s0, short s1, short s2, short s3,
        out short t0, out short t1, out short t2, out short t3)
    {
        int a1 = s0 + s3;
        int b1 = s1 + s2;
        int c1 = s1 - s2;
        int d1 = s0 - s3;
        t0 = (short)((a1 + b1 + 7) >> 4);
        t2 = (short)((a1 - b1 + 7) >> 4);
        // libvpx: ((c*2217 + d*5352 + 12000) >> 16) + (d != 0)
        t1 = (short)(((c1 * 2217 + d1 * 5352 + 12000) >> 16) + (d1 != 0 ? 1 : 0));
        t3 = (short)((d1 * 2217 - c1 * 5352 + 51000) >> 16);
    }

    /// <summary>Release kernel resources. Does NOT dispose the accelerator.</summary>
    public void Dispose() { /* auto-grouped kernels don't need explicit disposal */ }
}
