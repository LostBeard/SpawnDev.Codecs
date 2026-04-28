// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// ILGPU kernel for the VP9 forward DCT 4x4. Bit-exact mirror of
// Vp9ForwardDct4x4.Transform (the libvpx vpx_fdct4x4_c port). Runs on
// every ILGPU backend - CPU emulator, CUDA, OpenCL, WebGPU, WebGL, Wasm.
// Batched: one thread per 4x4 block, N blocks in parallel.
//
// VP9 is a normative bitstream so the kernel must produce bit-identical
// output to the reference function across every backend. Tests assert
// this directly via Vp9ForwardDct4x4KernelTests (cross-backend).
//
// Per-thread layout: 16 input samples (short) + 16 intermediate ints (in
// registers, no LocalMemory). Pass 1 multiplies by 16 with a +1 bias on
// the (0,0) DC slot when non-zero (libvpx-specific rounding bias). Pass 2
// reads from intermediate and applies the final post-pass `(x + 1) >> 2`.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// Batched ILGPU kernel that runs the VP9 forward DCT 4x4 across N
/// independent 4x4 blocks in parallel. Bit-exact mirror of
/// <see cref="Vp9ForwardDct4x4.Transform"/>.
/// </summary>
public sealed class Vp9ForwardDct4x4Kernel : IDisposable
{
    // Q14 cosine constants per VP9 spec sec 8.7.1.2. Must match Reference.
    private const int CosPi8_64  = 15137;
    private const int CosPi16_64 = 11585;
    private const int CosPi24_64 = 6270;

    private const int DctConstBits = 14;
    private const int DctConstRounding = 1 << (DctConstBits - 1);

    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<short>, ArrayView<int>, int> _kernel;

    /// <summary>Compile the kernel onto <paramref name="accelerator"/>.</summary>
    public Vp9ForwardDct4x4Kernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<short>, ArrayView<int>, int>(FdctKernel);
    }

    /// <summary>
    /// Run the FDCT on <paramref name="blockCount"/> blocks. Each block
    /// occupies 16 contiguous shorts in <paramref name="input"/> and
    /// 16 contiguous ints in <paramref name="output"/>.
    /// </summary>
    public void Run(ArrayView<short> input, ArrayView<int> output, int blockCount)
    {
        if (blockCount < 0) throw new ArgumentOutOfRangeException(nameof(blockCount));
        if (blockCount == 0) return;
        if (input.Length < blockCount * 16L)
            throw new ArgumentException($"input must hold at least blockCount*16 shorts (got {input.Length}).", nameof(input));
        if (output.Length < blockCount * 16L)
            throw new ArgumentException($"output must hold at least blockCount*16 ints (got {output.Length}).", nameof(output));
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
        using var dIn = _accelerator.Allocate1D<short>(blockCount * 16);
        using var dOut = _accelerator.Allocate1D<int>(blockCount * 16);
        dIn.View.CopyFromCPU(input.Span.ToArray());
        _kernel(blockCount, dIn.View, dOut.View, blockCount);
        await _accelerator.SynchronizeAsync();
        var readBack = await dOut.CopyToHostAsync();
        readBack.AsSpan(0, blockCount * 16).CopyTo(output.Span);
    }

    /// <summary>Kernel body. One thread per 4x4 block.</summary>
    private static void FdctKernel(
        Index1D blockIdx,
        ArrayView<short> input,
        ArrayView<int> output,
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

        // Pass 1: column DCT (input *= 16 with +1 bias on DC slot).
        // libvpx pass 1 reads `input[inputOffset + r*rowStride]` for r=0..3
        // with inputOffset incrementing by 1 per col. Result stored in the
        // intermediate buffer at position outOffset+0/1/2/3 (raster row,
        // column = i). After pass 1, intermediate[r*4+c] holds the
        // transposed DCT output (result[r=output_row][c=column_index]).
        //
        // For the kernel: column 0 -> i00, i10, i20, i30; column 1 -> i01,
        // i11, i21, i31; etc. The (0,0) DC slot is the c=0 case; libvpx
        // applies the +1 bias to inHigh[0] (= input[col=0, row=0]) only.

        // Column 0 (input *= 16; +1 bias on DC slot when non-zero per libvpx)
        FdctRow(
            i00 * 16, i10 * 16, i20 * 16, i30 * 16, addOneIfNonZero: true,
            out int t00_p1, out int t01_p1, out int t02_p1, out int t03_p1);
        // Column 1 (input *= 16)
        FdctRow(
            i01 * 16, i11 * 16, i21 * 16, i31 * 16, addOneIfNonZero: false,
            out int t10_p1, out int t11_p1, out int t12_p1, out int t13_p1);
        // Column 2 (input *= 16)
        FdctRow(
            i02 * 16, i12 * 16, i22 * 16, i32 * 16, addOneIfNonZero: false,
            out int t20_p1, out int t21_p1, out int t22_p1, out int t23_p1);
        // Column 3 (input *= 16)
        FdctRow(
            i03 * 16, i13 * 16, i23 * 16, i33 * 16, addOneIfNonZero: false,
            out int t30_p1, out int t31_p1, out int t32_p1, out int t33_p1);

        // intermediate layout in libvpx: intermediate[outOffset + 0..3]
        //   for column c, stores 4 results at positions [c*4..c*4+3].
        // Pass 2 reads `intermediate[i + r*4]` to traverse one column of
        // intermediate, which corresponds to the output position r of
        // pass 1 for column index i. So:
        //   intermediate[i + 0*4] = pass-1-col[i].out0  -> [c0_o0,c1_o0,c2_o0,c3_o0]
        //   intermediate[i + 1*4] = pass-1-col[i].out1  -> [c0_o1,c1_o1,c2_o1,c3_o1]
        //   intermediate[i + 2*4] = pass-1-col[i].out2
        //   intermediate[i + 3*4] = pass-1-col[i].out3
        //
        // Encode as named locals: row r of pass 2 reads
        //   intermediate[r + 0*4] = pass1_col0_or
        //   intermediate[r + 1*4] = pass1_col1_or
        //   intermediate[r + 2*4] = pass1_col2_or
        //   intermediate[r + 3*4] = pass1_col3_or
        // i.e. the row r of pass 2 takes the r-th output of every column.

        // Pass 2 row 0: reads outputs[0] from columns 0,1,2,3
        FdctRow(
            t00_p1, t10_p1, t20_p1, t30_p1, addOneIfNonZero: false,
            out int r0_o0, out int r0_o1, out int r0_o2, out int r0_o3);
        // Pass 2 row 1: reads outputs[1] from columns 0,1,2,3
        FdctRow(
            t01_p1, t11_p1, t21_p1, t31_p1, addOneIfNonZero: false,
            out int r1_o0, out int r1_o1, out int r1_o2, out int r1_o3);
        // Pass 2 row 2: reads outputs[2] from columns 0,1,2,3
        FdctRow(
            t02_p1, t12_p1, t22_p1, t32_p1, addOneIfNonZero: false,
            out int r2_o0, out int r2_o1, out int r2_o2, out int r2_o3);
        // Pass 2 row 3: reads outputs[3] from columns 0,1,2,3
        FdctRow(
            t03_p1, t13_p1, t23_p1, t33_p1, addOneIfNonZero: false,
            out int r3_o0, out int r3_o1, out int r3_o2, out int r3_o3);

        // Final post-pass: (x + 1) >> 2.
        output[outBase + 0]  = (r0_o0 + 1) >> 2;
        output[outBase + 1]  = (r0_o1 + 1) >> 2;
        output[outBase + 2]  = (r0_o2 + 1) >> 2;
        output[outBase + 3]  = (r0_o3 + 1) >> 2;
        output[outBase + 4]  = (r1_o0 + 1) >> 2;
        output[outBase + 5]  = (r1_o1 + 1) >> 2;
        output[outBase + 6]  = (r1_o2 + 1) >> 2;
        output[outBase + 7]  = (r1_o3 + 1) >> 2;
        output[outBase + 8]  = (r2_o0 + 1) >> 2;
        output[outBase + 9]  = (r2_o1 + 1) >> 2;
        output[outBase + 10] = (r2_o2 + 1) >> 2;
        output[outBase + 11] = (r2_o3 + 1) >> 2;
        output[outBase + 12] = (r3_o0 + 1) >> 2;
        output[outBase + 13] = (r3_o1 + 1) >> 2;
        output[outBase + 14] = (r3_o2 + 1) >> 2;
        output[outBase + 15] = (r3_o3 + 1) >> 2;
    }

    /// <summary>
    /// 4-point pass of the VP9 forward DCT. Caller pre-multiplies pass-1
    /// inputs by 16 before invoking. <paramref name="addOneIfNonZero"/>
    /// implements libvpx's pass-1 DC-slot rounding bias - set only for
    /// the column-0 of pass 1 (libvpx `if (i == 0 &amp;&amp; inHigh[0] != 0)
    /// inHigh[0]++`).
    /// </summary>
    private static void FdctRow(
        int s0, int s1, int s2, int s3, bool addOneIfNonZero,
        out int o0, out int o1, out int o2, out int o3)
    {
        if (addOneIfNonZero && s0 != 0) s0++;

        int x0 = s0 + s3;
        int x1 = s1 + s2;
        int x2 = s1 - s2;
        int x3 = s0 - s3;

        o0 = (int)(((long)(x0 + x1) * CosPi16_64 + DctConstRounding) >> DctConstBits);
        o2 = (int)(((long)(x0 - x1) * CosPi16_64 + DctConstRounding) >> DctConstBits);
        o1 = (int)(((long)x2 * CosPi24_64 + (long)x3 * CosPi8_64 + DctConstRounding) >> DctConstBits);
        o3 = (int)(((long)(-x2) * CosPi8_64 + (long)x3 * CosPi24_64 + DctConstRounding) >> DctConstBits);
    }

    /// <summary>Release kernel resources. Does NOT dispose the accelerator.</summary>
    public void Dispose() { /* auto-grouped kernels don't need explicit disposal */ }
}
