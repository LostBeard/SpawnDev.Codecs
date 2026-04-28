// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// ILGPU kernel for the VP8 forward Walsh-Hadamard 4x4 (Y2 second-order
// transform). Bit-exact mirror of Vp8ForwardTransform.ShortWalsh4x4
// (libvpx vp8_short_walsh4x4_c port). Batched: one thread per 4x4
// block, N blocks in parallel.
//
// One Walsh per macroblock (the Y2 block holds the 16 Y4 DCs); at FullHD
// that's 8160 Walsh transforms per frame.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>
/// Batched ILGPU kernel for the VP8 forward Walsh-Hadamard 4x4 transform.
/// Bit-exact mirror of <see cref="Vp8ForwardTransform.ShortWalsh4x4"/>.
/// </summary>
public sealed class Vp8ForwardWalsh4x4Kernel : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<short>, ArrayView<short>, int> _kernel;

    /// <summary>Compile the kernel onto <paramref name="accelerator"/>.</summary>
    public Vp8ForwardWalsh4x4Kernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<short>, ArrayView<short>, int>(WalshKernel);
    }

    /// <summary>
    /// Run the Walsh forward transform on <paramref name="blockCount"/>
    /// blocks. Each block is 16 contiguous shorts (input + output).
    /// </summary>
    public void Run(ArrayView<short> input, ArrayView<short> output, int blockCount)
    {
        if (blockCount < 0) throw new ArgumentOutOfRangeException(nameof(blockCount));
        if (blockCount == 0) return;
        if (input.Length < blockCount * 16L)
            throw new ArgumentException($"input must hold blockCount*16 shorts.", nameof(input));
        if (output.Length < blockCount * 16L)
            throw new ArgumentException($"output must hold blockCount*16 shorts.", nameof(output));
        _kernel(blockCount, input, output, blockCount);
    }

    /// <summary>Kernel body. One thread per 4x4 block.</summary>
    private static void WalshKernel(
        Index1D blockIdx,
        ArrayView<short> input,
        ArrayView<short> output,
        int blockCount)
    {
        int idx = blockIdx;
        if (idx >= blockCount) return;
        long inBase = (long)idx * 16;
        long outBase = (long)idx * 16;

        // Pull 16 input samples.
        short i00 = input[inBase + 0],  i01 = input[inBase + 1],
              i02 = input[inBase + 2],  i03 = input[inBase + 3];
        short i10 = input[inBase + 4],  i11 = input[inBase + 5],
              i12 = input[inBase + 6],  i13 = input[inBase + 7];
        short i20 = input[inBase + 8],  i21 = input[inBase + 9],
              i22 = input[inBase + 10], i23 = input[inBase + 11];
        short i30 = input[inBase + 12], i31 = input[inBase + 13],
              i32 = input[inBase + 14], i33 = input[inBase + 15];

        // Pass 1: rows -> stage1.
        WalshRow(i00, i01, i02, i03, out short s00, out short s01, out short s02, out short s03);
        WalshRow(i10, i11, i12, i13, out short s10, out short s11, out short s12, out short s13);
        WalshRow(i20, i21, i22, i23, out short s20, out short s21, out short s22, out short s23);
        WalshRow(i30, i31, i32, i33, out short s30, out short s31, out short s32, out short s33);

        // Pass 2: columns. libvpx pattern:
        //   a1 = stage1[i+0] + stage1[i+8]
        //   d1 = stage1[i+4] + stage1[i+12]
        //   c1 = stage1[i+4] - stage1[i+12]
        //   b1 = stage1[i+0] - stage1[i+8]
        //   a2/b2/c2/d2 = a1+d1, b1+c1, b1-c1, a1-d1
        //   apply (val += val < 0 ? 1 : 0) round-to-zero adjustment
        //   output[i + N] = (val + 3) >> 3 with N = 0/4/8/12
        // Column 0
        WalshCol(s00, s10, s20, s30, out short o00, out short o10, out short o20, out short o30);
        output[outBase + 0]  = o00;
        output[outBase + 4]  = o10;
        output[outBase + 8]  = o20;
        output[outBase + 12] = o30;

        // Column 1
        WalshCol(s01, s11, s21, s31, out short o01, out short o11, out short o21, out short o31);
        output[outBase + 1]  = o01;
        output[outBase + 5]  = o11;
        output[outBase + 9]  = o21;
        output[outBase + 13] = o31;

        // Column 2
        WalshCol(s02, s12, s22, s32, out short o02, out short o12, out short o22, out short o32);
        output[outBase + 2]  = o02;
        output[outBase + 6]  = o12;
        output[outBase + 10] = o22;
        output[outBase + 14] = o32;

        // Column 3
        WalshCol(s03, s13, s23, s33, out short o03, out short o13, out short o23, out short o33);
        output[outBase + 3]  = o03;
        output[outBase + 7]  = o13;
        output[outBase + 11] = o23;
        output[outBase + 15] = o33;
    }

    /// <summary>Row pass of the Walsh transform.</summary>
    private static void WalshRow(
        short s0, short s1, short s2, short s3,
        out short t0, out short t1, out short t2, out short t3)
    {
        // libvpx layout: a1 uses [0,2], d1 uses [1,3], c1 uses [1,3] diff, b1 uses [0,2] diff.
        int a1 = (s0 + s2) * 4;
        int d1 = (s1 + s3) * 4;
        int c1 = (s1 - s3) * 4;
        int b1 = (s0 - s2) * 4;
        t0 = (short)(a1 + d1 + (a1 != 0 ? 1 : 0));
        t1 = (short)(b1 + c1);
        t2 = (short)(b1 - c1);
        t3 = (short)(a1 - d1);
    }

    /// <summary>Column pass of the Walsh transform.</summary>
    private static void WalshCol(
        short s0, short s1, short s2, short s3,
        out short t0, out short t1, out short t2, out short t3)
    {
        int a1 = s0 + s2;
        int d1 = s1 + s3;
        int c1 = s1 - s3;
        int b1 = s0 - s2;
        int a2 = a1 + d1;
        int b2 = b1 + c1;
        int c2 = b1 - c1;
        int d2 = a1 - d1;
        a2 += a2 < 0 ? 1 : 0;
        b2 += b2 < 0 ? 1 : 0;
        c2 += c2 < 0 ? 1 : 0;
        d2 += d2 < 0 ? 1 : 0;
        t0 = (short)((a2 + 3) >> 3);
        t1 = (short)((b2 + 3) >> 3);
        t2 = (short)((c2 + 3) >> 3);
        t3 = (short)((d2 + 3) >> 3);
    }

    /// <summary>Release kernel resources. Does NOT dispose the accelerator.</summary>
    public void Dispose() { /* auto-grouped kernels don't need explicit disposal */ }
}
