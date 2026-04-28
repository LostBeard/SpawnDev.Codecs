// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// ILGPU kernel for VP8 inverse Walsh-Hadamard 4x4. Bit-exact mirror of
// Vp8InverseTransform.ShortInvWalsh4x4 (libvpx vp8_short_inv_walsh4x4_c).
// One thread per 4x4 block. Decodes the Y2 second-order block back into
// 16 Y4 DC values.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>
/// Batched ILGPU inverse Walsh-Hadamard 4x4 transform for VP8 Y2.
/// Bit-exact mirror of <see cref="Vp8InverseTransform.ShortInvWalsh4x4"/>.
/// </summary>
public sealed class Vp8InverseWalsh4x4Kernel : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<short>, ArrayView<short>, int> _kernel;

    /// <summary>Compile the kernel.</summary>
    public Vp8InverseWalsh4x4Kernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<short>, ArrayView<short>, int>(InvWalshKernel);
    }

    /// <summary>Run on N blocks. 16 shorts in, 16 shorts out per block.</summary>
    public void Run(ArrayView<short> input, ArrayView<short> output, int blockCount)
    {
        if (blockCount < 0) throw new ArgumentOutOfRangeException(nameof(blockCount));
        if (blockCount == 0) return;
        if (input.Length < blockCount * 16L)
            throw new ArgumentException("input must hold blockCount*16 shorts.", nameof(input));
        if (output.Length < blockCount * 16L)
            throw new ArgumentException("output must hold blockCount*16 shorts.", nameof(output));
        _kernel(blockCount, input, output, blockCount);
    }

    private static void InvWalshKernel(
        Index1D blockIdx,
        ArrayView<short> input,
        ArrayView<short> output,
        int blockCount)
    {
        int idx = blockIdx;
        if (idx >= blockCount) return;
        long inBase = (long)idx * 16;
        long outBase = (long)idx * 16;

        // Pull input into 16 short registers.
        short i00 = input[inBase + 0],  i01 = input[inBase + 1],
              i02 = input[inBase + 2],  i03 = input[inBase + 3];
        short i10 = input[inBase + 4],  i11 = input[inBase + 5],
              i12 = input[inBase + 6],  i13 = input[inBase + 7];
        short i20 = input[inBase + 8],  i21 = input[inBase + 9],
              i22 = input[inBase + 10], i23 = input[inBase + 11];
        short i30 = input[inBase + 12], i31 = input[inBase + 13],
              i32 = input[inBase + 14], i33 = input[inBase + 15];

        // Column pass (per i = 0..3 column): ip[i+0], ip[i+4], ip[i+8], ip[i+12]
        // libvpx pattern:
        //   a1 = ip[0] + ip[12]; b1 = ip[4] + ip[8]; c1 = ip[4] - ip[8]; d1 = ip[0] - ip[12];
        //   op[0] = a1 + b1; op[4] = c1 + d1; op[8] = a1 - b1; op[12] = d1 - c1;
        InvWalshCol(i00, i10, i20, i30, out short s00, out short s10, out short s20, out short s30);
        InvWalshCol(i01, i11, i21, i31, out short s01, out short s11, out short s21, out short s31);
        InvWalshCol(i02, i12, i22, i32, out short s02, out short s12, out short s22, out short s32);
        InvWalshCol(i03, i13, i23, i33, out short s03, out short s13, out short s23, out short s33);

        // Row pass with +3 round, >>3 shift.
        // libvpx pattern:
        //   a1 = op[0] + op[3]; b1 = op[1] + op[2]; c1 = op[1] - op[2]; d1 = op[0] - op[3];
        //   a2 = a1 + b1; b2 = c1 + d1; c2 = a1 - b1; d2 = d1 - c1;
        //   mbDqCoeff[0] = (a2+3)>>3; [1]=(b2+3)>>3; [2]=(c2+3)>>3; [3]=(d2+3)>>3;
        InvWalshRowFinal(s00, s01, s02, s03, output, outBase + 0);
        InvWalshRowFinal(s10, s11, s12, s13, output, outBase + 4);
        InvWalshRowFinal(s20, s21, s22, s23, output, outBase + 8);
        InvWalshRowFinal(s30, s31, s32, s33, output, outBase + 12);
    }

    private static void InvWalshCol(
        short i0, short i1, short i2, short i3,
        out short o0, out short o1, out short o2, out short o3)
    {
        int a1 = i0 + i3;
        int b1 = i1 + i2;
        int c1 = i1 - i2;
        int d1 = i0 - i3;
        o0 = (short)(a1 + b1);
        o1 = (short)(c1 + d1);
        o2 = (short)(a1 - b1);
        o3 = (short)(d1 - c1);
    }

    private static void InvWalshRowFinal(
        short s0, short s1, short s2, short s3,
        ArrayView<short> output, long outBase)
    {
        int a1 = s0 + s3;
        int b1 = s1 + s2;
        int c1 = s1 - s2;
        int d1 = s0 - s3;
        int a2 = a1 + b1;
        int b2 = c1 + d1;
        int c2 = a1 - b1;
        int d2 = d1 - c1;
        output[outBase + 0] = (short)((a2 + 3) >> 3);
        output[outBase + 1] = (short)((b2 + 3) >> 3);
        output[outBase + 2] = (short)((c2 + 3) >> 3);
        output[outBase + 3] = (short)((d2 + 3) >> 3);
    }

    /// <summary>Release kernel resources. Does NOT dispose the accelerator.</summary>
    public void Dispose() { /* auto-grouped */ }
}
