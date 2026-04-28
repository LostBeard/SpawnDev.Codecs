// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// ILGPU kernel for the VP9 forward DCT 32x32. Bit-exact mirror of
// Vp9ForwardDct32x32.Transform (the libvpx vpx_fdct32x32_c port). Batched:
// one thread per 32x32 block.
//
// VP9 32x32 always uses DCT_DCT - per spec sec 8.7, ADST is not defined
// for 32x32. So the kernel only needs the DCT path.
//
// Heavy kernel. 1024 longs (8KB) intermediate + per-pass tempIn/tempOut
// (256 + 256 bytes) + Fdct32 step scratch (256 bytes). Total per-thread
// LocalMemory ~9KB. WebGL is expected to gate out at the runner level
// (varying-vector limits) like the iDCT/iADST 16x16 kernels do; the
// other 5 backends (CPU, CUDA, OpenCL, WebGPU, Wasm) handle the 9KB of
// per-thread memory cleanly.
//
// libvpx vpx_fdct32x32_c is itself a C reference - the kernel mirrors
// it staircase by staircase rather than try to fuse the two passes,
// keeping the code readable and trivially auditable against the spec.

using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>Batched ILGPU kernel for the VP9 forward DCT 32x32 (DCT_DCT only).</summary>
public sealed class Vp9ForwardDct32x32Kernel : IDisposable
{
    private const int CosPi1_64  = 16364;
    private const int CosPi2_64  = 16305;
    private const int CosPi3_64  = 16207;
    private const int CosPi4_64  = 16069;
    private const int CosPi5_64  = 15893;
    private const int CosPi6_64  = 15679;
    private const int CosPi7_64  = 15426;
    private const int CosPi8_64  = 15137;
    private const int CosPi9_64  = 14811;
    private const int CosPi10_64 = 14449;
    private const int CosPi11_64 = 14053;
    private const int CosPi12_64 = 13623;
    private const int CosPi13_64 = 13160;
    private const int CosPi14_64 = 12665;
    private const int CosPi15_64 = 12140;
    private const int CosPi16_64 = 11585;
    private const int CosPi17_64 = 11003;
    private const int CosPi18_64 = 10394;
    private const int CosPi19_64 = 9760;
    private const int CosPi20_64 = 9102;
    private const int CosPi21_64 = 8423;
    private const int CosPi22_64 = 7723;
    private const int CosPi23_64 = 7005;
    private const int CosPi24_64 = 6270;
    private const int CosPi25_64 = 5520;
    private const int CosPi26_64 = 4756;
    private const int CosPi27_64 = 3981;
    private const int CosPi28_64 = 3196;
    private const int CosPi29_64 = 2404;
    private const int CosPi30_64 = 1606;
    private const int CosPi31_64 = 804;

    private const int DctConstBits = 14;
    private const int DctConstRounding = 1 << (DctConstBits - 1);

    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<short>, ArrayView<int>, int> _kernel;

    /// <summary>Compile the kernel onto <paramref name="accelerator"/>.</summary>
    public Vp9ForwardDct32x32Kernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<short>, ArrayView<int>, int>(FdctKernel);
    }

    /// <summary>
    /// Run the FDCT on <paramref name="blockCount"/> 32x32 blocks. Each
    /// block: 1024 contiguous shorts in / 1024 contiguous ints out.
    /// </summary>
    public void Run(ArrayView<short> input, ArrayView<int> output, int blockCount)
    {
        if (blockCount < 0) throw new ArgumentOutOfRangeException(nameof(blockCount));
        if (blockCount == 0) return;
        if (input.Length < blockCount * 1024L)
            throw new ArgumentException($"input must hold at least blockCount*1024 shorts.", nameof(input));
        if (output.Length < blockCount * 1024L)
            throw new ArgumentException($"output must hold at least blockCount*1024 ints.", nameof(output));
        _kernel(blockCount, input, output, blockCount);
    }

    /// <summary>
    /// Convenience: allocate temporary GPU buffers, run, copy back.
    /// </summary>
    public async Task RunAsync(
        ReadOnlyMemory<short> input, Memory<int> output, int blockCount)
    {
        if (blockCount <= 0) return;
        using var dIn = _accelerator.Allocate1D<short>(blockCount * 1024);
        using var dOut = _accelerator.Allocate1D<int>(blockCount * 1024);
        dIn.View.CopyFromCPU(input.Span.ToArray());
        _kernel(blockCount, dIn.View, dOut.View, blockCount);
        await _accelerator.SynchronizeAsync();
        var readBack = await dOut.CopyToHostAsync();
        readBack.AsSpan(0, blockCount * 1024).CopyTo(output.Span);
    }

    private static void FdctKernel(
        Index1D blockIdx,
        ArrayView<short> input,
        ArrayView<int> output,
        int blockCount)
    {
        int idx = blockIdx;
        if (idx >= blockCount) return;

        long inBase = (long)idx * 1024;
        long outBase = (long)idx * 1024;

        // Per-thread scratch.
        //   intermediate[1024] - between pass 1 and pass 2.
        //   tempIn[32]         - input vector for current Fdct32 call.
        //   tempOut[32]        - output vector from current Fdct32 call.
        //   step[32]           - working scratch inside Fdct32.
        var intermediate = LocalMemory.Allocate<long>(1024);
        var tempIn = LocalMemory.Allocate<long>(32);
        var tempOut = LocalMemory.Allocate<long>(32);
        var step = LocalMemory.Allocate<long>(32);

        // Pass 1 (columns): input *= 4, Fdct32 round=false, then
        // PositiveBiasShift on each output.
        for (int i = 0; i < 32; i++)
        {
            for (int j = 0; j < 32; j++) tempIn[j] = (long)input[inBase + j * 32 + i] * 4L;
            Fdct32(tempIn, tempOut, step);
            for (int j = 0; j < 32; j++)
                intermediate[j * 32 + i] = PositiveBiasShift(tempOut[j]);
        }

        // Pass 2 (rows): Fdct32 round=false, then HalfRoundShift cast to int.
        for (int i = 0; i < 32; i++)
        {
            for (int j = 0; j < 32; j++) tempIn[j] = intermediate[j + i * 32];
            Fdct32(tempIn, tempOut, step);
            for (int j = 0; j < 32; j++)
                output[outBase + j + i * 32] = (int)HalfRoundShift(tempOut[j]);
        }
    }

    /// <summary>libvpx <c>half_round_shift</c>: <c>(input + 1 + (input&lt;0)) &gt;&gt; 2</c>.</summary>
    private static long HalfRoundShift(long input) =>
        (input + 1 + (input < 0 ? 1 : 0)) >> 2;

    /// <summary>libvpx between-pass shift: <c>(input + 1 + (input&gt;0)) &gt;&gt; 2</c>.</summary>
    private static long PositiveBiasShift(long input) =>
        (input + 1 + (input > 0 ? 1 : 0)) >> 2;

    /// <summary>libvpx <c>dct_32_round</c>: rounded right shift by 14.</summary>
    private static long DctRound(long input) =>
        (input + DctConstRounding) >> DctConstBits;

    /// <summary>
    /// 1D 32-point forward DCT. Mirrors libvpx <c>vpx_fdct32</c>.
    /// 7 stages with cospi multiplications + bit-reversed final permutation.
    /// </summary>
    /// <remarks>
    /// NoInlining keeps WGSL shader size sane - same reasoning as
    /// Vp9Idct16x16Kernel.Idct16Row. Without it the 7-stage 32-point
    /// butterfly inlines at every call site (32 row passes + 32 column
    /// passes per block) and explodes the shader.
    /// </remarks>
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void Fdct32(ArrayView<long> input, ArrayView<long> output, ArrayView<long> step)
    {
        // Stage 1
        step[0]  = input[0]  + input[31];
        step[1]  = input[1]  + input[30];
        step[2]  = input[2]  + input[29];
        step[3]  = input[3]  + input[28];
        step[4]  = input[4]  + input[27];
        step[5]  = input[5]  + input[26];
        step[6]  = input[6]  + input[25];
        step[7]  = input[7]  + input[24];
        step[8]  = input[8]  + input[23];
        step[9]  = input[9]  + input[22];
        step[10] = input[10] + input[21];
        step[11] = input[11] + input[20];
        step[12] = input[12] + input[19];
        step[13] = input[13] + input[18];
        step[14] = input[14] + input[17];
        step[15] = input[15] + input[16];
        step[16] = -input[16] + input[15];
        step[17] = -input[17] + input[14];
        step[18] = -input[18] + input[13];
        step[19] = -input[19] + input[12];
        step[20] = -input[20] + input[11];
        step[21] = -input[21] + input[10];
        step[22] = -input[22] + input[9];
        step[23] = -input[23] + input[8];
        step[24] = -input[24] + input[7];
        step[25] = -input[25] + input[6];
        step[26] = -input[26] + input[5];
        step[27] = -input[27] + input[4];
        step[28] = -input[28] + input[3];
        step[29] = -input[29] + input[2];
        step[30] = -input[30] + input[1];
        step[31] = -input[31] + input[0];

        // Stage 2
        output[0]  = step[0]  + step[15];
        output[1]  = step[1]  + step[14];
        output[2]  = step[2]  + step[13];
        output[3]  = step[3]  + step[12];
        output[4]  = step[4]  + step[11];
        output[5]  = step[5]  + step[10];
        output[6]  = step[6]  + step[9];
        output[7]  = step[7]  + step[8];
        output[8]  = -step[8]  + step[7];
        output[9]  = -step[9]  + step[6];
        output[10] = -step[10] + step[5];
        output[11] = -step[11] + step[4];
        output[12] = -step[12] + step[3];
        output[13] = -step[13] + step[2];
        output[14] = -step[14] + step[1];
        output[15] = -step[15] + step[0];

        output[16] = step[16];
        output[17] = step[17];
        output[18] = step[18];
        output[19] = step[19];

        long c16 = CosPi16_64;
        output[20] = DctRound((-step[20] + step[27]) * c16);
        output[21] = DctRound((-step[21] + step[26]) * c16);
        output[22] = DctRound((-step[22] + step[25]) * c16);
        output[23] = DctRound((-step[23] + step[24]) * c16);
        output[24] = DctRound((step[24] + step[23]) * c16);
        output[25] = DctRound((step[25] + step[22]) * c16);
        output[26] = DctRound((step[26] + step[21]) * c16);
        output[27] = DctRound((step[27] + step[20]) * c16);

        output[28] = step[28];
        output[29] = step[29];
        output[30] = step[30];
        output[31] = step[31];

        // Stage 3
        step[0] = output[0] + output[7];
        step[1] = output[1] + output[6];
        step[2] = output[2] + output[5];
        step[3] = output[3] + output[4];
        step[4] = -output[4] + output[3];
        step[5] = -output[5] + output[2];
        step[6] = -output[6] + output[1];
        step[7] = -output[7] + output[0];
        step[8] = output[8];
        step[9] = output[9];
        step[10] = DctRound((-output[10] + output[13]) * c16);
        step[11] = DctRound((-output[11] + output[12]) * c16);
        step[12] = DctRound((output[12] + output[11]) * c16);
        step[13] = DctRound((output[13] + output[10]) * c16);
        step[14] = output[14];
        step[15] = output[15];

        step[16] = output[16] + output[23];
        step[17] = output[17] + output[22];
        step[18] = output[18] + output[21];
        step[19] = output[19] + output[20];
        step[20] = -output[20] + output[19];
        step[21] = -output[21] + output[18];
        step[22] = -output[22] + output[17];
        step[23] = -output[23] + output[16];
        step[24] = -output[24] + output[31];
        step[25] = -output[25] + output[30];
        step[26] = -output[26] + output[29];
        step[27] = -output[27] + output[28];
        step[28] = output[28] + output[27];
        step[29] = output[29] + output[26];
        step[30] = output[30] + output[25];
        step[31] = output[31] + output[24];

        // Stage 4
        long c8  = CosPi8_64;
        long c24 = CosPi24_64;

        output[0] = step[0] + step[3];
        output[1] = step[1] + step[2];
        output[2] = -step[2] + step[1];
        output[3] = -step[3] + step[0];
        output[4] = step[4];
        output[5] = DctRound((-step[5] + step[6]) * c16);
        output[6] = DctRound((step[6] + step[5]) * c16);
        output[7] = step[7];
        output[8] = step[8] + step[11];
        output[9] = step[9] + step[10];
        output[10] = -step[10] + step[9];
        output[11] = -step[11] + step[8];
        output[12] = -step[12] + step[15];
        output[13] = -step[13] + step[14];
        output[14] = step[14] + step[13];
        output[15] = step[15] + step[12];

        output[16] = step[16];
        output[17] = step[17];
        output[18] = DctRound(step[18] * -c8 + step[29] * c24);
        output[19] = DctRound(step[19] * -c8 + step[28] * c24);
        output[20] = DctRound(step[20] * -c24 + step[27] * -c8);
        output[21] = DctRound(step[21] * -c24 + step[26] * -c8);
        output[22] = step[22];
        output[23] = step[23];
        output[24] = step[24];
        output[25] = step[25];
        output[26] = DctRound(step[26] * c24 + step[21] * -c8);
        output[27] = DctRound(step[27] * c24 + step[20] * -c8);
        output[28] = DctRound(step[28] * c8 + step[19] * c24);
        output[29] = DctRound(step[29] * c8 + step[18] * c24);
        output[30] = step[30];
        output[31] = step[31];

        // Stage 5
        step[0] = DctRound((output[0] + output[1]) * c16);
        step[1] = DctRound((-output[1] + output[0]) * c16);
        step[2] = DctRound(output[2] * c24 + output[3] * c8);
        step[3] = DctRound(output[3] * c24 - output[2] * c8);
        step[4] = output[4] + output[5];
        step[5] = -output[5] + output[4];
        step[6] = -output[6] + output[7];
        step[7] = output[7] + output[6];
        step[8] = output[8];
        step[9] = DctRound(output[9] * -c8 + output[14] * c24);
        step[10] = DctRound(output[10] * -c24 + output[13] * -c8);
        step[11] = output[11];
        step[12] = output[12];
        step[13] = DctRound(output[13] * c24 + output[10] * -c8);
        step[14] = DctRound(output[14] * c8 + output[9] * c24);
        step[15] = output[15];

        step[16] = output[16] + output[19];
        step[17] = output[17] + output[18];
        step[18] = -output[18] + output[17];
        step[19] = -output[19] + output[16];
        step[20] = -output[20] + output[23];
        step[21] = -output[21] + output[22];
        step[22] = output[22] + output[21];
        step[23] = output[23] + output[20];
        step[24] = output[24] + output[27];
        step[25] = output[25] + output[26];
        step[26] = -output[26] + output[25];
        step[27] = -output[27] + output[24];
        step[28] = -output[28] + output[31];
        step[29] = -output[29] + output[30];
        step[30] = output[30] + output[29];
        step[31] = output[31] + output[28];

        // Stage 6
        long c4  = CosPi4_64;
        long c28 = CosPi28_64;
        long c12 = CosPi12_64;
        long c20 = CosPi20_64;

        output[0] = step[0];
        output[1] = step[1];
        output[2] = step[2];
        output[3] = step[3];
        output[4] = DctRound(step[4] * c28 + step[7] * c4);
        output[5] = DctRound(step[5] * c12 + step[6] * c20);
        output[6] = DctRound(step[6] * c12 + step[5] * -c20);
        output[7] = DctRound(step[7] * c28 + step[4] * -c4);
        output[8] = step[8] + step[9];
        output[9] = -step[9] + step[8];
        output[10] = -step[10] + step[11];
        output[11] = step[11] + step[10];
        output[12] = step[12] + step[13];
        output[13] = -step[13] + step[12];
        output[14] = -step[14] + step[15];
        output[15] = step[15] + step[14];

        output[16] = step[16];
        output[17] = DctRound(step[17] * -c4 + step[30] * c28);
        output[18] = DctRound(step[18] * -c28 + step[29] * -c4);
        output[19] = step[19];
        output[20] = step[20];
        output[21] = DctRound(step[21] * -c20 + step[26] * c12);
        output[22] = DctRound(step[22] * -c12 + step[25] * -c20);
        output[23] = step[23];
        output[24] = step[24];
        output[25] = DctRound(step[25] * c12 + step[22] * -c20);
        output[26] = DctRound(step[26] * c20 + step[21] * c12);
        output[27] = step[27];
        output[28] = step[28];
        output[29] = DctRound(step[29] * c28 + step[18] * -c4);
        output[30] = DctRound(step[30] * c4 + step[17] * c28);
        output[31] = step[31];

        // Stage 7
        long c2  = CosPi2_64;
        long c30 = CosPi30_64;
        long c14 = CosPi14_64;
        long c18 = CosPi18_64;
        long c10 = CosPi10_64;
        long c22 = CosPi22_64;
        long c26 = CosPi26_64;
        long c6  = CosPi6_64;

        step[0] = output[0];
        step[1] = output[1];
        step[2] = output[2];
        step[3] = output[3];
        step[4] = output[4];
        step[5] = output[5];
        step[6] = output[6];
        step[7] = output[7];
        step[8]  = DctRound(output[8]  * c30 + output[15] * c2);
        step[9]  = DctRound(output[9]  * c14 + output[14] * c18);
        step[10] = DctRound(output[10] * c22 + output[13] * c10);
        step[11] = DctRound(output[11] * c6  + output[12] * c26);
        step[12] = DctRound(output[12] * c6  + output[11] * -c26);
        step[13] = DctRound(output[13] * c22 + output[10] * -c10);
        step[14] = DctRound(output[14] * c14 + output[9]  * -c18);
        step[15] = DctRound(output[15] * c30 + output[8]  * -c2);

        step[16] = output[16] + output[17];
        step[17] = -output[17] + output[16];
        step[18] = -output[18] + output[19];
        step[19] = output[19] + output[18];
        step[20] = output[20] + output[21];
        step[21] = -output[21] + output[20];
        step[22] = -output[22] + output[23];
        step[23] = output[23] + output[22];
        step[24] = output[24] + output[25];
        step[25] = -output[25] + output[24];
        step[26] = -output[26] + output[27];
        step[27] = output[27] + output[26];
        step[28] = output[28] + output[29];
        step[29] = -output[29] + output[28];
        step[30] = -output[30] + output[31];
        step[31] = output[31] + output[30];

        // Final stage --- bit-reversed output indices.
        long c1  = CosPi1_64;
        long c31 = CosPi31_64;
        long c15 = CosPi15_64;
        long c17 = CosPi17_64;
        long c23 = CosPi23_64;
        long c9  = CosPi9_64;
        long c7  = CosPi7_64;
        long c25 = CosPi25_64;
        long c27 = CosPi27_64;
        long c5  = CosPi5_64;
        long c11 = CosPi11_64;
        long c21 = CosPi21_64;
        long c19 = CosPi19_64;
        long c13 = CosPi13_64;
        long c3  = CosPi3_64;
        long c29 = CosPi29_64;

        output[0]  = step[0];
        output[16] = step[1];
        output[8]  = step[2];
        output[24] = step[3];
        output[4]  = step[4];
        output[20] = step[5];
        output[12] = step[6];
        output[28] = step[7];
        output[2]  = step[8];
        output[18] = step[9];
        output[10] = step[10];
        output[26] = step[11];
        output[6]  = step[12];
        output[22] = step[13];
        output[14] = step[14];
        output[30] = step[15];

        output[1]  = DctRound(step[16] * c31 + step[31] * c1);
        output[17] = DctRound(step[17] * c15 + step[30] * c17);
        output[9]  = DctRound(step[18] * c23 + step[29] * c9);
        output[25] = DctRound(step[19] * c7  + step[28] * c25);
        output[5]  = DctRound(step[20] * c27 + step[27] * c5);
        output[21] = DctRound(step[21] * c11 + step[26] * c21);
        output[13] = DctRound(step[22] * c19 + step[25] * c13);
        output[29] = DctRound(step[23] * c3  + step[24] * c29);
        output[3]  = DctRound(step[24] * c3  + step[23] * -c29);
        output[19] = DctRound(step[25] * c19 + step[22] * -c13);
        output[11] = DctRound(step[26] * c11 + step[21] * -c21);
        output[27] = DctRound(step[27] * c27 + step[20] * -c5);
        output[7]  = DctRound(step[28] * c7  + step[19] * -c25);
        output[23] = DctRound(step[29] * c23 + step[18] * -c9);
        output[15] = DctRound(step[30] * c15 + step[17] * -c17);
        output[31] = DctRound(step[31] * c31 + step[16] * -c1);
    }

    /// <summary>Release kernel resources. Does NOT dispose the accelerator.</summary>
    public void Dispose() { /* auto-grouped */ }
}
