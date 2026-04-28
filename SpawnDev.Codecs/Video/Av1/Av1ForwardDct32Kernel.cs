// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// ILGPU kernel for the AV1 32-point forward DCT (1D). Bit-exact mirror
// of Av1ForwardDct32.Transform - one thread per 32-element 1D
// transform.
//
// 9 stages with cospi-driven half_btf rotations + final bit-reversed
// scatter to output. Uses LocalMemory<int>(32) for per-thread scratch
// (bf0/bf1 ping-pong buffers).
//
// Cospi default cos_bit for fdct32 is 12 (libaom). All cos_bit values
// 10..13 are supported. Cospi values are loaded from a per-thread
// 32-element table populated by branch-on-cos_bit; the 32-element
// table covers every odd index from 1..63 plus all even indices fdct32
// touches.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>
/// Batched ILGPU kernel for the AV1 32-point forward DCT (1D). Bit-exact
/// mirror of <see cref="Av1ForwardDct32.Transform"/>.
/// </summary>
public sealed class Av1ForwardDct32Kernel : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, int, int> _kernel;

    /// <summary>Compile the kernel onto <paramref name="accelerator"/>.</summary>
    public Av1ForwardDct32Kernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, ArrayView<int>, int, int>(FdctKernel);
    }

    /// <summary>
    /// Run the FDCT on <paramref name="transformCount"/> independent
    /// 32-element transforms.
    /// </summary>
    public void Run(ArrayView<int> input, ArrayView<int> output, int transformCount, int cosBit = Av1ForwardDct32.DefaultCosBit)
    {
        if (transformCount < 0) throw new ArgumentOutOfRangeException(nameof(transformCount));
        if (transformCount == 0) return;
        if (cosBit < 10 || cosBit > 13) throw new ArgumentOutOfRangeException(nameof(cosBit), "must be in [10, 13]");
        if (input.Length < transformCount * 32L)
            throw new ArgumentException($"input must hold at least transformCount*32 ints (got {input.Length}).", nameof(input));
        if (output.Length < transformCount * 32L)
            throw new ArgumentException($"output must hold at least transformCount*32 ints (got {output.Length}).", nameof(output));
        _kernel(transformCount, input, output, transformCount, cosBit);
    }

    /// <summary>Kernel body. One thread per 32-element transform.</summary>
    private static void FdctKernel(
        Index1D transformIdx,
        ArrayView<int> input,
        ArrayView<int> output,
        int transformCount,
        int cosBit)
    {
        int idx = transformIdx;
        if (idx >= transformCount) return;
        long inBase = (long)idx * 32;
        long outBase = (long)idx * 32;

        // 64-element cospi table populated per cos_bit. Stored in
        // LocalMemory<int>(64) so kernel doesn't need branches inside
        // the butterfly hot path.
        var cospi = LocalMemory.Allocate<int>(64);
        FillCospi(cospi, cosBit);

        var bf0 = LocalMemory.Allocate<int>(32);
        var bf1 = LocalMemory.Allocate<int>(32);

        // Stage 1
        bf0[0]  = input[inBase + 0]  + input[inBase + 31];
        bf0[1]  = input[inBase + 1]  + input[inBase + 30];
        bf0[2]  = input[inBase + 2]  + input[inBase + 29];
        bf0[3]  = input[inBase + 3]  + input[inBase + 28];
        bf0[4]  = input[inBase + 4]  + input[inBase + 27];
        bf0[5]  = input[inBase + 5]  + input[inBase + 26];
        bf0[6]  = input[inBase + 6]  + input[inBase + 25];
        bf0[7]  = input[inBase + 7]  + input[inBase + 24];
        bf0[8]  = input[inBase + 8]  + input[inBase + 23];
        bf0[9]  = input[inBase + 9]  + input[inBase + 22];
        bf0[10] = input[inBase + 10] + input[inBase + 21];
        bf0[11] = input[inBase + 11] + input[inBase + 20];
        bf0[12] = input[inBase + 12] + input[inBase + 19];
        bf0[13] = input[inBase + 13] + input[inBase + 18];
        bf0[14] = input[inBase + 14] + input[inBase + 17];
        bf0[15] = input[inBase + 15] + input[inBase + 16];
        bf0[16] = -input[inBase + 16] + input[inBase + 15];
        bf0[17] = -input[inBase + 17] + input[inBase + 14];
        bf0[18] = -input[inBase + 18] + input[inBase + 13];
        bf0[19] = -input[inBase + 19] + input[inBase + 12];
        bf0[20] = -input[inBase + 20] + input[inBase + 11];
        bf0[21] = -input[inBase + 21] + input[inBase + 10];
        bf0[22] = -input[inBase + 22] + input[inBase + 9];
        bf0[23] = -input[inBase + 23] + input[inBase + 8];
        bf0[24] = -input[inBase + 24] + input[inBase + 7];
        bf0[25] = -input[inBase + 25] + input[inBase + 6];
        bf0[26] = -input[inBase + 26] + input[inBase + 5];
        bf0[27] = -input[inBase + 27] + input[inBase + 4];
        bf0[28] = -input[inBase + 28] + input[inBase + 3];
        bf0[29] = -input[inBase + 29] + input[inBase + 2];
        bf0[30] = -input[inBase + 30] + input[inBase + 1];
        bf0[31] = -input[inBase + 31] + input[inBase + 0];

        // Stage 2
        bf1[0]  = bf0[0]  + bf0[15];
        bf1[1]  = bf0[1]  + bf0[14];
        bf1[2]  = bf0[2]  + bf0[13];
        bf1[3]  = bf0[3]  + bf0[12];
        bf1[4]  = bf0[4]  + bf0[11];
        bf1[5]  = bf0[5]  + bf0[10];
        bf1[6]  = bf0[6]  + bf0[9];
        bf1[7]  = bf0[7]  + bf0[8];
        bf1[8]  = -bf0[8]  + bf0[7];
        bf1[9]  = -bf0[9]  + bf0[6];
        bf1[10] = -bf0[10] + bf0[5];
        bf1[11] = -bf0[11] + bf0[4];
        bf1[12] = -bf0[12] + bf0[3];
        bf1[13] = -bf0[13] + bf0[2];
        bf1[14] = -bf0[14] + bf0[1];
        bf1[15] = -bf0[15] + bf0[0];
        bf1[16] = bf0[16];
        bf1[17] = bf0[17];
        bf1[18] = bf0[18];
        bf1[19] = bf0[19];
        bf1[20] = HalfBtf(-cospi[32], bf0[20], cospi[32], bf0[27], cosBit);
        bf1[21] = HalfBtf(-cospi[32], bf0[21], cospi[32], bf0[26], cosBit);
        bf1[22] = HalfBtf(-cospi[32], bf0[22], cospi[32], bf0[25], cosBit);
        bf1[23] = HalfBtf(-cospi[32], bf0[23], cospi[32], bf0[24], cosBit);
        bf1[24] = HalfBtf( cospi[32], bf0[24], cospi[32], bf0[23], cosBit);
        bf1[25] = HalfBtf( cospi[32], bf0[25], cospi[32], bf0[22], cosBit);
        bf1[26] = HalfBtf( cospi[32], bf0[26], cospi[32], bf0[21], cosBit);
        bf1[27] = HalfBtf( cospi[32], bf0[27], cospi[32], bf0[20], cosBit);
        bf1[28] = bf0[28];
        bf1[29] = bf0[29];
        bf1[30] = bf0[30];
        bf1[31] = bf0[31];

        // Stage 3
        bf0[0]  = bf1[0] + bf1[7];
        bf0[1]  = bf1[1] + bf1[6];
        bf0[2]  = bf1[2] + bf1[5];
        bf0[3]  = bf1[3] + bf1[4];
        bf0[4]  = -bf1[4] + bf1[3];
        bf0[5]  = -bf1[5] + bf1[2];
        bf0[6]  = -bf1[6] + bf1[1];
        bf0[7]  = -bf1[7] + bf1[0];
        bf0[8]  = bf1[8];
        bf0[9]  = bf1[9];
        bf0[10] = HalfBtf(-cospi[32], bf1[10], cospi[32], bf1[13], cosBit);
        bf0[11] = HalfBtf(-cospi[32], bf1[11], cospi[32], bf1[12], cosBit);
        bf0[12] = HalfBtf( cospi[32], bf1[12], cospi[32], bf1[11], cosBit);
        bf0[13] = HalfBtf( cospi[32], bf1[13], cospi[32], bf1[10], cosBit);
        bf0[14] = bf1[14];
        bf0[15] = bf1[15];
        bf0[16] = bf1[16] + bf1[23];
        bf0[17] = bf1[17] + bf1[22];
        bf0[18] = bf1[18] + bf1[21];
        bf0[19] = bf1[19] + bf1[20];
        bf0[20] = -bf1[20] + bf1[19];
        bf0[21] = -bf1[21] + bf1[18];
        bf0[22] = -bf1[22] + bf1[17];
        bf0[23] = -bf1[23] + bf1[16];
        bf0[24] = -bf1[24] + bf1[31];
        bf0[25] = -bf1[25] + bf1[30];
        bf0[26] = -bf1[26] + bf1[29];
        bf0[27] = -bf1[27] + bf1[28];
        bf0[28] = bf1[28] + bf1[27];
        bf0[29] = bf1[29] + bf1[26];
        bf0[30] = bf1[30] + bf1[25];
        bf0[31] = bf1[31] + bf1[24];

        // Stage 4
        bf1[0]  = bf0[0] + bf0[3];
        bf1[1]  = bf0[1] + bf0[2];
        bf1[2]  = -bf0[2] + bf0[1];
        bf1[3]  = -bf0[3] + bf0[0];
        bf1[4]  = bf0[4];
        bf1[5]  = HalfBtf(-cospi[32], bf0[5], cospi[32], bf0[6], cosBit);
        bf1[6]  = HalfBtf( cospi[32], bf0[6], cospi[32], bf0[5], cosBit);
        bf1[7]  = bf0[7];
        bf1[8]  = bf0[8] + bf0[11];
        bf1[9]  = bf0[9] + bf0[10];
        bf1[10] = -bf0[10] + bf0[9];
        bf1[11] = -bf0[11] + bf0[8];
        bf1[12] = -bf0[12] + bf0[15];
        bf1[13] = -bf0[13] + bf0[14];
        bf1[14] = bf0[14] + bf0[13];
        bf1[15] = bf0[15] + bf0[12];
        bf1[16] = bf0[16];
        bf1[17] = bf0[17];
        bf1[18] = HalfBtf(-cospi[16], bf0[18],  cospi[48], bf0[29], cosBit);
        bf1[19] = HalfBtf(-cospi[16], bf0[19],  cospi[48], bf0[28], cosBit);
        bf1[20] = HalfBtf(-cospi[48], bf0[20], -cospi[16], bf0[27], cosBit);
        bf1[21] = HalfBtf(-cospi[48], bf0[21], -cospi[16], bf0[26], cosBit);
        bf1[22] = bf0[22];
        bf1[23] = bf0[23];
        bf1[24] = bf0[24];
        bf1[25] = bf0[25];
        bf1[26] = HalfBtf( cospi[48], bf0[26], -cospi[16], bf0[21], cosBit);
        bf1[27] = HalfBtf( cospi[48], bf0[27], -cospi[16], bf0[20], cosBit);
        bf1[28] = HalfBtf( cospi[16], bf0[28],  cospi[48], bf0[19], cosBit);
        bf1[29] = HalfBtf( cospi[16], bf0[29],  cospi[48], bf0[18], cosBit);
        bf1[30] = bf0[30];
        bf1[31] = bf0[31];

        // Stage 5
        bf0[0]  = HalfBtf( cospi[32], bf1[0], cospi[32], bf1[1], cosBit);
        bf0[1]  = HalfBtf(-cospi[32], bf1[1], cospi[32], bf1[0], cosBit);
        bf0[2]  = HalfBtf( cospi[48], bf1[2], cospi[16], bf1[3], cosBit);
        bf0[3]  = HalfBtf( cospi[48], bf1[3], -cospi[16], bf1[2], cosBit);
        bf0[4]  = bf1[4] + bf1[5];
        bf0[5]  = -bf1[5] + bf1[4];
        bf0[6]  = -bf1[6] + bf1[7];
        bf0[7]  = bf1[7] + bf1[6];
        bf0[8]  = bf1[8];
        bf0[9]  = HalfBtf(-cospi[16], bf1[9],  cospi[48], bf1[14], cosBit);
        bf0[10] = HalfBtf(-cospi[48], bf1[10], -cospi[16], bf1[13], cosBit);
        bf0[11] = bf1[11];
        bf0[12] = bf1[12];
        bf0[13] = HalfBtf( cospi[48], bf1[13], -cospi[16], bf1[10], cosBit);
        bf0[14] = HalfBtf( cospi[16], bf1[14],  cospi[48], bf1[9],  cosBit);
        bf0[15] = bf1[15];
        bf0[16] = bf1[16] + bf1[19];
        bf0[17] = bf1[17] + bf1[18];
        bf0[18] = -bf1[18] + bf1[17];
        bf0[19] = -bf1[19] + bf1[16];
        bf0[20] = -bf1[20] + bf1[23];
        bf0[21] = -bf1[21] + bf1[22];
        bf0[22] = bf1[22] + bf1[21];
        bf0[23] = bf1[23] + bf1[20];
        bf0[24] = bf1[24] + bf1[27];
        bf0[25] = bf1[25] + bf1[26];
        bf0[26] = -bf1[26] + bf1[25];
        bf0[27] = -bf1[27] + bf1[24];
        bf0[28] = -bf1[28] + bf1[31];
        bf0[29] = -bf1[29] + bf1[30];
        bf0[30] = bf1[30] + bf1[29];
        bf0[31] = bf1[31] + bf1[28];

        // Stage 6
        bf1[0]  = bf0[0];
        bf1[1]  = bf0[1];
        bf1[2]  = bf0[2];
        bf1[3]  = bf0[3];
        bf1[4]  = HalfBtf( cospi[56], bf0[4],  cospi[8],  bf0[7], cosBit);
        bf1[5]  = HalfBtf( cospi[24], bf0[5],  cospi[40], bf0[6], cosBit);
        bf1[6]  = HalfBtf( cospi[24], bf0[6], -cospi[40], bf0[5], cosBit);
        bf1[7]  = HalfBtf( cospi[56], bf0[7], -cospi[8],  bf0[4], cosBit);
        bf1[8]  = bf0[8] + bf0[9];
        bf1[9]  = -bf0[9] + bf0[8];
        bf1[10] = -bf0[10] + bf0[11];
        bf1[11] = bf0[11] + bf0[10];
        bf1[12] = bf0[12] + bf0[13];
        bf1[13] = -bf0[13] + bf0[12];
        bf1[14] = -bf0[14] + bf0[15];
        bf1[15] = bf0[15] + bf0[14];
        bf1[16] = bf0[16];
        bf1[17] = HalfBtf(-cospi[8],  bf0[17],  cospi[56], bf0[30], cosBit);
        bf1[18] = HalfBtf(-cospi[56], bf0[18], -cospi[8],  bf0[29], cosBit);
        bf1[19] = bf0[19];
        bf1[20] = bf0[20];
        bf1[21] = HalfBtf(-cospi[40], bf0[21],  cospi[24], bf0[26], cosBit);
        bf1[22] = HalfBtf(-cospi[24], bf0[22], -cospi[40], bf0[25], cosBit);
        bf1[23] = bf0[23];
        bf1[24] = bf0[24];
        bf1[25] = HalfBtf( cospi[24], bf0[25], -cospi[40], bf0[22], cosBit);
        bf1[26] = HalfBtf( cospi[40], bf0[26],  cospi[24], bf0[21], cosBit);
        bf1[27] = bf0[27];
        bf1[28] = bf0[28];
        bf1[29] = HalfBtf( cospi[56], bf0[29], -cospi[8],  bf0[18], cosBit);
        bf1[30] = HalfBtf( cospi[8],  bf0[30],  cospi[56], bf0[17], cosBit);
        bf1[31] = bf0[31];

        // Stage 7
        bf0[0]  = bf1[0];
        bf0[1]  = bf1[1];
        bf0[2]  = bf1[2];
        bf0[3]  = bf1[3];
        bf0[4]  = bf1[4];
        bf0[5]  = bf1[5];
        bf0[6]  = bf1[6];
        bf0[7]  = bf1[7];
        bf0[8]  = HalfBtf( cospi[60], bf1[8],   cospi[4],  bf1[15], cosBit);
        bf0[9]  = HalfBtf( cospi[28], bf1[9],   cospi[36], bf1[14], cosBit);
        bf0[10] = HalfBtf( cospi[44], bf1[10],  cospi[20], bf1[13], cosBit);
        bf0[11] = HalfBtf( cospi[12], bf1[11],  cospi[52], bf1[12], cosBit);
        bf0[12] = HalfBtf( cospi[12], bf1[12], -cospi[52], bf1[11], cosBit);
        bf0[13] = HalfBtf( cospi[44], bf1[13], -cospi[20], bf1[10], cosBit);
        bf0[14] = HalfBtf( cospi[28], bf1[14], -cospi[36], bf1[9],  cosBit);
        bf0[15] = HalfBtf( cospi[60], bf1[15], -cospi[4],  bf1[8],  cosBit);
        bf0[16] = bf1[16] + bf1[17];
        bf0[17] = -bf1[17] + bf1[16];
        bf0[18] = -bf1[18] + bf1[19];
        bf0[19] = bf1[19] + bf1[18];
        bf0[20] = bf1[20] + bf1[21];
        bf0[21] = -bf1[21] + bf1[20];
        bf0[22] = -bf1[22] + bf1[23];
        bf0[23] = bf1[23] + bf1[22];
        bf0[24] = bf1[24] + bf1[25];
        bf0[25] = -bf1[25] + bf1[24];
        bf0[26] = -bf1[26] + bf1[27];
        bf0[27] = bf1[27] + bf1[26];
        bf0[28] = bf1[28] + bf1[29];
        bf0[29] = -bf1[29] + bf1[28];
        bf0[30] = -bf1[30] + bf1[31];
        bf0[31] = bf1[31] + bf1[30];

        // Stage 8
        bf1[0]  = bf0[0];
        bf1[1]  = bf0[1];
        bf1[2]  = bf0[2];
        bf1[3]  = bf0[3];
        bf1[4]  = bf0[4];
        bf1[5]  = bf0[5];
        bf1[6]  = bf0[6];
        bf1[7]  = bf0[7];
        bf1[8]  = bf0[8];
        bf1[9]  = bf0[9];
        bf1[10] = bf0[10];
        bf1[11] = bf0[11];
        bf1[12] = bf0[12];
        bf1[13] = bf0[13];
        bf1[14] = bf0[14];
        bf1[15] = bf0[15];
        bf1[16] = HalfBtf(cospi[62], bf0[16], cospi[2],  bf0[31], cosBit);
        bf1[17] = HalfBtf(cospi[30], bf0[17], cospi[34], bf0[30], cosBit);
        bf1[18] = HalfBtf(cospi[46], bf0[18], cospi[18], bf0[29], cosBit);
        bf1[19] = HalfBtf(cospi[14], bf0[19], cospi[50], bf0[28], cosBit);
        bf1[20] = HalfBtf(cospi[54], bf0[20], cospi[10], bf0[27], cosBit);
        bf1[21] = HalfBtf(cospi[22], bf0[21], cospi[42], bf0[26], cosBit);
        bf1[22] = HalfBtf(cospi[38], bf0[22], cospi[26], bf0[25], cosBit);
        bf1[23] = HalfBtf(cospi[6],  bf0[23], cospi[58], bf0[24], cosBit);
        bf1[24] = HalfBtf(cospi[6],  bf0[24], -cospi[58], bf0[23], cosBit);
        bf1[25] = HalfBtf(cospi[38], bf0[25], -cospi[26], bf0[22], cosBit);
        bf1[26] = HalfBtf(cospi[22], bf0[26], -cospi[42], bf0[21], cosBit);
        bf1[27] = HalfBtf(cospi[54], bf0[27], -cospi[10], bf0[20], cosBit);
        bf1[28] = HalfBtf(cospi[14], bf0[28], -cospi[50], bf0[19], cosBit);
        bf1[29] = HalfBtf(cospi[46], bf0[29], -cospi[18], bf0[18], cosBit);
        bf1[30] = HalfBtf(cospi[30], bf0[30], -cospi[34], bf0[17], cosBit);
        bf1[31] = HalfBtf(cospi[62], bf0[31], -cospi[2],  bf0[16], cosBit);

        // Stage 9: bit-reversed scatter
        output[outBase + 0]  = bf1[0];
        output[outBase + 1]  = bf1[16];
        output[outBase + 2]  = bf1[8];
        output[outBase + 3]  = bf1[24];
        output[outBase + 4]  = bf1[4];
        output[outBase + 5]  = bf1[20];
        output[outBase + 6]  = bf1[12];
        output[outBase + 7]  = bf1[28];
        output[outBase + 8]  = bf1[2];
        output[outBase + 9]  = bf1[18];
        output[outBase + 10] = bf1[10];
        output[outBase + 11] = bf1[26];
        output[outBase + 12] = bf1[6];
        output[outBase + 13] = bf1[22];
        output[outBase + 14] = bf1[14];
        output[outBase + 15] = bf1[30];
        output[outBase + 16] = bf1[1];
        output[outBase + 17] = bf1[17];
        output[outBase + 18] = bf1[9];
        output[outBase + 19] = bf1[25];
        output[outBase + 20] = bf1[5];
        output[outBase + 21] = bf1[21];
        output[outBase + 22] = bf1[13];
        output[outBase + 23] = bf1[29];
        output[outBase + 24] = bf1[3];
        output[outBase + 25] = bf1[19];
        output[outBase + 26] = bf1[11];
        output[outBase + 27] = bf1[27];
        output[outBase + 28] = bf1[7];
        output[outBase + 29] = bf1[23];
        output[outBase + 30] = bf1[15];
        output[outBase + 31] = bf1[31];
    }

    /// <summary>libaom <c>half_btf</c>: kernel-safe variant.</summary>
    private static int HalfBtf(int w0, int in0, int w1, int in1, int bit)
    {
        long result = (long)w0 * in0 + (long)w1 * in1;
        result += 1L << (bit - 1);
        return (int)(result >> bit);
    }

    /// <summary>
    /// Populates a 64-element cospi table per cos_bit. Mirrors
    /// Av1ForwardDct4.CospiArrData[cos_bit - 10] entries 0..63.
    /// </summary>
    private static void FillCospi(ArrayView<int> cospi, int cosBit)
    {
        if (cosBit == 13)
        {
            cospi[0] = 8192;  cospi[1] = 8190;  cospi[2] = 8182;  cospi[3] = 8170;
            cospi[4] = 8153;  cospi[5] = 8130;  cospi[6] = 8103;  cospi[7] = 8071;
            cospi[8] = 8035;  cospi[9] = 7993;  cospi[10] = 7946; cospi[11] = 7895;
            cospi[12] = 7839; cospi[13] = 7779; cospi[14] = 7713; cospi[15] = 7643;
            cospi[16] = 7568; cospi[17] = 7489; cospi[18] = 7405; cospi[19] = 7317;
            cospi[20] = 7225; cospi[21] = 7128; cospi[22] = 7027; cospi[23] = 6921;
            cospi[24] = 6811; cospi[25] = 6698; cospi[26] = 6580; cospi[27] = 6458;
            cospi[28] = 6333; cospi[29] = 6203; cospi[30] = 6070; cospi[31] = 5933;
            cospi[32] = 5793; cospi[33] = 5649; cospi[34] = 5501; cospi[35] = 5351;
            cospi[36] = 5197; cospi[37] = 5040; cospi[38] = 4880; cospi[39] = 4717;
            cospi[40] = 4551; cospi[41] = 4383; cospi[42] = 4212; cospi[43] = 4038;
            cospi[44] = 3862; cospi[45] = 3683; cospi[46] = 3503; cospi[47] = 3320;
            cospi[48] = 3135; cospi[49] = 2948; cospi[50] = 2760; cospi[51] = 2570;
            cospi[52] = 2378; cospi[53] = 2185; cospi[54] = 1990; cospi[55] = 1795;
            cospi[56] = 1598; cospi[57] = 1401; cospi[58] = 1202; cospi[59] = 1003;
            cospi[60] = 803;  cospi[61] = 603;  cospi[62] = 402;  cospi[63] = 201;
        }
        else if (cosBit == 12)
        {
            cospi[0] = 4096; cospi[1] = 4095; cospi[2] = 4091; cospi[3] = 4085;
            cospi[4] = 4076; cospi[5] = 4065; cospi[6] = 4052; cospi[7] = 4036;
            cospi[8] = 4017; cospi[9] = 3996; cospi[10] = 3973; cospi[11] = 3948;
            cospi[12] = 3920; cospi[13] = 3889; cospi[14] = 3857; cospi[15] = 3822;
            cospi[16] = 3784; cospi[17] = 3745; cospi[18] = 3703; cospi[19] = 3659;
            cospi[20] = 3612; cospi[21] = 3564; cospi[22] = 3513; cospi[23] = 3461;
            cospi[24] = 3406; cospi[25] = 3349; cospi[26] = 3290; cospi[27] = 3229;
            cospi[28] = 3166; cospi[29] = 3102; cospi[30] = 3035; cospi[31] = 2967;
            cospi[32] = 2896; cospi[33] = 2824; cospi[34] = 2751; cospi[35] = 2675;
            cospi[36] = 2598; cospi[37] = 2520; cospi[38] = 2440; cospi[39] = 2359;
            cospi[40] = 2276; cospi[41] = 2191; cospi[42] = 2106; cospi[43] = 2019;
            cospi[44] = 1931; cospi[45] = 1842; cospi[46] = 1751; cospi[47] = 1660;
            cospi[48] = 1567; cospi[49] = 1474; cospi[50] = 1380; cospi[51] = 1285;
            cospi[52] = 1189; cospi[53] = 1092; cospi[54] = 995;  cospi[55] = 897;
            cospi[56] = 799;  cospi[57] = 700;  cospi[58] = 601;  cospi[59] = 501;
            cospi[60] = 401;  cospi[61] = 301;  cospi[62] = 201;  cospi[63] = 101;
        }
        else if (cosBit == 11)
        {
            cospi[0] = 2048; cospi[1] = 2047; cospi[2] = 2046; cospi[3] = 2042;
            cospi[4] = 2038; cospi[5] = 2033; cospi[6] = 2026; cospi[7] = 2018;
            cospi[8] = 2009; cospi[9] = 1998; cospi[10] = 1987; cospi[11] = 1974;
            cospi[12] = 1960; cospi[13] = 1945; cospi[14] = 1928; cospi[15] = 1911;
            cospi[16] = 1892; cospi[17] = 1872; cospi[18] = 1851; cospi[19] = 1829;
            cospi[20] = 1806; cospi[21] = 1782; cospi[22] = 1757; cospi[23] = 1730;
            cospi[24] = 1703; cospi[25] = 1674; cospi[26] = 1645; cospi[27] = 1615;
            cospi[28] = 1583; cospi[29] = 1551; cospi[30] = 1517; cospi[31] = 1483;
            cospi[32] = 1448; cospi[33] = 1412; cospi[34] = 1375; cospi[35] = 1338;
            cospi[36] = 1299; cospi[37] = 1260; cospi[38] = 1220; cospi[39] = 1179;
            cospi[40] = 1138; cospi[41] = 1096; cospi[42] = 1053; cospi[43] = 1009;
            cospi[44] = 965;  cospi[45] = 921;  cospi[46] = 876;  cospi[47] = 830;
            cospi[48] = 784;  cospi[49] = 737;  cospi[50] = 690;  cospi[51] = 642;
            cospi[52] = 595;  cospi[53] = 546;  cospi[54] = 498;  cospi[55] = 449;
            cospi[56] = 400;  cospi[57] = 350;  cospi[58] = 301;  cospi[59] = 251;
            cospi[60] = 201;  cospi[61] = 151;  cospi[62] = 100;  cospi[63] = 50;
        }
        else // cosBit == 10
        {
            cospi[0] = 1024; cospi[1] = 1024; cospi[2] = 1023; cospi[3] = 1021;
            cospi[4] = 1019; cospi[5] = 1016; cospi[6] = 1013; cospi[7] = 1009;
            cospi[8] = 1004; cospi[9] = 999;  cospi[10] = 993; cospi[11] = 987;
            cospi[12] = 980; cospi[13] = 972; cospi[14] = 964; cospi[15] = 955;
            cospi[16] = 946; cospi[17] = 936; cospi[18] = 926; cospi[19] = 915;
            cospi[20] = 903; cospi[21] = 891; cospi[22] = 878; cospi[23] = 865;
            cospi[24] = 851; cospi[25] = 837; cospi[26] = 822; cospi[27] = 807;
            cospi[28] = 792; cospi[29] = 775; cospi[30] = 759; cospi[31] = 742;
            cospi[32] = 724; cospi[33] = 706; cospi[34] = 688; cospi[35] = 669;
            cospi[36] = 650; cospi[37] = 630; cospi[38] = 610; cospi[39] = 590;
            cospi[40] = 569; cospi[41] = 548; cospi[42] = 526; cospi[43] = 505;
            cospi[44] = 483; cospi[45] = 460; cospi[46] = 438; cospi[47] = 415;
            cospi[48] = 392; cospi[49] = 369; cospi[50] = 345; cospi[51] = 321;
            cospi[52] = 297; cospi[53] = 273; cospi[54] = 249; cospi[55] = 224;
            cospi[56] = 200; cospi[57] = 175; cospi[58] = 150; cospi[59] = 125;
            cospi[60] = 100; cospi[61] = 75;  cospi[62] = 50;  cospi[63] = 25;
        }
    }

    /// <summary>Release kernel resources. Does NOT dispose the accelerator.</summary>
    public void Dispose() { /* auto-grouped */ }
}
