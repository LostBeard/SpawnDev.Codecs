// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// ILGPU kernel for the AV1 16-point forward Asymmetric DST (1D).
// Bit-exact mirror of Av1ForwardAdst16.Transform - one thread per
// 16-element 1D ADST. Runs on every ILGPU backend.
//
// 9 stages with cospi-driven half_btf rotations + final scatter.
// Uses LocalMemory<int>(16) for stage scratch buffers.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>
/// Batched ILGPU kernel for the AV1 16-point forward ADST (1D). Bit-exact
/// mirror of <see cref="Av1ForwardAdst16.Transform"/>.
/// </summary>
public sealed class Av1ForwardAdst16Kernel : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, int, int> _kernel;

    /// <summary>Compile the kernel onto <paramref name="accelerator"/>.</summary>
    public Av1ForwardAdst16Kernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, ArrayView<int>, int, int>(FadstKernel);
    }

    /// <summary>
    /// Run the FADST on <paramref name="transformCount"/> independent
    /// 16-element transforms.
    /// </summary>
    public void Run(ArrayView<int> input, ArrayView<int> output, int transformCount, int cosBit = Av1ForwardAdst16.DefaultCosBit)
    {
        if (transformCount < 0) throw new ArgumentOutOfRangeException(nameof(transformCount));
        if (transformCount == 0) return;
        if (cosBit < 10 || cosBit > 13) throw new ArgumentOutOfRangeException(nameof(cosBit), "must be in [10, 13]");
        if (input.Length < transformCount * 16L)
            throw new ArgumentException($"input must hold at least transformCount*16 ints (got {input.Length}).", nameof(input));
        if (output.Length < transformCount * 16L)
            throw new ArgumentException($"output must hold at least transformCount*16 ints (got {output.Length}).", nameof(output));
        _kernel(transformCount, input, output, transformCount, cosBit);
    }

    /// <summary>Kernel body. One thread per 16-element transform.</summary>
    private static void FadstKernel(
        Index1D transformIdx,
        ArrayView<int> input,
        ArrayView<int> output,
        int transformCount,
        int cosBit)
    {
        int idx = transformIdx;
        if (idx >= transformCount) return;
        long inBase = (long)idx * 16;
        long outBase = (long)idx * 16;

        // 64-element cospi table per cos_bit. fadst16 uses many indices
        // (every multiple of 2 from 2..62), so the simplest path is
        // populate the full cospi table once.
        var cospi = LocalMemory.Allocate<int>(64);
        FillCospi(cospi, cosBit);

        var step = LocalMemory.Allocate<int>(16);
        var bf1 = LocalMemory.Allocate<int>(16);

        // Stage 1: input remap with sign flips.
        bf1[0]  =  input[inBase + 0];
        bf1[1]  = -input[inBase + 15];
        bf1[2]  = -input[inBase + 7];
        bf1[3]  =  input[inBase + 8];
        bf1[4]  = -input[inBase + 3];
        bf1[5]  =  input[inBase + 12];
        bf1[6]  =  input[inBase + 4];
        bf1[7]  = -input[inBase + 11];
        bf1[8]  = -input[inBase + 1];
        bf1[9]  =  input[inBase + 14];
        bf1[10] =  input[inBase + 6];
        bf1[11] = -input[inBase + 9];
        bf1[12] =  input[inBase + 2];
        bf1[13] = -input[inBase + 13];
        bf1[14] = -input[inBase + 5];
        bf1[15] =  input[inBase + 10];

        // Stage 2: cospi[32] rotations on (2,3), (6,7), (10,11), (14,15).
        step[0]  = bf1[0];
        step[1]  = bf1[1];
        step[2]  = HalfBtf(cospi[32], bf1[2],  cospi[32], bf1[3], cosBit);
        step[3]  = HalfBtf(cospi[32], bf1[2], -cospi[32], bf1[3], cosBit);
        step[4]  = bf1[4];
        step[5]  = bf1[5];
        step[6]  = HalfBtf(cospi[32], bf1[6],  cospi[32], bf1[7], cosBit);
        step[7]  = HalfBtf(cospi[32], bf1[6], -cospi[32], bf1[7], cosBit);
        step[8]  = bf1[8];
        step[9]  = bf1[9];
        step[10] = HalfBtf(cospi[32], bf1[10],  cospi[32], bf1[11], cosBit);
        step[11] = HalfBtf(cospi[32], bf1[10], -cospi[32], bf1[11], cosBit);
        step[12] = bf1[12];
        step[13] = bf1[13];
        step[14] = HalfBtf(cospi[32], bf1[14],  cospi[32], bf1[15], cosBit);
        step[15] = HalfBtf(cospi[32], bf1[14], -cospi[32], bf1[15], cosBit);

        // Stage 3: butterfly 4-element groups.
        bf1[0]  = step[0]  + step[2];
        bf1[1]  = step[1]  + step[3];
        bf1[2]  = step[0]  - step[2];
        bf1[3]  = step[1]  - step[3];
        bf1[4]  = step[4]  + step[6];
        bf1[5]  = step[5]  + step[7];
        bf1[6]  = step[4]  - step[6];
        bf1[7]  = step[5]  - step[7];
        bf1[8]  = step[8]  + step[10];
        bf1[9]  = step[9]  + step[11];
        bf1[10] = step[8]  - step[10];
        bf1[11] = step[9]  - step[11];
        bf1[12] = step[12] + step[14];
        bf1[13] = step[13] + step[15];
        bf1[14] = step[12] - step[14];
        bf1[15] = step[13] - step[15];

        // Stage 4: cospi[16/48] on (4,5), (6,7), (12,13), (14,15).
        step[0]  = bf1[0];
        step[1]  = bf1[1];
        step[2]  = bf1[2];
        step[3]  = bf1[3];
        step[4]  = HalfBtf( cospi[16], bf1[4],  cospi[48], bf1[5], cosBit);
        step[5]  = HalfBtf( cospi[48], bf1[4], -cospi[16], bf1[5], cosBit);
        step[6]  = HalfBtf(-cospi[48], bf1[6],  cospi[16], bf1[7], cosBit);
        step[7]  = HalfBtf( cospi[16], bf1[6],  cospi[48], bf1[7], cosBit);
        step[8]  = bf1[8];
        step[9]  = bf1[9];
        step[10] = bf1[10];
        step[11] = bf1[11];
        step[12] = HalfBtf( cospi[16], bf1[12],  cospi[48], bf1[13], cosBit);
        step[13] = HalfBtf( cospi[48], bf1[12], -cospi[16], bf1[13], cosBit);
        step[14] = HalfBtf(-cospi[48], bf1[14],  cospi[16], bf1[15], cosBit);
        step[15] = HalfBtf( cospi[16], bf1[14],  cospi[48], bf1[15], cosBit);

        // Stage 5: butterfly across halves.
        bf1[0]  = step[0]  + step[4];
        bf1[1]  = step[1]  + step[5];
        bf1[2]  = step[2]  + step[6];
        bf1[3]  = step[3]  + step[7];
        bf1[4]  = step[0]  - step[4];
        bf1[5]  = step[1]  - step[5];
        bf1[6]  = step[2]  - step[6];
        bf1[7]  = step[3]  - step[7];
        bf1[8]  = step[8]  + step[12];
        bf1[9]  = step[9]  + step[13];
        bf1[10] = step[10] + step[14];
        bf1[11] = step[11] + step[15];
        bf1[12] = step[8]  - step[12];
        bf1[13] = step[9]  - step[13];
        bf1[14] = step[10] - step[14];
        bf1[15] = step[11] - step[15];

        // Stage 6: cospi[8/56/40/24] rotations on the upper 8.
        step[0]  = bf1[0];
        step[1]  = bf1[1];
        step[2]  = bf1[2];
        step[3]  = bf1[3];
        step[4]  = bf1[4];
        step[5]  = bf1[5];
        step[6]  = bf1[6];
        step[7]  = bf1[7];
        step[8]  = HalfBtf( cospi[8],  bf1[8],   cospi[56], bf1[9],  cosBit);
        step[9]  = HalfBtf( cospi[56], bf1[8],  -cospi[8],  bf1[9],  cosBit);
        step[10] = HalfBtf( cospi[40], bf1[10],  cospi[24], bf1[11], cosBit);
        step[11] = HalfBtf( cospi[24], bf1[10], -cospi[40], bf1[11], cosBit);
        step[12] = HalfBtf(-cospi[56], bf1[12],  cospi[8],  bf1[13], cosBit);
        step[13] = HalfBtf( cospi[8],  bf1[12],  cospi[56], bf1[13], cosBit);
        step[14] = HalfBtf(-cospi[24], bf1[14],  cospi[40], bf1[15], cosBit);
        step[15] = HalfBtf( cospi[40], bf1[14],  cospi[24], bf1[15], cosBit);

        // Stage 7: butterfly across full 16-element width.
        bf1[0]  = step[0] + step[8];
        bf1[1]  = step[1] + step[9];
        bf1[2]  = step[2] + step[10];
        bf1[3]  = step[3] + step[11];
        bf1[4]  = step[4] + step[12];
        bf1[5]  = step[5] + step[13];
        bf1[6]  = step[6] + step[14];
        bf1[7]  = step[7] + step[15];
        bf1[8]  = step[0] - step[8];
        bf1[9]  = step[1] - step[9];
        bf1[10] = step[2] - step[10];
        bf1[11] = step[3] - step[11];
        bf1[12] = step[4] - step[12];
        bf1[13] = step[5] - step[13];
        bf1[14] = step[6] - step[14];
        bf1[15] = step[7] - step[15];

        // Stage 8: cospi[2/62/10/54/18/46/26/38/34/30/42/22/50/14/58/6] rotations.
        step[0]  = HalfBtf( cospi[2],  bf1[0],   cospi[62], bf1[1],  cosBit);
        step[1]  = HalfBtf( cospi[62], bf1[0],  -cospi[2],  bf1[1],  cosBit);
        step[2]  = HalfBtf( cospi[10], bf1[2],   cospi[54], bf1[3],  cosBit);
        step[3]  = HalfBtf( cospi[54], bf1[2],  -cospi[10], bf1[3],  cosBit);
        step[4]  = HalfBtf( cospi[18], bf1[4],   cospi[46], bf1[5],  cosBit);
        step[5]  = HalfBtf( cospi[46], bf1[4],  -cospi[18], bf1[5],  cosBit);
        step[6]  = HalfBtf( cospi[26], bf1[6],   cospi[38], bf1[7],  cosBit);
        step[7]  = HalfBtf( cospi[38], bf1[6],  -cospi[26], bf1[7],  cosBit);
        step[8]  = HalfBtf( cospi[34], bf1[8],   cospi[30], bf1[9],  cosBit);
        step[9]  = HalfBtf( cospi[30], bf1[8],  -cospi[34], bf1[9],  cosBit);
        step[10] = HalfBtf( cospi[42], bf1[10],  cospi[22], bf1[11], cosBit);
        step[11] = HalfBtf( cospi[22], bf1[10], -cospi[42], bf1[11], cosBit);
        step[12] = HalfBtf( cospi[50], bf1[12],  cospi[14], bf1[13], cosBit);
        step[13] = HalfBtf( cospi[14], bf1[12], -cospi[50], bf1[13], cosBit);
        step[14] = HalfBtf( cospi[58], bf1[14],  cospi[6],  bf1[15], cosBit);
        step[15] = HalfBtf( cospi[6],  bf1[14], -cospi[58], bf1[15], cosBit);

        // Stage 9: final scatter to output (libaom permutation).
        output[outBase + 0]  = step[1];
        output[outBase + 1]  = step[14];
        output[outBase + 2]  = step[3];
        output[outBase + 3]  = step[12];
        output[outBase + 4]  = step[5];
        output[outBase + 5]  = step[10];
        output[outBase + 6]  = step[7];
        output[outBase + 7]  = step[8];
        output[outBase + 8]  = step[9];
        output[outBase + 9]  = step[6];
        output[outBase + 10] = step[11];
        output[outBase + 11] = step[4];
        output[outBase + 12] = step[13];
        output[outBase + 13] = step[2];
        output[outBase + 14] = step[15];
        output[outBase + 15] = step[0];
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
