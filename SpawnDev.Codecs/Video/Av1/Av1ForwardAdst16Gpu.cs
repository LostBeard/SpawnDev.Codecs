// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 16-point forward Asymmetric DST (1D), GPU-callable form for
// in-kernel reuse. Bit-exact mirror of Av1ForwardAdst16.Transform
// (libaom av1/encoder/av1_fwd_txfm1d.c av1_fadst16 port).
//
// Av1ForwardAdst16Kernel uses LocalMemory<int>(64) for the cospi
// table + LocalMemory<int>(16) twice for step/bf1. This static
// helper translates those into scalar locals (32 for step+bf1, 23
// for the cospi entries fadst16 actually needs). 55 ints fit
// comfortably in registers across every backend.
//
// 9 stages with cospi-driven half_btf rotations + final scatter.

using ILGPU;

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>
/// GPU-callable AV1 16-point forward ADST helper. Bit-exact mirror of
/// <see cref="Av1ForwardAdst16"/> for in-kernel use.
/// </summary>
public static class Av1ForwardAdst16Gpu
{
    /// <summary>Default cosine-precision bits (libaom).</summary>
    public const int DefaultCosBit = Av1ForwardAdst16.DefaultCosBit;

    /// <summary>
    /// Apply the 16-point forward ADST to one 16-element 1D row/column.
    /// Reads <paramref name="input"/> starting at <paramref name="inBase"/>;
    /// writes 16 ints to <paramref name="output"/> starting at
    /// <paramref name="outBase"/>.
    /// </summary>
    public static void Forward16(
        ArrayView<int> input, long inBase,
        ArrayView<int> output, long outBase,
        int cosBit)
    {
        ResolveCospi(cosBit,
            out int c2,  out int c6,  out int c8,  out int c10,
            out int c14, out int c16, out int c18, out int c22,
            out int c24, out int c26, out int c30, out int c32,
            out int c34, out int c38, out int c40, out int c42,
            out int c46, out int c48, out int c50, out int c54,
            out int c56, out int c58, out int c62);

        // Stage 1: input remap with sign flips.
        int b0  =  input[inBase + 0];
        int b1  = -input[inBase + 15];
        int b2  = -input[inBase + 7];
        int b3  =  input[inBase + 8];
        int b4  = -input[inBase + 3];
        int b5  =  input[inBase + 12];
        int b6  =  input[inBase + 4];
        int b7  = -input[inBase + 11];
        int b8  = -input[inBase + 1];
        int b9  =  input[inBase + 14];
        int b10 =  input[inBase + 6];
        int b11 = -input[inBase + 9];
        int b12 =  input[inBase + 2];
        int b13 = -input[inBase + 13];
        int b14 = -input[inBase + 5];
        int b15 =  input[inBase + 10];

        // Stage 2: cospi[32] rotations on (2,3), (6,7), (10,11), (14,15).
        int s0  = b0;
        int s1  = b1;
        int s2  = HalfBtf(c32, b2,  c32, b3, cosBit);
        int s3  = HalfBtf(c32, b2, -c32, b3, cosBit);
        int s4  = b4;
        int s5  = b5;
        int s6  = HalfBtf(c32, b6,  c32, b7, cosBit);
        int s7  = HalfBtf(c32, b6, -c32, b7, cosBit);
        int s8  = b8;
        int s9  = b9;
        int s10 = HalfBtf(c32, b10,  c32, b11, cosBit);
        int s11 = HalfBtf(c32, b10, -c32, b11, cosBit);
        int s12 = b12;
        int s13 = b13;
        int s14 = HalfBtf(c32, b14,  c32, b15, cosBit);
        int s15 = HalfBtf(c32, b14, -c32, b15, cosBit);

        // Stage 3: butterfly 4-element groups.
        b0  = s0  + s2;
        b1  = s1  + s3;
        b2  = s0  - s2;
        b3  = s1  - s3;
        b4  = s4  + s6;
        b5  = s5  + s7;
        b6  = s4  - s6;
        b7  = s5  - s7;
        b8  = s8  + s10;
        b9  = s9  + s11;
        b10 = s8  - s10;
        b11 = s9  - s11;
        b12 = s12 + s14;
        b13 = s13 + s15;
        b14 = s12 - s14;
        b15 = s13 - s15;

        // Stage 4: cospi[16/48] on (4,5), (6,7), (12,13), (14,15).
        s0  = b0;
        s1  = b1;
        s2  = b2;
        s3  = b3;
        s4  = HalfBtf( c16, b4,  c48, b5, cosBit);
        s5  = HalfBtf( c48, b4, -c16, b5, cosBit);
        s6  = HalfBtf(-c48, b6,  c16, b7, cosBit);
        s7  = HalfBtf( c16, b6,  c48, b7, cosBit);
        s8  = b8;
        s9  = b9;
        s10 = b10;
        s11 = b11;
        s12 = HalfBtf( c16, b12,  c48, b13, cosBit);
        s13 = HalfBtf( c48, b12, -c16, b13, cosBit);
        s14 = HalfBtf(-c48, b14,  c16, b15, cosBit);
        s15 = HalfBtf( c16, b14,  c48, b15, cosBit);

        // Stage 5: butterfly across halves.
        b0  = s0  + s4;
        b1  = s1  + s5;
        b2  = s2  + s6;
        b3  = s3  + s7;
        b4  = s0  - s4;
        b5  = s1  - s5;
        b6  = s2  - s6;
        b7  = s3  - s7;
        b8  = s8  + s12;
        b9  = s9  + s13;
        b10 = s10 + s14;
        b11 = s11 + s15;
        b12 = s8  - s12;
        b13 = s9  - s13;
        b14 = s10 - s14;
        b15 = s11 - s15;

        // Stage 6: cospi[8/56/40/24] rotations on the upper 8.
        s0  = b0;
        s1  = b1;
        s2  = b2;
        s3  = b3;
        s4  = b4;
        s5  = b5;
        s6  = b6;
        s7  = b7;
        s8  = HalfBtf( c8,  b8,   c56, b9,  cosBit);
        s9  = HalfBtf( c56, b8,  -c8,  b9,  cosBit);
        s10 = HalfBtf( c40, b10,  c24, b11, cosBit);
        s11 = HalfBtf( c24, b10, -c40, b11, cosBit);
        s12 = HalfBtf(-c56, b12,  c8,  b13, cosBit);
        s13 = HalfBtf( c8,  b12,  c56, b13, cosBit);
        s14 = HalfBtf(-c24, b14,  c40, b15, cosBit);
        s15 = HalfBtf( c40, b14,  c24, b15, cosBit);

        // Stage 7: butterfly across full 16-element width.
        b0  = s0 + s8;
        b1  = s1 + s9;
        b2  = s2 + s10;
        b3  = s3 + s11;
        b4  = s4 + s12;
        b5  = s5 + s13;
        b6  = s6 + s14;
        b7  = s7 + s15;
        b8  = s0 - s8;
        b9  = s1 - s9;
        b10 = s2 - s10;
        b11 = s3 - s11;
        b12 = s4 - s12;
        b13 = s5 - s13;
        b14 = s6 - s14;
        b15 = s7 - s15;

        // Stage 8: cospi[2/62/10/54/18/46/26/38/34/30/42/22/50/14/58/6] rotations.
        s0  = HalfBtf( c2,  b0,   c62, b1,  cosBit);
        s1  = HalfBtf( c62, b0,  -c2,  b1,  cosBit);
        s2  = HalfBtf( c10, b2,   c54, b3,  cosBit);
        s3  = HalfBtf( c54, b2,  -c10, b3,  cosBit);
        s4  = HalfBtf( c18, b4,   c46, b5,  cosBit);
        s5  = HalfBtf( c46, b4,  -c18, b5,  cosBit);
        s6  = HalfBtf( c26, b6,   c38, b7,  cosBit);
        s7  = HalfBtf( c38, b6,  -c26, b7,  cosBit);
        s8  = HalfBtf( c34, b8,   c30, b9,  cosBit);
        s9  = HalfBtf( c30, b8,  -c34, b9,  cosBit);
        s10 = HalfBtf( c42, b10,  c22, b11, cosBit);
        s11 = HalfBtf( c22, b10, -c42, b11, cosBit);
        s12 = HalfBtf( c50, b12,  c14, b13, cosBit);
        s13 = HalfBtf( c14, b12, -c50, b13, cosBit);
        s14 = HalfBtf( c58, b14,  c6,  b15, cosBit);
        s15 = HalfBtf( c6,  b14, -c58, b15, cosBit);

        // Stage 9: final scatter to output (libaom permutation).
        output[outBase + 0]  = s1;
        output[outBase + 1]  = s14;
        output[outBase + 2]  = s3;
        output[outBase + 3]  = s12;
        output[outBase + 4]  = s5;
        output[outBase + 5]  = s10;
        output[outBase + 6]  = s7;
        output[outBase + 7]  = s8;
        output[outBase + 8]  = s9;
        output[outBase + 9]  = s6;
        output[outBase + 10] = s11;
        output[outBase + 11] = s4;
        output[outBase + 12] = s13;
        output[outBase + 13] = s2;
        output[outBase + 14] = s15;
        output[outBase + 15] = s0;
    }

    /// <summary>libaom <c>half_btf</c>: kernel-safe variant.</summary>
    private static int HalfBtf(int w0, int in0, int w1, int in1, int bit)
    {
        long result = (long)w0 * in0 + (long)w1 * in1;
        result += 1L << (bit - 1);
        return (int)(result >> bit);
    }

    /// <summary>
    /// Resolves the 23 cospi entries fadst16 needs (every multiple
    /// of 8 in the low half: 8, 16, 24, 32, 40, 48, 56; every
    /// multiple of 4 in the high half: 2, 6, 10, 14, 18, 22, 26,
    /// 30, 34, 38, 42, 46, 50, 54, 58, 62). Inlined as branches so
    /// the kernel does not have to read a 64-element table buffer.
    /// </summary>
    private static void ResolveCospi(int cosBit,
        out int c2,  out int c6,  out int c8,  out int c10,
        out int c14, out int c16, out int c18, out int c22,
        out int c24, out int c26, out int c30, out int c32,
        out int c34, out int c38, out int c40, out int c42,
        out int c46, out int c48, out int c50, out int c54,
        out int c56, out int c58, out int c62)
    {
        if (cosBit == 13)
        {
            c2  = 8182; c6  = 8103; c8  = 8035; c10 = 7946;
            c14 = 7713; c16 = 7568; c18 = 7405; c22 = 7027;
            c24 = 6811; c26 = 6580; c30 = 6070; c32 = 5793;
            c34 = 5501; c38 = 4880; c40 = 4551; c42 = 4212;
            c46 = 3503; c48 = 3135; c50 = 2760; c54 = 1990;
            c56 = 1598; c58 = 1202; c62 = 402;
        }
        else if (cosBit == 12)
        {
            c2  = 4091; c6  = 4052; c8  = 4017; c10 = 3973;
            c14 = 3857; c16 = 3784; c18 = 3703; c22 = 3513;
            c24 = 3406; c26 = 3290; c30 = 3035; c32 = 2896;
            c34 = 2751; c38 = 2440; c40 = 2276; c42 = 2106;
            c46 = 1751; c48 = 1567; c50 = 1380; c54 = 995;
            c56 = 799;  c58 = 601;  c62 = 201;
        }
        else if (cosBit == 11)
        {
            c2  = 2046; c6  = 2026; c8  = 2009; c10 = 1987;
            c14 = 1928; c16 = 1892; c18 = 1851; c22 = 1757;
            c24 = 1703; c26 = 1645; c30 = 1517; c32 = 1448;
            c34 = 1375; c38 = 1220; c40 = 1138; c42 = 1053;
            c46 = 876;  c48 = 784;  c50 = 690;  c54 = 498;
            c56 = 400;  c58 = 301;  c62 = 100;
        }
        else
        {
            c2  = 1023; c6  = 1013; c8  = 1004; c10 = 993;
            c14 = 964;  c16 = 946;  c18 = 926;  c22 = 878;
            c24 = 851;  c26 = 822;  c30 = 759;  c32 = 724;
            c34 = 688;  c38 = 610;  c40 = 569;  c42 = 526;
            c46 = 438;  c48 = 392;  c50 = 345;  c54 = 249;
            c56 = 200;  c58 = 150;  c62 = 50;
        }
    }
}
