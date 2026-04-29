// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 16-point inverse Asymmetric DST (1D), GPU-callable form for
// in-kernel reuse. Bit-exact mirror of Av1InverseAdst16.Transform
// (libaom av1/common/av1_inv_txfm1d.c av1_iadst16 port).
//
// Pairs with Av1ForwardAdst16Gpu (encoder-side) - now AV1 16-point ADST
// has GPU primitives in both directions.
//
// 9 stages with cospi-driven half_btf rotations + final permutation
// with sign flips.

using ILGPU;

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>
/// GPU-callable AV1 16-point inverse ADST helper. Bit-exact mirror of
/// <see cref="Av1InverseAdst16"/> for in-kernel use.
/// </summary>
public static class Av1InverseAdst16Gpu
{
    /// <summary>Default cosine-precision bits (libaom).</summary>
    public const int DefaultCosBit = Av1InverseAdst16.DefaultCosBit;

    /// <summary>
    /// Apply the 16-point inverse ADST to one 16-element 1D row/column.
    /// </summary>
    public static void Inverse16(
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

        // Stage 1: input permutation.
        int b0  = input[inBase + 15];
        int b1  = input[inBase + 0];
        int b2  = input[inBase + 13];
        int b3  = input[inBase + 2];
        int b4  = input[inBase + 11];
        int b5  = input[inBase + 4];
        int b6  = input[inBase + 9];
        int b7  = input[inBase + 6];
        int b8  = input[inBase + 7];
        int b9  = input[inBase + 8];
        int b10 = input[inBase + 5];
        int b11 = input[inBase + 10];
        int b12 = input[inBase + 3];
        int b13 = input[inBase + 12];
        int b14 = input[inBase + 1];
        int b15 = input[inBase + 14];

        // Stage 2.
        int s0  = HalfBtf(c2,  b0, c62, b1, cosBit);
        int s1  = HalfBtf(c62, b0, -c2, b1, cosBit);
        int s2  = HalfBtf(c10, b2, c54, b3, cosBit);
        int s3  = HalfBtf(c54, b2, -c10, b3, cosBit);
        int s4  = HalfBtf(c18, b4, c46, b5, cosBit);
        int s5  = HalfBtf(c46, b4, -c18, b5, cosBit);
        int s6  = HalfBtf(c26, b6, c38, b7, cosBit);
        int s7  = HalfBtf(c38, b6, -c26, b7, cosBit);
        int s8  = HalfBtf(c34, b8, c30, b9, cosBit);
        int s9  = HalfBtf(c30, b8, -c34, b9, cosBit);
        int s10 = HalfBtf(c42, b10, c22, b11, cosBit);
        int s11 = HalfBtf(c22, b10, -c42, b11, cosBit);
        int s12 = HalfBtf(c50, b12, c14, b13, cosBit);
        int s13 = HalfBtf(c14, b12, -c50, b13, cosBit);
        int s14 = HalfBtf(c58, b14, c6, b15, cosBit);
        int s15 = HalfBtf(c6, b14, -c58, b15, cosBit);

        // Stage 3: butterfly between 0..7 and 8..15.
        b0  = s0  + s8;
        b1  = s1  + s9;
        b2  = s2  + s10;
        b3  = s3  + s11;
        b4  = s4  + s12;
        b5  = s5  + s13;
        b6  = s6  + s14;
        b7  = s7  + s15;
        b8  = s0  - s8;
        b9  = s1  - s9;
        b10 = s2  - s10;
        b11 = s3  - s11;
        b12 = s4  - s12;
        b13 = s5  - s13;
        b14 = s6  - s14;
        b15 = s7  - s15;

        // Stage 4.
        s0 = b0;  s1 = b1;  s2 = b2;  s3 = b3;
        s4 = b4;  s5 = b5;  s6 = b6;  s7 = b7;
        s8  = HalfBtf(c8,  b8, c56, b9, cosBit);
        s9  = HalfBtf(c56, b8, -c8, b9, cosBit);
        s10 = HalfBtf(c40, b10, c24, b11, cosBit);
        s11 = HalfBtf(c24, b10, -c40, b11, cosBit);
        s12 = HalfBtf(-c56, b12, c8, b13, cosBit);
        s13 = HalfBtf(c8, b12, c56, b13, cosBit);
        s14 = HalfBtf(-c24, b14, c40, b15, cosBit);
        s15 = HalfBtf(c40, b14, c24, b15, cosBit);

        // Stage 5.
        b0  = s0 + s4;
        b1  = s1 + s5;
        b2  = s2 + s6;
        b3  = s3 + s7;
        b4  = s0 - s4;
        b5  = s1 - s5;
        b6  = s2 - s6;
        b7  = s3 - s7;
        b8  = s8  + s12;
        b9  = s9  + s13;
        b10 = s10 + s14;
        b11 = s11 + s15;
        b12 = s8  - s12;
        b13 = s9  - s13;
        b14 = s10 - s14;
        b15 = s11 - s15;

        // Stage 6.
        s0 = b0;  s1 = b1;  s2 = b2;  s3 = b3;
        s4  = HalfBtf(c16, b4, c48, b5, cosBit);
        s5  = HalfBtf(c48, b4, -c16, b5, cosBit);
        s6  = HalfBtf(-c48, b6, c16, b7, cosBit);
        s7  = HalfBtf(c16, b6, c48, b7, cosBit);
        s8  = b8;  s9  = b9;  s10 = b10; s11 = b11;
        s12 = HalfBtf(c16, b12, c48, b13, cosBit);
        s13 = HalfBtf(c48, b12, -c16, b13, cosBit);
        s14 = HalfBtf(-c48, b14, c16, b15, cosBit);
        s15 = HalfBtf(c16, b14, c48, b15, cosBit);

        // Stage 7.
        b0  = s0 + s2;
        b1  = s1 + s3;
        b2  = s0 - s2;
        b3  = s1 - s3;
        b4  = s4 + s6;
        b5  = s5 + s7;
        b6  = s4 - s6;
        b7  = s5 - s7;
        b8  = s8  + s10;
        b9  = s9  + s11;
        b10 = s8  - s10;
        b11 = s9  - s11;
        b12 = s12 + s14;
        b13 = s13 + s15;
        b14 = s12 - s14;
        b15 = s13 - s15;

        // Stage 8.
        s0 = b0;  s1 = b1;
        s2 = HalfBtf(c32, b2, c32, b3, cosBit);
        s3 = HalfBtf(c32, b2, -c32, b3, cosBit);
        s4 = b4; s5 = b5;
        s6 = HalfBtf(c32, b6, c32, b7, cosBit);
        s7 = HalfBtf(c32, b6, -c32, b7, cosBit);
        s8 = b8; s9 = b9;
        s10 = HalfBtf(c32, b10, c32, b11, cosBit);
        s11 = HalfBtf(c32, b10, -c32, b11, cosBit);
        s12 = b12; s13 = b13;
        s14 = HalfBtf(c32, b14, c32, b15, cosBit);
        s15 = HalfBtf(c32, b14, -c32, b15, cosBit);

        // Stage 9: final permutation with sign flips.
        output[outBase + 0]  =  s0;
        output[outBase + 1]  = -s8;
        output[outBase + 2]  =  s12;
        output[outBase + 3]  = -s4;
        output[outBase + 4]  =  s6;
        output[outBase + 5]  = -s14;
        output[outBase + 6]  =  s10;
        output[outBase + 7]  = -s2;
        output[outBase + 8]  =  s3;
        output[outBase + 9]  = -s11;
        output[outBase + 10] =  s15;
        output[outBase + 11] = -s7;
        output[outBase + 12] =  s5;
        output[outBase + 13] = -s13;
        output[outBase + 14] =  s9;
        output[outBase + 15] = -s1;
    }

    /// <summary>libaom <c>half_btf</c>: kernel-safe variant.</summary>
    private static int HalfBtf(int w0, int in0, int w1, int in1, int bit)
    {
        long result = (long)w0 * in0 + (long)w1 * in1;
        result += 1L << (bit - 1);
        return (int)(result >> bit);
    }

    /// <summary>Resolves the 23 cospi entries iadst16 needs.</summary>
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
