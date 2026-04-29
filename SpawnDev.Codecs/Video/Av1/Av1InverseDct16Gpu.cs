// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 16-point inverse DCT (1D), GPU-callable form for in-kernel
// reuse. Bit-exact mirror of Av1InverseDct16.Transform (libaom
// av1/common/av1_inv_txfm1d.c av1_idct16 port).
//
// Static helper exists so the upcoming Av1KeyframeDecodeKernel +
// Av1FrameSequentialEncodeKernel (recon path) can run the 1D inverse
// DCT16 inside a per-block 2D transform call chain without
// dispatching a separate kernel boundary.
//
// 7 stages of butterfly + cospi multiplications + final outer
// butterfly. Same 32-scalar-locals pattern as Av1ForwardDct16Gpu -
// 32 ints fit in registers on every backend.
//
// Default cos_bit = 12 (libaom inverse default).

using ILGPU;

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>
/// GPU-callable AV1 16-point inverse DCT helper. Bit-exact mirror of
/// <see cref="Av1InverseDct16"/> for in-kernel use.
/// </summary>
public static class Av1InverseDct16Gpu
{
    /// <summary>libaom default cos_bit for the inverse 16-point DCT.</summary>
    public const int DefaultCosBit = Av1InverseDct16.DefaultCosBit;

    /// <summary>
    /// Apply the 16-point inverse DCT to one 16-element 1D row/column.
    /// Reads <paramref name="input"/> starting at <paramref name="inBase"/>;
    /// writes 16 ints to <paramref name="output"/> starting at
    /// <paramref name="outBase"/>.
    /// </summary>
    public static void Inverse16(
        ArrayView<int> input, long inBase,
        ArrayView<int> output, long outBase,
        int cosBit)
    {
        ResolveCospi(cosBit, out int c4, out int c8, out int c12, out int c16,
            out int c20, out int c24, out int c28, out int c32, out int c36,
            out int c40, out int c44, out int c48, out int c52, out int c56,
            out int c60);

        // Stage 1: input permutation (bit reverse).
        int bf0  = input[inBase + 0];
        int bf1  = input[inBase + 8];
        int bf2  = input[inBase + 4];
        int bf3  = input[inBase + 12];
        int bf4  = input[inBase + 2];
        int bf5  = input[inBase + 10];
        int bf6  = input[inBase + 6];
        int bf7  = input[inBase + 14];
        int bf8  = input[inBase + 1];
        int bf9  = input[inBase + 9];
        int bf10 = input[inBase + 5];
        int bf11 = input[inBase + 13];
        int bf12 = input[inBase + 3];
        int bf13 = input[inBase + 11];
        int bf14 = input[inBase + 7];
        int bf15 = input[inBase + 15];

        // Stage 2: rotate upper half (8..15).
        int s0  = bf0;
        int s1  = bf1;
        int s2  = bf2;
        int s3  = bf3;
        int s4  = bf4;
        int s5  = bf5;
        int s6  = bf6;
        int s7  = bf7;
        int s8  = HalfBtf( c60, bf8,  -c4,  bf15, cosBit);
        int s9  = HalfBtf( c28, bf9,  -c36, bf14, cosBit);
        int s10 = HalfBtf( c44, bf10, -c20, bf13, cosBit);
        int s11 = HalfBtf( c12, bf11, -c52, bf12, cosBit);
        int s12 = HalfBtf( c52, bf11,  c12, bf12, cosBit);
        int s13 = HalfBtf( c20, bf10,  c44, bf13, cosBit);
        int s14 = HalfBtf( c36, bf9,   c28, bf14, cosBit);
        int s15 = HalfBtf( c4,  bf8,   c60, bf15, cosBit);

        // Stage 3: rotate middle (4..7), butterfly upper (8..15).
        bf0  = s0;
        bf1  = s1;
        bf2  = s2;
        bf3  = s3;
        bf4  = HalfBtf( c56, s4, -c8,  s7, cosBit);
        bf5  = HalfBtf( c24, s5, -c40, s6, cosBit);
        bf6  = HalfBtf( c40, s5,  c24, s6, cosBit);
        bf7  = HalfBtf( c8,  s4,  c56, s7, cosBit);
        bf8  =  s8  + s9;
        bf9  =  s8  - s9;
        bf10 = -s10 + s11;
        bf11 =  s10 + s11;
        bf12 =  s12 + s13;
        bf13 =  s12 - s13;
        bf14 = -s14 + s15;
        bf15 =  s14 + s15;

        // Stage 4
        s0  = HalfBtf( c32, bf0,  c32, bf1, cosBit);
        s1  = HalfBtf( c32, bf0, -c32, bf1, cosBit);
        s2  = HalfBtf( c48, bf2, -c16, bf3, cosBit);
        s3  = HalfBtf( c16, bf2,  c48, bf3, cosBit);
        s4  =  bf4 + bf5;
        s5  =  bf4 - bf5;
        s6  = -bf6 + bf7;
        s7  =  bf6 + bf7;
        s8  =  bf8;
        s9  = HalfBtf(-c16, bf9,   c48, bf14, cosBit);
        s10 = HalfBtf(-c48, bf10, -c16, bf13, cosBit);
        s11 =  bf11;
        s12 =  bf12;
        s13 = HalfBtf(-c16, bf10,  c48, bf13, cosBit);
        s14 = HalfBtf( c48, bf9,   c16, bf14, cosBit);
        s15 =  bf15;

        // Stage 5
        bf0  =  s0 + s3;
        bf1  =  s1 + s2;
        bf2  =  s1 - s2;
        bf3  =  s0 - s3;
        bf4  =  s4;
        bf5  = HalfBtf(-c32, s5, c32, s6, cosBit);
        bf6  = HalfBtf( c32, s5, c32, s6, cosBit);
        bf7  =  s7;
        bf8  =  s8  + s11;
        bf9  =  s9  + s10;
        bf10 =  s9  - s10;
        bf11 =  s8  - s11;
        bf12 = -s12 + s15;
        bf13 = -s13 + s14;
        bf14 =  s13 + s14;
        bf15 =  s12 + s15;

        // Stage 6
        s0  =  bf0 + bf7;
        s1  =  bf1 + bf6;
        s2  =  bf2 + bf5;
        s3  =  bf3 + bf4;
        s4  =  bf3 - bf4;
        s5  =  bf2 - bf5;
        s6  =  bf1 - bf6;
        s7  =  bf0 - bf7;
        s8  =  bf8;
        s9  =  bf9;
        s10 = HalfBtf(-c32, bf10, c32, bf13, cosBit);
        s11 = HalfBtf(-c32, bf11, c32, bf12, cosBit);
        s12 = HalfBtf( c32, bf11, c32, bf12, cosBit);
        s13 = HalfBtf( c32, bf10, c32, bf13, cosBit);
        s14 =  bf14;
        s15 =  bf15;

        // Stage 7: outer butterfly.
        output[outBase + 0]  = s0 + s15;
        output[outBase + 1]  = s1 + s14;
        output[outBase + 2]  = s2 + s13;
        output[outBase + 3]  = s3 + s12;
        output[outBase + 4]  = s4 + s11;
        output[outBase + 5]  = s5 + s10;
        output[outBase + 6]  = s6 + s9;
        output[outBase + 7]  = s7 + s8;
        output[outBase + 8]  = s7 - s8;
        output[outBase + 9]  = s6 - s9;
        output[outBase + 10] = s5 - s10;
        output[outBase + 11] = s4 - s11;
        output[outBase + 12] = s3 - s12;
        output[outBase + 13] = s2 - s13;
        output[outBase + 14] = s1 - s14;
        output[outBase + 15] = s0 - s15;
    }

    /// <summary>libaom <c>half_btf</c>: kernel-safe variant.</summary>
    private static int HalfBtf(int w0, int in0, int w1, int in1, int bit)
    {
        long result = (long)w0 * in0 + (long)w1 * in1;
        result += 1L << (bit - 1);
        return (int)(result >> bit);
    }

    /// <summary>
    /// Resolves the 15 cospi entries idct16 needs (every multiple of 4
    /// from 4..60). Same set as fdct16. Inlined as branches so the
    /// kernel does not have to read a 64-element table buffer.
    /// </summary>
    private static void ResolveCospi(int cosBit,
        out int c4, out int c8, out int c12, out int c16, out int c20,
        out int c24, out int c28, out int c32, out int c36, out int c40,
        out int c44, out int c48, out int c52, out int c56, out int c60)
    {
        if (cosBit == 13)
        {
            c4 = 8153; c8 = 8035; c12 = 7839; c16 = 7568; c20 = 7225;
            c24 = 6811; c28 = 6333; c32 = 5793; c36 = 5197; c40 = 4551;
            c44 = 3862; c48 = 3135; c52 = 2378; c56 = 1598; c60 = 803;
        }
        else if (cosBit == 12)
        {
            c4 = 4076; c8 = 4017; c12 = 3920; c16 = 3784; c20 = 3612;
            c24 = 3406; c28 = 3166; c32 = 2896; c36 = 2598; c40 = 2276;
            c44 = 1931; c48 = 1567; c52 = 1189; c56 = 799;  c60 = 401;
        }
        else if (cosBit == 11)
        {
            c4 = 2038; c8 = 2009; c12 = 1960; c16 = 1892; c20 = 1806;
            c24 = 1703; c28 = 1583; c32 = 1448; c36 = 1299; c40 = 1138;
            c44 = 965;  c48 = 784;  c52 = 595;  c56 = 400;  c60 = 201;
        }
        else
        {
            c4 = 1019; c8 = 1004; c12 = 980;  c16 = 946;  c20 = 903;
            c24 = 851;  c28 = 792; c32 = 724;  c36 = 650;  c40 = 569;
            c44 = 483;  c48 = 392; c52 = 297;  c56 = 200;  c60 = 100;
        }
    }
}
