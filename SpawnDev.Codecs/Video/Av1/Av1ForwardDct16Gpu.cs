// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 16-point forward DCT (1D), GPU-callable form for in-kernel
// reuse. Bit-exact mirror of Av1ForwardDct16.Transform (libaom
// av1/encoder/av1_fwd_txfm1d.c av1_fdct16 port).
//
// Av1ForwardDct16Kernel already wraps this as a standalone batched
// dispatch using LocalMemory<int>(16) twice for the stage scratch
// buffers. This static helper must be callable from inside another
// kernel (Av1FrameSequentialEncodeKernel), so it expands the two
// 16-int scratch arrays as 32 scalar locals (a0..a15 + b0..b15).
// 32 ints fit comfortably in registers on every backend and avoids
// the LocalMemory-must-be-kernel-scope restriction.
//
// 7 stages of butterfly + cospi multiplications + final scatter.

using ILGPU;

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>
/// GPU-callable AV1 16-point forward DCT helper. Bit-exact mirror of
/// <see cref="Av1ForwardDct16"/> for in-kernel use.
/// </summary>
public static class Av1ForwardDct16Gpu
{
    /// <summary>Default cosine-precision bits (libaom).</summary>
    public const int DefaultCosBit = Av1ForwardDct16.DefaultCosBit;

    /// <summary>
    /// Apply the 16-point forward DCT to one 16-element 1D row/column.
    /// Reads <paramref name="input"/> starting at <paramref name="inBase"/>;
    /// writes 16 ints to <paramref name="output"/> starting at
    /// <paramref name="outBase"/>.
    /// </summary>
    public static void Forward16(
        ArrayView<int> input, long inBase,
        ArrayView<int> output, long outBase,
        int cosBit)
    {
        ResolveCospi(cosBit, out int c4, out int c8, out int c12, out int c16,
            out int c20, out int c24, out int c28, out int c32, out int c36,
            out int c40, out int c44, out int c48, out int c52, out int c56,
            out int c60);

        int in0  = input[inBase + 0];
        int in1  = input[inBase + 1];
        int in2  = input[inBase + 2];
        int in3  = input[inBase + 3];
        int in4  = input[inBase + 4];
        int in5  = input[inBase + 5];
        int in6  = input[inBase + 6];
        int in7  = input[inBase + 7];
        int in8  = input[inBase + 8];
        int in9  = input[inBase + 9];
        int in10 = input[inBase + 10];
        int in11 = input[inBase + 11];
        int in12 = input[inBase + 12];
        int in13 = input[inBase + 13];
        int in14 = input[inBase + 14];
        int in15 = input[inBase + 15];

        // Stage 1
        int a0  =  in0  + in15;
        int a1  =  in1  + in14;
        int a2  =  in2  + in13;
        int a3  =  in3  + in12;
        int a4  =  in4  + in11;
        int a5  =  in5  + in10;
        int a6  =  in6  + in9;
        int a7  =  in7  + in8;
        int a8  = -in8  + in7;
        int a9  = -in9  + in6;
        int a10 = -in10 + in5;
        int a11 = -in11 + in4;
        int a12 = -in12 + in3;
        int a13 = -in13 + in2;
        int a14 = -in14 + in1;
        int a15 = -in15 + in0;

        // Stage 2
        int b0  =  a0 + a7;
        int b1  =  a1 + a6;
        int b2  =  a2 + a5;
        int b3  =  a3 + a4;
        int b4  = -a4 + a3;
        int b5  = -a5 + a2;
        int b6  = -a6 + a1;
        int b7  = -a7 + a0;
        int b8  =  a8;
        int b9  =  a9;
        int b10 = HalfBtf(-c32, a10,  c32, a13, cosBit);
        int b11 = HalfBtf(-c32, a11,  c32, a12, cosBit);
        int b12 = HalfBtf( c32, a12,  c32, a11, cosBit);
        int b13 = HalfBtf( c32, a13,  c32, a10, cosBit);
        int b14 =  a14;
        int b15 =  a15;

        // Stage 3
        a0  =  b0 + b3;
        a1  =  b1 + b2;
        a2  = -b2 + b1;
        a3  = -b3 + b0;
        a4  =  b4;
        a5  = HalfBtf(-c32, b5,  c32, b6, cosBit);
        a6  = HalfBtf( c32, b6,  c32, b5, cosBit);
        a7  =  b7;
        a8  =  b8 + b11;
        a9  =  b9 + b10;
        a10 = -b10 + b9;
        a11 = -b11 + b8;
        a12 = -b12 + b15;
        a13 = -b13 + b14;
        a14 =  b14 + b13;
        a15 =  b15 + b12;

        // Stage 4
        b0  = HalfBtf( c32, a0,  c32, a1, cosBit);
        b1  = HalfBtf(-c32, a1,  c32, a0, cosBit);
        b2  = HalfBtf( c48, a2,  c16, a3, cosBit);
        b3  = HalfBtf( c48, a3, -c16, a2, cosBit);
        b4  =  a4 + a5;
        b5  = -a5 + a4;
        b6  = -a6 + a7;
        b7  =  a7 + a6;
        b8  =  a8;
        b9  = HalfBtf(-c16, a9,   c48, a14, cosBit);
        b10 = HalfBtf(-c48, a10, -c16, a13, cosBit);
        b11 =  a11;
        b12 =  a12;
        b13 = HalfBtf( c48, a13, -c16, a10, cosBit);
        b14 = HalfBtf( c16, a14,  c48, a9,  cosBit);
        b15 =  a15;

        // Stage 5
        a0  =  b0;
        a1  =  b1;
        a2  =  b2;
        a3  =  b3;
        a4  = HalfBtf( c56, b4,  c8,  b7, cosBit);
        a5  = HalfBtf( c24, b5,  c40, b6, cosBit);
        a6  = HalfBtf( c24, b6, -c40, b5, cosBit);
        a7  = HalfBtf( c56, b7, -c8,  b4, cosBit);
        a8  =  b8 + b9;
        a9  = -b9 + b8;
        a10 = -b10 + b11;
        a11 =  b11 + b10;
        a12 =  b12 + b13;
        a13 = -b13 + b12;
        a14 = -b14 + b15;
        a15 =  b15 + b14;

        // Stage 6
        b0  = a0;  b1  = a1;  b2  = a2;  b3  = a3;
        b4  = a4;  b5  = a5;  b6  = a6;  b7  = a7;
        b8  = HalfBtf( c60, a8,   c4,  a15, cosBit);
        b9  = HalfBtf( c28, a9,   c36, a14, cosBit);
        b10 = HalfBtf( c44, a10,  c20, a13, cosBit);
        b11 = HalfBtf( c12, a11,  c52, a12, cosBit);
        b12 = HalfBtf( c12, a12, -c52, a11, cosBit);
        b13 = HalfBtf( c44, a13, -c20, a10, cosBit);
        b14 = HalfBtf( c28, a14, -c36, a9,  cosBit);
        b15 = HalfBtf( c60, a15, -c4,  a8,  cosBit);

        // Stage 7 (interleave / scatter)
        output[outBase + 0]  = b0;
        output[outBase + 1]  = b8;
        output[outBase + 2]  = b4;
        output[outBase + 3]  = b12;
        output[outBase + 4]  = b2;
        output[outBase + 5]  = b10;
        output[outBase + 6]  = b6;
        output[outBase + 7]  = b14;
        output[outBase + 8]  = b1;
        output[outBase + 9]  = b9;
        output[outBase + 10] = b5;
        output[outBase + 11] = b13;
        output[outBase + 12] = b3;
        output[outBase + 13] = b11;
        output[outBase + 14] = b7;
        output[outBase + 15] = b15;
    }

    /// <summary>libaom <c>half_btf</c>: kernel-safe variant.</summary>
    private static int HalfBtf(int w0, int in0, int w1, int in1, int bit)
    {
        long result = (long)w0 * in0 + (long)w1 * in1;
        result += 1L << (bit - 1);
        return (int)(result >> bit);
    }

    /// <summary>
    /// Resolves the 15 cospi entries fdct16 needs (every multiple of 4
    /// from 4..60). Inlined as branches so the kernel does not have to
    /// read a 64-element table buffer.
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
