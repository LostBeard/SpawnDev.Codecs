// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 8-point inverse Asymmetric DST (1D), GPU-callable form for
// in-kernel reuse. Bit-exact mirror of Av1InverseAdst8.Transform
// (libaom av1/common/av1_inv_txfm1d.c av1_iadst8 port).
//
// Pairs with Av1ForwardAdst8Gpu (encoder-side) - now AV1 8-point ADST
// has GPU primitives in both directions.
//
// 7 stages with cospi-driven half_btf rotations + final permutation
// with sign flips.

using ILGPU;

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>
/// GPU-callable AV1 8-point inverse ADST helper. Bit-exact mirror of
/// <see cref="Av1InverseAdst8"/> for in-kernel use.
/// </summary>
public static class Av1InverseAdst8Gpu
{
    /// <summary>Default cosine-precision bits (libaom).</summary>
    public const int DefaultCosBit = Av1InverseAdst8.DefaultCosBit;

    /// <summary>
    /// Apply the 8-point inverse ADST to one 8-element 1D row/column.
    /// Reads <paramref name="input"/> starting at <paramref name="inBase"/>;
    /// writes 8 ints to <paramref name="output"/> starting at
    /// <paramref name="outBase"/>.
    /// </summary>
    public static void Inverse8(
        ArrayView<int> input, long inBase,
        ArrayView<int> output, long outBase,
        int cosBit)
    {
        ResolveCospi(cosBit, out int c4, out int c12, out int c16, out int c20,
            out int c28, out int c32, out int c36, out int c44, out int c48,
            out int c52, out int c60);

        // Stage 1: input permutation.
        int bf0 = input[inBase + 7];
        int bf1 = input[inBase + 0];
        int bf2 = input[inBase + 5];
        int bf3 = input[inBase + 2];
        int bf4 = input[inBase + 3];
        int bf5 = input[inBase + 4];
        int bf6 = input[inBase + 1];
        int bf7 = input[inBase + 6];

        // Stage 2: cospi rotations (half_btf: w0*in0 + w1*in1, rounded down by cosBit).
        int s0 = HalfBtf(c4, bf0, c60, bf1, cosBit);
        int s1 = HalfBtf(c60, bf0, -c4, bf1, cosBit);
        int s2 = HalfBtf(c20, bf2, c44, bf3, cosBit);
        int s3 = HalfBtf(c44, bf2, -c20, bf3, cosBit);
        int s4 = HalfBtf(c36, bf4, c28, bf5, cosBit);
        int s5 = HalfBtf(c28, bf4, -c36, bf5, cosBit);
        int s6 = HalfBtf(c52, bf6, c12, bf7, cosBit);
        int s7 = HalfBtf(c12, bf6, -c52, bf7, cosBit);

        // Stage 3: butterfly (lower vs upper half).
        bf0 = s0 + s4;
        bf1 = s1 + s5;
        bf2 = s2 + s6;
        bf3 = s3 + s7;
        bf4 = s0 - s4;
        bf5 = s1 - s5;
        bf6 = s2 - s6;
        bf7 = s3 - s7;

        // Stage 4: cospi rotations on upper 4.
        s0 = bf0;
        s1 = bf1;
        s2 = bf2;
        s3 = bf3;
        s4 = HalfBtf(c16, bf4, c48, bf5, cosBit);
        s5 = HalfBtf(c48, bf4, -c16, bf5, cosBit);
        s6 = HalfBtf(-c48, bf6, c16, bf7, cosBit);
        s7 = HalfBtf(c16, bf6, c48, bf7, cosBit);

        // Stage 5: butterfly across 4-strides.
        bf0 = s0 + s2;
        bf1 = s1 + s3;
        bf2 = s0 - s2;
        bf3 = s1 - s3;
        bf4 = s4 + s6;
        bf5 = s5 + s7;
        bf6 = s4 - s6;
        bf7 = s5 - s7;

        // Stage 6: cospi rotation on a few entries.
        s0 = bf0;
        s1 = bf1;
        s2 = HalfBtf(c32, bf2, c32, bf3, cosBit);
        s3 = HalfBtf(c32, bf2, -c32, bf3, cosBit);
        s4 = bf4;
        s5 = bf5;
        s6 = HalfBtf(c32, bf6, c32, bf7, cosBit);
        s7 = HalfBtf(c32, bf6, -c32, bf7, cosBit);

        // Stage 7: final permutation with sign flips.
        output[outBase + 0] = s0;
        output[outBase + 1] = -s4;
        output[outBase + 2] = s6;
        output[outBase + 3] = -s2;
        output[outBase + 4] = s3;
        output[outBase + 5] = -s7;
        output[outBase + 6] = s5;
        output[outBase + 7] = -s1;
    }

    /// <summary>libaom <c>half_btf</c>: kernel-safe variant.</summary>
    private static int HalfBtf(int w0, int in0, int w1, int in1, int bit)
    {
        long result = (long)w0 * in0 + (long)w1 * in1;
        result += 1L << (bit - 1);
        return (int)(result >> bit);
    }

    /// <summary>Resolves the 11 cospi entries iadst8 needs (same as fadst8).</summary>
    private static void ResolveCospi(int cosBit,
        out int c4, out int c12, out int c16, out int c20, out int c28,
        out int c32, out int c36, out int c44, out int c48, out int c52,
        out int c60)
    {
        if (cosBit == 13)
        {
            c4 = 8153; c12 = 7839; c16 = 7568; c20 = 7225; c28 = 6333;
            c32 = 5793; c36 = 5197; c44 = 3862; c48 = 3135; c52 = 2378; c60 = 803;
        }
        else if (cosBit == 12)
        {
            c4 = 4076; c12 = 3920; c16 = 3784; c20 = 3612; c28 = 3166;
            c32 = 2896; c36 = 2598; c44 = 1931; c48 = 1567; c52 = 1189; c60 = 401;
        }
        else if (cosBit == 11)
        {
            c4 = 2038; c12 = 1960; c16 = 1892; c20 = 1806; c28 = 1583;
            c32 = 1448; c36 = 1299; c44 = 965;  c48 = 784;  c52 = 595;  c60 = 201;
        }
        else
        {
            c4 = 1019; c12 = 980;  c16 = 946;  c20 = 903;  c28 = 792;
            c32 = 724;  c36 = 650;  c44 = 483;  c48 = 392;  c52 = 297;  c60 = 100;
        }
    }
}
