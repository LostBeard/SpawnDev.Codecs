// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 8-point inverse DCT (1D), GPU-callable form for in-kernel
// reuse. Bit-exact mirror of Av1InverseDct8.Transform (libaom
// av1/common/av1_inv_txfm1d.c av1_idct8 port).
//
// Static helper exists so the upcoming Av1KeyframeDecodeKernel +
// Av1FrameSequentialEncodeKernel (recon path) can run the 1D inverse
// DCT8 inside a per-block 2D transform call chain without dispatching
// a separate kernel boundary - that's the v3 host-as-pure-coordinator
// pattern.
//
// 5 stages of butterfly + cospi multiplications.
// Default cos_bit = 12 (libaom inverse default).

using ILGPU;

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>
/// GPU-callable AV1 8-point inverse DCT helper. Bit-exact mirror of
/// <see cref="Av1InverseDct8"/> for in-kernel use.
/// </summary>
public static class Av1InverseDct8Gpu
{
    /// <summary>libaom default cos_bit for the inverse 8-point DCT.</summary>
    public const int DefaultCosBit = Av1InverseDct8.DefaultCosBit;

    /// <summary>
    /// Apply the 8-point inverse DCT to one 8-element 1D row/column.
    /// Reads <paramref name="input"/> starting at <paramref name="inBase"/>;
    /// writes 8 ints to <paramref name="output"/> starting at
    /// <paramref name="outBase"/>.
    /// </summary>
    public static void Inverse8(
        ArrayView<int> input, long inBase,
        ArrayView<int> output, long outBase,
        int cosBit)
    {
        ResolveCospi(cosBit, out int c8, out int c16, out int c24, out int c32,
            out int c40, out int c48, out int c56);

        // Stage 1: re-permute input.
        int bf0 = input[inBase + 0];
        int bf1 = input[inBase + 4];
        int bf2 = input[inBase + 2];
        int bf3 = input[inBase + 6];
        int bf4 = input[inBase + 1];
        int bf5 = input[inBase + 5];
        int bf6 = input[inBase + 3];
        int bf7 = input[inBase + 7];

        // Stage 2: cospi rotation on the upper half (4..7).
        int s0 = bf0;
        int s1 = bf1;
        int s2 = bf2;
        int s3 = bf3;
        int s4 = HalfBtf(c56, bf4, -c8,  bf7, cosBit);
        int s5 = HalfBtf(c24, bf5, -c40, bf6, cosBit);
        int s6 = HalfBtf(c40, bf5,  c24, bf6, cosBit);
        int s7 = HalfBtf(c8,  bf4,  c56, bf7, cosBit);

        // Stage 3: cospi butterfly on lower 4 + add/sub on upper 4.
        bf0 = HalfBtf(c32, s0,  c32, s1, cosBit);
        bf1 = HalfBtf(c32, s0, -c32, s1, cosBit);
        bf2 = HalfBtf(c48, s2, -c16, s3, cosBit);
        bf3 = HalfBtf(c16, s2,  c48, s3, cosBit);
        bf4 =  s4 + s5;
        bf5 =  s4 - s5;
        bf6 = -s6 + s7;
        bf7 =  s6 + s7;

        // Stage 4: butterfly + cospi rotation on middle 2 of upper.
        s0 = bf0 + bf3;
        s1 = bf1 + bf2;
        s2 = bf1 - bf2;
        s3 = bf0 - bf3;
        s4 = bf4;
        s5 = HalfBtf(-c32, bf5, c32, bf6, cosBit);
        s6 = HalfBtf( c32, bf5, c32, bf6, cosBit);
        s7 = bf7;

        // Stage 5: outer butterfly.
        output[outBase + 0] = s0 + s7;
        output[outBase + 1] = s1 + s6;
        output[outBase + 2] = s2 + s5;
        output[outBase + 3] = s3 + s4;
        output[outBase + 4] = s3 - s4;
        output[outBase + 5] = s2 - s5;
        output[outBase + 6] = s1 - s6;
        output[outBase + 7] = s0 - s7;
    }

    /// <summary>libaom <c>half_btf</c>: kernel-safe variant.</summary>
    private static int HalfBtf(int w0, int in0, int w1, int in1, int bit)
    {
        long result = (long)w0 * in0 + (long)w1 * in1;
        result += 1L << (bit - 1);
        return (int)(result >> bit);
    }

    /// <summary>
    /// Resolves the 7 cospi entries idct8 needs (8, 16, 24, 32, 40, 48,
    /// 56). Same 7 cospi values as the forward 8-point DCT (the
    /// transform is its own structural mirror), inlined per cos_bit.
    /// </summary>
    private static void ResolveCospi(int cosBit,
        out int c8, out int c16, out int c24, out int c32,
        out int c40, out int c48, out int c56)
    {
        if (cosBit == 13)      { c8 = 8035; c16 = 7568; c24 = 6811; c32 = 5793; c40 = 4551; c48 = 3135; c56 = 1598; }
        else if (cosBit == 12) { c8 = 4017; c16 = 3784; c24 = 3406; c32 = 2896; c40 = 2276; c48 = 1567; c56 = 799; }
        else if (cosBit == 11) { c8 = 2009; c16 = 1892; c24 = 1703; c32 = 1448; c40 = 1138; c48 = 784;  c56 = 400; }
        else                   { c8 = 1004; c16 = 946;  c24 = 851;  c32 = 724;  c40 = 569;  c48 = 392;  c56 = 200; }
    }
}
