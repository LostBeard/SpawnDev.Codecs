// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 8-point forward DCT (1D), GPU-callable form for in-kernel
// reuse. Bit-exact mirror of Av1ForwardDct8.Transform (libaom
// av1/encoder/av1_fwd_txfm1d.c av1_fdct8 port).
//
// Av1ForwardDct8Kernel already wraps this as a standalone batched
// dispatch (one thread per 8-element 1D transform). The static
// helper here exists so the upcoming Av1FrameSequentialEncodeKernel
// can run the 1D forward DCT8 inside a per-block 2D transform call
// chain without dispatching a separate kernel boundary - that's the
// v3 host-as-pure-coordinator pattern.
//
// 5 stages of butterfly + cospi multiplications + final interleave.
// Inlined cospi values via ResolveCospi keep the kernel free of
// 64-entry table lookups.

using ILGPU;

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>
/// GPU-callable AV1 8-point forward DCT helper. Bit-exact mirror of
/// <see cref="Av1ForwardDct8"/> for in-kernel use.
/// </summary>
public static class Av1ForwardDct8Gpu
{
    /// <summary>Default cosine-precision bits (libaom).</summary>
    public const int DefaultCosBit = Av1ForwardDct8.DefaultCosBit;

    /// <summary>
    /// Apply the 8-point forward DCT to one 8-element 1D row/column.
    /// Reads <paramref name="input"/> starting at <paramref name="inBase"/>;
    /// writes 8 ints to <paramref name="output"/> starting at
    /// <paramref name="outBase"/>.
    /// </summary>
    public static void Forward8(
        ArrayView<int> input, long inBase,
        ArrayView<int> output, long outBase,
        int cosBit)
    {
        ResolveCospi(cosBit, out int c8, out int c16, out int c24, out int c32,
            out int c40, out int c48, out int c56);

        int in0 = input[inBase + 0];
        int in1 = input[inBase + 1];
        int in2 = input[inBase + 2];
        int in3 = input[inBase + 3];
        int in4 = input[inBase + 4];
        int in5 = input[inBase + 5];
        int in6 = input[inBase + 6];
        int in7 = input[inBase + 7];

        // Stage 1
        int s10 =  in0 + in7;
        int s11 =  in1 + in6;
        int s12 =  in2 + in5;
        int s13 =  in3 + in4;
        int s14 = -in4 + in3;
        int s15 = -in5 + in2;
        int s16 = -in6 + in1;
        int s17 = -in7 + in0;

        // Stage 2
        int s20 = s10 + s13;
        int s21 = s11 + s12;
        int s22 = -s12 + s11;
        int s23 = -s13 + s10;
        int s24 = s14;
        int s25 = HalfBtf(-c32, s15,  c32, s16, cosBit);
        int s26 = HalfBtf( c32, s16,  c32, s15, cosBit);
        int s27 = s17;

        // Stage 3
        int s30 = HalfBtf( c32, s20,  c32, s21, cosBit);
        int s31 = HalfBtf(-c32, s21,  c32, s20, cosBit);
        int s32 = HalfBtf( c48, s22,  c16, s23, cosBit);
        int s33 = HalfBtf( c48, s23, -c16, s22, cosBit);
        int s34 = s24 + s25;
        int s35 = -s25 + s24;
        int s36 = -s26 + s27;
        int s37 = s27 + s26;

        // Stage 4
        int s40 = s30;
        int s41 = s31;
        int s42 = s32;
        int s43 = s33;
        int s44 = HalfBtf( c56, s34,  c8,  s37, cosBit);
        int s45 = HalfBtf( c24, s35,  c40, s36, cosBit);
        int s46 = HalfBtf( c24, s36, -c40, s35, cosBit);
        int s47 = HalfBtf( c56, s37, -c8,  s34, cosBit);

        // Stage 5 (interleave)
        output[outBase + 0] = s40;
        output[outBase + 1] = s44;
        output[outBase + 2] = s42;
        output[outBase + 3] = s46;
        output[outBase + 4] = s41;
        output[outBase + 5] = s45;
        output[outBase + 6] = s43;
        output[outBase + 7] = s47;
    }

    /// <summary>libaom <c>half_btf</c>: kernel-safe variant.</summary>
    private static int HalfBtf(int w0, int in0, int w1, int in1, int bit)
    {
        long result = (long)w0 * in0 + (long)w1 * in1;
        result += 1L << (bit - 1);
        return (int)(result >> bit);
    }

    /// <summary>
    /// Resolves the 7 cospi entries fdct8 needs (8, 16, 24, 32, 40, 48,
    /// 56) per cos_bit. Inlined as branches so the kernel does not have
    /// to read a 64-element table buffer.
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
