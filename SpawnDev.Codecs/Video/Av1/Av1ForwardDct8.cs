// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 8-point forward DCT (1D). Bit-exact port of libaom
// av1/encoder/av1_fwd_txfm1d.c av1_fdct8.
//
// 5 stages of butterfly + cospi multiplications + final interleave.
// Default cos_bit = 13 (libaom AV1 fwd txfm config).

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>AV1 8-point forward DCT (1D).</summary>
public static class Av1ForwardDct8
{
    /// <summary>Default cosine-precision bits (libaom).</summary>
    public const int DefaultCosBit = 13;

    /// <summary>8-point forward DCT. Mirrors libaom <c>av1_fdct8</c>.</summary>
    public static void Transform(ReadOnlySpan<int> input, Span<int> output, int cosBit = DefaultCosBit)
    {
        if (input.Length < 8) throw new ArgumentException("input must have 8 entries", nameof(input));
        if (output.Length < 8) throw new ArgumentException("output must have 8 entries", nameof(output));
        if (cosBit < 10 || cosBit > 13) throw new ArgumentOutOfRangeException(nameof(cosBit));

        var cospi = Av1ForwardDct4.CospiArr(cosBit);

        // Stage 1
        Span<int> s1 = stackalloc int[8];
        s1[0] = input[0] + input[7];
        s1[1] = input[1] + input[6];
        s1[2] = input[2] + input[5];
        s1[3] = input[3] + input[4];
        s1[4] = -input[4] + input[3];
        s1[5] = -input[5] + input[2];
        s1[6] = -input[6] + input[1];
        s1[7] = -input[7] + input[0];

        // Stage 2
        Span<int> s2 = stackalloc int[8];
        s2[0] = s1[0] + s1[3];
        s2[1] = s1[1] + s1[2];
        s2[2] = -s1[2] + s1[1];
        s2[3] = -s1[3] + s1[0];
        s2[4] = s1[4];
        s2[5] = Av1ForwardDct4.HalfBtf(-cospi[32], s1[5], cospi[32], s1[6], cosBit);
        s2[6] = Av1ForwardDct4.HalfBtf(cospi[32], s1[6], cospi[32], s1[5], cosBit);
        s2[7] = s1[7];

        // Stage 3
        Span<int> s3 = stackalloc int[8];
        s3[0] = Av1ForwardDct4.HalfBtf(cospi[32], s2[0], cospi[32], s2[1], cosBit);
        s3[1] = Av1ForwardDct4.HalfBtf(-cospi[32], s2[1], cospi[32], s2[0], cosBit);
        s3[2] = Av1ForwardDct4.HalfBtf(cospi[48], s2[2], cospi[16], s2[3], cosBit);
        s3[3] = Av1ForwardDct4.HalfBtf(cospi[48], s2[3], -cospi[16], s2[2], cosBit);
        s3[4] = s2[4] + s2[5];
        s3[5] = -s2[5] + s2[4];
        s3[6] = -s2[6] + s2[7];
        s3[7] = s2[7] + s2[6];

        // Stage 4
        Span<int> s4 = stackalloc int[8];
        s4[0] = s3[0];
        s4[1] = s3[1];
        s4[2] = s3[2];
        s4[3] = s3[3];
        s4[4] = Av1ForwardDct4.HalfBtf(cospi[56], s3[4], cospi[8], s3[7], cosBit);
        s4[5] = Av1ForwardDct4.HalfBtf(cospi[24], s3[5], cospi[40], s3[6], cosBit);
        s4[6] = Av1ForwardDct4.HalfBtf(cospi[24], s3[6], -cospi[40], s3[5], cosBit);
        s4[7] = Av1ForwardDct4.HalfBtf(cospi[56], s3[7], -cospi[8], s3[4], cosBit);

        // Stage 5 (interleave)
        output[0] = s4[0];
        output[1] = s4[4];
        output[2] = s4[2];
        output[3] = s4[6];
        output[4] = s4[1];
        output[5] = s4[5];
        output[6] = s4[3];
        output[7] = s4[7];
    }
}
