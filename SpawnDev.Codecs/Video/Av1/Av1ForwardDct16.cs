// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 16-point forward DCT (1D). Bit-exact port of libaom
// av1/encoder/av1_fwd_txfm1d.c av1_fdct16. 7 stages.

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>AV1 16-point forward DCT (1D).</summary>
public static class Av1ForwardDct16
{
    /// <summary>Default cosine-precision bits (libaom).</summary>
    public const int DefaultCosBit = 13;

    /// <summary>16-point forward DCT. Mirrors libaom <c>av1_fdct16</c>.</summary>
    public static void Transform(ReadOnlySpan<int> input, Span<int> output, int cosBit = DefaultCosBit)
    {
        if (input.Length < 16) throw new ArgumentException("input must have 16 entries", nameof(input));
        if (output.Length < 16) throw new ArgumentException("output must have 16 entries", nameof(output));
        if (cosBit < 10 || cosBit > 13) throw new ArgumentOutOfRangeException(nameof(cosBit));

        var cospi = Av1ForwardDct4.CospiArr(cosBit);
        Span<int> a = stackalloc int[16];
        Span<int> b = stackalloc int[16];

        // Stage 1: a = output (we use 'a' as temp)
        a[0] = input[0] + input[15];
        a[1] = input[1] + input[14];
        a[2] = input[2] + input[13];
        a[3] = input[3] + input[12];
        a[4] = input[4] + input[11];
        a[5] = input[5] + input[10];
        a[6] = input[6] + input[9];
        a[7] = input[7] + input[8];
        a[8] = -input[8] + input[7];
        a[9] = -input[9] + input[6];
        a[10] = -input[10] + input[5];
        a[11] = -input[11] + input[4];
        a[12] = -input[12] + input[3];
        a[13] = -input[13] + input[2];
        a[14] = -input[14] + input[1];
        a[15] = -input[15] + input[0];

        // Stage 2: b = step
        b[0] = a[0] + a[7];
        b[1] = a[1] + a[6];
        b[2] = a[2] + a[5];
        b[3] = a[3] + a[4];
        b[4] = -a[4] + a[3];
        b[5] = -a[5] + a[2];
        b[6] = -a[6] + a[1];
        b[7] = -a[7] + a[0];
        b[8] = a[8];
        b[9] = a[9];
        b[10] = Av1ForwardDct4.HalfBtf(-cospi[32], a[10], cospi[32], a[13], cosBit);
        b[11] = Av1ForwardDct4.HalfBtf(-cospi[32], a[11], cospi[32], a[12], cosBit);
        b[12] = Av1ForwardDct4.HalfBtf(cospi[32], a[12], cospi[32], a[11], cosBit);
        b[13] = Av1ForwardDct4.HalfBtf(cospi[32], a[13], cospi[32], a[10], cosBit);
        b[14] = a[14];
        b[15] = a[15];

        // Stage 3: a
        a[0] = b[0] + b[3];
        a[1] = b[1] + b[2];
        a[2] = -b[2] + b[1];
        a[3] = -b[3] + b[0];
        a[4] = b[4];
        a[5] = Av1ForwardDct4.HalfBtf(-cospi[32], b[5], cospi[32], b[6], cosBit);
        a[6] = Av1ForwardDct4.HalfBtf(cospi[32], b[6], cospi[32], b[5], cosBit);
        a[7] = b[7];
        a[8] = b[8] + b[11];
        a[9] = b[9] + b[10];
        a[10] = -b[10] + b[9];
        a[11] = -b[11] + b[8];
        a[12] = -b[12] + b[15];
        a[13] = -b[13] + b[14];
        a[14] = b[14] + b[13];
        a[15] = b[15] + b[12];

        // Stage 4: b
        b[0] = Av1ForwardDct4.HalfBtf(cospi[32], a[0], cospi[32], a[1], cosBit);
        b[1] = Av1ForwardDct4.HalfBtf(-cospi[32], a[1], cospi[32], a[0], cosBit);
        b[2] = Av1ForwardDct4.HalfBtf(cospi[48], a[2], cospi[16], a[3], cosBit);
        b[3] = Av1ForwardDct4.HalfBtf(cospi[48], a[3], -cospi[16], a[2], cosBit);
        b[4] = a[4] + a[5];
        b[5] = -a[5] + a[4];
        b[6] = -a[6] + a[7];
        b[7] = a[7] + a[6];
        b[8] = a[8];
        b[9] = Av1ForwardDct4.HalfBtf(-cospi[16], a[9], cospi[48], a[14], cosBit);
        b[10] = Av1ForwardDct4.HalfBtf(-cospi[48], a[10], -cospi[16], a[13], cosBit);
        b[11] = a[11];
        b[12] = a[12];
        b[13] = Av1ForwardDct4.HalfBtf(cospi[48], a[13], -cospi[16], a[10], cosBit);
        b[14] = Av1ForwardDct4.HalfBtf(cospi[16], a[14], cospi[48], a[9], cosBit);
        b[15] = a[15];

        // Stage 5: a
        a[0] = b[0];
        a[1] = b[1];
        a[2] = b[2];
        a[3] = b[3];
        a[4] = Av1ForwardDct4.HalfBtf(cospi[56], b[4], cospi[8], b[7], cosBit);
        a[5] = Av1ForwardDct4.HalfBtf(cospi[24], b[5], cospi[40], b[6], cosBit);
        a[6] = Av1ForwardDct4.HalfBtf(cospi[24], b[6], -cospi[40], b[5], cosBit);
        a[7] = Av1ForwardDct4.HalfBtf(cospi[56], b[7], -cospi[8], b[4], cosBit);
        a[8] = b[8] + b[9];
        a[9] = -b[9] + b[8];
        a[10] = -b[10] + b[11];
        a[11] = b[11] + b[10];
        a[12] = b[12] + b[13];
        a[13] = -b[13] + b[12];
        a[14] = -b[14] + b[15];
        a[15] = b[15] + b[14];

        // Stage 6: b
        b[0] = a[0]; b[1] = a[1]; b[2] = a[2]; b[3] = a[3];
        b[4] = a[4]; b[5] = a[5]; b[6] = a[6]; b[7] = a[7];
        b[8] = Av1ForwardDct4.HalfBtf(cospi[60], a[8], cospi[4], a[15], cosBit);
        b[9] = Av1ForwardDct4.HalfBtf(cospi[28], a[9], cospi[36], a[14], cosBit);
        b[10] = Av1ForwardDct4.HalfBtf(cospi[44], a[10], cospi[20], a[13], cosBit);
        b[11] = Av1ForwardDct4.HalfBtf(cospi[12], a[11], cospi[52], a[12], cosBit);
        b[12] = Av1ForwardDct4.HalfBtf(cospi[12], a[12], -cospi[52], a[11], cosBit);
        b[13] = Av1ForwardDct4.HalfBtf(cospi[44], a[13], -cospi[20], a[10], cosBit);
        b[14] = Av1ForwardDct4.HalfBtf(cospi[28], a[14], -cospi[36], a[9], cosBit);
        b[15] = Av1ForwardDct4.HalfBtf(cospi[60], a[15], -cospi[4], a[8], cosBit);

        // Stage 7 (interleave)
        output[0] = b[0]; output[1] = b[8]; output[2] = b[4]; output[3] = b[12];
        output[4] = b[2]; output[5] = b[10]; output[6] = b[6]; output[7] = b[14];
        output[8] = b[1]; output[9] = b[9]; output[10] = b[5]; output[11] = b[13];
        output[12] = b[3]; output[13] = b[11]; output[14] = b[7]; output[15] = b[15];
    }
}
