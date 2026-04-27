// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 16-point forward Asymmetric DST (1D). Bit-exact port of libaom
// av1/encoder/av1_fwd_txfm1d.c av1_fadst16.
//
// 9 stages with cospi_arr-driven half_btf rotations. Final scatter
// permutation matches libaom exactly.

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>AV1 16-point forward Asymmetric DST (1D).</summary>
public static class Av1ForwardAdst16
{
    /// <summary>Default cosine-precision bits (libaom).</summary>
    public const int DefaultCosBit = 13;

    /// <summary>16-point forward ADST. Mirrors libaom <c>av1_fadst16</c>.</summary>
    public static void Transform(ReadOnlySpan<int> input, Span<int> output, int cosBit = DefaultCosBit)
    {
        if (input.Length < 16) throw new ArgumentException("input must have 16 entries", nameof(input));
        if (output.Length < 16) throw new ArgumentException("output must have 16 entries", nameof(output));
        if (cosBit < 10 || cosBit > 13) throw new ArgumentOutOfRangeException(nameof(cosBit));

        var cospi = Av1ForwardDct4.CospiArr(cosBit);
        Span<int> step = stackalloc int[16];
        Span<int> bf1 = stackalloc int[16];

        // Stage 1: input remap with sign flips.
        bf1[0]  =  input[0];
        bf1[1]  = -input[15];
        bf1[2]  = -input[7];
        bf1[3]  =  input[8];
        bf1[4]  = -input[3];
        bf1[5]  =  input[12];
        bf1[6]  =  input[4];
        bf1[7]  = -input[11];
        bf1[8]  = -input[1];
        bf1[9]  =  input[14];
        bf1[10] =  input[6];
        bf1[11] = -input[9];
        bf1[12] =  input[2];
        bf1[13] = -input[13];
        bf1[14] = -input[5];
        bf1[15] =  input[10];

        // Stage 2: cospi[32] rotations on (2,3), (6,7), (10,11), (14,15).
        step[0]  = bf1[0];
        step[1]  = bf1[1];
        step[2]  = Av1ForwardDct4.HalfBtf(cospi[32], bf1[2],  cospi[32], bf1[3], cosBit);
        step[3]  = Av1ForwardDct4.HalfBtf(cospi[32], bf1[2], -cospi[32], bf1[3], cosBit);
        step[4]  = bf1[4];
        step[5]  = bf1[5];
        step[6]  = Av1ForwardDct4.HalfBtf(cospi[32], bf1[6],  cospi[32], bf1[7], cosBit);
        step[7]  = Av1ForwardDct4.HalfBtf(cospi[32], bf1[6], -cospi[32], bf1[7], cosBit);
        step[8]  = bf1[8];
        step[9]  = bf1[9];
        step[10] = Av1ForwardDct4.HalfBtf(cospi[32], bf1[10],  cospi[32], bf1[11], cosBit);
        step[11] = Av1ForwardDct4.HalfBtf(cospi[32], bf1[10], -cospi[32], bf1[11], cosBit);
        step[12] = bf1[12];
        step[13] = bf1[13];
        step[14] = Av1ForwardDct4.HalfBtf(cospi[32], bf1[14],  cospi[32], bf1[15], cosBit);
        step[15] = Av1ForwardDct4.HalfBtf(cospi[32], bf1[14], -cospi[32], bf1[15], cosBit);

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
        step[4]  = Av1ForwardDct4.HalfBtf( cospi[16], bf1[4],  cospi[48], bf1[5], cosBit);
        step[5]  = Av1ForwardDct4.HalfBtf( cospi[48], bf1[4], -cospi[16], bf1[5], cosBit);
        step[6]  = Av1ForwardDct4.HalfBtf(-cospi[48], bf1[6],  cospi[16], bf1[7], cosBit);
        step[7]  = Av1ForwardDct4.HalfBtf( cospi[16], bf1[6],  cospi[48], bf1[7], cosBit);
        step[8]  = bf1[8];
        step[9]  = bf1[9];
        step[10] = bf1[10];
        step[11] = bf1[11];
        step[12] = Av1ForwardDct4.HalfBtf( cospi[16], bf1[12],  cospi[48], bf1[13], cosBit);
        step[13] = Av1ForwardDct4.HalfBtf( cospi[48], bf1[12], -cospi[16], bf1[13], cosBit);
        step[14] = Av1ForwardDct4.HalfBtf(-cospi[48], bf1[14],  cospi[16], bf1[15], cosBit);
        step[15] = Av1ForwardDct4.HalfBtf( cospi[16], bf1[14],  cospi[48], bf1[15], cosBit);

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

        // Stage 6: cospi[8/56/40/24] rotations on the upper 8 elements.
        step[0]  = bf1[0];
        step[1]  = bf1[1];
        step[2]  = bf1[2];
        step[3]  = bf1[3];
        step[4]  = bf1[4];
        step[5]  = bf1[5];
        step[6]  = bf1[6];
        step[7]  = bf1[7];
        step[8]  = Av1ForwardDct4.HalfBtf( cospi[ 8], bf1[ 8],  cospi[56], bf1[ 9], cosBit);
        step[9]  = Av1ForwardDct4.HalfBtf( cospi[56], bf1[ 8], -cospi[ 8], bf1[ 9], cosBit);
        step[10] = Av1ForwardDct4.HalfBtf( cospi[40], bf1[10],  cospi[24], bf1[11], cosBit);
        step[11] = Av1ForwardDct4.HalfBtf( cospi[24], bf1[10], -cospi[40], bf1[11], cosBit);
        step[12] = Av1ForwardDct4.HalfBtf(-cospi[56], bf1[12],  cospi[ 8], bf1[13], cosBit);
        step[13] = Av1ForwardDct4.HalfBtf( cospi[ 8], bf1[12],  cospi[56], bf1[13], cosBit);
        step[14] = Av1ForwardDct4.HalfBtf(-cospi[24], bf1[14],  cospi[40], bf1[15], cosBit);
        step[15] = Av1ForwardDct4.HalfBtf( cospi[40], bf1[14],  cospi[24], bf1[15], cosBit);

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
        step[0]  = Av1ForwardDct4.HalfBtf( cospi[ 2], bf1[ 0],  cospi[62], bf1[ 1], cosBit);
        step[1]  = Av1ForwardDct4.HalfBtf( cospi[62], bf1[ 0], -cospi[ 2], bf1[ 1], cosBit);
        step[2]  = Av1ForwardDct4.HalfBtf( cospi[10], bf1[ 2],  cospi[54], bf1[ 3], cosBit);
        step[3]  = Av1ForwardDct4.HalfBtf( cospi[54], bf1[ 2], -cospi[10], bf1[ 3], cosBit);
        step[4]  = Av1ForwardDct4.HalfBtf( cospi[18], bf1[ 4],  cospi[46], bf1[ 5], cosBit);
        step[5]  = Av1ForwardDct4.HalfBtf( cospi[46], bf1[ 4], -cospi[18], bf1[ 5], cosBit);
        step[6]  = Av1ForwardDct4.HalfBtf( cospi[26], bf1[ 6],  cospi[38], bf1[ 7], cosBit);
        step[7]  = Av1ForwardDct4.HalfBtf( cospi[38], bf1[ 6], -cospi[26], bf1[ 7], cosBit);
        step[8]  = Av1ForwardDct4.HalfBtf( cospi[34], bf1[ 8],  cospi[30], bf1[ 9], cosBit);
        step[9]  = Av1ForwardDct4.HalfBtf( cospi[30], bf1[ 8], -cospi[34], bf1[ 9], cosBit);
        step[10] = Av1ForwardDct4.HalfBtf( cospi[42], bf1[10],  cospi[22], bf1[11], cosBit);
        step[11] = Av1ForwardDct4.HalfBtf( cospi[22], bf1[10], -cospi[42], bf1[11], cosBit);
        step[12] = Av1ForwardDct4.HalfBtf( cospi[50], bf1[12],  cospi[14], bf1[13], cosBit);
        step[13] = Av1ForwardDct4.HalfBtf( cospi[14], bf1[12], -cospi[50], bf1[13], cosBit);
        step[14] = Av1ForwardDct4.HalfBtf( cospi[58], bf1[14],  cospi[ 6], bf1[15], cosBit);
        step[15] = Av1ForwardDct4.HalfBtf( cospi[ 6], bf1[14], -cospi[58], bf1[15], cosBit);

        // Stage 9: final scatter to output (libaom permutation).
        output[0]  = step[1];
        output[1]  = step[14];
        output[2]  = step[3];
        output[3]  = step[12];
        output[4]  = step[5];
        output[5]  = step[10];
        output[6]  = step[7];
        output[7]  = step[8];
        output[8]  = step[9];
        output[9]  = step[6];
        output[10] = step[11];
        output[11] = step[4];
        output[12] = step[13];
        output[13] = step[2];
        output[14] = step[15];
        output[15] = step[0];
    }
}
