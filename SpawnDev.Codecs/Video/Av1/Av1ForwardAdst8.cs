// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 8-point forward Asymmetric DST (1D). Bit-exact port of libaom
// av1/encoder/av1_fwd_txfm1d.c av1_fadst8.
//
// 7 stages with cospi_arr-driven half_btf rotations. Output reordered
// to libaom's final scatter pattern.

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>AV1 8-point forward Asymmetric DST (1D).</summary>
public static class Av1ForwardAdst8
{
    /// <summary>Default cosine-precision bits (libaom).</summary>
    public const int DefaultCosBit = 13;

    /// <summary>8-point forward ADST. Mirrors libaom <c>av1_fadst8</c>.</summary>
    public static void Transform(ReadOnlySpan<int> input, Span<int> output, int cosBit = DefaultCosBit)
    {
        if (input.Length < 8) throw new ArgumentException("input must have 8 entries", nameof(input));
        if (output.Length < 8) throw new ArgumentException("output must have 8 entries", nameof(output));
        if (cosBit < 10 || cosBit > 13) throw new ArgumentOutOfRangeException(nameof(cosBit));

        var cospi = Av1ForwardDct4.CospiArr(cosBit);
        Span<int> step = stackalloc int[8];

        // Stage 1: input remap with sign flips.
        Span<int> bf1 = stackalloc int[8];
        bf1[0] =  input[0];
        bf1[1] = -input[7];
        bf1[2] = -input[3];
        bf1[3] =  input[4];
        bf1[4] = -input[1];
        bf1[5] =  input[6];
        bf1[6] =  input[2];
        bf1[7] = -input[5];

        // Stage 2: cospi[32] rotations on (2,3) and (6,7).
        step[0] = bf1[0];
        step[1] = bf1[1];
        step[2] = Av1ForwardDct4.HalfBtf(cospi[32], bf1[2],  cospi[32], bf1[3], cosBit);
        step[3] = Av1ForwardDct4.HalfBtf(cospi[32], bf1[2], -cospi[32], bf1[3], cosBit);
        step[4] = bf1[4];
        step[5] = bf1[5];
        step[6] = Av1ForwardDct4.HalfBtf(cospi[32], bf1[6],  cospi[32], bf1[7], cosBit);
        step[7] = Av1ForwardDct4.HalfBtf(cospi[32], bf1[6], -cospi[32], bf1[7], cosBit);

        // Stage 3: butterfly into bf1 (acts as output buffer).
        bf1[0] = step[0] + step[2];
        bf1[1] = step[1] + step[3];
        bf1[2] = step[0] - step[2];
        bf1[3] = step[1] - step[3];
        bf1[4] = step[4] + step[6];
        bf1[5] = step[5] + step[7];
        bf1[6] = step[4] - step[6];
        bf1[7] = step[5] - step[7];

        // Stage 4: cospi[16]/[48] rotations on (4,5) and (6,7).
        step[0] = bf1[0];
        step[1] = bf1[1];
        step[2] = bf1[2];
        step[3] = bf1[3];
        step[4] = Av1ForwardDct4.HalfBtf( cospi[16], bf1[4],  cospi[48], bf1[5], cosBit);
        step[5] = Av1ForwardDct4.HalfBtf( cospi[48], bf1[4], -cospi[16], bf1[5], cosBit);
        step[6] = Av1ForwardDct4.HalfBtf(-cospi[48], bf1[6],  cospi[16], bf1[7], cosBit);
        step[7] = Av1ForwardDct4.HalfBtf( cospi[16], bf1[6],  cospi[48], bf1[7], cosBit);

        // Stage 5: butterfly across halves.
        bf1[0] = step[0] + step[4];
        bf1[1] = step[1] + step[5];
        bf1[2] = step[2] + step[6];
        bf1[3] = step[3] + step[7];
        bf1[4] = step[0] - step[4];
        bf1[5] = step[1] - step[5];
        bf1[6] = step[2] - step[6];
        bf1[7] = step[3] - step[7];

        // Stage 6: cospi[4/60/20/44/36/28/52/12] rotations.
        step[0] = Av1ForwardDct4.HalfBtf( cospi[ 4], bf1[0],  cospi[60], bf1[1], cosBit);
        step[1] = Av1ForwardDct4.HalfBtf( cospi[60], bf1[0], -cospi[ 4], bf1[1], cosBit);
        step[2] = Av1ForwardDct4.HalfBtf( cospi[20], bf1[2],  cospi[44], bf1[3], cosBit);
        step[3] = Av1ForwardDct4.HalfBtf( cospi[44], bf1[2], -cospi[20], bf1[3], cosBit);
        step[4] = Av1ForwardDct4.HalfBtf( cospi[36], bf1[4],  cospi[28], bf1[5], cosBit);
        step[5] = Av1ForwardDct4.HalfBtf( cospi[28], bf1[4], -cospi[36], bf1[5], cosBit);
        step[6] = Av1ForwardDct4.HalfBtf( cospi[52], bf1[6],  cospi[12], bf1[7], cosBit);
        step[7] = Av1ForwardDct4.HalfBtf( cospi[12], bf1[6], -cospi[52], bf1[7], cosBit);

        // Stage 7: final scatter to output (libaom permutation).
        output[0] = step[1];
        output[1] = step[6];
        output[2] = step[3];
        output[3] = step[4];
        output[4] = step[5];
        output[5] = step[2];
        output[6] = step[7];
        output[7] = step[0];
    }
}
