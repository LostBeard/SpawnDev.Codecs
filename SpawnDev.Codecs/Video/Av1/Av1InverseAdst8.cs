// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 8-point inverse Asymmetric DST (1D). Bit-exact port of libaom
// av1/common/av1_inv_txfm1d.c av1_iadst8.
//
// Upstream Copyright (c) 2016, Alliance for Open Media.
// Upstream license: BSD 2-Clause + AV1 Patent License 1.0. See NOTICE.md.

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>AV1 8-point inverse Asymmetric DST (1D).</summary>
public static class Av1InverseAdst8
{
    /// <summary>Default cos_bit for the inverse 8-point ADST (libaom).</summary>
    public const int DefaultCosBit = 12;

    /// <summary>
    /// 8-point inverse ADST. Mirrors libaom <c>av1_iadst8</c>.
    /// </summary>
    public static void Transform(ReadOnlySpan<int> input, Span<int> output, int cosBit = DefaultCosBit)
    {
        if (input.Length < 8) throw new ArgumentException("input must have 8 entries", nameof(input));
        if (output.Length < 8) throw new ArgumentException("output must have 8 entries", nameof(output));
        if (cosBit < 10 || cosBit > 13) throw new ArgumentOutOfRangeException(nameof(cosBit));

        var cospi = Av1ForwardDct4.CospiArr(cosBit);
        int Btf(int w0, int in0, int w1, int in1) =>
            Av1ForwardDct4.HalfBtf(w0, in0, w1, in1, cosBit);

        // Stage 1: input permutation
        Span<int> bf = stackalloc int[8];
        bf[0] = input[7];
        bf[1] = input[0];
        bf[2] = input[5];
        bf[3] = input[2];
        bf[4] = input[3];
        bf[5] = input[4];
        bf[6] = input[1];
        bf[7] = input[6];

        // Stage 2: cospi rotations
        Span<int> step = stackalloc int[8];
        step[0] = Btf(cospi[4], bf[0], cospi[60], bf[1]);
        step[1] = Btf(cospi[60], bf[0], -cospi[4], bf[1]);
        step[2] = Btf(cospi[20], bf[2], cospi[44], bf[3]);
        step[3] = Btf(cospi[44], bf[2], -cospi[20], bf[3]);
        step[4] = Btf(cospi[36], bf[4], cospi[28], bf[5]);
        step[5] = Btf(cospi[28], bf[4], -cospi[36], bf[5]);
        step[6] = Btf(cospi[52], bf[6], cospi[12], bf[7]);
        step[7] = Btf(cospi[12], bf[6], -cospi[52], bf[7]);

        // Stage 3: butterfly (lower vs upper half)
        bf[0] = step[0] + step[4];
        bf[1] = step[1] + step[5];
        bf[2] = step[2] + step[6];
        bf[3] = step[3] + step[7];
        bf[4] = step[0] - step[4];
        bf[5] = step[1] - step[5];
        bf[6] = step[2] - step[6];
        bf[7] = step[3] - step[7];

        // Stage 4: cospi rotations on upper 4
        step[0] = bf[0];
        step[1] = bf[1];
        step[2] = bf[2];
        step[3] = bf[3];
        step[4] = Btf(cospi[16], bf[4], cospi[48], bf[5]);
        step[5] = Btf(cospi[48], bf[4], -cospi[16], bf[5]);
        step[6] = Btf(-cospi[48], bf[6], cospi[16], bf[7]);
        step[7] = Btf(cospi[16], bf[6], cospi[48], bf[7]);

        // Stage 5: butterfly across 4-strides
        bf[0] = step[0] + step[2];
        bf[1] = step[1] + step[3];
        bf[2] = step[0] - step[2];
        bf[3] = step[1] - step[3];
        bf[4] = step[4] + step[6];
        bf[5] = step[5] + step[7];
        bf[6] = step[4] - step[6];
        bf[7] = step[5] - step[7];

        // Stage 6: cospi rotation on a few entries
        step[0] = bf[0];
        step[1] = bf[1];
        step[2] = Btf(cospi[32], bf[2], cospi[32], bf[3]);
        step[3] = Btf(cospi[32], bf[2], -cospi[32], bf[3]);
        step[4] = bf[4];
        step[5] = bf[5];
        step[6] = Btf(cospi[32], bf[6], cospi[32], bf[7]);
        step[7] = Btf(cospi[32], bf[6], -cospi[32], bf[7]);

        // Stage 7: final permutation with sign flips
        output[0] = step[0];
        output[1] = -step[4];
        output[2] = step[6];
        output[3] = -step[2];
        output[4] = step[3];
        output[5] = -step[7];
        output[6] = step[5];
        output[7] = -step[1];
    }
}
