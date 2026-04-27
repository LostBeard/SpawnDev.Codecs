// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 16-point inverse Asymmetric DST (1D). Bit-exact port of libaom
// av1/common/av1_inv_txfm1d.c av1_iadst16.
//
// Upstream Copyright (c) 2016, Alliance for Open Media.
// Upstream license: BSD 2-Clause + AV1 Patent License 1.0. See NOTICE.md.

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>AV1 16-point inverse Asymmetric DST (1D).</summary>
public static class Av1InverseAdst16
{
    /// <summary>Default cos_bit for the inverse 16-point ADST (libaom).</summary>
    public const int DefaultCosBit = 12;

    /// <summary>
    /// 16-point inverse ADST. Mirrors libaom <c>av1_iadst16</c>.
    /// </summary>
    public static void Transform(ReadOnlySpan<int> input, Span<int> output, int cosBit = DefaultCosBit)
    {
        if (input.Length < 16) throw new ArgumentException("input must have 16 entries", nameof(input));
        if (output.Length < 16) throw new ArgumentException("output must have 16 entries", nameof(output));
        if (cosBit < 10 || cosBit > 13) throw new ArgumentOutOfRangeException(nameof(cosBit));

        var cospi = Av1ForwardDct4.CospiArr(cosBit);
        int Btf(int w0, int in0, int w1, int in1) =>
            Av1ForwardDct4.HalfBtf(w0, in0, w1, in1, cosBit);

        // Stage 1: input permutation
        Span<int> bf = stackalloc int[16];
        bf[0] = input[15];
        bf[1] = input[0];
        bf[2] = input[13];
        bf[3] = input[2];
        bf[4] = input[11];
        bf[5] = input[4];
        bf[6] = input[9];
        bf[7] = input[6];
        bf[8] = input[7];
        bf[9] = input[8];
        bf[10] = input[5];
        bf[11] = input[10];
        bf[12] = input[3];
        bf[13] = input[12];
        bf[14] = input[1];
        bf[15] = input[14];

        // Stage 2
        Span<int> step = stackalloc int[16];
        step[0] = Btf(cospi[2], bf[0], cospi[62], bf[1]);
        step[1] = Btf(cospi[62], bf[0], -cospi[2], bf[1]);
        step[2] = Btf(cospi[10], bf[2], cospi[54], bf[3]);
        step[3] = Btf(cospi[54], bf[2], -cospi[10], bf[3]);
        step[4] = Btf(cospi[18], bf[4], cospi[46], bf[5]);
        step[5] = Btf(cospi[46], bf[4], -cospi[18], bf[5]);
        step[6] = Btf(cospi[26], bf[6], cospi[38], bf[7]);
        step[7] = Btf(cospi[38], bf[6], -cospi[26], bf[7]);
        step[8] = Btf(cospi[34], bf[8], cospi[30], bf[9]);
        step[9] = Btf(cospi[30], bf[8], -cospi[34], bf[9]);
        step[10] = Btf(cospi[42], bf[10], cospi[22], bf[11]);
        step[11] = Btf(cospi[22], bf[10], -cospi[42], bf[11]);
        step[12] = Btf(cospi[50], bf[12], cospi[14], bf[13]);
        step[13] = Btf(cospi[14], bf[12], -cospi[50], bf[13]);
        step[14] = Btf(cospi[58], bf[14], cospi[6], bf[15]);
        step[15] = Btf(cospi[6], bf[14], -cospi[58], bf[15]);

        // Stage 3: butterfly between 0..7 and 8..15
        bf[0] = step[0] + step[8];
        bf[1] = step[1] + step[9];
        bf[2] = step[2] + step[10];
        bf[3] = step[3] + step[11];
        bf[4] = step[4] + step[12];
        bf[5] = step[5] + step[13];
        bf[6] = step[6] + step[14];
        bf[7] = step[7] + step[15];
        bf[8] = step[0] - step[8];
        bf[9] = step[1] - step[9];
        bf[10] = step[2] - step[10];
        bf[11] = step[3] - step[11];
        bf[12] = step[4] - step[12];
        bf[13] = step[5] - step[13];
        bf[14] = step[6] - step[14];
        bf[15] = step[7] - step[15];

        // Stage 4
        step[0] = bf[0];
        step[1] = bf[1];
        step[2] = bf[2];
        step[3] = bf[3];
        step[4] = bf[4];
        step[5] = bf[5];
        step[6] = bf[6];
        step[7] = bf[7];
        step[8] = Btf(cospi[8], bf[8], cospi[56], bf[9]);
        step[9] = Btf(cospi[56], bf[8], -cospi[8], bf[9]);
        step[10] = Btf(cospi[40], bf[10], cospi[24], bf[11]);
        step[11] = Btf(cospi[24], bf[10], -cospi[40], bf[11]);
        step[12] = Btf(-cospi[56], bf[12], cospi[8], bf[13]);
        step[13] = Btf(cospi[8], bf[12], cospi[56], bf[13]);
        step[14] = Btf(-cospi[24], bf[14], cospi[40], bf[15]);
        step[15] = Btf(cospi[40], bf[14], cospi[24], bf[15]);

        // Stage 5
        bf[0] = step[0] + step[4];
        bf[1] = step[1] + step[5];
        bf[2] = step[2] + step[6];
        bf[3] = step[3] + step[7];
        bf[4] = step[0] - step[4];
        bf[5] = step[1] - step[5];
        bf[6] = step[2] - step[6];
        bf[7] = step[3] - step[7];
        bf[8] = step[8] + step[12];
        bf[9] = step[9] + step[13];
        bf[10] = step[10] + step[14];
        bf[11] = step[11] + step[15];
        bf[12] = step[8] - step[12];
        bf[13] = step[9] - step[13];
        bf[14] = step[10] - step[14];
        bf[15] = step[11] - step[15];

        // Stage 6
        step[0] = bf[0];
        step[1] = bf[1];
        step[2] = bf[2];
        step[3] = bf[3];
        step[4] = Btf(cospi[16], bf[4], cospi[48], bf[5]);
        step[5] = Btf(cospi[48], bf[4], -cospi[16], bf[5]);
        step[6] = Btf(-cospi[48], bf[6], cospi[16], bf[7]);
        step[7] = Btf(cospi[16], bf[6], cospi[48], bf[7]);
        step[8] = bf[8];
        step[9] = bf[9];
        step[10] = bf[10];
        step[11] = bf[11];
        step[12] = Btf(cospi[16], bf[12], cospi[48], bf[13]);
        step[13] = Btf(cospi[48], bf[12], -cospi[16], bf[13]);
        step[14] = Btf(-cospi[48], bf[14], cospi[16], bf[15]);
        step[15] = Btf(cospi[16], bf[14], cospi[48], bf[15]);

        // Stage 7
        bf[0] = step[0] + step[2];
        bf[1] = step[1] + step[3];
        bf[2] = step[0] - step[2];
        bf[3] = step[1] - step[3];
        bf[4] = step[4] + step[6];
        bf[5] = step[5] + step[7];
        bf[6] = step[4] - step[6];
        bf[7] = step[5] - step[7];
        bf[8] = step[8] + step[10];
        bf[9] = step[9] + step[11];
        bf[10] = step[8] - step[10];
        bf[11] = step[9] - step[11];
        bf[12] = step[12] + step[14];
        bf[13] = step[13] + step[15];
        bf[14] = step[12] - step[14];
        bf[15] = step[13] - step[15];

        // Stage 8
        step[0] = bf[0];
        step[1] = bf[1];
        step[2] = Btf(cospi[32], bf[2], cospi[32], bf[3]);
        step[3] = Btf(cospi[32], bf[2], -cospi[32], bf[3]);
        step[4] = bf[4];
        step[5] = bf[5];
        step[6] = Btf(cospi[32], bf[6], cospi[32], bf[7]);
        step[7] = Btf(cospi[32], bf[6], -cospi[32], bf[7]);
        step[8] = bf[8];
        step[9] = bf[9];
        step[10] = Btf(cospi[32], bf[10], cospi[32], bf[11]);
        step[11] = Btf(cospi[32], bf[10], -cospi[32], bf[11]);
        step[12] = bf[12];
        step[13] = bf[13];
        step[14] = Btf(cospi[32], bf[14], cospi[32], bf[15]);
        step[15] = Btf(cospi[32], bf[14], -cospi[32], bf[15]);

        // Stage 9: final permutation with sign flips
        output[0] = step[0];
        output[1] = -step[8];
        output[2] = step[12];
        output[3] = -step[4];
        output[4] = step[6];
        output[5] = -step[14];
        output[6] = step[10];
        output[7] = -step[2];
        output[8] = step[3];
        output[9] = -step[11];
        output[10] = step[15];
        output[11] = -step[7];
        output[12] = step[5];
        output[13] = -step[13];
        output[14] = step[9];
        output[15] = -step[1];
    }
}
