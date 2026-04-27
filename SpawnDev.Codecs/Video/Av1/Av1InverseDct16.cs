// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 16-point inverse DCT (1D). Bit-exact port of libaom
// av1/common/av1_inv_txfm1d.c av1_idct16.
//
// Upstream Copyright (c) 2016, Alliance for Open Media.
// Upstream license: BSD 2-Clause + AV1 Patent License 1.0. See NOTICE.md.

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>AV1 16-point inverse DCT (1D building block).</summary>
public static class Av1InverseDct16
{
    /// <summary>libaom default cos_bit for the inverse 16-point DCT.</summary>
    public const int DefaultCosBit = 12;

    /// <summary>
    /// 16-point inverse DCT. Mirrors libaom <c>av1_idct16</c>.
    /// </summary>
    public static void Transform(ReadOnlySpan<int> input, Span<int> output, int cosBit = DefaultCosBit)
    {
        if (input.Length < 16) throw new ArgumentException("input must have 16 entries", nameof(input));
        if (output.Length < 16) throw new ArgumentException("output must have 16 entries", nameof(output));
        if (cosBit < 10 || cosBit > 13) throw new ArgumentOutOfRangeException(nameof(cosBit), "must be in [10, 13]");

        int Btf(int w0, int in0, int w1, int in1) =>
            Av1ForwardDct4.HalfBtf(w0, in0, w1, in1, cosBit);

        var cospi = Av1ForwardDct4.CospiArr(cosBit);

        // Stage 1: input permutation (bit reverse)
        Span<int> bf = stackalloc int[16];
        bf[0] = input[0];
        bf[1] = input[8];
        bf[2] = input[4];
        bf[3] = input[12];
        bf[4] = input[2];
        bf[5] = input[10];
        bf[6] = input[6];
        bf[7] = input[14];
        bf[8] = input[1];
        bf[9] = input[9];
        bf[10] = input[5];
        bf[11] = input[13];
        bf[12] = input[3];
        bf[13] = input[11];
        bf[14] = input[7];
        bf[15] = input[15];

        // Stage 2: rotate upper half (8..15)
        Span<int> step = stackalloc int[16];
        step[0] = bf[0];
        step[1] = bf[1];
        step[2] = bf[2];
        step[3] = bf[3];
        step[4] = bf[4];
        step[5] = bf[5];
        step[6] = bf[6];
        step[7] = bf[7];
        step[8] = Btf(cospi[60], bf[8], -cospi[4], bf[15]);
        step[9] = Btf(cospi[28], bf[9], -cospi[36], bf[14]);
        step[10] = Btf(cospi[44], bf[10], -cospi[20], bf[13]);
        step[11] = Btf(cospi[12], bf[11], -cospi[52], bf[12]);
        step[12] = Btf(cospi[52], bf[11], cospi[12], bf[12]);
        step[13] = Btf(cospi[20], bf[10], cospi[44], bf[13]);
        step[14] = Btf(cospi[36], bf[9], cospi[28], bf[14]);
        step[15] = Btf(cospi[4], bf[8], cospi[60], bf[15]);

        // Stage 3: rotate middle (4..7), butterfly upper (8..15)
        bf[0] = step[0];
        bf[1] = step[1];
        bf[2] = step[2];
        bf[3] = step[3];
        bf[4] = Btf(cospi[56], step[4], -cospi[8], step[7]);
        bf[5] = Btf(cospi[24], step[5], -cospi[40], step[6]);
        bf[6] = Btf(cospi[40], step[5], cospi[24], step[6]);
        bf[7] = Btf(cospi[8], step[4], cospi[56], step[7]);
        bf[8] = step[8] + step[9];
        bf[9] = step[8] - step[9];
        bf[10] = -step[10] + step[11];
        bf[11] = step[10] + step[11];
        bf[12] = step[12] + step[13];
        bf[13] = step[12] - step[13];
        bf[14] = -step[14] + step[15];
        bf[15] = step[14] + step[15];

        // Stage 4
        step[0] = Btf(cospi[32], bf[0], cospi[32], bf[1]);
        step[1] = Btf(cospi[32], bf[0], -cospi[32], bf[1]);
        step[2] = Btf(cospi[48], bf[2], -cospi[16], bf[3]);
        step[3] = Btf(cospi[16], bf[2], cospi[48], bf[3]);
        step[4] = bf[4] + bf[5];
        step[5] = bf[4] - bf[5];
        step[6] = -bf[6] + bf[7];
        step[7] = bf[6] + bf[7];
        step[8] = bf[8];
        step[9] = Btf(-cospi[16], bf[9], cospi[48], bf[14]);
        step[10] = Btf(-cospi[48], bf[10], -cospi[16], bf[13]);
        step[11] = bf[11];
        step[12] = bf[12];
        step[13] = Btf(-cospi[16], bf[10], cospi[48], bf[13]);
        step[14] = Btf(cospi[48], bf[9], cospi[16], bf[14]);
        step[15] = bf[15];

        // Stage 5
        bf[0] = step[0] + step[3];
        bf[1] = step[1] + step[2];
        bf[2] = step[1] - step[2];
        bf[3] = step[0] - step[3];
        bf[4] = step[4];
        bf[5] = Btf(-cospi[32], step[5], cospi[32], step[6]);
        bf[6] = Btf(cospi[32], step[5], cospi[32], step[6]);
        bf[7] = step[7];
        bf[8] = step[8] + step[11];
        bf[9] = step[9] + step[10];
        bf[10] = step[9] - step[10];
        bf[11] = step[8] - step[11];
        bf[12] = -step[12] + step[15];
        bf[13] = -step[13] + step[14];
        bf[14] = step[13] + step[14];
        bf[15] = step[12] + step[15];

        // Stage 6
        step[0] = bf[0] + bf[7];
        step[1] = bf[1] + bf[6];
        step[2] = bf[2] + bf[5];
        step[3] = bf[3] + bf[4];
        step[4] = bf[3] - bf[4];
        step[5] = bf[2] - bf[5];
        step[6] = bf[1] - bf[6];
        step[7] = bf[0] - bf[7];
        step[8] = bf[8];
        step[9] = bf[9];
        step[10] = Btf(-cospi[32], bf[10], cospi[32], bf[13]);
        step[11] = Btf(-cospi[32], bf[11], cospi[32], bf[12]);
        step[12] = Btf(cospi[32], bf[11], cospi[32], bf[12]);
        step[13] = Btf(cospi[32], bf[10], cospi[32], bf[13]);
        step[14] = bf[14];
        step[15] = bf[15];

        // Stage 7: outer butterfly
        output[0] = step[0] + step[15];
        output[1] = step[1] + step[14];
        output[2] = step[2] + step[13];
        output[3] = step[3] + step[12];
        output[4] = step[4] + step[11];
        output[5] = step[5] + step[10];
        output[6] = step[6] + step[9];
        output[7] = step[7] + step[8];
        output[8] = step[7] - step[8];
        output[9] = step[6] - step[9];
        output[10] = step[5] - step[10];
        output[11] = step[4] - step[11];
        output[12] = step[3] - step[12];
        output[13] = step[2] - step[13];
        output[14] = step[1] - step[14];
        output[15] = step[0] - step[15];
    }
}
