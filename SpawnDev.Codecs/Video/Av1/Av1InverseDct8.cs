// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 8-point inverse DCT (1D). Bit-exact port of libaom
// av1/common/av1_inv_txfm1d.c av1_idct8.
//
// Upstream Copyright (c) 2016, Alliance for Open Media.
// Upstream license: BSD 2-Clause + AV1 Patent License 1.0. See NOTICE.md.

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>AV1 8-point inverse DCT (1D building block).</summary>
public static class Av1InverseDct8
{
    /// <summary>libaom default cos_bit for the inverse 8-point DCT.</summary>
    public const int DefaultCosBit = 12;

    /// <summary>
    /// 8-point inverse DCT. Mirrors libaom <c>av1_idct8</c>.
    /// </summary>
    public static void Transform(ReadOnlySpan<int> input, Span<int> output, int cosBit = DefaultCosBit)
    {
        if (input.Length < 8) throw new ArgumentException("input must have 8 entries", nameof(input));
        if (output.Length < 8) throw new ArgumentException("output must have 8 entries", nameof(output));
        if (cosBit < 10 || cosBit > 13) throw new ArgumentOutOfRangeException(nameof(cosBit), "must be in [10, 13]");

        var cospi = Av1ForwardDct4.CospiArr(cosBit);

        // Stage 1: re-permute input.
        Span<int> bf = stackalloc int[8];
        bf[0] = input[0];
        bf[1] = input[4];
        bf[2] = input[2];
        bf[3] = input[6];
        bf[4] = input[1];
        bf[5] = input[5];
        bf[6] = input[3];
        bf[7] = input[7];

        // Stage 2: cospi rotation on the upper half (4..7).
        Span<int> step = stackalloc int[8];
        step[0] = bf[0];
        step[1] = bf[1];
        step[2] = bf[2];
        step[3] = bf[3];
        step[4] = Av1ForwardDct4.HalfBtf(cospi[56], bf[4], -cospi[8], bf[7], cosBit);
        step[5] = Av1ForwardDct4.HalfBtf(cospi[24], bf[5], -cospi[40], bf[6], cosBit);
        step[6] = Av1ForwardDct4.HalfBtf(cospi[40], bf[5], cospi[24], bf[6], cosBit);
        step[7] = Av1ForwardDct4.HalfBtf(cospi[8], bf[4], cospi[56], bf[7], cosBit);

        // Stage 3: cospi butterfly on lower 4 + add/sub on upper 4.
        bf[0] = Av1ForwardDct4.HalfBtf(cospi[32], step[0], cospi[32], step[1], cosBit);
        bf[1] = Av1ForwardDct4.HalfBtf(cospi[32], step[0], -cospi[32], step[1], cosBit);
        bf[2] = Av1ForwardDct4.HalfBtf(cospi[48], step[2], -cospi[16], step[3], cosBit);
        bf[3] = Av1ForwardDct4.HalfBtf(cospi[16], step[2], cospi[48], step[3], cosBit);
        bf[4] = step[4] + step[5];
        bf[5] = step[4] - step[5];
        bf[6] = -step[6] + step[7];
        bf[7] = step[6] + step[7];

        // Stage 4: butterfly + cospi rotation on middle 2 of upper.
        step[0] = bf[0] + bf[3];
        step[1] = bf[1] + bf[2];
        step[2] = bf[1] - bf[2];
        step[3] = bf[0] - bf[3];
        step[4] = bf[4];
        step[5] = Av1ForwardDct4.HalfBtf(-cospi[32], bf[5], cospi[32], bf[6], cosBit);
        step[6] = Av1ForwardDct4.HalfBtf(cospi[32], bf[5], cospi[32], bf[6], cosBit);
        step[7] = bf[7];

        // Stage 5: outer butterfly.
        output[0] = step[0] + step[7];
        output[1] = step[1] + step[6];
        output[2] = step[2] + step[5];
        output[3] = step[3] + step[4];
        output[4] = step[3] - step[4];
        output[5] = step[2] - step[5];
        output[6] = step[1] - step[6];
        output[7] = step[0] - step[7];
    }
}
