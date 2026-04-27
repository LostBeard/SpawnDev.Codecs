// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 4-point inverse DCT (1D). Bit-exact port of libaom
// av1/common/av1_inv_txfm1d.c av1_idct4.
//
// Upstream Copyright (c) 2016, Alliance for Open Media.
// Upstream license: BSD 2-Clause + AV1 Patent License 1.0. See NOTICE.md.
// Upstream source: https://aomedia.googlesource.com/aom (av1/common/av1_inv_txfm1d.c)
//
// AV1 inverse 1D transforms reverse a forward 1D pass: the forward
// stages produce coefficients in low-frequency-first / interleave order;
// the inverse stages re-permute the input then apply the reverse
// butterfly + cospi multiplication chain.
//
// Cosine constants and half_btf primitive are shared with the forward
// transforms in <see cref="Av1ForwardDct4"/>. Cosine precision (cos_bit)
// is configurable per call; the AV1 inverse transform config table picks
// a per-tx-size cos_bit, but for the 4-point DCT default is 12.

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>AV1 4-point inverse DCT (1D building block).</summary>
public static class Av1InverseDct4
{
    /// <summary>libaom default cos_bit for the inverse 4-point DCT.</summary>
    public const int DefaultCosBit = 12;

    /// <summary>
    /// 4-point inverse DCT. Mirrors libaom <c>av1_idct4</c>.
    /// <paramref name="output"/> must NOT alias <paramref name="input"/>.
    /// </summary>
    public static void Transform(ReadOnlySpan<int> input, Span<int> output, int cosBit = DefaultCosBit)
    {
        if (input.Length < 4) throw new ArgumentException("input must have 4 entries", nameof(input));
        if (output.Length < 4) throw new ArgumentException("output must have 4 entries", nameof(output));
        if (cosBit < 10 || cosBit > 13) throw new ArgumentOutOfRangeException(nameof(cosBit), "must be in [10, 13]");

        var cospi = Av1ForwardDct4.CospiArr(cosBit);

        // Stage 1: re-permute input
        Span<int> bf = stackalloc int[4];
        bf[0] = input[0];
        bf[1] = input[2];
        bf[2] = input[1];
        bf[3] = input[3];

        // Stage 2: cospi butterfly (top 2 + bottom 2 with rotated weights)
        Span<int> step = stackalloc int[4];
        step[0] = Av1ForwardDct4.HalfBtf(cospi[32], bf[0], cospi[32], bf[1], cosBit);
        step[1] = Av1ForwardDct4.HalfBtf(cospi[32], bf[0], -cospi[32], bf[1], cosBit);
        step[2] = Av1ForwardDct4.HalfBtf(cospi[48], bf[2], -cospi[16], bf[3], cosBit);
        step[3] = Av1ForwardDct4.HalfBtf(cospi[16], bf[2], cospi[48], bf[3], cosBit);

        // Stage 3: outer butterfly (mirror of stage 1 of forward transform)
        output[0] = step[0] + step[3];
        output[1] = step[1] + step[2];
        output[2] = step[1] - step[2];
        output[3] = step[0] - step[3];
    }
}
