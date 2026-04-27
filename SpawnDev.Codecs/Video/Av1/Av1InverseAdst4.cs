// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 4-point inverse Asymmetric DST (1D). Bit-exact port of libaom
// av1/common/av1_inv_txfm1d.c av1_iadst4.
//
// Upstream Copyright (c) 2016, Alliance for Open Media.
// Upstream license: BSD 2-Clause + AV1 Patent License 1.0. See NOTICE.md.
//
// ADST = Asymmetric Discrete Sine Transform. AV1 pairs ADST with DCT
// in directional intra prediction modes - the DC mode uses DCT-DCT,
// while angled-from-edge modes use ADST/DCT or DCT/ADST or ADST/ADST
// depending on direction relative to the row/col axis.
//
// Reuses sinpi constants from <see cref="Av1ForwardAdst4"/>.

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>AV1 4-point inverse Asymmetric DST (1D).</summary>
public static class Av1InverseAdst4
{
    /// <summary>Default cos_bit for the inverse 4-point ADST (libaom).</summary>
    public const int DefaultCosBit = 12;

    /// <summary>
    /// 4-point inverse ADST. Mirrors libaom <c>av1_iadst4</c>.
    /// Uses 64-bit intermediates as libaom does because per-stage
    /// magnitudes can exceed 32-bit range before the final round_shift.
    /// </summary>
    public static void Transform(ReadOnlySpan<int> input, Span<int> output, int cosBit = DefaultCosBit)
    {
        if (input.Length < 4) throw new ArgumentException("input must have 4 entries", nameof(input));
        if (output.Length < 4) throw new ArgumentException("output must have 4 entries", nameof(output));
        if (cosBit < 10 || cosBit > 13) throw new ArgumentOutOfRangeException(nameof(cosBit));

        var sinpi = Av1ForwardAdst4.SinpiArr(cosBit);

        long x0 = input[0];
        long x1 = input[1];
        long x2 = input[2];
        long x3 = input[3];

        if ((x0 | x1 | x2 | x3) == 0)
        {
            output[0] = output[1] = output[2] = output[3] = 0;
            return;
        }

        // libaom invariant: sinpi[1] + sinpi[2] == sinpi[4]. Documented assert.

        // Stage 1
        long s0 = (long)sinpi[1] * x0;
        long s1 = (long)sinpi[2] * x0;
        long s2 = (long)sinpi[3] * x1;
        long s3 = (long)sinpi[4] * x2;
        long s4 = (long)sinpi[1] * x2;
        long s5 = (long)sinpi[2] * x3;
        long s6 = (long)sinpi[4] * x3;

        // Stage 2: s7 is in input domain (no per-bit scaling)
        long s7 = (x0 - x2) + x3;

        // Stage 3
        s0 = s0 + s3;
        s1 = s1 - s4;
        long sNew3 = s2;                       // overwrite s3
        long sNew2 = (long)sinpi[3] * s7;      // overwrite s2

        // Stage 4
        s0 = s0 + s5;
        s1 = s1 - s6;

        // Stage 5
        long y0 = s0 + sNew3;
        long y1 = s1 + sNew3;
        long y2 = sNew2;
        long y3 = s0 + s1;

        // Stage 6
        y3 = y3 - sNew3;

        output[0] = RoundShift(y0, cosBit);
        output[1] = RoundShift(y1, cosBit);
        output[2] = RoundShift(y2, cosBit);
        output[3] = RoundShift(y3, cosBit);
    }

    /// <summary>libaom <c>round_shift</c>: arithmetic round-half-up by <paramref name="bit"/>.</summary>
    public static int RoundShift(long value, int bit)
    {
        return (int)((value + (1L << (bit - 1))) >> bit);
    }
}
