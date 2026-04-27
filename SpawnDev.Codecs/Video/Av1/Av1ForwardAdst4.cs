// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 4-point forward Asymmetric DST (1D). Bit-exact port of libaom
// av1/encoder/av1_fwd_txfm1d.c av1_fadst4.
//
// Uses sinpi constants from libaom av1_txfm.c sinpi_arr_data[4][5]
// (5 sin values for the ADST). The ADST is paired with DCT for
// directional intra prediction modes in AV1.

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>AV1 4-point forward Asymmetric DST (1D).</summary>
public static class Av1ForwardAdst4
{
    /// <summary>Default cosine-precision bits (libaom).</summary>
    public const int DefaultCosBit = 13;

    /// <summary>
    /// libaom <c>av1_sinpi_arr_data</c> for cos_bit=10..13.
    /// 5 values per row representing sin(pi*i/9) * 2^cos_bit for i in 0..4.
    /// </summary>
    public static readonly int[][] SinpiArrData = new int[][]
    {
        new int[] { 0,  330,  621,  836,  951 }, // bit 10
        new int[] { 0,  660, 1241, 1672, 1901 }, // bit 11
        new int[] { 0, 1321, 2482, 3344, 3803 }, // bit 12
        new int[] { 0, 2642, 4964, 6688, 7606 }, // bit 13
    };

    /// <summary>libaom <c>sinpi_arr(cos_bit)</c>.</summary>
    public static int[] SinpiArr(int cosBit) => SinpiArrData[cosBit - 10];

    /// <summary>4-point forward ADST. Mirrors libaom <c>av1_fadst4</c>.</summary>
    public static void Transform(ReadOnlySpan<int> input, Span<int> output, int cosBit = DefaultCosBit)
    {
        if (input.Length < 4) throw new ArgumentException("input must have 4 entries", nameof(input));
        if (output.Length < 4) throw new ArgumentException("output must have 4 entries", nameof(output));
        if (cosBit < 10 || cosBit > 13) throw new ArgumentOutOfRangeException(nameof(cosBit));

        var sinpi = SinpiArr(cosBit);

        int x0 = input[0];
        int x1 = input[1];
        int x2 = input[2];
        int x3 = input[3];

        if ((x0 | x1 | x2 | x3) == 0)
        {
            output[0] = output[1] = output[2] = output[3] = 0;
            return;
        }

        // Stage 1
        long s0 = (long)sinpi[1] * x0;
        long s1 = (long)sinpi[4] * x0;
        long s2 = (long)sinpi[2] * x1;
        long s3 = (long)sinpi[1] * x1;
        long s4 = (long)sinpi[3] * x2;
        long s5 = (long)sinpi[4] * x3;
        long s6 = (long)sinpi[2] * x3;
        long s7 = x0 + x1;

        // Stage 2
        s7 = s7 - x3;

        // Stage 3
        long y0 = s0 + s2;
        long y1 = (long)sinpi[3] * s7;
        long y2 = s1 - s3;
        long y3 = s4;

        // Stage 4
        y0 += s5;
        y2 += s6;

        // Stage 5
        long t0 = y0 + y3;
        long t1 = y1;
        long t2 = y2 - y3;
        long t3 = y2 - y0;

        // Stage 6
        t3 += y3;

        // 1-D ADST scaling factor: round_shift by cos_bit.
        output[0] = RoundShift(t0, cosBit);
        output[1] = RoundShift(t1, cosBit);
        output[2] = RoundShift(t2, cosBit);
        output[3] = RoundShift(t3, cosBit);
    }

    private static int RoundShift(long value, int bit)
    {
        return (int)((value + (1L << (bit - 1))) >> bit);
    }
}
