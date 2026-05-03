// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 4-point forward Asymmetric DST. Bit-exact port of libvpx
// vp9/encoder/vp9_dct.c fadst4. Uses sinpi_<i>_9 constants from
// vpx_dsp/txfm_common.h.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 4-point forward ADST (encoder side).</summary>
public static class Vp9ForwardAdst4
{
    /// <summary>4-point forward ADST. Mirrors libvpx <c>fadst4</c>.</summary>
    public static void Transform(ReadOnlySpan<int> input, Span<int> output)
    {
        if (input.Length < 4) throw new ArgumentException("input must have 4 entries", nameof(input));
        if (output.Length < 4) throw new ArgumentException("output must have 4 entries", nameof(output));

        int x0 = input[0];
        int x1 = input[1];
        int x2 = input[2];
        int x3 = input[3];

        if ((x0 | x1 | x2 | x3) == 0)
        {
            output[0] = output[1] = output[2] = output[3] = 0;
            return;
        }

        long s0 = (long)Vp9CospiConstants.Sinpi1_9 * x0;
        long s1 = (long)Vp9CospiConstants.Sinpi4_9 * x0;
        long s2 = (long)Vp9CospiConstants.Sinpi2_9 * x1;
        long s3 = (long)Vp9CospiConstants.Sinpi1_9 * x1;
        long s4 = (long)Vp9CospiConstants.Sinpi3_9 * x2;
        long s5 = (long)Vp9CospiConstants.Sinpi4_9 * x3;
        long s6 = (long)Vp9CospiConstants.Sinpi2_9 * x3;
        long s7 = x0 + x1 - x3;

        long y0 = s0 + s2 + s5;
        long y1 = (long)Vp9CospiConstants.Sinpi3_9 * s7;
        long y2 = s1 - s3 + s6;
        long y3 = s4;

        long t0 = y0 + y3;
        long t1 = y1;
        long t2 = y2 - y3;
        long t3 = y2 - y0 + y3;

        output[0] = Vp9CospiConstants.RoundShift(t0);
        output[1] = Vp9CospiConstants.RoundShift(t1);
        output[2] = Vp9CospiConstants.RoundShift(t2);
        output[3] = Vp9CospiConstants.RoundShift(t3);
    }
}
