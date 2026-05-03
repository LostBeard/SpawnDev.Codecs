// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 8-point forward Asymmetric DST. Bit-exact port of libvpx
// vp9/encoder/vp9_dct.c fadst8. 3 stages with cospi multiplications,
// final negation pattern.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 8-point forward ADST (encoder side).</summary>
public static class Vp9ForwardAdst8
{
    /// <summary>8-point forward ADST. Mirrors libvpx <c>fadst8</c>.</summary>
    public static void Transform(ReadOnlySpan<int> input, Span<int> output)
    {
        if (input.Length < 8) throw new ArgumentException("input must have 8 entries", nameof(input));
        if (output.Length < 8) throw new ArgumentException("output must have 8 entries", nameof(output));

        // Input remap (libvpx convention: x_i indexes shuffled).
        long x0 = input[7], x1 = input[0], x2 = input[5], x3 = input[2];
        long x4 = input[3], x5 = input[4], x6 = input[1], x7 = input[6];

        // Stage 1
        long s0 = (long)Vp9CospiConstants.Cospi2_64 * x0 + (long)Vp9CospiConstants.Cospi30_64 * x1;
        long s1 = (long)Vp9CospiConstants.Cospi30_64 * x0 - (long)Vp9CospiConstants.Cospi2_64 * x1;
        long s2 = (long)Vp9CospiConstants.Cospi10_64 * x2 + (long)Vp9CospiConstants.Cospi22_64 * x3;
        long s3 = (long)Vp9CospiConstants.Cospi22_64 * x2 - (long)Vp9CospiConstants.Cospi10_64 * x3;
        long s4 = (long)Vp9CospiConstants.Cospi18_64 * x4 + (long)Vp9CospiConstants.Cospi14_64 * x5;
        long s5 = (long)Vp9CospiConstants.Cospi14_64 * x4 - (long)Vp9CospiConstants.Cospi18_64 * x5;
        long s6 = (long)Vp9CospiConstants.Cospi26_64 * x6 + (long)Vp9CospiConstants.Cospi6_64 * x7;
        long s7 = (long)Vp9CospiConstants.Cospi6_64 * x6 - (long)Vp9CospiConstants.Cospi26_64 * x7;

        x0 = Vp9CospiConstants.RoundShift(s0 + s4);
        x1 = Vp9CospiConstants.RoundShift(s1 + s5);
        x2 = Vp9CospiConstants.RoundShift(s2 + s6);
        x3 = Vp9CospiConstants.RoundShift(s3 + s7);
        x4 = Vp9CospiConstants.RoundShift(s0 - s4);
        x5 = Vp9CospiConstants.RoundShift(s1 - s5);
        x6 = Vp9CospiConstants.RoundShift(s2 - s6);
        x7 = Vp9CospiConstants.RoundShift(s3 - s7);

        // Stage 2
        long t0 = x0, t1 = x1, t2 = x2, t3 = x3;
        long t4 = (long)Vp9CospiConstants.Cospi8_64 * x4 + (long)Vp9CospiConstants.Cospi24_64 * x5;
        long t5 = (long)Vp9CospiConstants.Cospi24_64 * x4 - (long)Vp9CospiConstants.Cospi8_64 * x5;
        long t6 = -(long)Vp9CospiConstants.Cospi24_64 * x6 + (long)Vp9CospiConstants.Cospi8_64 * x7;
        long t7 = (long)Vp9CospiConstants.Cospi8_64 * x6 + (long)Vp9CospiConstants.Cospi24_64 * x7;

        x0 = t0 + t2;
        x1 = t1 + t3;
        x2 = t0 - t2;
        x3 = t1 - t3;
        x4 = Vp9CospiConstants.RoundShift(t4 + t6);
        x5 = Vp9CospiConstants.RoundShift(t5 + t7);
        x6 = Vp9CospiConstants.RoundShift(t4 - t6);
        x7 = Vp9CospiConstants.RoundShift(t5 - t7);

        // Stage 3
        long u2 = (long)Vp9CospiConstants.Cospi16_64 * (x2 + x3);
        long u3 = (long)Vp9CospiConstants.Cospi16_64 * (x2 - x3);
        long u6 = (long)Vp9CospiConstants.Cospi16_64 * (x6 + x7);
        long u7 = (long)Vp9CospiConstants.Cospi16_64 * (x6 - x7);

        x2 = Vp9CospiConstants.RoundShift(u2);
        x3 = Vp9CospiConstants.RoundShift(u3);
        x6 = Vp9CospiConstants.RoundShift(u6);
        x7 = Vp9CospiConstants.RoundShift(u7);

        // Output (libvpx negation pattern)
        output[0] = (int)x0;
        output[1] = (int)-x4;
        output[2] = (int)x6;
        output[3] = (int)-x2;
        output[4] = (int)x3;
        output[5] = (int)-x7;
        output[6] = (int)x5;
        output[7] = (int)-x1;
    }
}
