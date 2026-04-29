// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 8-point forward Asymmetric DST, GPU-callable form for in-kernel
// reuse. Bit-exact mirror of Vp9ForwardAdst8.Transform (libvpx
// vp9/encoder/vp9_dct.c fadst8 port). 3 stages with cospi
// multiplications + final negation pattern.
//
// Pairs with the existing Vp9Iadst8x8Gpu (decoder side) - now both
// directions of the VP9 8-point ADST have GPU primitives.

using ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// GPU-callable VP9 8-point forward ADST helper. Bit-exact mirror of
/// <see cref="Vp9ForwardAdst8"/> for in-kernel use.
/// </summary>
public static class Vp9ForwardAdst8Gpu
{
    private const int Cospi2_64 = 16305;
    private const int Cospi6_64 = 15679;
    private const int Cospi8_64 = 15137;
    private const int Cospi10_64 = 14449;
    private const int Cospi14_64 = 12665;
    private const int Cospi16_64 = 11585;
    private const int Cospi18_64 = 10394;
    private const int Cospi22_64 = 7723;
    private const int Cospi24_64 = 6270;
    private const int Cospi26_64 = 4756;
    private const int Cospi30_64 = 1606;
    private const int DctConstBits = 14;
    private const int DctConstRounding = 1 << (DctConstBits - 1);

    /// <summary>
    /// Apply the 8-point forward ADST to one 8-element 1D row/column.
    /// </summary>
    public static void Forward8(
        ArrayView<int> input, long inBase,
        ArrayView<int> output, long outBase)
    {
        // Input remap (libvpx convention).
        long x0 = input[inBase + 7];
        long x1 = input[inBase + 0];
        long x2 = input[inBase + 5];
        long x3 = input[inBase + 2];
        long x4 = input[inBase + 3];
        long x5 = input[inBase + 4];
        long x6 = input[inBase + 1];
        long x7 = input[inBase + 6];

        // Stage 1.
        long s0 = (long)Cospi2_64 * x0 + (long)Cospi30_64 * x1;
        long s1 = (long)Cospi30_64 * x0 - (long)Cospi2_64 * x1;
        long s2 = (long)Cospi10_64 * x2 + (long)Cospi22_64 * x3;
        long s3 = (long)Cospi22_64 * x2 - (long)Cospi10_64 * x3;
        long s4 = (long)Cospi18_64 * x4 + (long)Cospi14_64 * x5;
        long s5 = (long)Cospi14_64 * x4 - (long)Cospi18_64 * x5;
        long s6 = (long)Cospi26_64 * x6 + (long)Cospi6_64 * x7;
        long s7 = (long)Cospi6_64 * x6 - (long)Cospi26_64 * x7;

        x0 = RoundShift(s0 + s4);
        x1 = RoundShift(s1 + s5);
        x2 = RoundShift(s2 + s6);
        x3 = RoundShift(s3 + s7);
        x4 = RoundShift(s0 - s4);
        x5 = RoundShift(s1 - s5);
        x6 = RoundShift(s2 - s6);
        x7 = RoundShift(s3 - s7);

        // Stage 2.
        long t0 = x0, t1 = x1, t2 = x2, t3 = x3;
        long t4 = (long)Cospi8_64 * x4 + (long)Cospi24_64 * x5;
        long t5 = (long)Cospi24_64 * x4 - (long)Cospi8_64 * x5;
        long t6 = -(long)Cospi24_64 * x6 + (long)Cospi8_64 * x7;
        long t7 = (long)Cospi8_64 * x6 + (long)Cospi24_64 * x7;

        x0 = t0 + t2;
        x1 = t1 + t3;
        x2 = t0 - t2;
        x3 = t1 - t3;
        x4 = RoundShift(t4 + t6);
        x5 = RoundShift(t5 + t7);
        x6 = RoundShift(t4 - t6);
        x7 = RoundShift(t5 - t7);

        // Stage 3.
        long u2 = (long)Cospi16_64 * (x2 + x3);
        long u3 = (long)Cospi16_64 * (x2 - x3);
        long u6 = (long)Cospi16_64 * (x6 + x7);
        long u7 = (long)Cospi16_64 * (x6 - x7);

        x2 = RoundShift(u2);
        x3 = RoundShift(u3);
        x6 = RoundShift(u6);
        x7 = RoundShift(u7);

        // Output with libvpx negation pattern.
        output[outBase + 0] = (int)x0;
        output[outBase + 1] = (int)-x4;
        output[outBase + 2] = (int)x6;
        output[outBase + 3] = (int)-x2;
        output[outBase + 4] = (int)x3;
        output[outBase + 5] = (int)-x7;
        output[outBase + 6] = (int)x5;
        output[outBase + 7] = (int)-x1;
    }

    /// <summary>libvpx round_shift: (input + 1 &lt;&lt; 13) &gt;&gt; 14.</summary>
    private static long RoundShift(long input) =>
        (input + DctConstRounding) >> DctConstBits;
}
