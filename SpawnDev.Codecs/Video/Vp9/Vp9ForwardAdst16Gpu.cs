// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 16-point forward Asymmetric DST, GPU-callable form for in-kernel
// reuse. Bit-exact mirror of Vp9ForwardAdst16.Transform (libvpx
// vp9/encoder/vp9_dct.c fadst16 port). 4 stages with cospi
// multiplications + final negation pattern.
//
// Pairs with the existing Vp9Iadst16x16Gpu (decoder side) - now both
// directions of the VP9 16-point ADST have GPU primitives. Combined
// with the just-shipped Vp9ForwardAdst4Gpu and Vp9ForwardAdst8Gpu, the
// VP9 forward ADST family is now GPU-callable at all 3 valid sizes
// (4 / 8 / 16); 32-point is iDCT-only per VP9 spec.

using ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// GPU-callable VP9 16-point forward ADST helper. Bit-exact mirror of
/// <see cref="Vp9ForwardAdst16"/> for in-kernel use.
/// </summary>
public static class Vp9ForwardAdst16Gpu
{
    private const int Cospi1_64 = 16364;
    private const int Cospi3_64 = 16207;
    private const int Cospi4_64 = 16069;
    private const int Cospi5_64 = 15893;
    private const int Cospi7_64 = 15426;
    private const int Cospi8_64 = 15137;
    private const int Cospi9_64 = 14811;
    private const int Cospi11_64 = 14053;
    private const int Cospi12_64 = 13623;
    private const int Cospi13_64 = 13160;
    private const int Cospi15_64 = 12140;
    private const int Cospi16_64 = 11585;
    private const int Cospi17_64 = 11003;
    private const int Cospi19_64 = 9760;
    private const int Cospi20_64 = 9102;
    private const int Cospi21_64 = 8423;
    private const int Cospi23_64 = 7005;
    private const int Cospi24_64 = 6270;
    private const int Cospi25_64 = 5520;
    private const int Cospi27_64 = 3981;
    private const int Cospi28_64 = 3196;
    private const int Cospi29_64 = 2404;
    private const int Cospi31_64 = 804;
    private const int DctConstBits = 14;
    private const int DctConstRounding = 1 << (DctConstBits - 1);

    /// <summary>
    /// Apply the 16-point forward ADST to one 16-element 1D row/column.
    /// </summary>
    public static void Forward16(
        ArrayView<int> input, long inBase,
        ArrayView<int> output, long outBase)
    {
        // Input remap (libvpx index pattern).
        long x0  = input[inBase + 15];
        long x1  = input[inBase + 0];
        long x2  = input[inBase + 13];
        long x3  = input[inBase + 2];
        long x4  = input[inBase + 11];
        long x5  = input[inBase + 4];
        long x6  = input[inBase + 9];
        long x7  = input[inBase + 6];
        long x8  = input[inBase + 7];
        long x9  = input[inBase + 8];
        long x10 = input[inBase + 5];
        long x11 = input[inBase + 10];
        long x12 = input[inBase + 3];
        long x13 = input[inBase + 12];
        long x14 = input[inBase + 1];
        long x15 = input[inBase + 14];

        // Stage 1.
        long s0  = x0 * Cospi1_64  + x1 * Cospi31_64;
        long s1  = x0 * Cospi31_64 - x1 * Cospi1_64;
        long s2  = x2 * Cospi5_64  + x3 * Cospi27_64;
        long s3  = x2 * Cospi27_64 - x3 * Cospi5_64;
        long s4  = x4 * Cospi9_64  + x5 * Cospi23_64;
        long s5  = x4 * Cospi23_64 - x5 * Cospi9_64;
        long s6  = x6 * Cospi13_64 + x7 * Cospi19_64;
        long s7  = x6 * Cospi19_64 - x7 * Cospi13_64;
        long s8  = x8 * Cospi17_64 + x9 * Cospi15_64;
        long s9  = x8 * Cospi15_64 - x9 * Cospi17_64;
        long s10 = x10 * Cospi21_64 + x11 * Cospi11_64;
        long s11 = x10 * Cospi11_64 - x11 * Cospi21_64;
        long s12 = x12 * Cospi25_64 + x13 * Cospi7_64;
        long s13 = x12 * Cospi7_64  - x13 * Cospi25_64;
        long s14 = x14 * Cospi29_64 + x15 * Cospi3_64;
        long s15 = x14 * Cospi3_64  - x15 * Cospi29_64;

        x0  = RoundShift(s0 + s8);
        x1  = RoundShift(s1 + s9);
        x2  = RoundShift(s2 + s10);
        x3  = RoundShift(s3 + s11);
        x4  = RoundShift(s4 + s12);
        x5  = RoundShift(s5 + s13);
        x6  = RoundShift(s6 + s14);
        x7  = RoundShift(s7 + s15);
        x8  = RoundShift(s0 - s8);
        x9  = RoundShift(s1 - s9);
        x10 = RoundShift(s2 - s10);
        x11 = RoundShift(s3 - s11);
        x12 = RoundShift(s4 - s12);
        x13 = RoundShift(s5 - s13);
        x14 = RoundShift(s6 - s14);
        x15 = RoundShift(s7 - s15);

        // Stage 2.
        s0 = x0; s1 = x1; s2 = x2; s3 = x3;
        s4 = x4; s5 = x5; s6 = x6; s7 = x7;
        s8  = x8 * Cospi4_64   + x9 * Cospi28_64;
        s9  = x8 * Cospi28_64  - x9 * Cospi4_64;
        s10 = x10 * Cospi20_64 + x11 * Cospi12_64;
        s11 = x10 * Cospi12_64 - x11 * Cospi20_64;
        s12 = -x12 * Cospi28_64 + x13 * Cospi4_64;
        s13 = x12 * Cospi4_64   + x13 * Cospi28_64;
        s14 = -x14 * Cospi12_64 + x15 * Cospi20_64;
        s15 = x14 * Cospi20_64  + x15 * Cospi12_64;

        x0 = s0 + s4;
        x1 = s1 + s5;
        x2 = s2 + s6;
        x3 = s3 + s7;
        x4 = s0 - s4;
        x5 = s1 - s5;
        x6 = s2 - s6;
        x7 = s3 - s7;
        x8  = RoundShift(s8 + s12);
        x9  = RoundShift(s9 + s13);
        x10 = RoundShift(s10 + s14);
        x11 = RoundShift(s11 + s15);
        x12 = RoundShift(s8 - s12);
        x13 = RoundShift(s9 - s13);
        x14 = RoundShift(s10 - s14);
        x15 = RoundShift(s11 - s15);

        // Stage 3.
        s0 = x0; s1 = x1; s2 = x2; s3 = x3;
        s4 = x4 * Cospi8_64  + x5 * Cospi24_64;
        s5 = x4 * Cospi24_64 - x5 * Cospi8_64;
        s6 = -x6 * Cospi24_64 + x7 * Cospi8_64;
        s7 = x6 * Cospi8_64   + x7 * Cospi24_64;
        s8 = x8; s9 = x9; s10 = x10; s11 = x11;
        s12 = x12 * Cospi8_64  + x13 * Cospi24_64;
        s13 = x12 * Cospi24_64 - x13 * Cospi8_64;
        s14 = -x14 * Cospi24_64 + x15 * Cospi8_64;
        s15 = x14 * Cospi8_64   + x15 * Cospi24_64;

        x0 = s0 + s2;
        x1 = s1 + s3;
        x2 = s0 - s2;
        x3 = s1 - s3;
        x4 = RoundShift(s4 + s6);
        x5 = RoundShift(s5 + s7);
        x6 = RoundShift(s4 - s6);
        x7 = RoundShift(s5 - s7);
        x8 = s8 + s10;
        x9 = s9 + s11;
        x10 = s8 - s10;
        x11 = s9 - s11;
        x12 = RoundShift(s12 + s14);
        x13 = RoundShift(s13 + s15);
        x14 = RoundShift(s12 - s14);
        x15 = RoundShift(s13 - s15);

        // Stage 4.
        s2  = -(long)Cospi16_64 * (x2 + x3);
        s3  = (long)Cospi16_64 * (x2 - x3);
        s6  = (long)Cospi16_64 * (x6 + x7);
        s7  = (long)Cospi16_64 * (-x6 + x7);
        s10 = (long)Cospi16_64 * (x10 + x11);
        s11 = (long)Cospi16_64 * (-x10 + x11);
        s14 = -(long)Cospi16_64 * (x14 + x15);
        s15 = (long)Cospi16_64 * (x14 - x15);

        x2  = RoundShift(s2);
        x3  = RoundShift(s3);
        x6  = RoundShift(s6);
        x7  = RoundShift(s7);
        x10 = RoundShift(s10);
        x11 = RoundShift(s11);
        x14 = RoundShift(s14);
        x15 = RoundShift(s15);

        // Output with libvpx negation pattern.
        output[outBase + 0]  = (int)x0;
        output[outBase + 1]  = (int)-x8;
        output[outBase + 2]  = (int)x12;
        output[outBase + 3]  = (int)-x4;
        output[outBase + 4]  = (int)x6;
        output[outBase + 5]  = (int)x14;
        output[outBase + 6]  = (int)x10;
        output[outBase + 7]  = (int)x2;
        output[outBase + 8]  = (int)x3;
        output[outBase + 9]  = (int)x11;
        output[outBase + 10] = (int)x15;
        output[outBase + 11] = (int)x7;
        output[outBase + 12] = (int)x5;
        output[outBase + 13] = (int)-x13;
        output[outBase + 14] = (int)x9;
        output[outBase + 15] = (int)-x1;
    }

    /// <summary>libvpx round_shift: (input + 1 &lt;&lt; 13) &gt;&gt; 14.</summary>
    private static long RoundShift(long input) =>
        (input + DctConstRounding) >> DctConstBits;
}
