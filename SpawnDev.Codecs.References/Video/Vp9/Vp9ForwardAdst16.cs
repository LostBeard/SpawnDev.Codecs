// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 16-point forward Asymmetric DST. Bit-exact port of libvpx
// vp9/encoder/vp9_dct.c fadst16. Four stages with cospi_<i>_64
// multiplications and a final 4-stage cross-pattern.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 16-point forward ADST (encoder side).</summary>
public static class Vp9ForwardAdst16
{
    /// <summary>16-point forward ADST. Mirrors libvpx <c>fadst16</c>.</summary>
    public static void Transform(ReadOnlySpan<int> input, Span<int> output)
    {
        if (input.Length < 16) throw new ArgumentException("input must have 16 entries", nameof(input));
        if (output.Length < 16) throw new ArgumentException("output must have 16 entries", nameof(output));

        // Input remap (libvpx index pattern).
        long x0  = input[15];
        long x1  = input[0];
        long x2  = input[13];
        long x3  = input[2];
        long x4  = input[11];
        long x5  = input[4];
        long x6  = input[9];
        long x7  = input[6];
        long x8  = input[7];
        long x9  = input[8];
        long x10 = input[5];
        long x11 = input[10];
        long x12 = input[3];
        long x13 = input[12];
        long x14 = input[1];
        long x15 = input[14];

        long s0, s1, s2, s3, s4, s5, s6, s7;
        long s8, s9, s10, s11, s12, s13, s14, s15;

        // Stage 1
        s0  = x0 * Vp9CospiConstants.Cospi1_64  + x1 * Vp9CospiConstants.Cospi31_64;
        s1  = x0 * Vp9CospiConstants.Cospi31_64 - x1 * Vp9CospiConstants.Cospi1_64;
        s2  = x2 * Vp9CospiConstants.Cospi5_64  + x3 * Vp9CospiConstants.Cospi27_64;
        s3  = x2 * Vp9CospiConstants.Cospi27_64 - x3 * Vp9CospiConstants.Cospi5_64;
        s4  = x4 * Vp9CospiConstants.Cospi9_64  + x5 * Vp9CospiConstants.Cospi23_64;
        s5  = x4 * Vp9CospiConstants.Cospi23_64 - x5 * Vp9CospiConstants.Cospi9_64;
        s6  = x6 * Vp9CospiConstants.Cospi13_64 + x7 * Vp9CospiConstants.Cospi19_64;
        s7  = x6 * Vp9CospiConstants.Cospi19_64 - x7 * Vp9CospiConstants.Cospi13_64;
        s8  = x8 * Vp9CospiConstants.Cospi17_64 + x9 * Vp9CospiConstants.Cospi15_64;
        s9  = x8 * Vp9CospiConstants.Cospi15_64 - x9 * Vp9CospiConstants.Cospi17_64;
        s10 = x10 * Vp9CospiConstants.Cospi21_64 + x11 * Vp9CospiConstants.Cospi11_64;
        s11 = x10 * Vp9CospiConstants.Cospi11_64 - x11 * Vp9CospiConstants.Cospi21_64;
        s12 = x12 * Vp9CospiConstants.Cospi25_64 + x13 * Vp9CospiConstants.Cospi7_64;
        s13 = x12 * Vp9CospiConstants.Cospi7_64  - x13 * Vp9CospiConstants.Cospi25_64;
        s14 = x14 * Vp9CospiConstants.Cospi29_64 + x15 * Vp9CospiConstants.Cospi3_64;
        s15 = x14 * Vp9CospiConstants.Cospi3_64  - x15 * Vp9CospiConstants.Cospi29_64;

        x0  = Vp9CospiConstants.RoundShift(s0 + s8);
        x1  = Vp9CospiConstants.RoundShift(s1 + s9);
        x2  = Vp9CospiConstants.RoundShift(s2 + s10);
        x3  = Vp9CospiConstants.RoundShift(s3 + s11);
        x4  = Vp9CospiConstants.RoundShift(s4 + s12);
        x5  = Vp9CospiConstants.RoundShift(s5 + s13);
        x6  = Vp9CospiConstants.RoundShift(s6 + s14);
        x7  = Vp9CospiConstants.RoundShift(s7 + s15);
        x8  = Vp9CospiConstants.RoundShift(s0 - s8);
        x9  = Vp9CospiConstants.RoundShift(s1 - s9);
        x10 = Vp9CospiConstants.RoundShift(s2 - s10);
        x11 = Vp9CospiConstants.RoundShift(s3 - s11);
        x12 = Vp9CospiConstants.RoundShift(s4 - s12);
        x13 = Vp9CospiConstants.RoundShift(s5 - s13);
        x14 = Vp9CospiConstants.RoundShift(s6 - s14);
        x15 = Vp9CospiConstants.RoundShift(s7 - s15);

        // Stage 2
        s0 = x0; s1 = x1; s2 = x2; s3 = x3;
        s4 = x4; s5 = x5; s6 = x6; s7 = x7;
        s8  = x8 * Vp9CospiConstants.Cospi4_64   + x9 * Vp9CospiConstants.Cospi28_64;
        s9  = x8 * Vp9CospiConstants.Cospi28_64  - x9 * Vp9CospiConstants.Cospi4_64;
        s10 = x10 * Vp9CospiConstants.Cospi20_64 + x11 * Vp9CospiConstants.Cospi12_64;
        s11 = x10 * Vp9CospiConstants.Cospi12_64 - x11 * Vp9CospiConstants.Cospi20_64;
        s12 = -x12 * Vp9CospiConstants.Cospi28_64 + x13 * Vp9CospiConstants.Cospi4_64;
        s13 = x12 * Vp9CospiConstants.Cospi4_64   + x13 * Vp9CospiConstants.Cospi28_64;
        s14 = -x14 * Vp9CospiConstants.Cospi12_64 + x15 * Vp9CospiConstants.Cospi20_64;
        s15 = x14 * Vp9CospiConstants.Cospi20_64  + x15 * Vp9CospiConstants.Cospi12_64;

        x0 = s0 + s4;
        x1 = s1 + s5;
        x2 = s2 + s6;
        x3 = s3 + s7;
        x4 = s0 - s4;
        x5 = s1 - s5;
        x6 = s2 - s6;
        x7 = s3 - s7;
        x8  = Vp9CospiConstants.RoundShift(s8 + s12);
        x9  = Vp9CospiConstants.RoundShift(s9 + s13);
        x10 = Vp9CospiConstants.RoundShift(s10 + s14);
        x11 = Vp9CospiConstants.RoundShift(s11 + s15);
        x12 = Vp9CospiConstants.RoundShift(s8 - s12);
        x13 = Vp9CospiConstants.RoundShift(s9 - s13);
        x14 = Vp9CospiConstants.RoundShift(s10 - s14);
        x15 = Vp9CospiConstants.RoundShift(s11 - s15);

        // Stage 3
        s0 = x0; s1 = x1; s2 = x2; s3 = x3;
        s4 = x4 * Vp9CospiConstants.Cospi8_64  + x5 * Vp9CospiConstants.Cospi24_64;
        s5 = x4 * Vp9CospiConstants.Cospi24_64 - x5 * Vp9CospiConstants.Cospi8_64;
        s6 = -x6 * Vp9CospiConstants.Cospi24_64 + x7 * Vp9CospiConstants.Cospi8_64;
        s7 = x6 * Vp9CospiConstants.Cospi8_64   + x7 * Vp9CospiConstants.Cospi24_64;
        s8 = x8; s9 = x9; s10 = x10; s11 = x11;
        s12 = x12 * Vp9CospiConstants.Cospi8_64  + x13 * Vp9CospiConstants.Cospi24_64;
        s13 = x12 * Vp9CospiConstants.Cospi24_64 - x13 * Vp9CospiConstants.Cospi8_64;
        s14 = -x14 * Vp9CospiConstants.Cospi24_64 + x15 * Vp9CospiConstants.Cospi8_64;
        s15 = x14 * Vp9CospiConstants.Cospi8_64   + x15 * Vp9CospiConstants.Cospi24_64;

        x0 = s0 + s2;
        x1 = s1 + s3;
        x2 = s0 - s2;
        x3 = s1 - s3;
        x4 = Vp9CospiConstants.RoundShift(s4 + s6);
        x5 = Vp9CospiConstants.RoundShift(s5 + s7);
        x6 = Vp9CospiConstants.RoundShift(s4 - s6);
        x7 = Vp9CospiConstants.RoundShift(s5 - s7);
        x8 = s8 + s10;
        x9 = s9 + s11;
        x10 = s8 - s10;
        x11 = s9 - s11;
        x12 = Vp9CospiConstants.RoundShift(s12 + s14);
        x13 = Vp9CospiConstants.RoundShift(s13 + s15);
        x14 = Vp9CospiConstants.RoundShift(s12 - s14);
        x15 = Vp9CospiConstants.RoundShift(s13 - s15);

        // Stage 4
        s2  = -(long)Vp9CospiConstants.Cospi16_64 * (x2 + x3);
        s3  = (long)Vp9CospiConstants.Cospi16_64 * (x2 - x3);
        s6  = (long)Vp9CospiConstants.Cospi16_64 * (x6 + x7);
        s7  = (long)Vp9CospiConstants.Cospi16_64 * (-x6 + x7);
        s10 = (long)Vp9CospiConstants.Cospi16_64 * (x10 + x11);
        s11 = (long)Vp9CospiConstants.Cospi16_64 * (-x10 + x11);
        s14 = -(long)Vp9CospiConstants.Cospi16_64 * (x14 + x15);
        s15 = (long)Vp9CospiConstants.Cospi16_64 * (x14 - x15);

        x2  = Vp9CospiConstants.RoundShift(s2);
        x3  = Vp9CospiConstants.RoundShift(s3);
        x6  = Vp9CospiConstants.RoundShift(s6);
        x7  = Vp9CospiConstants.RoundShift(s7);
        x10 = Vp9CospiConstants.RoundShift(s10);
        x11 = Vp9CospiConstants.RoundShift(s11);
        x14 = Vp9CospiConstants.RoundShift(s14);
        x15 = Vp9CospiConstants.RoundShift(s15);

        // Output (libvpx negation pattern).
        output[0]  = (int)x0;
        output[1]  = (int)-x8;
        output[2]  = (int)x12;
        output[3]  = (int)-x4;
        output[4]  = (int)x6;
        output[5]  = (int)x14;
        output[6]  = (int)x10;
        output[7]  = (int)x2;
        output[8]  = (int)x3;
        output[9]  = (int)x11;
        output[10] = (int)x15;
        output[11] = (int)x7;
        output[12] = (int)x5;
        output[13] = (int)-x13;
        output[14] = (int)x9;
        output[15] = (int)-x1;
    }
}
