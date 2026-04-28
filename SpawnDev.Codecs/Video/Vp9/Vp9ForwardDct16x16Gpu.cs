// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 16x16 forward DCT, GPU-callable form for in-kernel reuse.
// Bit-exact mirror of Vp9ForwardDct16x16.Transform (the libvpx
// vpx_fdct16x16_c port).
//
// Vp9ForwardDct16x16Kernel already wraps this math as a standalone
// dispatch handling batches of independent blocks. The static helper
// here exists so the per-frame sequential encode kernel can run FDCT
// for ONE block at a time without dispatching a separate kernel
// boundary - that's the v3 host-as-pure-coordinator pattern.
//
// Two-pass shape:
//   Pass 1: column FDCT (input * 4). Stores intermediate transposed
//           into the caller-supplied scratch buffer.
//   Pass 2: row FDCT, reading intermediate column-major (which equals
//           pass-1 transposed). Inputs are half-round-shifted via
//           ((x + 1) >> 2) before the butterfly.
//
// The 16-point butterfly is identical between the two passes; it
// lives in <see cref="Butterfly16"/> with [MethodImpl(NoInlining)]
// to keep the WGSL shader size sane (same trick the standalone
// Vp9ForwardDct16x16Kernel uses).

using System.Runtime.CompilerServices;
using ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// GPU-callable VP9 16x16 forward DCT helper. Bit-exact mirror of
/// <see cref="Vp9ForwardDct16x16"/> for in-kernel use.
/// </summary>
public static class Vp9ForwardDct16x16Gpu
{
    private const int CosPi2_64  = 16305;
    private const int CosPi4_64  = 16069;
    private const int CosPi6_64  = 15679;
    private const int CosPi8_64  = 15137;
    private const int CosPi10_64 = 14449;
    private const int CosPi12_64 = 13623;
    private const int CosPi14_64 = 12665;
    private const int CosPi16_64 = 11585;
    private const int CosPi18_64 = 10394;
    private const int CosPi20_64 = 9102;
    private const int CosPi22_64 = 7723;
    private const int CosPi24_64 = 6270;
    private const int CosPi26_64 = 4756;
    private const int CosPi28_64 = 3196;
    private const int CosPi30_64 = 1606;

    private const int DctConstBits = 14;
    private const int DctConstRounding = 1 << (DctConstBits - 1);

    /// <summary>
    /// Forward DCT one 16x16 block. Reads <paramref name="input"/>
    /// starting at <paramref name="inBase"/> with row stride
    /// <paramref name="inStride"/>; writes 256 ints to
    /// <paramref name="output"/> starting at <paramref name="outBase"/>
    /// in row-major layout (stride = 16). The
    /// <paramref name="scratch"/> view must hold at least 256 ints
    /// and serves as the inter-pass intermediate buffer.
    /// </summary>
    public static void Forward16x16(
        ArrayView<short> input, long inBase, int inStride,
        ArrayView<int> output, long outBase,
        ArrayView<int> scratch)
    {
        // Pass 1: 16 column DCTs (input * 4). scratch[col*16 + j]
        // holds the j-th output of column c (transposed).
        for (int col = 0; col < 16; col++)
        {
            int ih0 = (input[inBase + col +  0L * inStride] + input[inBase + col + 15L * inStride]) * 4;
            int ih1 = (input[inBase + col +  1L * inStride] + input[inBase + col + 14L * inStride]) * 4;
            int ih2 = (input[inBase + col +  2L * inStride] + input[inBase + col + 13L * inStride]) * 4;
            int ih3 = (input[inBase + col +  3L * inStride] + input[inBase + col + 12L * inStride]) * 4;
            int ih4 = (input[inBase + col +  4L * inStride] + input[inBase + col + 11L * inStride]) * 4;
            int ih5 = (input[inBase + col +  5L * inStride] + input[inBase + col + 10L * inStride]) * 4;
            int ih6 = (input[inBase + col +  6L * inStride] + input[inBase + col +  9L * inStride]) * 4;
            int ih7 = (input[inBase + col +  7L * inStride] + input[inBase + col +  8L * inStride]) * 4;
            int s10 = (input[inBase + col +  7L * inStride] - input[inBase + col +  8L * inStride]) * 4;
            int s11 = (input[inBase + col +  6L * inStride] - input[inBase + col +  9L * inStride]) * 4;
            int s12 = (input[inBase + col +  5L * inStride] - input[inBase + col + 10L * inStride]) * 4;
            int s13 = (input[inBase + col +  4L * inStride] - input[inBase + col + 11L * inStride]) * 4;
            int s14 = (input[inBase + col +  3L * inStride] - input[inBase + col + 12L * inStride]) * 4;
            int s15 = (input[inBase + col +  2L * inStride] - input[inBase + col + 13L * inStride]) * 4;
            int s16 = (input[inBase + col +  1L * inStride] - input[inBase + col + 14L * inStride]) * 4;
            int s17 = (input[inBase + col +  0L * inStride] - input[inBase + col + 15L * inStride]) * 4;

            Butterfly16(
                ih0, ih1, ih2, ih3, ih4, ih5, ih6, ih7,
                s10, s11, s12, s13, s14, s15, s16, s17,
                out int o0,  out int o1,  out int o2,  out int o3,
                out int o4,  out int o5,  out int o6,  out int o7,
                out int o8,  out int o9,  out int o10, out int o11,
                out int o12, out int o13, out int o14, out int o15);

            int b = col * 16;
            scratch[b +  0] = o0;  scratch[b +  1] = o1;  scratch[b +  2] = o2;  scratch[b +  3] = o3;
            scratch[b +  4] = o4;  scratch[b +  5] = o5;  scratch[b +  6] = o6;  scratch[b +  7] = o7;
            scratch[b +  8] = o8;  scratch[b +  9] = o9;  scratch[b + 10] = o10; scratch[b + 11] = o11;
            scratch[b + 12] = o12; scratch[b + 13] = o13; scratch[b + 14] = o14; scratch[b + 15] = o15;
        }

        // Pass 2: 16 row DCTs - input rounded via (x + 1) >> 2.
        for (int col = 0; col < 16; col++)
        {
            int ih0 = ((scratch[col +  0 * 16] + 1) >> 2) + ((scratch[col + 15 * 16] + 1) >> 2);
            int ih1 = ((scratch[col +  1 * 16] + 1) >> 2) + ((scratch[col + 14 * 16] + 1) >> 2);
            int ih2 = ((scratch[col +  2 * 16] + 1) >> 2) + ((scratch[col + 13 * 16] + 1) >> 2);
            int ih3 = ((scratch[col +  3 * 16] + 1) >> 2) + ((scratch[col + 12 * 16] + 1) >> 2);
            int ih4 = ((scratch[col +  4 * 16] + 1) >> 2) + ((scratch[col + 11 * 16] + 1) >> 2);
            int ih5 = ((scratch[col +  5 * 16] + 1) >> 2) + ((scratch[col + 10 * 16] + 1) >> 2);
            int ih6 = ((scratch[col +  6 * 16] + 1) >> 2) + ((scratch[col +  9 * 16] + 1) >> 2);
            int ih7 = ((scratch[col +  7 * 16] + 1) >> 2) + ((scratch[col +  8 * 16] + 1) >> 2);
            int s10 = ((scratch[col +  7 * 16] + 1) >> 2) - ((scratch[col +  8 * 16] + 1) >> 2);
            int s11 = ((scratch[col +  6 * 16] + 1) >> 2) - ((scratch[col +  9 * 16] + 1) >> 2);
            int s12 = ((scratch[col +  5 * 16] + 1) >> 2) - ((scratch[col + 10 * 16] + 1) >> 2);
            int s13 = ((scratch[col +  4 * 16] + 1) >> 2) - ((scratch[col + 11 * 16] + 1) >> 2);
            int s14 = ((scratch[col +  3 * 16] + 1) >> 2) - ((scratch[col + 12 * 16] + 1) >> 2);
            int s15 = ((scratch[col +  2 * 16] + 1) >> 2) - ((scratch[col + 13 * 16] + 1) >> 2);
            int s16 = ((scratch[col +  1 * 16] + 1) >> 2) - ((scratch[col + 14 * 16] + 1) >> 2);
            int s17 = ((scratch[col +  0 * 16] + 1) >> 2) - ((scratch[col + 15 * 16] + 1) >> 2);

            Butterfly16(
                ih0, ih1, ih2, ih3, ih4, ih5, ih6, ih7,
                s10, s11, s12, s13, s14, s15, s16, s17,
                out int o0,  out int o1,  out int o2,  out int o3,
                out int o4,  out int o5,  out int o6,  out int o7,
                out int o8,  out int o9,  out int o10, out int o11,
                out int o12, out int o13, out int o14, out int o15);

            long b = outBase + col * 16;
            output[b +  0] = o0;  output[b +  1] = o1;  output[b +  2] = o2;  output[b +  3] = o3;
            output[b +  4] = o4;  output[b +  5] = o5;  output[b +  6] = o6;  output[b +  7] = o7;
            output[b +  8] = o8;  output[b +  9] = o9;  output[b + 10] = o10; output[b + 11] = o11;
            output[b + 12] = o12; output[b + 13] = o13; output[b + 14] = o14; output[b + 15] = o15;
        }
    }

    /// <summary>
    /// 16-point 1D forward DCT butterfly. Bit-exact against the libvpx
    /// reference's inner loop. NoInlining keeps WGSL shader size
    /// manageable - same reasoning as Vp9Idct16x16Kernel.Idct16Row.
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void Butterfly16(
        int ih0, int ih1, int ih2, int ih3, int ih4, int ih5, int ih6, int ih7,
        int s10, int s11, int s12, int s13, int s14, int s15, int s16, int s17,
        out int o0,  out int o1,  out int o2,  out int o3,
        out int o4,  out int o5,  out int o6,  out int o7,
        out int o8,  out int o9,  out int o10, out int o11,
        out int o12, out int o13, out int o14, out int o15)
    {
        // Even half: fdct8 on inHigh[0..7] (locals ih0..ih7).
        int s0 = ih0 + ih7;
        int s1 = ih1 + ih6;
        int s2 = ih2 + ih5;
        int s3 = ih3 + ih4;
        int s4 = ih3 - ih4;
        int s5 = ih2 - ih5;
        int s6 = ih1 - ih6;
        int s7 = ih0 - ih7;

        int x0 = s0 + s3, x1 = s1 + s2, x2 = s1 - s2, x3 = s0 - s3;
        long t0 = (long)(x0 + x1) * CosPi16_64;
        long t1 = (long)(x0 - x1) * CosPi16_64;
        long t2 = (long)x3 * CosPi8_64 + (long)x2 * CosPi24_64;
        long t3 = (long)x3 * CosPi24_64 - (long)x2 * CosPi8_64;
        o0  = RoundShift(t0);
        o4  = RoundShift(t2);
        o8  = RoundShift(t1);
        o12 = RoundShift(t3);

        long u0 = (long)(s6 - s5) * CosPi16_64;
        long u1 = (long)(s6 + s5) * CosPi16_64;
        int v2 = RoundShift(u0);
        int v3 = RoundShift(u1);

        int y0 = s4 + v2, y1 = s4 - v2, y2 = s7 - v3, y3 = s7 + v3;
        long w0 = (long)y0 * CosPi28_64 + (long)y3 * CosPi4_64;
        long w1 = (long)y1 * CosPi12_64 + (long)y2 * CosPi20_64;
        long w2 = (long)y2 * CosPi12_64 + (long)y1 * (-CosPi20_64);
        long w3 = (long)y3 * CosPi28_64 + (long)y0 * (-CosPi4_64);
        o2  = RoundShift(w0);
        o6  = RoundShift(w2);
        o10 = RoundShift(w1);
        o14 = RoundShift(w3);

        // Odd half. step1[0..7] = locals s10..s17 (s1X where X is the index).
        long temp1 = (long)(s15 - s12) * CosPi16_64;
        long temp2 = (long)(s14 - s13) * CosPi16_64;
        int sa2 = RoundShift(temp1);
        int sa3 = RoundShift(temp2);
        temp1 = (long)(s14 + s13) * CosPi16_64;
        temp2 = (long)(s15 + s12) * CosPi16_64;
        int sa4 = RoundShift(temp1);
        int sa5 = RoundShift(temp2);

        int sb0 = s10 + sa3;
        int sb1 = s11 + sa2;
        int sb2 = s11 - sa2;
        int sb3 = s10 - sa3;
        int sb4 = s17 - sa4;
        int sb5 = s16 - sa5;
        int sb6 = s16 + sa5;
        int sb7 = s17 + sa4;

        temp1 = (long)sb1 * (-CosPi8_64) + (long)sb6 * CosPi24_64;
        temp2 = (long)sb2 * CosPi24_64 + (long)sb5 * CosPi8_64;
        int sc1 = RoundShift(temp1);
        int sc2 = RoundShift(temp2);
        temp1 = (long)sb2 * CosPi8_64 - (long)sb5 * CosPi24_64;
        temp2 = (long)sb1 * CosPi24_64 + (long)sb6 * CosPi8_64;
        int sc5 = RoundShift(temp1);
        int sc6 = RoundShift(temp2);

        int sd0 = sb0 + sc1;
        int sd1 = sb0 - sc1;
        int sd2 = sb3 + sc2;
        int sd3 = sb3 - sc2;
        int sd4 = sb4 - sc5;
        int sd5 = sb4 + sc5;
        int sd6 = sb7 - sc6;
        int sd7 = sb7 + sc6;

        temp1 = (long)sd0 * CosPi30_64 + (long)sd7 * CosPi2_64;
        temp2 = (long)sd1 * CosPi14_64 + (long)sd6 * CosPi18_64;
        o1 = RoundShift(temp1);
        o9 = RoundShift(temp2);
        temp1 = (long)sd2 * CosPi22_64 + (long)sd5 * CosPi10_64;
        temp2 = (long)sd3 * CosPi6_64 + (long)sd4 * CosPi26_64;
        o5  = RoundShift(temp1);
        o13 = RoundShift(temp2);
        temp1 = (long)sd3 * (-CosPi26_64) + (long)sd4 * CosPi6_64;
        temp2 = (long)sd2 * (-CosPi10_64) + (long)sd5 * CosPi22_64;
        o3  = RoundShift(temp1);
        o11 = RoundShift(temp2);
        temp1 = (long)sd1 * (-CosPi18_64) + (long)sd6 * CosPi14_64;
        temp2 = (long)sd0 * (-CosPi2_64) + (long)sd7 * CosPi30_64;
        o7  = RoundShift(temp1);
        o15 = RoundShift(temp2);
    }

    private static int RoundShift(long input) =>
        (int)((input + DctConstRounding) >> DctConstBits);
}
