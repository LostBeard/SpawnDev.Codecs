// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 8x8 forward DCT, GPU-callable form for in-kernel reuse.
// Bit-exact mirror of Vp9ForwardDct8x8.Transform (the libvpx
// vpx_fdct8x8_c port).
//
// Vp9ForwardDct8x8Kernel already wraps this math as a standalone
// dispatch handling batches of independent blocks. The static helper
// here exists so the per-frame sequential encode kernel can run FDCT
// 8x8 (chroma transform for v1 keyframes) for ONE block at a time
// inside the per-frame walk - that's the v3 host-as-pure-coordinator
// pattern.
//
// Two-pass shape:
//   Pass 1: column FDCT (input * 4). Stores intermediate transposed
//           into the caller-supplied scratch buffer.
//   Pass 2: row FDCT, reading intermediate column-major (no input
//           scaling). Final output divided by 2 (truncate-toward-zero
//           per libvpx final_output[i] /= 2).

using ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// GPU-callable VP9 8x8 forward DCT helper. Bit-exact mirror of
/// <see cref="Vp9ForwardDct8x8"/> for in-kernel use.
/// </summary>
public static class Vp9ForwardDct8x8Gpu
{
    private const int CosPi4_64  = 16069;
    private const int CosPi8_64  = 15137;
    private const int CosPi12_64 = 13623;
    private const int CosPi16_64 = 11585;
    private const int CosPi20_64 = 9102;
    private const int CosPi24_64 = 6270;
    private const int CosPi28_64 = 3196;

    private const int DctConstBits = 14;
    private const int DctConstRounding = 1 << (DctConstBits - 1);

    /// <summary>
    /// Forward DCT one 8x8 block. Reads <paramref name="input"/>
    /// starting at <paramref name="inBase"/> with row stride
    /// <paramref name="inStride"/>; writes 64 ints to
    /// <paramref name="output"/> starting at <paramref name="outBase"/>
    /// in row-major layout (stride = 8). The
    /// <paramref name="scratch"/> view must hold at least 64 ints
    /// and serves as the inter-pass intermediate buffer.
    /// </summary>
    public static void Forward8x8(
        ArrayView<short> input, long inBase, int inStride,
        ArrayView<int> output, long outBase,
        ArrayView<int> scratch)
    {
        // Pass 1: 8 column DCTs (input * 4).
        for (int col = 0; col < 8; col++)
        {
            int s0 = (input[inBase + col + 0L * inStride] + input[inBase + col + 7L * inStride]) * 4;
            int s1 = (input[inBase + col + 1L * inStride] + input[inBase + col + 6L * inStride]) * 4;
            int s2 = (input[inBase + col + 2L * inStride] + input[inBase + col + 5L * inStride]) * 4;
            int s3 = (input[inBase + col + 3L * inStride] + input[inBase + col + 4L * inStride]) * 4;
            int s4 = (input[inBase + col + 3L * inStride] - input[inBase + col + 4L * inStride]) * 4;
            int s5 = (input[inBase + col + 2L * inStride] - input[inBase + col + 5L * inStride]) * 4;
            int s6 = (input[inBase + col + 1L * inStride] - input[inBase + col + 6L * inStride]) * 4;
            int s7 = (input[inBase + col + 0L * inStride] - input[inBase + col + 7L * inStride]) * 4;

            Butterfly8(s0, s1, s2, s3, s4, s5, s6, s7,
                out int o0, out int o1, out int o2, out int o3,
                out int o4, out int o5, out int o6, out int o7);

            scratch[col * 8 + 0] = o0;
            scratch[col * 8 + 1] = o1;
            scratch[col * 8 + 2] = o2;
            scratch[col * 8 + 3] = o3;
            scratch[col * 8 + 4] = o4;
            scratch[col * 8 + 5] = o5;
            scratch[col * 8 + 6] = o6;
            scratch[col * 8 + 7] = o7;
        }

        // Pass 2: 8 row DCTs (no input scaling). Read intermediate
        // column-major (which equals pass-1 transposed). Output is
        // divided by 2 (truncate-toward-zero) per libvpx convention.
        for (int col = 0; col < 8; col++)
        {
            int s0 = scratch[col + 0 * 8] + scratch[col + 7 * 8];
            int s1 = scratch[col + 1 * 8] + scratch[col + 6 * 8];
            int s2 = scratch[col + 2 * 8] + scratch[col + 5 * 8];
            int s3 = scratch[col + 3 * 8] + scratch[col + 4 * 8];
            int s4 = scratch[col + 3 * 8] - scratch[col + 4 * 8];
            int s5 = scratch[col + 2 * 8] - scratch[col + 5 * 8];
            int s6 = scratch[col + 1 * 8] - scratch[col + 6 * 8];
            int s7 = scratch[col + 0 * 8] - scratch[col + 7 * 8];

            Butterfly8(s0, s1, s2, s3, s4, s5, s6, s7,
                out int o0, out int o1, out int o2, out int o3,
                out int o4, out int o5, out int o6, out int o7);

            long b = outBase + col * 8;
            // libvpx final_output[i] /= 2 - signed integer division
            // truncates toward zero. SpawnDev.ILGPU 4.9.2-rc.28
            // shipped a fix for the IR-level Div-by-pow2 -> Shr
            // rewrite on signed dividends, but a follow-up test
            // (2026-04-28 21:57) showed IlgpuIntDivByTwoRepro STILL
            // fails identically on rc.28. WORKAROUND retained until
            // root cause is found. Pinged Geordi.
            output[b + 0] = HalveTruncateTowardZero(o0);
            output[b + 1] = HalveTruncateTowardZero(o1);
            output[b + 2] = HalveTruncateTowardZero(o2);
            output[b + 3] = HalveTruncateTowardZero(o3);
            output[b + 4] = HalveTruncateTowardZero(o4);
            output[b + 5] = HalveTruncateTowardZero(o5);
            output[b + 6] = HalveTruncateTowardZero(o6);
            output[b + 7] = HalveTruncateTowardZero(o7);
        }
    }

    /// <summary>
    /// WORKAROUND: bit-exact mirror of C# <c>x / 2</c> (truncate toward
    /// zero). rc.28's gate-on-Unsigned fix should make plain `int / 2`
    /// work but verification 2026-04-28 21:57 showed the test still
    /// fails identically. Helper retained until the root cause is found.
    /// </summary>
    private static int HalveTruncateTowardZero(int x)
    {
        if (x >= 0) return x >> 1;
        return -((-x) >> 1);
    }

    /// <summary>
    /// 8-point 1D forward DCT butterfly. Mirrors libvpx
    /// <c>vpx_fdct8x8_c</c>'s inner loop bit-for-bit.
    /// </summary>
    private static void Butterfly8(
        int s0, int s1, int s2, int s3, int s4, int s5, int s6, int s7,
        out int o0, out int o1, out int o2, out int o3,
        out int o4, out int o5, out int o6, out int o7)
    {
        // fdct4(s0..s3) - even half.
        int x0 = s0 + s3;
        int x1 = s1 + s2;
        int x2 = s1 - s2;
        int x3 = s0 - s3;
        long t0 = (long)(x0 + x1) * CosPi16_64;
        long t1 = (long)(x0 - x1) * CosPi16_64;
        long t2 = (long)x2 * CosPi24_64 + (long)x3 * CosPi8_64;
        long t3 = (long)(-x2) * CosPi8_64 + (long)x3 * CosPi24_64;
        o0 = RoundShift(t0);
        o2 = RoundShift(t2);
        o4 = RoundShift(t1);
        o6 = RoundShift(t3);

        // s4..s7 stages - odd half.
        long u0 = (long)(s6 - s5) * CosPi16_64;
        long u1 = (long)(s6 + s5) * CosPi16_64;
        int v2 = RoundShift(u0);
        int v3 = RoundShift(u1);

        int y0 = s4 + v2;
        int y1 = s4 - v2;
        int y2 = s7 - v3;
        int y3 = s7 + v3;

        long w0 = (long)y0 * CosPi28_64 + (long)y3 * CosPi4_64;
        long w1 = (long)y1 * CosPi12_64 + (long)y2 * CosPi20_64;
        long w2 = (long)y2 * CosPi12_64 + (long)y1 * (-CosPi20_64);
        long w3 = (long)y3 * CosPi28_64 + (long)y0 * (-CosPi4_64);
        o1 = RoundShift(w0);
        o3 = RoundShift(w2);
        o5 = RoundShift(w1);
        o7 = RoundShift(w3);
    }

    private static int RoundShift(long input) =>
        (int)((input + DctConstRounding) >> DctConstBits);
}
