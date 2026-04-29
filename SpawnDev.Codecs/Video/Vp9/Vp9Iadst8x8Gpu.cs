// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 inverse ADST 8x8, GPU-callable form for in-kernel reuse. Bit-exact
// mirror of Vp9Iadst8x8Reference (libvpx vp9_iht8x8_64_add tx_type=3
// ADST_ADST port).
//
// Pairs with the existing Vp9Idct8x8Gpu - both 8x8 transform types in
// VP9 (DCT_DCT and ADST_ADST) now have GPU primitives.
//
// Two-pass shape (matches Vp9Iadst4x4Gpu / Vp9Idct8x8Gpu):
//   Row pass: 8 row-1D iADSTs, intermediate stored in scratch as short.
//   Column pass: 8 column-1D iADSTs that add (colOut + 16) >> 5 to the
//                predictor pixel and clip to [0, 255].
//
// libvpx input reorder: x0=in[7], x1=in[0], x2=in[5], x3=in[2],
//                       x4=in[3], x5=in[4], x6=in[1], x7=in[6].

using ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// GPU-callable VP9 inverse ADST 8x8 helper. Bit-exact mirror of
/// <see cref="Vp9Iadst8x8Reference"/> for in-kernel use.
/// </summary>
public static class Vp9Iadst8x8Gpu
{
    private const int CosPi2_64 = 16305;
    private const int CosPi6_64 = 15679;
    private const int CosPi8_64 = 15137;
    private const int CosPi10_64 = 14449;
    private const int CosPi14_64 = 12665;
    private const int CosPi16_64 = 11585;
    private const int CosPi18_64 = 10394;
    private const int CosPi22_64 = 7723;
    private const int CosPi24_64 = 6270;
    private const int CosPi26_64 = 4756;
    private const int CosPi30_64 = 1606;

    /// <summary>
    /// Inverse-ADST one 8x8 block (ADST_ADST) and add the residual to
    /// <paramref name="dest"/> in place.
    /// </summary>
    public static void Iadst8x8(
        ArrayView<short> coefs, long coefBase,
        ArrayView<byte> dest, long destBase, int destStride,
        ArrayView<short> scratch)
    {
        // Row pass.
        for (int row = 0; row < 8; row++)
        {
            long rBase = coefBase + row * 8;
            Iadst8(
                coefs[rBase + 0], coefs[rBase + 1], coefs[rBase + 2], coefs[rBase + 3],
                coefs[rBase + 4], coefs[rBase + 5], coefs[rBase + 6], coefs[rBase + 7],
                out short o0, out short o1, out short o2, out short o3,
                out short o4, out short o5, out short o6, out short o7);
            int rb = row * 8;
            scratch[rb + 0] = o0;
            scratch[rb + 1] = o1;
            scratch[rb + 2] = o2;
            scratch[rb + 3] = o3;
            scratch[rb + 4] = o4;
            scratch[rb + 5] = o5;
            scratch[rb + 6] = o6;
            scratch[rb + 7] = o7;
        }

        // Column pass + residual add + clip.
        for (int col = 0; col < 8; col++)
        {
            Iadst8(
                scratch[0 * 8 + col], scratch[1 * 8 + col],
                scratch[2 * 8 + col], scratch[3 * 8 + col],
                scratch[4 * 8 + col], scratch[5 * 8 + col],
                scratch[6 * 8 + col], scratch[7 * 8 + col],
                out short c0, out short c1, out short c2, out short c3,
                out short c4, out short c5, out short c6, out short c7);

            ApplyResidual(c0, dest, destBase + 0L * destStride + col);
            ApplyResidual(c1, dest, destBase + 1L * destStride + col);
            ApplyResidual(c2, dest, destBase + 2L * destStride + col);
            ApplyResidual(c3, dest, destBase + 3L * destStride + col);
            ApplyResidual(c4, dest, destBase + 4L * destStride + col);
            ApplyResidual(c5, dest, destBase + 5L * destStride + col);
            ApplyResidual(c6, dest, destBase + 6L * destStride + col);
            ApplyResidual(c7, dest, destBase + 7L * destStride + col);
        }
    }

    /// <summary>One-dimensional 8-point iADST butterfly with libvpx input reorder.</summary>
    private static void Iadst8(
        short i0, short i1, short i2, short i3, short i4, short i5, short i6, short i7,
        out short o0, out short o1, out short o2, out short o3,
        out short o4, out short o5, out short o6, out short o7)
    {
        // libvpx input reorder.
        int x0 = i7;
        int x1 = i0;
        int x2 = i5;
        int x3 = i2;
        int x4 = i3;
        int x5 = i4;
        int x6 = i1;
        int x7 = i6;

        if ((x0 | x1 | x2 | x3 | x4 | x5 | x6 | x7) == 0)
        {
            o0 = 0; o1 = 0; o2 = 0; o3 = 0;
            o4 = 0; o5 = 0; o6 = 0; o7 = 0;
            return;
        }

        // Stage 1.
        int s0 = CosPi2_64 * x0 + CosPi30_64 * x1;
        int s1 = CosPi30_64 * x0 - CosPi2_64 * x1;
        int s2 = CosPi10_64 * x2 + CosPi22_64 * x3;
        int s3 = CosPi22_64 * x2 - CosPi10_64 * x3;
        int s4 = CosPi18_64 * x4 + CosPi14_64 * x5;
        int s5 = CosPi14_64 * x4 - CosPi18_64 * x5;
        int s6 = CosPi26_64 * x6 + CosPi6_64 * x7;
        int s7 = CosPi6_64 * x6 - CosPi26_64 * x7;

        x0 = Rs14(s0 + s4);
        x1 = Rs14(s1 + s5);
        x2 = Rs14(s2 + s6);
        x3 = Rs14(s3 + s7);
        x4 = Rs14(s0 - s4);
        x5 = Rs14(s1 - s5);
        x6 = Rs14(s2 - s6);
        x7 = Rs14(s3 - s7);

        // Stage 2.
        s0 = x0;
        s1 = x1;
        s2 = x2;
        s3 = x3;
        s4 = CosPi8_64 * x4 + CosPi24_64 * x5;
        s5 = CosPi24_64 * x4 - CosPi8_64 * x5;
        s6 = -CosPi24_64 * x6 + CosPi8_64 * x7;
        s7 = CosPi8_64 * x6 + CosPi24_64 * x7;

        x0 = (short)(s0 + s2);
        x1 = (short)(s1 + s3);
        x2 = (short)(s0 - s2);
        x3 = (short)(s1 - s3);
        x4 = Rs14(s4 + s6);
        x5 = Rs14(s5 + s7);
        x6 = Rs14(s4 - s6);
        x7 = Rs14(s5 - s7);

        // Stage 3.
        s2 = CosPi16_64 * (x2 + x3);
        s3 = CosPi16_64 * (x2 - x3);
        s6 = CosPi16_64 * (x6 + x7);
        s7 = CosPi16_64 * (x6 - x7);

        x2 = Rs14(s2);
        x3 = Rs14(s3);
        x6 = Rs14(s6);
        x7 = Rs14(s7);

        // Output with sign inversions at odd positions.
        o0 = (short)x0;
        o1 = (short)-x4;
        o2 = (short)x6;
        o3 = (short)-x2;
        o4 = (short)x3;
        o5 = (short)-x7;
        o6 = (short)x5;
        o7 = (short)-x1;
    }

    /// <summary>Q14 rounded narrow-to-int16.</summary>
    private static short Rs14(int value) => (short)((value + (1 << 13)) >> 14);

    private static void ApplyResidual(short residual, ArrayView<byte> dest, long destIdx)
    {
        // ROUND_POWER_OF_TWO(x, 5) = (x + 16) >> 5
        int rounded = (residual + 16) >> 5;
        int sum = dest[destIdx] + rounded;
        if (sum < 0) sum = 0;
        else if (sum > 255) sum = 255;
        dest[destIdx] = (byte)sum;
    }
}
