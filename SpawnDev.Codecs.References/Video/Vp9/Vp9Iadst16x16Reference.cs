// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 inverse ADST 16x16 - bit-exact CPU oracle for libvpx iadst16_c
// plus the 2D ADST_ADST variant. Largest iADST VP9 uses (32x32 is
// iDCT-only per spec, so iHT dispatcher coverage stops at 16x16).
//
// libvpx reference: vpx_dsp/inv_txfm.c (iadst16_c).
//
// Structure
//   - 4 stages. Input reordering puts odd/even positions into specific
//     stage-1 slots (matching libvpx exactly):
//       x0..x7  = input[15,0,13,2,11,4,9,6]
//       x8..x15 = input[7,8,5,10,3,12,1,14]
//   - Stage 1: 8 cospi rotation pairs, Q14 rounded.
//   - Stage 2: 4 rotation pairs on 8..15, add/sub on 0..7.
//   - Stage 3: 2 rotation pairs each on 4..7 and 12..15, add/sub
//     on 0..3 and 8..11.
//   - Stage 4: 4 rotation pairs on 2,3 / 6,7 / 10,11 / 14,15 using
//     cospi_16_64 (the DC rotation constant).
//   - Output has sign inversions at specific positions:
//     [x0, -x8, x12, -x4, x6, x14, x10, x2, x3, x11, x15, x7, x5, -x13, x9, -x1]
//
// 2D variant (ADST_ADST, tx_type=3):
//   - Row then column pass, final round (x + 32) >> 6 (same as iDCT 16x16).

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>CPU oracle for VP9 inverse ADST 16x16.</summary>
public static class Vp9Iadst16x16Reference
{
    // Full Q14 cosine table - the 16-point iADST uses every odd + even
    // cospi constant from 1 through 31.
    private const int CosPi1_64 = 16364;
    private const int CosPi3_64 = 16207;
    private const int CosPi4_64 = 16069;
    private const int CosPi5_64 = 15893;
    private const int CosPi7_64 = 15426;
    private const int CosPi8_64 = 15137;
    private const int CosPi9_64 = 14811;
    private const int CosPi11_64 = 14053;
    private const int CosPi12_64 = 13623;
    private const int CosPi13_64 = 13160;
    private const int CosPi15_64 = 12140;
    private const int CosPi16_64 = 11585;
    private const int CosPi17_64 = 11003;
    private const int CosPi19_64 = 9760;
    private const int CosPi20_64 = 9102;
    private const int CosPi21_64 = 8423;
    private const int CosPi23_64 = 7005;
    private const int CosPi24_64 = 6270;
    private const int CosPi25_64 = 5520;
    private const int CosPi27_64 = 3981;
    private const int CosPi28_64 = 3196;
    private const int CosPi29_64 = 2404;
    private const int CosPi31_64 = 804;

    /// <summary>
    /// Apply <paramref name="input"/> (256 coefficients, row-major 16x16)
    /// via 2D iADST (rows then columns) as a residual to
    /// <paramref name="dest"/>. Matches libvpx vp9_iht16x16_256_add with
    /// tx_type = 3 (ADST_ADST).
    /// </summary>
    public static void IadstAdst16x16_256_Add(
        ReadOnlySpan<short> input, Span<byte> dest, int stride)
    {
        if (input.Length < 256)
            throw new ArgumentException("input must have >= 256 coefficients", nameof(input));
        if (stride < 16)
            throw new ArgumentException("stride must be >= 16", nameof(stride));
        if (dest.Length < 15 * stride + 16)
            throw new ArgumentException("dest too small for 16 rows at the given stride", nameof(dest));

        Span<short> tmp = stackalloc short[256];

        for (int row = 0; row < 16; row++)
        {
            Iadst16_1d(input.Slice(row * 16, 16), tmp.Slice(row * 16, 16));
        }

        Span<short> colIn = stackalloc short[16];
        Span<short> colOut = stackalloc short[16];
        for (int col = 0; col < 16; col++)
        {
            for (int j = 0; j < 16; j++) colIn[j] = tmp[j * 16 + col];
            Iadst16_1d(colIn, colOut);
            for (int j = 0; j < 16; j++)
            {
                // ROUND_POWER_OF_TWO(x, 6) = (x + 32) >> 6.
                int residual = (colOut[j] + 32) >> 6;
                int predictor = dest[j * stride + col];
                int sum = predictor + residual;
                if (sum < 0) sum = 0;
                else if (sum > 255) sum = 255;
                dest[j * stride + col] = (byte)sum;
            }
        }
    }

    /// <summary>One-dimensional 16-point iADST per libvpx iadst16_c.</summary>
    internal static void Iadst16_1d(ReadOnlySpan<short> input, Span<short> output)
    {
        int x0 = input[15];
        int x1 = input[0];
        int x2 = input[13];
        int x3 = input[2];
        int x4 = input[11];
        int x5 = input[4];
        int x6 = input[9];
        int x7 = input[6];
        int x8 = input[7];
        int x9 = input[8];
        int x10 = input[5];
        int x11 = input[10];
        int x12 = input[3];
        int x13 = input[12];
        int x14 = input[1];
        int x15 = input[14];

        if ((x0 | x1 | x2 | x3 | x4 | x5 | x6 | x7 |
             x8 | x9 | x10 | x11 | x12 | x13 | x14 | x15) == 0)
        {
            for (int i = 0; i < 16; i++) output[i] = 0;
            return;
        }

        // Stage 1.
        int s0 = x0 * CosPi1_64 + x1 * CosPi31_64;
        int s1 = x0 * CosPi31_64 - x1 * CosPi1_64;
        int s2 = x2 * CosPi5_64 + x3 * CosPi27_64;
        int s3 = x2 * CosPi27_64 - x3 * CosPi5_64;
        int s4 = x4 * CosPi9_64 + x5 * CosPi23_64;
        int s5 = x4 * CosPi23_64 - x5 * CosPi9_64;
        int s6 = x6 * CosPi13_64 + x7 * CosPi19_64;
        int s7 = x6 * CosPi19_64 - x7 * CosPi13_64;
        int s8 = x8 * CosPi17_64 + x9 * CosPi15_64;
        int s9 = x8 * CosPi15_64 - x9 * CosPi17_64;
        int s10 = x10 * CosPi21_64 + x11 * CosPi11_64;
        int s11 = x10 * CosPi11_64 - x11 * CosPi21_64;
        int s12 = x12 * CosPi25_64 + x13 * CosPi7_64;
        int s13 = x12 * CosPi7_64 - x13 * CosPi25_64;
        int s14 = x14 * CosPi29_64 + x15 * CosPi3_64;
        int s15 = x14 * CosPi3_64 - x15 * CosPi29_64;

        x0 = Rs14(s0 + s8);
        x1 = Rs14(s1 + s9);
        x2 = Rs14(s2 + s10);
        x3 = Rs14(s3 + s11);
        x4 = Rs14(s4 + s12);
        x5 = Rs14(s5 + s13);
        x6 = Rs14(s6 + s14);
        x7 = Rs14(s7 + s15);
        x8 = Rs14(s0 - s8);
        x9 = Rs14(s1 - s9);
        x10 = Rs14(s2 - s10);
        x11 = Rs14(s3 - s11);
        x12 = Rs14(s4 - s12);
        x13 = Rs14(s5 - s13);
        x14 = Rs14(s6 - s14);
        x15 = Rs14(s7 - s15);

        // Stage 2.
        s0 = x0; s1 = x1; s2 = x2; s3 = x3;
        s4 = x4; s5 = x5; s6 = x6; s7 = x7;
        s8 = x8 * CosPi4_64 + x9 * CosPi28_64;
        s9 = x8 * CosPi28_64 - x9 * CosPi4_64;
        s10 = x10 * CosPi20_64 + x11 * CosPi12_64;
        s11 = x10 * CosPi12_64 - x11 * CosPi20_64;
        s12 = -x12 * CosPi28_64 + x13 * CosPi4_64;
        s13 = x12 * CosPi4_64 + x13 * CosPi28_64;
        s14 = -x14 * CosPi12_64 + x15 * CosPi20_64;
        s15 = x14 * CosPi20_64 + x15 * CosPi12_64;

        x0 = (short)(s0 + s4);
        x1 = (short)(s1 + s5);
        x2 = (short)(s2 + s6);
        x3 = (short)(s3 + s7);
        x4 = (short)(s0 - s4);
        x5 = (short)(s1 - s5);
        x6 = (short)(s2 - s6);
        x7 = (short)(s3 - s7);
        x8 = Rs14(s8 + s12);
        x9 = Rs14(s9 + s13);
        x10 = Rs14(s10 + s14);
        x11 = Rs14(s11 + s15);
        x12 = Rs14(s8 - s12);
        x13 = Rs14(s9 - s13);
        x14 = Rs14(s10 - s14);
        x15 = Rs14(s11 - s15);

        // Stage 3.
        s0 = x0; s1 = x1; s2 = x2; s3 = x3;
        s4 = x4 * CosPi8_64 + x5 * CosPi24_64;
        s5 = x4 * CosPi24_64 - x5 * CosPi8_64;
        s6 = -x6 * CosPi24_64 + x7 * CosPi8_64;
        s7 = x6 * CosPi8_64 + x7 * CosPi24_64;
        s8 = x8; s9 = x9; s10 = x10; s11 = x11;
        s12 = x12 * CosPi8_64 + x13 * CosPi24_64;
        s13 = x12 * CosPi24_64 - x13 * CosPi8_64;
        s14 = -x14 * CosPi24_64 + x15 * CosPi8_64;
        s15 = x14 * CosPi8_64 + x15 * CosPi24_64;

        x0 = (short)(s0 + s2);
        x1 = (short)(s1 + s3);
        x2 = (short)(s0 - s2);
        x3 = (short)(s1 - s3);
        x4 = Rs14(s4 + s6);
        x5 = Rs14(s5 + s7);
        x6 = Rs14(s4 - s6);
        x7 = Rs14(s5 - s7);
        x8 = (short)(s8 + s10);
        x9 = (short)(s9 + s11);
        x10 = (short)(s8 - s10);
        x11 = (short)(s9 - s11);
        x12 = Rs14(s12 + s14);
        x13 = Rs14(s13 + s15);
        x14 = Rs14(s12 - s14);
        x15 = Rs14(s13 - s15);

        // Stage 4.
        s2 = -CosPi16_64 * (x2 + x3);
        s3 = CosPi16_64 * (x2 - x3);
        s6 = CosPi16_64 * (x6 + x7);
        s7 = CosPi16_64 * (-x6 + x7);
        s10 = CosPi16_64 * (x10 + x11);
        s11 = CosPi16_64 * (-x10 + x11);
        s14 = -CosPi16_64 * (x14 + x15);
        s15 = CosPi16_64 * (x14 - x15);

        x2 = Rs14(s2);
        x3 = Rs14(s3);
        x6 = Rs14(s6);
        x7 = Rs14(s7);
        x10 = Rs14(s10);
        x11 = Rs14(s11);
        x14 = Rs14(s14);
        x15 = Rs14(s15);

        // Output with sign inversions at specific positions.
        output[0] = (short)x0;
        output[1] = (short)-x8;
        output[2] = (short)x12;
        output[3] = (short)-x4;
        output[4] = (short)x6;
        output[5] = (short)x14;
        output[6] = (short)x10;
        output[7] = (short)x2;
        output[8] = (short)x3;
        output[9] = (short)x11;
        output[10] = (short)x15;
        output[11] = (short)x7;
        output[12] = (short)x5;
        output[13] = (short)-x13;
        output[14] = (short)x9;
        output[15] = (short)-x1;
    }

    private static int Rs14(int value) => (short)((value + (1 << 13)) >> 14);
}
