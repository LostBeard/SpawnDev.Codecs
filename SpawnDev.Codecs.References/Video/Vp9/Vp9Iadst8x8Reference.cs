// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 inverse ADST 8x8 - bit-exact CPU oracle for libvpx iadst8_c
// plus the 2D ADST_ADST variant. Unlike iADST 4x4 which has a 3-stage
// butterfly with sinpi-based rotation constants, iADST 8x8 uses the
// standard cospi table with a non-trivial input reordering:
//
//     x0 = input[7], x1 = input[0], x2 = input[5], x3 = input[2],
//     x4 = input[3], x5 = input[4], x6 = input[1], x7 = input[6]
//
// libvpx reference: vpx_dsp/inv_txfm.c (iadst8_c).
//
// Structure
//   - Stage 1: 8 cospi multiplies into s0..s7 with Q14 rounding,
//     followed by +/- combining.
//   - Stage 2: passthrough on 0..3, 4 more cospi multiplies on 4..7
//     combined with Q14 rounding.
//   - Stage 3: 4 rotation multiplies on positions 2, 3, 6, 7 using
//     cospi_16_64 (the "DC" rotation constant).
//   - Output has negations at odd slots (-x4, -x2, -x7, -x1); the
//     layout is output = [x0, -x4, x6, -x2, x3, -x7, x5, -x1].
//
// 2D variant (ADST_ADST, tx_type=3):
//   - Row pass iADST each row; column pass iADST each column.
//   - Final round is (x + 16) >> 5 - same as iDCT 8x8 (8-point scale).

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>CPU oracle for VP9 inverse ADST 8x8.</summary>
public static class Vp9Iadst8x8Reference
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
    /// Apply <paramref name="input"/> (64 coefficients, row-major 8x8)
    /// via 2D iADST (rows then columns) as a residual to
    /// <paramref name="dest"/>. Matches libvpx vp9_iht8x8_64_add with
    /// tx_type = 3 (ADST_ADST).
    /// </summary>
    public static void IadstAdst8x8_64_Add(
        ReadOnlySpan<short> input, Span<byte> dest, int stride)
    {
        if (input.Length < 64)
            throw new ArgumentException("input must have >= 64 coefficients", nameof(input));
        if (stride < 8)
            throw new ArgumentException("stride must be >= 8", nameof(stride));
        if (dest.Length < 7 * stride + 8)
            throw new ArgumentException("dest too small for 8 rows at the given stride", nameof(dest));

        Span<short> tmp = stackalloc short[64];

        for (int row = 0; row < 8; row++)
        {
            Iadst8_1d(
                input.Slice(row * 8, 8),
                tmp.Slice(row * 8, 8));
        }

        Span<short> colIn = stackalloc short[8];
        Span<short> colOut = stackalloc short[8];
        for (int col = 0; col < 8; col++)
        {
            for (int j = 0; j < 8; j++) colIn[j] = tmp[j * 8 + col];
            Iadst8_1d(colIn, colOut);
            for (int j = 0; j < 8; j++)
            {
                // ROUND_POWER_OF_TWO(x, 5) = (x + 16) >> 5.
                int residual = (colOut[j] + 16) >> 5;
                int predictor = dest[j * stride + col];
                int sum = predictor + residual;
                if (sum < 0) sum = 0;
                else if (sum > 255) sum = 255;
                dest[j * stride + col] = (byte)sum;
            }
        }
    }

    /// <summary>
    /// One-dimensional 8-point iADST butterfly per libvpx iadst8_c.
    /// Internal so the iHT 8x8 tx_type dispatcher can share it.
    /// </summary>
    internal static void Iadst8_1d(ReadOnlySpan<short> input, Span<short> output)
    {
        // libvpx input reordering (NOT the natural order).
        int x0 = input[7];
        int x1 = input[0];
        int x2 = input[5];
        int x3 = input[2];
        int x4 = input[3];
        int x5 = input[4];
        int x6 = input[1];
        int x7 = input[6];

        if ((x0 | x1 | x2 | x3 | x4 | x5 | x6 | x7) == 0)
        {
            for (int i = 0; i < 8; i++) output[i] = 0;
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

        // Stage 3: rotation on positions 2,3,6,7 through cospi_16_64.
        s2 = CosPi16_64 * (x2 + x3);
        s3 = CosPi16_64 * (x2 - x3);
        s6 = CosPi16_64 * (x6 + x7);
        s7 = CosPi16_64 * (x6 - x7);

        x2 = Rs14(s2);
        x3 = Rs14(s3);
        x6 = Rs14(s6);
        x7 = Rs14(s7);

        // Output with sign inversions at odd positions.
        output[0] = (short)x0;
        output[1] = (short)-x4;
        output[2] = (short)x6;
        output[3] = (short)-x2;
        output[4] = (short)x3;
        output[5] = (short)-x7;
        output[6] = (short)x5;
        output[7] = (short)-x1;
    }

    /// <summary>Q14 rounded narrow-to-int16.</summary>
    private static int Rs14(int value) => (short)((value + (1 << 13)) >> 14);
}
