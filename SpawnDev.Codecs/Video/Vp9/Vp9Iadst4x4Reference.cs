// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 inverse ADST 4x4 - bit-exact CPU reference port of libvpx iadst4_c
// followed by the ADST_ADST 2D transform (vp9_iht4x4_16_add with
// tx_type = 3). iADST is VP9's second block-transform, paired with iDCT
// for intra-prediction modes. The 2D variant supports four tx_type
// combinations; this file implements the pure-ADST one. Mixed types
// (iADST rows + iDCT columns, etc.) land in a follow-up slice with a
// tx_type dispatcher.
//
// Spec: VP9 Bitstream Specification sec 8.7.1.5 "Inverse 4x4 ADST".
// libvpx reference: vp9/common/vp9_idct.c (iadst4_c / vp9_iht4x4_16_add_c).
//
// Constants
//   sinpi_1_9 = 5283   (sin(pi*1/9) * 2^14)
//   sinpi_2_9 = 9929   (sin(pi*2/9) * 2^14)
//   sinpi_3_9 = 13377  (sin(pi*3/9) * 2^14)
//   sinpi_4_9 = 15212  (sin(pi*4/9) * 2^14)

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// CPU reference for VP9 inverse ADST 4x4. Bit-exact against libvpx.
/// </summary>
public static class Vp9Iadst4x4Reference
{
    private const int SinPi1_9 = 5283;
    private const int SinPi2_9 = 9929;
    private const int SinPi3_9 = 13377;
    private const int SinPi4_9 = 15212;

    /// <summary>
    /// Apply <paramref name="input"/> (16 coefficients, row-major 4x4) via
    /// 2D iADST (rows then columns) as a residual to <paramref name="dest"/>.
    /// Matches libvpx <c>vp9_iht4x4_16_add</c> with <c>tx_type = 3</c>
    /// (ADST_ADST).
    /// </summary>
    public static void IadstAdst4x4_16_Add(
        ReadOnlySpan<short> input, Span<byte> dest, int stride)
    {
        if (input.Length < 16)
            throw new ArgumentException("input must have >= 16 coefficients", nameof(input));
        if (stride < 4)
            throw new ArgumentException("stride must be >= 4", nameof(stride));
        if (dest.Length < 3 * stride + 4)
            throw new ArgumentException("dest too small for 4 rows at the given stride", nameof(dest));

        Span<short> tmp = stackalloc short[16];

        // Row pass: iADST each row.
        for (int row = 0; row < 4; row++)
        {
            Iadst4_1d(
                input.Slice(row * 4, 4),
                tmp.Slice(row * 4, 4));
        }

        // Column pass: iADST each column; final round + residual-add + clip.
        Span<short> colIn = stackalloc short[4];
        Span<short> colOut = stackalloc short[4];
        for (int col = 0; col < 4; col++)
        {
            for (int j = 0; j < 4; j++) colIn[j] = tmp[j * 4 + col];
            Iadst4_1d(colIn, colOut);
            for (int j = 0; j < 4; j++)
            {
                // Same final >>4 rounding as iDCT 4x4 - the 4-point
                // transform pair shares the overall scale factor.
                int residual = (colOut[j] + 8) >> 4;
                int predictor = dest[j * stride + col];
                int sum = predictor + residual;
                if (sum < 0) sum = 0;
                else if (sum > 255) sum = 255;
                dest[j * stride + col] = (byte)sum;
            }
        }
    }

    /// <summary>
    /// One-dimensional 4-point iADST butterfly per libvpx <c>iadst4_c</c>.
    /// Fast-exits with zero output when all 4 inputs are zero (matches the
    /// libvpx short-circuit; not strictly required for correctness but
    /// skips pointless arithmetic).
    /// </summary>
    private static void Iadst4_1d(ReadOnlySpan<short> input, Span<short> output)
    {
        int x0 = input[0];
        int x1 = input[1];
        int x2 = input[2];
        int x3 = input[3];

        if ((x0 | x1 | x2 | x3) == 0)
        {
            output[0] = output[1] = output[2] = output[3] = 0;
            return;
        }

        // Per-input sinpi multiplies.
        int s0 = SinPi1_9 * x0;
        int s1 = SinPi2_9 * x0;
        int s2 = SinPi3_9 * x1;
        int s3 = SinPi4_9 * x2;
        int s4 = SinPi1_9 * x2;
        int s5 = SinPi2_9 * x3;
        int s6 = SinPi4_9 * x3;
        int s7 = x0 - x2 + x3;

        // Combine.
        int c0 = s0 + s3 + s5;
        int c1 = s1 - s4 - s6;
        int c3 = s2;
        int c2 = SinPi3_9 * s7;

        // 1D transform scaling factor is sqrt(2); Q14 rounding finalises
        // each output short.
        output[0] = RoundShift14(c0 + c3);
        output[1] = RoundShift14(c1 + c3);
        output[2] = RoundShift14(c2);
        output[3] = RoundShift14(c0 + c1 - c3);
    }

    private static short RoundShift14(int value) => (short)((value + (1 << 13)) >> 14);
}
