// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 inverse Hybrid Transform 4x4 - bit-exact CPU reference port of
// libvpx vp9_iht4x4_16_add_c.
//
// VP9's intra-prediction modes select one of four 2D transform type
// combinations at 4x4 block size (VP9 Bitstream Spec sec 8.7.1.6):
//
//   tx_type = 0  DCT_DCT    (iDCT  rows, iDCT  cols)
//   tx_type = 1  ADST_DCT   (iADST rows, iDCT  cols)
//   tx_type = 2  DCT_ADST   (iDCT  rows, iADST cols)
//   tx_type = 3  ADST_ADST  (iADST rows, iADST cols)
//
// This dispatcher picks the row and column 1D transforms per tx_type,
// runs the row pass then the column pass, applies ROUND_POWER_OF_TWO(,4)
// to each coefficient, adds to the predictor, and clips to [0, 255].

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 4x4 transform-type code (spec sec 8.7.1.6).</summary>
public enum Vp9TxType4x4
{
    /// <summary>Both dimensions use iDCT.</summary>
    DctDct = 0,
    /// <summary>Rows use iADST, columns use iDCT.</summary>
    AdstDct = 1,
    /// <summary>Rows use iDCT, columns use iADST.</summary>
    DctAdst = 2,
    /// <summary>Both dimensions use iADST.</summary>
    AdstAdst = 3,
}

/// <summary>
/// CPU reference for VP9 inverse 2D 4x4 hybrid transform. Matches libvpx
/// <c>vp9_iht4x4_16_add_c</c> bit-exactly across all four tx_types.
/// </summary>
public static class Vp9Iht4x4Reference
{
    /// <summary>
    /// Apply the 2D inverse transform indicated by <paramref name="txType"/>
    /// to <paramref name="input"/> (16 coefficients, row-major 4x4) as a
    /// residual to <paramref name="dest"/> (4x4 block of 8-bit pixels with
    /// <paramref name="stride"/> bytes per row).
    /// </summary>
    public static void Iht4x4_16_Add(
        Vp9TxType4x4 txType,
        ReadOnlySpan<short> input, Span<byte> dest, int stride)
    {
        if (input.Length < 16)
            throw new ArgumentException("input must have >= 16 coefficients", nameof(input));
        if (stride < 4)
            throw new ArgumentException("stride must be >= 4", nameof(stride));
        if (dest.Length < 3 * stride + 4)
            throw new ArgumentException("dest too small for 4 rows at the given stride", nameof(dest));

        // Row pass - select row transform by tx_type. In libvpx the
        // low bit of tx_type chooses the ROW transform (0/2 -> iDCT,
        // 1/3 -> iADST), and the high bit chooses the column transform
        // (0/1 -> iDCT, 2/3 -> iADST).
        bool rowIsAdst = ((int)txType & 1) != 0;
        bool colIsAdst = ((int)txType & 2) != 0;

        Span<short> tmp = stackalloc short[16];

        for (int row = 0; row < 4; row++)
        {
            var rowIn = input.Slice(row * 4, 4);
            var rowOut = tmp.Slice(row * 4, 4);
            if (rowIsAdst)
                Vp9Iadst4x4Reference.Iadst4_1d(rowIn, rowOut);
            else
                Vp9Idct4x4Reference.Idct4_1d(rowIn, rowOut);
        }

        Span<short> colIn = stackalloc short[4];
        Span<short> colOut = stackalloc short[4];
        for (int col = 0; col < 4; col++)
        {
            for (int j = 0; j < 4; j++) colIn[j] = tmp[j * 4 + col];
            if (colIsAdst)
                Vp9Iadst4x4Reference.Iadst4_1d(colIn, colOut);
            else
                Vp9Idct4x4Reference.Idct4_1d(colIn, colOut);
            for (int j = 0; j < 4; j++)
            {
                int residual = (colOut[j] + 8) >> 4;
                int predictor = dest[j * stride + col];
                int sum = predictor + residual;
                if (sum < 0) sum = 0;
                else if (sum > 255) sum = 255;
                dest[j * stride + col] = (byte)sum;
            }
        }
    }
}
