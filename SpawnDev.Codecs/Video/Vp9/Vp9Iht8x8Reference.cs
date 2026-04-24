// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 inverse Hybrid Transform 8x8 - bit-exact CPU oracle for libvpx
// vp9_iht8x8_64_add_c.
//
// Same shape as slice 122's 4x4 dispatcher, applied at 8x8 block size
// using the 1D transforms from slice 119 (iDCT 8x8) and slice 125
// (iADST 8x8).
//
// VP9 intra-prediction at 8x8 selects one of:
//   tx_type = 0  DCT_DCT    (iDCT  rows, iDCT  cols)
//   tx_type = 1  ADST_DCT   (iADST rows, iDCT  cols)
//   tx_type = 2  DCT_ADST   (iDCT  rows, iADST cols)
//   tx_type = 3  ADST_ADST  (iADST rows, iADST cols)
//
// Final residual round is (x + 16) >> 5, same as standalone iDCT 8x8
// and iADST 8x8 - all three share the 8-point scale factor.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 8x8 transform-type code.</summary>
public enum Vp9TxType8x8
{
    /// <summary>Both dimensions iDCT.</summary>
    DctDct = 0,
    /// <summary>Rows iADST, columns iDCT.</summary>
    AdstDct = 1,
    /// <summary>Rows iDCT, columns iADST.</summary>
    DctAdst = 2,
    /// <summary>Both dimensions iADST.</summary>
    AdstAdst = 3,
}

/// <summary>CPU oracle for VP9 inverse 2D 8x8 hybrid transform.</summary>
public static class Vp9Iht8x8Reference
{
    /// <summary>
    /// Apply the 2D inverse transform indicated by <paramref name="txType"/>
    /// to <paramref name="input"/> (64 coefficients, row-major 8x8) as a
    /// residual to <paramref name="dest"/>.
    /// </summary>
    public static void Iht8x8_64_Add(
        Vp9TxType8x8 txType,
        ReadOnlySpan<short> input, Span<byte> dest, int stride)
    {
        if (input.Length < 64)
            throw new ArgumentException("input must have >= 64 coefficients", nameof(input));
        if (stride < 8)
            throw new ArgumentException("stride must be >= 8", nameof(stride));
        if (dest.Length < 7 * stride + 8)
            throw new ArgumentException("dest too small for 8 rows at the given stride", nameof(dest));

        // Per libvpx convention: low bit of tx_type selects ROW transform
        // (0/2 -> iDCT, 1/3 -> iADST); high bit selects COLUMN transform
        // (0/1 -> iDCT, 2/3 -> iADST).
        bool rowIsAdst = ((int)txType & 1) != 0;
        bool colIsAdst = ((int)txType & 2) != 0;

        Span<short> tmp = stackalloc short[64];
        for (int row = 0; row < 8; row++)
        {
            var rowIn = input.Slice(row * 8, 8);
            var rowOut = tmp.Slice(row * 8, 8);
            if (rowIsAdst) Vp9Iadst8x8Reference.Iadst8_1d(rowIn, rowOut);
            else           Vp9Idct8x8Reference.Idct8_1d(rowIn, rowOut);
        }

        Span<short> colIn = stackalloc short[8];
        Span<short> colOut = stackalloc short[8];
        for (int col = 0; col < 8; col++)
        {
            for (int j = 0; j < 8; j++) colIn[j] = tmp[j * 8 + col];
            if (colIsAdst) Vp9Iadst8x8Reference.Iadst8_1d(colIn, colOut);
            else           Vp9Idct8x8Reference.Idct8_1d(colIn, colOut);
            for (int j = 0; j < 8; j++)
            {
                int residual = (colOut[j] + 16) >> 5;
                int predictor = dest[j * stride + col];
                int sum = predictor + residual;
                if (sum < 0) sum = 0;
                else if (sum > 255) sum = 255;
                dest[j * stride + col] = (byte)sum;
            }
        }
    }
}
