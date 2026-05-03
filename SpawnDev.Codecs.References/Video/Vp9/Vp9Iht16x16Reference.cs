// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 inverse Hybrid Transform 16x16 - bit-exact CPU oracle for libvpx
// vp9_iht16x16_256_add_c. Largest block size supporting hybrid iADST/iDCT
// per VP9 spec (32x32 is iDCT-only).
//
// Structure mirrors slice 122's 4x4 and slice 126's 8x8 dispatchers:
//   tx_type = 0  DCT_DCT    (iDCT  rows, iDCT  cols)
//   tx_type = 1  ADST_DCT   (iADST rows, iDCT  cols)
//   tx_type = 2  DCT_ADST   (iDCT  rows, iADST cols)
//   tx_type = 3  ADST_ADST  (iADST rows, iADST cols)
//
// Final residual round is (x + 32) >> 6 (same as all 16x16 transforms).

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 16x16 transform-type code.</summary>
public enum Vp9TxType16x16
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

/// <summary>CPU oracle for VP9 inverse 2D 16x16 hybrid transform.</summary>
public static class Vp9Iht16x16Reference
{
    /// <summary>
    /// Apply the 2D inverse transform indicated by <paramref name="txType"/>
    /// to <paramref name="input"/> (256 coefficients, row-major 16x16) as a
    /// residual to <paramref name="dest"/>.
    /// </summary>
    public static void Iht16x16_256_Add(
        Vp9TxType16x16 txType,
        ReadOnlySpan<short> input, Span<byte> dest, int stride)
    {
        if (input.Length < 256)
            throw new ArgumentException("input must have >= 256 coefficients", nameof(input));
        if (stride < 16)
            throw new ArgumentException("stride must be >= 16", nameof(stride));
        if (dest.Length < 15 * stride + 16)
            throw new ArgumentException("dest too small for 16 rows at the given stride", nameof(dest));

        bool rowIsAdst = ((int)txType & 1) != 0;
        bool colIsAdst = ((int)txType & 2) != 0;

        Span<short> tmp = stackalloc short[256];
        for (int row = 0; row < 16; row++)
        {
            var rowIn = input.Slice(row * 16, 16);
            var rowOut = tmp.Slice(row * 16, 16);
            if (rowIsAdst) Vp9Iadst16x16Reference.Iadst16_1d(rowIn, rowOut);
            else           Vp9Idct16x16Reference.Idct16_1d(rowIn, rowOut);
        }

        Span<short> colIn = stackalloc short[16];
        Span<short> colOut = stackalloc short[16];
        for (int col = 0; col < 16; col++)
        {
            for (int j = 0; j < 16; j++) colIn[j] = tmp[j * 16 + col];
            if (colIsAdst) Vp9Iadst16x16Reference.Iadst16_1d(colIn, colOut);
            else           Vp9Idct16x16Reference.Idct16_1d(colIn, colOut);
            for (int j = 0; j < 16; j++)
            {
                int residual = (colOut[j] + 32) >> 6;
                int predictor = dest[j * stride + col];
                int sum = predictor + residual;
                if (sum < 0) sum = 0;
                else if (sum > 255) sum = 255;
                dest[j * stride + col] = (byte)sum;
            }
        }
    }
}
