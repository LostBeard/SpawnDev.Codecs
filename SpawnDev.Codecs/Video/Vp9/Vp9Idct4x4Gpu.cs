// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 4x4 inverse DCT, GPU-callable form for in-kernel reuse.
// Bit-exact mirror of Vp9Idct4x4Reference (the libvpx vp9_idct4x4_16_add
// port).
//
// Pairs with the existing Vp9Idct8x8Gpu and Vp9Idct16x16Gpu helpers -
// completes the standalone single-block iDCT primitives needed by the
// v3 sequential decoder/encoder reconstruction loops.
//
// Two-pass shape (matches Vp9Idct4x4Reference):
//   Row pass: 4 row-1D iDCTs, intermediate stored in scratch as short.
//   Column pass: 4 column-1D iDCTs that add (colOut + 8) >> 4 to the
//                predictor pixel + clip to [0, 255].
//
// Caller supplies a 16-short scratch view for the inter-pass intermediate.

using ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// GPU-callable VP9 4x4 inverse DCT helper. Bit-exact mirror of
/// <see cref="Vp9Idct4x4Reference"/> for in-kernel use.
/// </summary>
public static class Vp9Idct4x4Gpu
{
    private const int CosPi16_64 = 11585;
    private const int CosPi8_64 = 15137;
    private const int CosPi24_64 = 6270;

    /// <summary>
    /// Inverse-DCT one 4x4 block and add the residual to
    /// <paramref name="dest"/> in place. Reads <paramref name="coefs"/>
    /// starting at <paramref name="coefBase"/> (16 contiguous shorts,
    /// row-major); writes back to <paramref name="dest"/> starting at
    /// <paramref name="destBase"/> with row stride <paramref name="destStride"/>.
    /// Each output pixel is <c>clip3(0, 255, dest + (residual + 8) &gt;&gt; 4)</c>.
    ///
    /// <paramref name="scratch"/> must hold at least 16 shorts for the
    /// inter-pass intermediate buffer.
    /// </summary>
    public static void Idct4x4(
        ArrayView<short> coefs, long coefBase,
        ArrayView<byte> dest, long destBase, int destStride,
        ArrayView<short> scratch)
    {
        // Row pass.
        for (int row = 0; row < 4; row++)
        {
            long rBase = coefBase + row * 4;
            Idct4(
                coefs[rBase + 0], coefs[rBase + 1],
                coefs[rBase + 2], coefs[rBase + 3],
                out short o0, out short o1, out short o2, out short o3);
            int rb = row * 4;
            scratch[rb + 0] = o0;
            scratch[rb + 1] = o1;
            scratch[rb + 2] = o2;
            scratch[rb + 3] = o3;
        }

        // Column pass + residual add + clip.
        for (int col = 0; col < 4; col++)
        {
            Idct4(
                scratch[0 * 4 + col], scratch[1 * 4 + col],
                scratch[2 * 4 + col], scratch[3 * 4 + col],
                out short c0, out short c1, out short c2, out short c3);

            ApplyResidual(c0, dest, destBase + 0L * destStride + col);
            ApplyResidual(c1, dest, destBase + 1L * destStride + col);
            ApplyResidual(c2, dest, destBase + 2L * destStride + col);
            ApplyResidual(c3, dest, destBase + 3L * destStride + col);
        }
    }

    /// <summary>One-dimensional 4-point iDCT butterfly.</summary>
    private static void Idct4(
        short i0, short i1, short i2, short i3,
        out short o0, out short o1, out short o2, out short o3)
    {
        int t1 = (i0 + i2) * CosPi16_64;
        int t2 = (i0 - i2) * CosPi16_64;
        short step0 = DctConstRoundShift(t1);
        short step1 = DctConstRoundShift(t2);
        int t3 = i1 * CosPi24_64 - i3 * CosPi8_64;
        int t4 = i1 * CosPi8_64 + i3 * CosPi24_64;
        short step2 = DctConstRoundShift(t3);
        short step3 = DctConstRoundShift(t4);

        o0 = (short)(step0 + step3);
        o1 = (short)(step1 + step2);
        o2 = (short)(step1 - step2);
        o3 = (short)(step0 - step3);
    }

    private static short DctConstRoundShift(int value)
    {
        int rounded = (value + (1 << 13)) >> 14;
        return (short)rounded;
    }

    private static void ApplyResidual(short residual, ArrayView<byte> dest, long destIdx)
    {
        // ROUND_POWER_OF_TWO(x, 4) = (x + 8) >> 4
        int rounded = (residual + 8) >> 4;
        int sum = dest[destIdx] + rounded;
        if (sum < 0) sum = 0;
        else if (sum > 255) sum = 255;
        dest[destIdx] = (byte)sum;
    }
}
