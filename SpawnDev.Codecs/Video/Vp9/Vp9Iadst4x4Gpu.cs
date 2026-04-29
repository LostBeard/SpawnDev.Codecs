// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 inverse ADST 4x4, GPU-callable form for in-kernel reuse. Bit-exact
// mirror of Vp9Iadst4x4Reference (libvpx vp9_iht4x4_16_add tx_type=3
// ADST_ADST port).
//
// Pairs with the existing Vp9Idct4x4Gpu (just shipped) - now both 4x4
// transform types in VP9 (DCT_DCT and ADST_ADST) have GPU primitives.
//
// Two-pass shape (same as Vp9Idct4x4Gpu):
//   Row pass: 4 row-1D iADSTs, intermediate stored in scratch as short.
//   Column pass: 4 column-1D iADSTs that add (colOut + 8) >> 4 to the
//                predictor pixel and clip to [0, 255].
//
// Caller supplies a 16-short scratch view for the inter-pass intermediate.

using ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// GPU-callable VP9 inverse ADST 4x4 helper. Bit-exact mirror of
/// <see cref="Vp9Iadst4x4Reference"/> for in-kernel use.
/// </summary>
public static class Vp9Iadst4x4Gpu
{
    private const int SinPi1_9 = 5283;
    private const int SinPi2_9 = 9929;
    private const int SinPi3_9 = 13377;
    private const int SinPi4_9 = 15212;

    /// <summary>
    /// Inverse-ADST one 4x4 block (ADST_ADST) and add the residual to
    /// <paramref name="dest"/> in place. Reads <paramref name="coefs"/>
    /// starting at <paramref name="coefBase"/> (16 contiguous shorts,
    /// row-major); writes back to <paramref name="dest"/> with row stride
    /// <paramref name="destStride"/>.
    /// </summary>
    public static void Iadst4x4(
        ArrayView<short> coefs, long coefBase,
        ArrayView<byte> dest, long destBase, int destStride,
        ArrayView<short> scratch)
    {
        // Row pass.
        for (int row = 0; row < 4; row++)
        {
            long rBase = coefBase + row * 4;
            Iadst4(
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
            Iadst4(
                scratch[0 * 4 + col], scratch[1 * 4 + col],
                scratch[2 * 4 + col], scratch[3 * 4 + col],
                out short c0, out short c1, out short c2, out short c3);

            ApplyResidual(c0, dest, destBase + 0L * destStride + col);
            ApplyResidual(c1, dest, destBase + 1L * destStride + col);
            ApplyResidual(c2, dest, destBase + 2L * destStride + col);
            ApplyResidual(c3, dest, destBase + 3L * destStride + col);
        }
    }

    /// <summary>One-dimensional 4-point iADST butterfly with zero-input early-out.</summary>
    private static void Iadst4(
        short i0, short i1, short i2, short i3,
        out short o0, out short o1, out short o2, out short o3)
    {
        int x0 = i0;
        int x1 = i1;
        int x2 = i2;
        int x3 = i3;

        if ((x0 | x1 | x2 | x3) == 0)
        {
            o0 = 0;
            o1 = 0;
            o2 = 0;
            o3 = 0;
            return;
        }

        int s0 = SinPi1_9 * x0;
        int s1 = SinPi2_9 * x0;
        int s2 = SinPi3_9 * x1;
        int s3 = SinPi4_9 * x2;
        int s4 = SinPi1_9 * x2;
        int s5 = SinPi2_9 * x3;
        int s6 = SinPi4_9 * x3;
        int s7 = x0 - x2 + x3;

        int c0 = s0 + s3 + s5;
        int c1 = s1 - s4 - s6;
        int c3 = s2;
        int c2 = SinPi3_9 * s7;

        o0 = RoundShift14(c0 + c3);
        o1 = RoundShift14(c1 + c3);
        o2 = RoundShift14(c2);
        o3 = RoundShift14(c0 + c1 - c3);
    }

    private static short RoundShift14(int value) => (short)((value + (1 << 13)) >> 14);

    private static void ApplyResidual(short residual, ArrayView<byte> dest, long destIdx)
    {
        int rounded = (residual + 8) >> 4;
        int sum = dest[destIdx] + rounded;
        if (sum < 0) sum = 0;
        else if (sum > 255) sum = 255;
        dest[destIdx] = (byte)sum;
    }
}
