// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 inverse ADST 16x16, GPU-callable form for in-kernel reuse.
// Bit-exact mirror of Vp9Iadst16x16Reference (libvpx vp9_iht16x16_256_add
// tx_type=3 ADST_ADST port).
//
// Pairs with the existing Vp9Idct16x16Gpu - now both 16x16 transform
// types in VP9 (DCT_DCT and ADST_ADST) have GPU primitives. Combined
// with the just-shipped Vp9Iadst4x4Gpu and Vp9Iadst8x8Gpu, the VP9
// inverse ADST family is now complete at all 3 valid sizes (4/8/16);
// 32x32 is iDCT-only per VP9 spec.
//
// Two-pass shape (matches Vp9Idct16x16Gpu):
//   Row pass: 16 row-1D iADSTs, intermediate stored in scratch as short.
//   Column pass: 16 column-1D iADSTs that add (colOut + 32) >> 6 to the
//                predictor pixel and clip to [0, 255].

using ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// GPU-callable VP9 inverse ADST 16x16 helper. Bit-exact mirror of
/// <see cref="Vp9Iadst16x16Reference"/> for in-kernel use.
/// </summary>
public static class Vp9Iadst16x16Gpu
{
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
    /// Inverse-ADST one 16x16 block (ADST_ADST) and add the residual to
    /// <paramref name="dest"/> in place. Caller supplies a 256-short
    /// scratch view for the inter-pass intermediate.
    /// </summary>
    public static void Iadst16x16(
        ArrayView<short> coefs, long coefBase,
        ArrayView<byte> dest, long destBase, int destStride,
        ArrayView<short> scratch)
    {
        // Row pass into scratch.
        for (int row = 0; row < 16; row++)
        {
            Iadst16(coefs, coefBase + row * 16L, scratch, row * 16L);
        }

        // Column pass + residual add + clip.
        for (int col = 0; col < 16; col++)
        {
            // Column pass uses scratch as input; we re-use scratch by reading column-wise.
            // Apply Iadst16 to column 'col' of scratch, write directly into a per-call
            // 16-short stack via the kernel-friendly Iadst16Col helper.
            Iadst16Col(scratch, col, dest, destBase, destStride);
        }
    }

    /// <summary>One-dimensional 16-point iADST per libvpx <c>iadst16_c</c>, row form.</summary>
    private static void Iadst16(
        ArrayView<short> input, long inBase,
        ArrayView<short> output, long outBase)
    {
        int x0 = input[inBase + 15];
        int x1 = input[inBase + 0];
        int x2 = input[inBase + 13];
        int x3 = input[inBase + 2];
        int x4 = input[inBase + 11];
        int x5 = input[inBase + 4];
        int x6 = input[inBase + 9];
        int x7 = input[inBase + 6];
        int x8 = input[inBase + 7];
        int x9 = input[inBase + 8];
        int x10 = input[inBase + 5];
        int x11 = input[inBase + 10];
        int x12 = input[inBase + 3];
        int x13 = input[inBase + 12];
        int x14 = input[inBase + 1];
        int x15 = input[inBase + 14];

        if ((x0 | x1 | x2 | x3 | x4 | x5 | x6 | x7 |
             x8 | x9 | x10 | x11 | x12 | x13 | x14 | x15) == 0)
        {
            for (int i = 0; i < 16; i++) output[outBase + i] = 0;
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
        output[outBase + 0] = (short)x0;
        output[outBase + 1] = (short)-x8;
        output[outBase + 2] = (short)x12;
        output[outBase + 3] = (short)-x4;
        output[outBase + 4] = (short)x6;
        output[outBase + 5] = (short)x14;
        output[outBase + 6] = (short)x10;
        output[outBase + 7] = (short)x2;
        output[outBase + 8] = (short)x3;
        output[outBase + 9] = (short)x11;
        output[outBase + 10] = (short)x15;
        output[outBase + 11] = (short)x7;
        output[outBase + 12] = (short)x5;
        output[outBase + 13] = (short)-x13;
        output[outBase + 14] = (short)x9;
        output[outBase + 15] = (short)-x1;
    }

    /// <summary>Column-pass variant: reads scratch column-wise, applies iADST, writes to dest with residual add.</summary>
    private static void Iadst16Col(
        ArrayView<short> scratch, int col,
        ArrayView<byte> dest, long destBase, int destStride)
    {
        // Read column from scratch.
        int x0 = scratch[15 * 16 + col];
        int x1 = scratch[0 * 16 + col];
        int x2 = scratch[13 * 16 + col];
        int x3 = scratch[2 * 16 + col];
        int x4 = scratch[11 * 16 + col];
        int x5 = scratch[4 * 16 + col];
        int x6 = scratch[9 * 16 + col];
        int x7 = scratch[6 * 16 + col];
        int x8 = scratch[7 * 16 + col];
        int x9 = scratch[8 * 16 + col];
        int x10 = scratch[5 * 16 + col];
        int x11 = scratch[10 * 16 + col];
        int x12 = scratch[3 * 16 + col];
        int x13 = scratch[12 * 16 + col];
        int x14 = scratch[1 * 16 + col];
        int x15 = scratch[14 * 16 + col];

        if ((x0 | x1 | x2 | x3 | x4 | x5 | x6 | x7 |
             x8 | x9 | x10 | x11 | x12 | x13 | x14 | x15) == 0)
        {
            // All-zero column: rounding (0 + 32) >> 6 = 0; predictor unchanged.
            return;
        }

        // (Stages 1-4: identical to Iadst16 above. Inlined here to keep ILGPU
        // happy with no helper allocation; the math is bit-identical.)
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

        // Output with sign inversions, residual add to dest pixels.
        ApplyResidual(x0,   dest, destBase + 0L * destStride + col);
        ApplyResidual(-x8,  dest, destBase + 1L * destStride + col);
        ApplyResidual(x12,  dest, destBase + 2L * destStride + col);
        ApplyResidual(-x4,  dest, destBase + 3L * destStride + col);
        ApplyResidual(x6,   dest, destBase + 4L * destStride + col);
        ApplyResidual(x14,  dest, destBase + 5L * destStride + col);
        ApplyResidual(x10,  dest, destBase + 6L * destStride + col);
        ApplyResidual(x2,   dest, destBase + 7L * destStride + col);
        ApplyResidual(x3,   dest, destBase + 8L * destStride + col);
        ApplyResidual(x11,  dest, destBase + 9L * destStride + col);
        ApplyResidual(x15,  dest, destBase + 10L * destStride + col);
        ApplyResidual(x7,   dest, destBase + 11L * destStride + col);
        ApplyResidual(x5,   dest, destBase + 12L * destStride + col);
        ApplyResidual(-x13, dest, destBase + 13L * destStride + col);
        ApplyResidual(x9,   dest, destBase + 14L * destStride + col);
        ApplyResidual(-x1,  dest, destBase + 15L * destStride + col);
    }

    private static short Rs14(int value) => (short)((value + (1 << 13)) >> 14);

    private static void ApplyResidual(int residualInt, ArrayView<byte> dest, long destIdx)
    {
        // ROUND_POWER_OF_TWO(x, 6) = (x + 32) >> 6
        int residual = (residualInt + 32) >> 6;
        int sum = dest[destIdx] + residual;
        if (sum < 0) sum = 0;
        else if (sum > 255) sum = 255;
        dest[destIdx] = (byte)sum;
    }
}
