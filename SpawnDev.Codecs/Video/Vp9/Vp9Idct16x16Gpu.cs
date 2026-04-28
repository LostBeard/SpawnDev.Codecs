// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 16x16 inverse DCT, GPU-callable form for in-kernel reuse.
// Bit-exact mirror of Vp9Idct16x16Reference (the libvpx
// vp9_idct16x16_256_add_c port).
//
// Vp9Idct16x16Kernel already wraps this math as a standalone batched
// dispatch (one thread per 16x16 block, requires SpawnDev.ILGPU
// rc.12+ for the LoopUnrolling cap that makes WGSL compile cheaply).
// This helper is the in-kernel companion for the v3 sequential
// encoder/decoder path: the per-frame kernel iterates blocks
// sequentially and adds residual to recon inline.
//
// The 16-point butterfly carries [MethodImpl(NoInlining)] so the
// WGSL codegen routes through the function-definition path
// (SpawnDev.ILGPU rc.14 commit 1cb4f6c) - without it the WGSL
// shader inlines the 7-stage 16-point butterfly at all 32 call
// sites and the validator chokes (~30s+ per kernel instance).
//
// Two-pass shape:
//   Row pass: 16 row-1D iDCTs, intermediate stored as int (the
//             int16 narrowing at each butterfly sub-step reproduces
//             libvpx WRAPLOW() semantics).
//   Column pass: 16 column-1D iDCTs that add `(colOut + 32) >> 6`
//                to the predictor pixel + clip to [0, 255].

using System.Runtime.CompilerServices;
using ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// GPU-callable VP9 16x16 inverse DCT helper. Bit-exact mirror of
/// <see cref="Vp9Idct16x16Reference"/> for in-kernel use.
/// </summary>
public static class Vp9Idct16x16Gpu
{
    private const int CosPi16_64 = 11585;
    private const int CosPi8_64 = 15137;
    private const int CosPi24_64 = 6270;
    private const int CosPi4_64 = 16069;
    private const int CosPi12_64 = 13623;
    private const int CosPi20_64 = 9102;
    private const int CosPi28_64 = 3196;
    private const int CosPi2_64 = 16305;
    private const int CosPi6_64 = 15679;
    private const int CosPi10_64 = 14449;
    private const int CosPi14_64 = 12665;
    private const int CosPi18_64 = 10394;
    private const int CosPi22_64 = 7723;
    private const int CosPi26_64 = 4756;
    private const int CosPi30_64 = 1606;

    /// <summary>
    /// Inverse-DCT one 16x16 block and add the residual to
    /// <paramref name="dest"/> in place. <paramref name="scratch"/>
    /// must hold at least 256 ints for the inter-pass intermediate
    /// buffer.
    /// </summary>
    public static void Idct16x16(
        ArrayView<short> coefs, long coefBase,
        ArrayView<byte> dest, long destBase, int destStride,
        ArrayView<int> scratch)
    {
        // Row pass.
        for (int row = 0; row < 16; row++)
        {
            long rBase = coefBase + row * 16;
            Idct16Row(
                coefs[rBase + 0],  coefs[rBase + 1],  coefs[rBase + 2],  coefs[rBase + 3],
                coefs[rBase + 4],  coefs[rBase + 5],  coefs[rBase + 6],  coefs[rBase + 7],
                coefs[rBase + 8],  coefs[rBase + 9],  coefs[rBase + 10], coefs[rBase + 11],
                coefs[rBase + 12], coefs[rBase + 13], coefs[rBase + 14], coefs[rBase + 15],
                out int o0,  out int o1,  out int o2,  out int o3,
                out int o4,  out int o5,  out int o6,  out int o7,
                out int o8,  out int o9,  out int o10, out int o11,
                out int o12, out int o13, out int o14, out int o15);
            int rTmp = row * 16;
            scratch[rTmp + 0] = o0;   scratch[rTmp + 1] = o1;   scratch[rTmp + 2] = o2;   scratch[rTmp + 3] = o3;
            scratch[rTmp + 4] = o4;   scratch[rTmp + 5] = o5;   scratch[rTmp + 6] = o6;   scratch[rTmp + 7] = o7;
            scratch[rTmp + 8] = o8;   scratch[rTmp + 9] = o9;   scratch[rTmp + 10] = o10; scratch[rTmp + 11] = o11;
            scratch[rTmp + 12] = o12; scratch[rTmp + 13] = o13; scratch[rTmp + 14] = o14; scratch[rTmp + 15] = o15;
        }

        // Column pass + residual add + clip.
        for (int col = 0; col < 16; col++)
        {
            Idct16Row(
                (short)scratch[ 0 * 16 + col], (short)scratch[ 1 * 16 + col],
                (short)scratch[ 2 * 16 + col], (short)scratch[ 3 * 16 + col],
                (short)scratch[ 4 * 16 + col], (short)scratch[ 5 * 16 + col],
                (short)scratch[ 6 * 16 + col], (short)scratch[ 7 * 16 + col],
                (short)scratch[ 8 * 16 + col], (short)scratch[ 9 * 16 + col],
                (short)scratch[10 * 16 + col], (short)scratch[11 * 16 + col],
                (short)scratch[12 * 16 + col], (short)scratch[13 * 16 + col],
                (short)scratch[14 * 16 + col], (short)scratch[15 * 16 + col],
                out int co0,  out int co1,  out int co2,  out int co3,
                out int co4,  out int co5,  out int co6,  out int co7,
                out int co8,  out int co9,  out int co10, out int co11,
                out int co12, out int co13, out int co14, out int co15);

            ApplyResidualAndClip(dest, destBase +  0L * destStride + col, co0);
            ApplyResidualAndClip(dest, destBase +  1L * destStride + col, co1);
            ApplyResidualAndClip(dest, destBase +  2L * destStride + col, co2);
            ApplyResidualAndClip(dest, destBase +  3L * destStride + col, co3);
            ApplyResidualAndClip(dest, destBase +  4L * destStride + col, co4);
            ApplyResidualAndClip(dest, destBase +  5L * destStride + col, co5);
            ApplyResidualAndClip(dest, destBase +  6L * destStride + col, co6);
            ApplyResidualAndClip(dest, destBase +  7L * destStride + col, co7);
            ApplyResidualAndClip(dest, destBase +  8L * destStride + col, co8);
            ApplyResidualAndClip(dest, destBase +  9L * destStride + col, co9);
            ApplyResidualAndClip(dest, destBase + 10L * destStride + col, co10);
            ApplyResidualAndClip(dest, destBase + 11L * destStride + col, co11);
            ApplyResidualAndClip(dest, destBase + 12L * destStride + col, co12);
            ApplyResidualAndClip(dest, destBase + 13L * destStride + col, co13);
            ApplyResidualAndClip(dest, destBase + 14L * destStride + col, co14);
            ApplyResidualAndClip(dest, destBase + 15L * destStride + col, co15);
        }
    }

    /// <summary>
    /// 16-point 1D iDCT butterfly, bit-exact against
    /// Vp9Idct16x16Reference.Idct16_1d. 7 stages.
    /// NoInlining keeps WGSL shader size manageable - same reasoning
    /// as the standalone Vp9Idct16x16Kernel.
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void Idct16Row(
        short i0,  short i1,  short i2,  short i3,
        short i4,  short i5,  short i6,  short i7,
        short i8,  short i9,  short i10, short i11,
        short i12, short i13, short i14, short i15,
        out int o0,  out int o1,  out int o2,  out int o3,
        out int o4,  out int o5,  out int o6,  out int o7,
        out int o8,  out int o9,  out int o10, out int o11,
        out int o12, out int o13, out int o14, out int o15)
    {
        // Stage 1: bit-reversal-style input reordering.
        short s1_0 = i0;
        short s1_1 = i8;
        short s1_2 = i4;
        short s1_3 = i12;
        short s1_4 = i2;
        short s1_5 = i10;
        short s1_6 = i6;
        short s1_7 = i14;
        short s1_8 = i1;
        short s1_9 = i9;
        short s1_10 = i5;
        short s1_11 = i13;
        short s1_12 = i3;
        short s1_13 = i11;
        short s1_14 = i7;
        short s1_15 = i15;

        // Stage 2.
        short s2_0 = s1_0;
        short s2_1 = s1_1;
        short s2_2 = s1_2;
        short s2_3 = s1_3;
        short s2_4 = s1_4;
        short s2_5 = s1_5;
        short s2_6 = s1_6;
        short s2_7 = s1_7;

        int t8a  = s1_8 * CosPi30_64 - s1_15 * CosPi2_64;
        int t8b  = s1_8 * CosPi2_64  + s1_15 * CosPi30_64;
        short s2_8  = (short)((t8a + (1 << 13)) >> 14);
        short s2_15 = (short)((t8b + (1 << 13)) >> 14);

        int t9a  = s1_9 * CosPi14_64 - s1_14 * CosPi18_64;
        int t9b  = s1_9 * CosPi18_64 + s1_14 * CosPi14_64;
        short s2_9  = (short)((t9a + (1 << 13)) >> 14);
        short s2_14 = (short)((t9b + (1 << 13)) >> 14);

        int t10a = s1_10 * CosPi22_64 - s1_13 * CosPi10_64;
        int t10b = s1_10 * CosPi10_64 + s1_13 * CosPi22_64;
        short s2_10 = (short)((t10a + (1 << 13)) >> 14);
        short s2_13 = (short)((t10b + (1 << 13)) >> 14);

        int t11a = s1_11 * CosPi6_64  - s1_12 * CosPi26_64;
        int t11b = s1_11 * CosPi26_64 + s1_12 * CosPi6_64;
        short s2_11 = (short)((t11a + (1 << 13)) >> 14);
        short s2_12 = (short)((t11b + (1 << 13)) >> 14);

        // Stage 3.
        short s3_0 = s2_0;
        short s3_1 = s2_1;
        short s3_2 = s2_2;
        short s3_3 = s2_3;

        int t4a = s2_4 * CosPi28_64 - s2_7 * CosPi4_64;
        int t4b = s2_4 * CosPi4_64  + s2_7 * CosPi28_64;
        short s3_4 = (short)((t4a + (1 << 13)) >> 14);
        short s3_7 = (short)((t4b + (1 << 13)) >> 14);

        int t5a = s2_5 * CosPi12_64 - s2_6 * CosPi20_64;
        int t5b = s2_5 * CosPi20_64 + s2_6 * CosPi12_64;
        short s3_5 = (short)((t5a + (1 << 13)) >> 14);
        short s3_6 = (short)((t5b + (1 << 13)) >> 14);

        short s3_8  = (short)( s2_8  + s2_9);
        short s3_9  = (short)( s2_8  - s2_9);
        short s3_10 = (short)(-s2_10 + s2_11);
        short s3_11 = (short)( s2_10 + s2_11);
        short s3_12 = (short)( s2_12 + s2_13);
        short s3_13 = (short)( s2_12 - s2_13);
        short s3_14 = (short)(-s2_14 + s2_15);
        short s3_15 = (short)( s2_14 + s2_15);

        // Stage 4.
        int t01a = (s3_0 + s3_1) * CosPi16_64;
        int t01b = (s3_0 - s3_1) * CosPi16_64;
        short s4_0 = (short)((t01a + (1 << 13)) >> 14);
        short s4_1 = (short)((t01b + (1 << 13)) >> 14);

        int t23a = s3_2 * CosPi24_64 - s3_3 * CosPi8_64;
        int t23b = s3_2 * CosPi8_64  + s3_3 * CosPi24_64;
        short s4_2 = (short)((t23a + (1 << 13)) >> 14);
        short s4_3 = (short)((t23b + (1 << 13)) >> 14);

        short s4_4 = (short)( s3_4 + s3_5);
        short s4_5 = (short)( s3_4 - s3_5);
        short s4_6 = (short)(-s3_6 + s3_7);
        short s4_7 = (short)( s3_6 + s3_7);

        short s4_8 = s3_8;
        short s4_15 = s3_15;

        int t9c  = -s3_9  * CosPi8_64  + s3_14 * CosPi24_64;
        int t9d  =  s3_9  * CosPi24_64 + s3_14 * CosPi8_64;
        short s4_9  = (short)((t9c + (1 << 13)) >> 14);
        short s4_14 = (short)((t9d + (1 << 13)) >> 14);

        int t10c = -s3_10 * CosPi24_64 - s3_13 * CosPi8_64;
        int t10d = -s3_10 * CosPi8_64  + s3_13 * CosPi24_64;
        short s4_10 = (short)((t10c + (1 << 13)) >> 14);
        short s4_13 = (short)((t10d + (1 << 13)) >> 14);

        short s4_11 = s3_11;
        short s4_12 = s3_12;

        // Stage 5.
        short s5_0 = (short)(s4_0 + s4_3);
        short s5_1 = (short)(s4_1 + s4_2);
        short s5_2 = (short)(s4_1 - s4_2);
        short s5_3 = (short)(s4_0 - s4_3);
        short s5_4 = s4_4;

        int t56a = (s4_6 - s4_5) * CosPi16_64;
        int t56b = (s4_5 + s4_6) * CosPi16_64;
        short s5_5 = (short)((t56a + (1 << 13)) >> 14);
        short s5_6 = (short)((t56b + (1 << 13)) >> 14);
        short s5_7 = s4_7;

        short s5_8  = (short)( s4_8  + s4_11);
        short s5_9  = (short)( s4_9  + s4_10);
        short s5_10 = (short)( s4_9  - s4_10);
        short s5_11 = (short)( s4_8  - s4_11);
        short s5_12 = (short)(-s4_12 + s4_15);
        short s5_13 = (short)(-s4_13 + s4_14);
        short s5_14 = (short)( s4_13 + s4_14);
        short s5_15 = (short)( s4_12 + s4_15);

        // Stage 6.
        short s6_0 = (short)(s5_0 + s5_7);
        short s6_1 = (short)(s5_1 + s5_6);
        short s6_2 = (short)(s5_2 + s5_5);
        short s6_3 = (short)(s5_3 + s5_4);
        short s6_4 = (short)(s5_3 - s5_4);
        short s6_5 = (short)(s5_2 - s5_5);
        short s6_6 = (short)(s5_1 - s5_6);
        short s6_7 = (short)(s5_0 - s5_7);
        short s6_8  = s5_8;
        short s6_9  = s5_9;

        int t1013a = (-s5_10 + s5_13) * CosPi16_64;
        int t1013b = ( s5_10 + s5_13) * CosPi16_64;
        short s6_10 = (short)((t1013a + (1 << 13)) >> 14);
        short s6_13 = (short)((t1013b + (1 << 13)) >> 14);

        int t1112a = (-s5_11 + s5_12) * CosPi16_64;
        int t1112b = ( s5_11 + s5_12) * CosPi16_64;
        short s6_11 = (short)((t1112a + (1 << 13)) >> 14);
        short s6_12 = (short)((t1112b + (1 << 13)) >> 14);

        short s6_14 = s5_14;
        short s6_15 = s5_15;

        // Stage 7: final combining butterfly.
        o0  = (short)(s6_0  + s6_15);
        o1  = (short)(s6_1  + s6_14);
        o2  = (short)(s6_2  + s6_13);
        o3  = (short)(s6_3  + s6_12);
        o4  = (short)(s6_4  + s6_11);
        o5  = (short)(s6_5  + s6_10);
        o6  = (short)(s6_6  + s6_9);
        o7  = (short)(s6_7  + s6_8);
        o8  = (short)(s6_7  - s6_8);
        o9  = (short)(s6_6  - s6_9);
        o10 = (short)(s6_5  - s6_10);
        o11 = (short)(s6_4  - s6_11);
        o12 = (short)(s6_3  - s6_12);
        o13 = (short)(s6_2  - s6_13);
        o14 = (short)(s6_1  - s6_14);
        o15 = (short)(s6_0  - s6_15);
    }

    private static void ApplyResidualAndClip(ArrayView<byte> dest, long offset, int colOut)
    {
        int residual = (colOut + 32) >> 6;
        int sum = dest[offset] + residual;
        if (sum < 0) sum = 0;
        else if (sum > 255) sum = 255;
        dest[offset] = (byte)sum;
    }
}
