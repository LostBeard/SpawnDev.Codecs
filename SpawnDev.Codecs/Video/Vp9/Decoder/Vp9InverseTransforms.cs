// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt). See NOTICE.md for upstream attributions.
//
// VP9 inverse transforms + reconstruction add (8-bit, profile 0), kernel-callable. A port of libvpx v1.17.0
// (BSD-3-Clause) vpx_dsp/inv_txfm.c (the non-high-bitdepth half: idct4/8/16/32, iadst4/8/16, vpx_iwht4x4_16_add,
// vpx_iwht4x4_1_add, the vpx_idct*_add eob variants) and vp9/common/vp9_idct.c (vp9_iht*_add / vp9_idct*_add /
// vp9_iwht4x4_add, the dispatch the decoder calls), with the constants of vpx_dsp/txfm_common.h.
//
// Integer semantics of libvpx's 8-bit build, exactly: tran_low_t is int16 (every store into a coefficient array, an
// int16_t step array or a tran_low_t variable truncates to 16 bits: (short) here), tran_high_t is int32 (products and
// sums wrap at 32 bits: C# unchecked int), WRAPLOW(x) is (int32_t)x (no CONFIG_EMULATE_HARDWARE), dct_const_round_shift
// is ROUND_POWER_OF_TWO(x, 14) in int32, clip_pixel_add is clamp(dest + v, 0, 255). The 1D transform bodies are
// libvpx's statement for statement (step1[k] = a{k}, step2[k] = b{k}, input[k] = in{k}).
//
// Kernel rules: static methods only, no allocation, no arrays (constants are const ints), no recursion; the row pass
// output and the column vector live in the caller's scratch (ScratchInts ints per thread).

using System.Runtime.CompilerServices;
using ILGPU;

namespace SpawnDev.Codecs.Video.Vp9.Decoder;

/// <summary>
/// VP9 inverse transform + add of one transform block (libvpx vp9_idct.c dispatch over vpx_dsp/inv_txfm.c, 8-bit).
/// </summary>
internal static class Vp9Itx
{
    /// <summary>Scratch ints one thread needs: the row pass output (32 x 32) + one column vector (32).</summary>
    public const int ScratchInts = 32 * 32 + 32;

    // libvpx tx_type (vp9_enums.h TX_TYPE)
    public const int DctDct = 0, AdstDct = 1, DctAdst = 2, AdstAdst = 3;

    // cospi_k_64 = round(16384 * cos(k * pi / 64)) and sinpi_k_9 (vpx_dsp/txfm_common.h)
    private const int C1 = 16364, C2 = 16305, C3 = 16207, C4 = 16069, C5 = 15893, C6 = 15679, C7 = 15426, C8 = 15137;
    private const int C9 = 14811, C10 = 14449, C11 = 14053, C12 = 13623, C13 = 13160, C14 = 12665, C15 = 12140, C16 = 11585;
    private const int C17 = 11003, C18 = 10394, C19 = 9760, C20 = 9102, C21 = 8423, C22 = 7723, C23 = 7005, C24 = 6270;
    private const int C25 = 5520, C26 = 4756, C27 = 3981, C28 = 3196, C29 = 2404, C30 = 1606, C31 = 804;
    private const int Sin1 = 5283, Sin2 = 9929, Sin3 = 13377, Sin4 = 15212;

    /// <summary>libvpx dct_const_round_shift: ROUND_POWER_OF_TWO(x, DCT_CONST_BITS = 14) on tran_high_t (int32, wraps).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int R14(int x) => (x + (1 << 13)) >> 14;

    /// <summary>libvpx clip_pixel_add: clip_pixel(dest + (int)WRAPLOW(trans)).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static byte ClipPixelAdd(byte dest, int trans)
    {
        int v = dest + trans;
        return (byte)(v < 0 ? 0 : (v > 255 ? 255 : v));
    }

    /// <summary>
    /// Inverse transform of one block's dequantized coefficients, added to the prediction in <paramref name="dst"/>
    /// (libvpx's decoder: lossless ? vp9_iwht4x4_add : vp9_iht{4x4,8x8,16x16}_add(tx_type) / vp9_idct32x32_add, each with
    /// its eob-based variant choice).
    /// </summary>
    /// <param name="coef">The block's dequantized coefficients (libvpx dqcoeff), n * n in raster order, int16 range; read only.</param>
    /// <param name="txSize">0 TX_4X4, 1 TX_8X8, 2 TX_16X16, 3 TX_32X32.</param>
    /// <param name="txType">0 DCT_DCT, 1 ADST_DCT, 2 DCT_ADST, 3 ADST_ADST (32x32 and lossless: DCT_DCT, ignored).</param>
    /// <param name="lossless">Segment lossless (base_q_idx 0, no deltas): the 4x4 Walsh-Hadamard transform.</param>
    /// <param name="eob">End of block: 1 + the scan position of the last coefficient (libvpx eobs[]).</param>
    /// <param name="dst">Byte plane; the block's top-left pixel at <paramref name="dstAt"/>, rows <paramref name="stride"/> apart.</param>
    /// <param name="scratch">At least <see cref="ScratchInts"/> ints at <paramref name="scratchAt"/>, this thread's own.</param>
    public static void InverseTransformAdd(ArrayView<int> coef, long coefAt, int txSize, int txType, bool lossless, int eob,
        ArrayView<byte> dst, long dstAt, int stride, ArrayView<int> scratch, long scratchAt)
    {
        if (lossless)
        {
            // vp9_iwht4x4_add
            if (eob > 1) Iwht4x4_16Add(coef, coefAt, dst, dstAt, stride, scratch, scratchAt);
            else Iwht4x4_1Add(coef, coefAt, dst, dstAt, stride);
            return;
        }
        if (txSize == 0)
        {
            // vp9_iht4x4_add -> vp9_idct4x4_add (DCT_DCT) / vp9_iht4x4_16_add
            if (txType != DctDct) Transform2DAdd(coef, coefAt, 4, txType, 4, false, 4, dst, dstAt, stride, scratch, scratchAt);
            else if (eob > 1) Transform2DAdd(coef, coefAt, 4, DctDct, 4, false, 4, dst, dstAt, stride, scratch, scratchAt);   // vpx_idct4x4_16_add
            else DcOnlyAdd(coef, coefAt, 4, 4, dst, dstAt, stride);                                                          // vpx_idct4x4_1_add
        }
        else if (txSize == 1)
        {
            // vp9_iht8x8_add -> vp9_idct8x8_add (DCT_DCT) / vp9_iht8x8_64_add
            if (txType != DctDct) Transform2DAdd(coef, coefAt, 8, txType, 8, false, 5, dst, dstAt, stride, scratch, scratchAt);
            else if (eob == 1) DcOnlyAdd(coef, coefAt, 8, 5, dst, dstAt, stride);                                             // vpx_idct8x8_1_add
            else if (eob <= 12) Transform2DAdd(coef, coefAt, 8, DctDct, 4, false, 5, dst, dstAt, stride, scratch, scratchAt);  // vpx_idct8x8_12_add
            else Transform2DAdd(coef, coefAt, 8, DctDct, 8, false, 5, dst, dstAt, stride, scratch, scratchAt);                 // vpx_idct8x8_64_add
        }
        else if (txSize == 2)
        {
            // vp9_iht16x16_add -> vp9_idct16x16_add (DCT_DCT) / vp9_iht16x16_256_add
            if (txType != DctDct) Transform2DAdd(coef, coefAt, 16, txType, 16, false, 6, dst, dstAt, stride, scratch, scratchAt);
            else if (eob == 1) DcOnlyAdd(coef, coefAt, 16, 6, dst, dstAt, stride);                                             // vpx_idct16x16_1_add
            else if (eob <= 10) Transform2DAdd(coef, coefAt, 16, DctDct, 4, false, 6, dst, dstAt, stride, scratch, scratchAt);  // vpx_idct16x16_10_add
            else if (eob <= 38) Transform2DAdd(coef, coefAt, 16, DctDct, 8, false, 6, dst, dstAt, stride, scratch, scratchAt);  // vpx_idct16x16_38_add
            else Transform2DAdd(coef, coefAt, 16, DctDct, 16, false, 6, dst, dstAt, stride, scratch, scratchAt);                // vpx_idct16x16_256_add
        }
        else
        {
            // vp9_idct32x32_add
            if (eob == 1) DcOnlyAdd(coef, coefAt, 32, 6, dst, dstAt, stride);                                                  // vpx_idct32x32_1_add
            else if (eob <= 34) Transform2DAdd(coef, coefAt, 32, DctDct, 8, false, 6, dst, dstAt, stride, scratch, scratchAt);  // vpx_idct32x32_34_add
            else if (eob <= 135) Transform2DAdd(coef, coefAt, 32, DctDct, 16, false, 6, dst, dstAt, stride, scratch, scratchAt); // vpx_idct32x32_135_add
            else Transform2DAdd(coef, coefAt, 32, DctDct, 32, true, 6, dst, dstAt, stride, scratch, scratchAt);                 // vpx_idct32x32_1024_add
        }
    }

    /// <summary>
    /// The row / column structure every libvpx n x n inverse transform shares: rows (only the first <paramref name="rows"/>;
    /// the rest of the row output is zero, as the eob variants' out[] = { 0 }; <paramref name="skipZeroRows"/>: an
    /// all-zero input row gives a zero output row without the transform, as vpx_idct32x32_1024_add), then each column,
    /// added to dst after ROUND_POWER_OF_TWO(out, <paramref name="shift"/>). tx_type picks the 1D transforms as libvpx's
    /// IHT tables { cols, rows }: rows ADST for DCT_ADST / ADST_ADST, columns ADST for ADST_DCT / ADST_ADST.
    /// </summary>
    private static void Transform2DAdd(ArrayView<int> coef, long coefAt, int n, int txType, int rows, bool skipZeroRows, int shift,
        ArrayView<byte> dst, long dstAt, int stride, ArrayView<int> scratch, long scratchAt)
    {
        bool rowAdst = txType == DctAdst || txType == AdstAdst;
        bool colAdst = txType == AdstDct || txType == AdstAdst;
        for (int r = 0; r < n; r++)
        {
            long o = scratchAt + r * n;
            bool zero = r >= rows;
            if (!zero && skipZeroRows)
            {
                int any = 0;
                for (int j = 0; j < n; j++) any |= (short)coef[coefAt + r * n + j];
                zero = any == 0;
            }
            if (zero)
            {
                for (int j = 0; j < n; j++) scratch[o + j] = 0;
                continue;
            }
            Transform1D(n, rowAdst, coef, coefAt + r * n, 1, scratch, o, 1);
        }
        long col = scratchAt + n * n;
        int half = 1 << (shift - 1);
        for (int i = 0; i < n; i++)
        {
            Transform1D(n, colAdst, scratch, scratchAt + i, n, scratch, col, 1);
            for (int j = 0; j < n; j++)
            {
                long p = dstAt + (long)j * stride + i;
                dst[p] = ClipPixelAdd(dst[p], (scratch[col + j] + half) >> shift);
            }
        }
    }

    private static void Transform1D(int n, bool adst, ArrayView<int> src, long sAt, int sStep, ArrayView<int> dst, long dAt, int dStep)
    {
        if (n == 4)
        {
            if (adst) Iadst4(src, sAt, sStep, dst, dAt, dStep);
            else Idct4(src, sAt, sStep, dst, dAt, dStep);
        }
        else if (n == 8)
        {
            if (adst) Iadst8(src, sAt, sStep, dst, dAt, dStep);
            else Idct8(src, sAt, sStep, dst, dAt, dStep);
        }
        else if (n == 16)
        {
            if (adst) Iadst16(src, sAt, sStep, dst, dAt, dStep);
            else Idct16(src, sAt, sStep, dst, dAt, dStep);
        }
        else Idct32(src, sAt, sStep, dst, dAt, dStep);
    }

    /// <summary>vpx_idct{4x4,8x8,16x16,32x32}_1_add: DC only.</summary>
    private static void DcOnlyAdd(ArrayView<int> coef, long coefAt, int n, int shift, ArrayView<byte> dst, long dstAt, int stride)
    {
        int o = (short)R14((short)coef[coefAt] * C16);   // tran_low_t out
        o = (short)R14(o * C16);
        int a1 = (o + (1 << (shift - 1))) >> shift;
        for (int j = 0; j < n; j++)
            for (int i = 0; i < n; i++)
            {
                long p = dstAt + (long)j * stride + i;
                dst[p] = ClipPixelAdd(dst[p], a1);
            }
    }

    /// <summary>vpx_iwht4x4_16_add_c: the lossless 4x4 inverse Walsh-Hadamard transform.</summary>
    private static void Iwht4x4_16Add(ArrayView<int> coef, long coefAt, ArrayView<byte> dst, long dstAt, int stride,
        ArrayView<int> scratch, long scratchAt)
    {
        int a1, b1, c1, d1, e1;
        for (int i = 0; i < 4; i++)
        {
            long ip = coefAt + i * 4;
            a1 = (short)coef[ip] >> UnitQuantShift;
            c1 = (short)coef[ip + 1] >> UnitQuantShift;
            d1 = (short)coef[ip + 2] >> UnitQuantShift;
            b1 = (short)coef[ip + 3] >> UnitQuantShift;
            a1 += c1;
            d1 -= b1;
            e1 = (a1 - d1) >> 1;
            b1 = e1 - b1;
            c1 = e1 - c1;
            a1 -= b1;
            d1 += c1;
            long op = scratchAt + i * 4;
            scratch[op] = (short)a1;
            scratch[op + 1] = (short)b1;
            scratch[op + 2] = (short)c1;
            scratch[op + 3] = (short)d1;
        }
        for (int i = 0; i < 4; i++)
        {
            long ip = scratchAt + i;
            a1 = scratch[ip];
            c1 = scratch[ip + 4];
            d1 = scratch[ip + 8];
            b1 = scratch[ip + 12];
            a1 += c1;
            d1 -= b1;
            e1 = (a1 - d1) >> 1;
            b1 = e1 - b1;
            c1 = e1 - c1;
            a1 -= b1;
            d1 += c1;
            long p = dstAt + i;
            dst[p] = ClipPixelAdd(dst[p], a1);
            dst[p + stride] = ClipPixelAdd(dst[p + stride], b1);
            dst[p + 2L * stride] = ClipPixelAdd(dst[p + 2L * stride], c1);
            dst[p + 3L * stride] = ClipPixelAdd(dst[p + 3L * stride], d1);
        }
    }

    /// <summary>vpx_iwht4x4_1_add_c: the lossless transform of a DC-only block.</summary>
    private static void Iwht4x4_1Add(ArrayView<int> coef, long coefAt, ArrayView<byte> dst, long dstAt, int stride)
    {
        int a1 = (short)coef[coefAt] >> UnitQuantShift;
        int e1 = a1 >> 1;
        a1 -= e1;
        int tmp0 = (short)a1, tmp1 = (short)e1;   // tran_low_t tmp[4] = { a1, e1, e1, e1 }
        for (int i = 0; i < 4; i++)
        {
            int ip = i == 0 ? tmp0 : tmp1;
            e1 = ip >> 1;
            a1 = ip - e1;
            long p = dstAt + i;
            dst[p] = ClipPixelAdd(dst[p], a1);
            dst[p + stride] = ClipPixelAdd(dst[p + stride], e1);
            dst[p + 2L * stride] = ClipPixelAdd(dst[p + 2L * stride], e1);
            dst[p + 3L * stride] = ClipPixelAdd(dst[p + 3L * stride], e1);
        }
    }

    private const int UnitQuantShift = 2;   // UNIT_QUANT_SHIFT

    // ---- 1D transforms: vpx_dsp/inv_txfm.c, statement for statement. Inputs are tran_low_t (int16) reads; every output
    // is a tran_low_t store (int16).

    /// <summary>libvpx idct4_c.</summary>
    private static void Idct4(ArrayView<int> src, long sAt, int sStep, ArrayView<int> dst, long dAt, int dStep)
    {
        int in0 = (short)src[sAt + 0 * sStep]; int in1 = (short)src[sAt + 1 * sStep]; int in2 = (short)src[sAt + 2 * sStep]; int in3 = (short)src[sAt + 3 * sStep];
        int a0, a1, a2, a3, t1, t2;
        t1 = (in0 + in2) * C16;
        t2 = (in0 - in2) * C16;
        a0 = (short)(R14(t1));
        a1 = (short)(R14(t2));
        t1 = in1 * C24 - in3 * C8;
        t2 = in1 * C8 + in3 * C24;
        a2 = (short)(R14(t1));
        a3 = (short)(R14(t2));
        dst[dAt + 0 * dStep] = (short)(a0 + a3);
        dst[dAt + 1 * dStep] = (short)(a1 + a2);
        dst[dAt + 2 * dStep] = (short)(a1 - a2);
        dst[dAt + 3 * dStep] = (short)(a0 - a3);
    }

    /// <summary>libvpx iadst4_c (all-zero input: zero output, as libvpx's early out).</summary>
    private static void Iadst4(ArrayView<int> src, long sAt, int sStep, ArrayView<int> dst, long dAt, int dStep)
    {
        int in0 = (short)src[sAt + 0 * sStep]; int in1 = (short)src[sAt + 1 * sStep]; int in2 = (short)src[sAt + 2 * sStep]; int in3 = (short)src[sAt + 3 * sStep];
        int s0, s1, s2, s3, s4, s5, s6, s7;
        int x0 = in0;
        int x1 = in1;
        int x2 = in2;
        int x3 = in3;
        if ((x0 | x1 | x2 | x3) == 0)
        {
            for (int k = 0; k < 4; k++) dst[dAt + k * dStep] = 0;
            return;
        }
        s0 = Sin1 * x0;
        s1 = Sin2 * x0;
        s2 = Sin3 * x1;
        s3 = Sin4 * x2;
        s4 = Sin1 * x2;
        s5 = Sin2 * x3;
        s6 = Sin4 * x3;
        s7 = (x0 - x2 + x3);
        s0 = s0 + s3 + s5;
        s1 = s1 - s4 - s6;
        s3 = s2;
        s2 = Sin3 * s7;
        dst[dAt + 0 * dStep] = (short)(R14(s0 + s3));
        dst[dAt + 1 * dStep] = (short)(R14(s1 + s3));
        dst[dAt + 2 * dStep] = (short)(R14(s2));
        dst[dAt + 3 * dStep] = (short)(R14(s0 + s1 - s3));
    }

    /// <summary>libvpx idct8_c.</summary>
    private static void Idct8(ArrayView<int> src, long sAt, int sStep, ArrayView<int> dst, long dAt, int dStep)
    {
        int in0 = (short)src[sAt + 0 * sStep]; int in1 = (short)src[sAt + 1 * sStep]; int in2 = (short)src[sAt + 2 * sStep]; int in3 = (short)src[sAt + 3 * sStep];
        int in4 = (short)src[sAt + 4 * sStep]; int in5 = (short)src[sAt + 5 * sStep]; int in6 = (short)src[sAt + 6 * sStep]; int in7 = (short)src[sAt + 7 * sStep];
        int a0, a1, a2, a3, a4, a5, a6, a7, b0, b1, b2, b3, b4, b5, b6, b7, t1, t2;
        a0 = in0;
        a2 = in4;
        a1 = in2;
        a3 = in6;
        t1 = in1 * C28 - in7 * C4;
        t2 = in1 * C4 + in7 * C28;
        a4 = (short)(R14(t1));
        a7 = (short)(R14(t2));
        t1 = in5 * C12 - in3 * C20;
        t2 = in5 * C20 + in3 * C12;
        a5 = (short)(R14(t1));
        a6 = (short)(R14(t2));
        t1 = (a0 + a2) * C16;
        t2 = (a0 - a2) * C16;
        b0 = (short)(R14(t1));
        b1 = (short)(R14(t2));
        t1 = a1 * C24 - a3 * C8;
        t2 = a1 * C8 + a3 * C24;
        b2 = (short)(R14(t1));
        b3 = (short)(R14(t2));
        b4 = (short)(a4 + a5);
        b5 = (short)(a4 - a5);
        b6 = (short)(-a6 + a7);
        b7 = (short)(a6 + a7);
        a0 = (short)(b0 + b3);
        a1 = (short)(b1 + b2);
        a2 = (short)(b1 - b2);
        a3 = (short)(b0 - b3);
        a4 = b4;
        t1 = (b6 - b5) * C16;
        t2 = (b5 + b6) * C16;
        a5 = (short)(R14(t1));
        a6 = (short)(R14(t2));
        a7 = b7;
        dst[dAt + 0 * dStep] = (short)(a0 + a7);
        dst[dAt + 1 * dStep] = (short)(a1 + a6);
        dst[dAt + 2 * dStep] = (short)(a2 + a5);
        dst[dAt + 3 * dStep] = (short)(a3 + a4);
        dst[dAt + 4 * dStep] = (short)(a3 - a4);
        dst[dAt + 5 * dStep] = (short)(a2 - a5);
        dst[dAt + 6 * dStep] = (short)(a1 - a6);
        dst[dAt + 7 * dStep] = (short)(a0 - a7);
    }

    /// <summary>libvpx iadst8_c. x / s are tran_high_t / int (32-bit, never truncated); only the outputs are int16.</summary>
    private static void Iadst8(ArrayView<int> src, long sAt, int sStep, ArrayView<int> dst, long dAt, int dStep)
    {
        int in0 = (short)src[sAt + 0 * sStep]; int in1 = (short)src[sAt + 1 * sStep]; int in2 = (short)src[sAt + 2 * sStep]; int in3 = (short)src[sAt + 3 * sStep];
        int in4 = (short)src[sAt + 4 * sStep]; int in5 = (short)src[sAt + 5 * sStep]; int in6 = (short)src[sAt + 6 * sStep]; int in7 = (short)src[sAt + 7 * sStep];
        int s0, s1, s2, s3, s4, s5, s6, s7;
        int x0 = in7;
        int x1 = in0;
        int x2 = in5;
        int x3 = in2;
        int x4 = in3;
        int x5 = in4;
        int x6 = in1;
        int x7 = in6;
        if ((x0 | x1 | x2 | x3 | x4 | x5 | x6 | x7) == 0)
        {
            for (int k = 0; k < 8; k++) dst[dAt + k * dStep] = 0;
            return;
        }
        s0 = (C2 * x0 + C30 * x1);
        s1 = (C30 * x0 - C2 * x1);
        s2 = (C10 * x2 + C22 * x3);
        s3 = (C22 * x2 - C10 * x3);
        s4 = (C18 * x4 + C14 * x5);
        s5 = (C14 * x4 - C18 * x5);
        s6 = (C26 * x6 + C6 * x7);
        s7 = (C6 * x6 - C26 * x7);
        x0 = (R14(s0 + s4));
        x1 = (R14(s1 + s5));
        x2 = (R14(s2 + s6));
        x3 = (R14(s3 + s7));
        x4 = (R14(s0 - s4));
        x5 = (R14(s1 - s5));
        x6 = (R14(s2 - s6));
        x7 = (R14(s3 - s7));
        s0 = (int)x0;
        s1 = (int)x1;
        s2 = (int)x2;
        s3 = (int)x3;
        s4 = (C8 * x4 + C24 * x5);
        s5 = (C24 * x4 - C8 * x5);
        s6 = (-C24 * x6 + C8 * x7);
        s7 = (C8 * x6 + C24 * x7);
        x0 = (s0 + s2);
        x1 = (s1 + s3);
        x2 = (s0 - s2);
        x3 = (s1 - s3);
        x4 = (R14(s4 + s6));
        x5 = (R14(s5 + s7));
        x6 = (R14(s4 - s6));
        x7 = (R14(s5 - s7));
        s2 = (C16 * (x2 + x3));
        s3 = (C16 * (x2 - x3));
        s6 = (C16 * (x6 + x7));
        s7 = (C16 * (x6 - x7));
        x2 = (R14(s2));
        x3 = (R14(s3));
        x6 = (R14(s6));
        x7 = (R14(s7));
        dst[dAt + 0 * dStep] = (short)(x0);
        dst[dAt + 1 * dStep] = (short)(-x4);
        dst[dAt + 2 * dStep] = (short)(x6);
        dst[dAt + 3 * dStep] = (short)(-x2);
        dst[dAt + 4 * dStep] = (short)(x3);
        dst[dAt + 5 * dStep] = (short)(-x7);
        dst[dAt + 6 * dStep] = (short)(x5);
        dst[dAt + 7 * dStep] = (short)(-x1);
    }

    /// <summary>libvpx idct16_c.</summary>
    private static void Idct16(ArrayView<int> src, long sAt, int sStep, ArrayView<int> dst, long dAt, int dStep)
    {
        int in0 = (short)src[sAt + 0 * sStep]; int in1 = (short)src[sAt + 1 * sStep]; int in2 = (short)src[sAt + 2 * sStep]; int in3 = (short)src[sAt + 3 * sStep];
        int in4 = (short)src[sAt + 4 * sStep]; int in5 = (short)src[sAt + 5 * sStep]; int in6 = (short)src[sAt + 6 * sStep]; int in7 = (short)src[sAt + 7 * sStep];
        int in8 = (short)src[sAt + 8 * sStep]; int in9 = (short)src[sAt + 9 * sStep]; int in10 = (short)src[sAt + 10 * sStep]; int in11 = (short)src[sAt + 11 * sStep];
        int in12 = (short)src[sAt + 12 * sStep]; int in13 = (short)src[sAt + 13 * sStep]; int in14 = (short)src[sAt + 14 * sStep]; int in15 = (short)src[sAt + 15 * sStep];
        int a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, t1, t2;
        a0 = in0;
        a1 = in8;
        a2 = in4;
        a3 = in12;
        a4 = in2;
        a5 = in10;
        a6 = in6;
        a7 = in14;
        a8 = in1;
        a9 = in9;
        a10 = in5;
        a11 = in13;
        a12 = in3;
        a13 = in11;
        a14 = in7;
        a15 = in15;
        b0 = a0;
        b1 = a1;
        b2 = a2;
        b3 = a3;
        b4 = a4;
        b5 = a5;
        b6 = a6;
        b7 = a7;
        t1 = a8 * C30 - a15 * C2;
        t2 = a8 * C2 + a15 * C30;
        b8 = (short)(R14(t1));
        b15 = (short)(R14(t2));
        t1 = a9 * C14 - a14 * C18;
        t2 = a9 * C18 + a14 * C14;
        b9 = (short)(R14(t1));
        b14 = (short)(R14(t2));
        t1 = a10 * C22 - a13 * C10;
        t2 = a10 * C10 + a13 * C22;
        b10 = (short)(R14(t1));
        b13 = (short)(R14(t2));
        t1 = a11 * C6 - a12 * C26;
        t2 = a11 * C26 + a12 * C6;
        b11 = (short)(R14(t1));
        b12 = (short)(R14(t2));
        a0 = b0;
        a1 = b1;
        a2 = b2;
        a3 = b3;
        t1 = b4 * C28 - b7 * C4;
        t2 = b4 * C4 + b7 * C28;
        a4 = (short)(R14(t1));
        a7 = (short)(R14(t2));
        t1 = b5 * C12 - b6 * C20;
        t2 = b5 * C20 + b6 * C12;
        a5 = (short)(R14(t1));
        a6 = (short)(R14(t2));
        a8 = (short)(b8 + b9);
        a9 = (short)(b8 - b9);
        a10 = (short)(-b10 + b11);
        a11 = (short)(b10 + b11);
        a12 = (short)(b12 + b13);
        a13 = (short)(b12 - b13);
        a14 = (short)(-b14 + b15);
        a15 = (short)(b14 + b15);
        t1 = (a0 + a1) * C16;
        t2 = (a0 - a1) * C16;
        b0 = (short)(R14(t1));
        b1 = (short)(R14(t2));
        t1 = a2 * C24 - a3 * C8;
        t2 = a2 * C8 + a3 * C24;
        b2 = (short)(R14(t1));
        b3 = (short)(R14(t2));
        b4 = (short)(a4 + a5);
        b5 = (short)(a4 - a5);
        b6 = (short)(-a6 + a7);
        b7 = (short)(a6 + a7);
        b8 = a8;
        b15 = a15;
        t1 = -a9 * C8 + a14 * C24;
        t2 = a9 * C24 + a14 * C8;
        b9 = (short)(R14(t1));
        b14 = (short)(R14(t2));
        t1 = -a10 * C24 - a13 * C8;
        t2 = -a10 * C8 + a13 * C24;
        b10 = (short)(R14(t1));
        b13 = (short)(R14(t2));
        b11 = a11;
        b12 = a12;
        a0 = (short)(b0 + b3);
        a1 = (short)(b1 + b2);
        a2 = (short)(b1 - b2);
        a3 = (short)(b0 - b3);
        a4 = b4;
        t1 = (b6 - b5) * C16;
        t2 = (b5 + b6) * C16;
        a5 = (short)(R14(t1));
        a6 = (short)(R14(t2));
        a7 = b7;
        a8 = (short)(b8 + b11);
        a9 = (short)(b9 + b10);
        a10 = (short)(b9 - b10);
        a11 = (short)(b8 - b11);
        a12 = (short)(-b12 + b15);
        a13 = (short)(-b13 + b14);
        a14 = (short)(b13 + b14);
        a15 = (short)(b12 + b15);
        b0 = (short)(a0 + a7);
        b1 = (short)(a1 + a6);
        b2 = (short)(a2 + a5);
        b3 = (short)(a3 + a4);
        b4 = (short)(a3 - a4);
        b5 = (short)(a2 - a5);
        b6 = (short)(a1 - a6);
        b7 = (short)(a0 - a7);
        b8 = a8;
        b9 = a9;
        t1 = (-a10 + a13) * C16;
        t2 = (a10 + a13) * C16;
        b10 = (short)(R14(t1));
        b13 = (short)(R14(t2));
        t1 = (-a11 + a12) * C16;
        t2 = (a11 + a12) * C16;
        b11 = (short)(R14(t1));
        b12 = (short)(R14(t2));
        b14 = a14;
        b15 = a15;
        dst[dAt + 0 * dStep] = (short)(b0 + b15);
        dst[dAt + 1 * dStep] = (short)(b1 + b14);
        dst[dAt + 2 * dStep] = (short)(b2 + b13);
        dst[dAt + 3 * dStep] = (short)(b3 + b12);
        dst[dAt + 4 * dStep] = (short)(b4 + b11);
        dst[dAt + 5 * dStep] = (short)(b5 + b10);
        dst[dAt + 6 * dStep] = (short)(b6 + b9);
        dst[dAt + 7 * dStep] = (short)(b7 + b8);
        dst[dAt + 8 * dStep] = (short)(b7 - b8);
        dst[dAt + 9 * dStep] = (short)(b6 - b9);
        dst[dAt + 10 * dStep] = (short)(b5 - b10);
        dst[dAt + 11 * dStep] = (short)(b4 - b11);
        dst[dAt + 12 * dStep] = (short)(b3 - b12);
        dst[dAt + 13 * dStep] = (short)(b2 - b13);
        dst[dAt + 14 * dStep] = (short)(b1 - b14);
        dst[dAt + 15 * dStep] = (short)(b0 - b15);
    }

    /// <summary>libvpx iadst16_c. x / s are tran_high_t (32-bit, never truncated); only the outputs are int16.</summary>
    private static void Iadst16(ArrayView<int> src, long sAt, int sStep, ArrayView<int> dst, long dAt, int dStep)
    {
        int in0 = (short)src[sAt + 0 * sStep]; int in1 = (short)src[sAt + 1 * sStep]; int in2 = (short)src[sAt + 2 * sStep]; int in3 = (short)src[sAt + 3 * sStep];
        int in4 = (short)src[sAt + 4 * sStep]; int in5 = (short)src[sAt + 5 * sStep]; int in6 = (short)src[sAt + 6 * sStep]; int in7 = (short)src[sAt + 7 * sStep];
        int in8 = (short)src[sAt + 8 * sStep]; int in9 = (short)src[sAt + 9 * sStep]; int in10 = (short)src[sAt + 10 * sStep]; int in11 = (short)src[sAt + 11 * sStep];
        int in12 = (short)src[sAt + 12 * sStep]; int in13 = (short)src[sAt + 13 * sStep]; int in14 = (short)src[sAt + 14 * sStep]; int in15 = (short)src[sAt + 15 * sStep];
        int s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15;
        int x0 = in15;
        int x1 = in0;
        int x2 = in13;
        int x3 = in2;
        int x4 = in11;
        int x5 = in4;
        int x6 = in9;
        int x7 = in6;
        int x8 = in7;
        int x9 = in8;
        int x10 = in5;
        int x11 = in10;
        int x12 = in3;
        int x13 = in12;
        int x14 = in1;
        int x15 = in14;
        if ((x0 | x1 | x2 | x3 | x4 | x5 | x6 | x7 | x8 | x9 | x10 | x11 | x12 | x13 | x14 | x15) == 0)
        {
            for (int k = 0; k < 16; k++) dst[dAt + k * dStep] = 0;
            return;
        }
        s0 = x0 * C1 + x1 * C31;
        s1 = x0 * C31 - x1 * C1;
        s2 = x2 * C5 + x3 * C27;
        s3 = x2 * C27 - x3 * C5;
        s4 = x4 * C9 + x5 * C23;
        s5 = x4 * C23 - x5 * C9;
        s6 = x6 * C13 + x7 * C19;
        s7 = x6 * C19 - x7 * C13;
        s8 = x8 * C17 + x9 * C15;
        s9 = x8 * C15 - x9 * C17;
        s10 = x10 * C21 + x11 * C11;
        s11 = x10 * C11 - x11 * C21;
        s12 = x12 * C25 + x13 * C7;
        s13 = x12 * C7 - x13 * C25;
        s14 = x14 * C29 + x15 * C3;
        s15 = x14 * C3 - x15 * C29;
        x0 = (R14(s0 + s8));
        x1 = (R14(s1 + s9));
        x2 = (R14(s2 + s10));
        x3 = (R14(s3 + s11));
        x4 = (R14(s4 + s12));
        x5 = (R14(s5 + s13));
        x6 = (R14(s6 + s14));
        x7 = (R14(s7 + s15));
        x8 = (R14(s0 - s8));
        x9 = (R14(s1 - s9));
        x10 = (R14(s2 - s10));
        x11 = (R14(s3 - s11));
        x12 = (R14(s4 - s12));
        x13 = (R14(s5 - s13));
        x14 = (R14(s6 - s14));
        x15 = (R14(s7 - s15));
        s0 = x0;
        s1 = x1;
        s2 = x2;
        s3 = x3;
        s4 = x4;
        s5 = x5;
        s6 = x6;
        s7 = x7;
        s8 = x8 * C4 + x9 * C28;
        s9 = x8 * C28 - x9 * C4;
        s10 = x10 * C20 + x11 * C12;
        s11 = x10 * C12 - x11 * C20;
        s12 = -x12 * C28 + x13 * C4;
        s13 = x12 * C4 + x13 * C28;
        s14 = -x14 * C12 + x15 * C20;
        s15 = x14 * C20 + x15 * C12;
        x0 = (s0 + s4);
        x1 = (s1 + s5);
        x2 = (s2 + s6);
        x3 = (s3 + s7);
        x4 = (s0 - s4);
        x5 = (s1 - s5);
        x6 = (s2 - s6);
        x7 = (s3 - s7);
        x8 = (R14(s8 + s12));
        x9 = (R14(s9 + s13));
        x10 = (R14(s10 + s14));
        x11 = (R14(s11 + s15));
        x12 = (R14(s8 - s12));
        x13 = (R14(s9 - s13));
        x14 = (R14(s10 - s14));
        x15 = (R14(s11 - s15));
        s0 = x0;
        s1 = x1;
        s2 = x2;
        s3 = x3;
        s4 = x4 * C8 + x5 * C24;
        s5 = x4 * C24 - x5 * C8;
        s6 = -x6 * C24 + x7 * C8;
        s7 = x6 * C8 + x7 * C24;
        s8 = x8;
        s9 = x9;
        s10 = x10;
        s11 = x11;
        s12 = x12 * C8 + x13 * C24;
        s13 = x12 * C24 - x13 * C8;
        s14 = -x14 * C24 + x15 * C8;
        s15 = x14 * C8 + x15 * C24;
        x0 = (s0 + s2);
        x1 = (s1 + s3);
        x2 = (s0 - s2);
        x3 = (s1 - s3);
        x4 = (R14(s4 + s6));
        x5 = (R14(s5 + s7));
        x6 = (R14(s4 - s6));
        x7 = (R14(s5 - s7));
        x8 = (s8 + s10);
        x9 = (s9 + s11);
        x10 = (s8 - s10);
        x11 = (s9 - s11);
        x12 = (R14(s12 + s14));
        x13 = (R14(s13 + s15));
        x14 = (R14(s12 - s14));
        x15 = (R14(s13 - s15));
        s2 = (-C16) * (x2 + x3);
        s3 = C16 * (x2 - x3);
        s6 = C16 * (x6 + x7);
        s7 = C16 * (-x6 + x7);
        s10 = C16 * (x10 + x11);
        s11 = C16 * (-x10 + x11);
        s14 = (-C16) * (x14 + x15);
        s15 = C16 * (x14 - x15);
        x2 = (R14(s2));
        x3 = (R14(s3));
        x6 = (R14(s6));
        x7 = (R14(s7));
        x10 = (R14(s10));
        x11 = (R14(s11));
        x14 = (R14(s14));
        x15 = (R14(s15));
        dst[dAt + 0 * dStep] = (short)(x0);
        dst[dAt + 1 * dStep] = (short)(-x8);
        dst[dAt + 2 * dStep] = (short)(x12);
        dst[dAt + 3 * dStep] = (short)(-x4);
        dst[dAt + 4 * dStep] = (short)(x6);
        dst[dAt + 5 * dStep] = (short)(x14);
        dst[dAt + 6 * dStep] = (short)(x10);
        dst[dAt + 7 * dStep] = (short)(x2);
        dst[dAt + 8 * dStep] = (short)(x3);
        dst[dAt + 9 * dStep] = (short)(x11);
        dst[dAt + 10 * dStep] = (short)(x15);
        dst[dAt + 11 * dStep] = (short)(x7);
        dst[dAt + 12 * dStep] = (short)(x5);
        dst[dAt + 13 * dStep] = (short)(-x13);
        dst[dAt + 14 * dStep] = (short)(x9);
        dst[dAt + 15 * dStep] = (short)(-x1);
    }

    /// <summary>libvpx idct32_c.</summary>
    private static void Idct32(ArrayView<int> src, long sAt, int sStep, ArrayView<int> dst, long dAt, int dStep)
    {
        int in0 = (short)src[sAt + 0 * sStep]; int in1 = (short)src[sAt + 1 * sStep]; int in2 = (short)src[sAt + 2 * sStep]; int in3 = (short)src[sAt + 3 * sStep];
        int in4 = (short)src[sAt + 4 * sStep]; int in5 = (short)src[sAt + 5 * sStep]; int in6 = (short)src[sAt + 6 * sStep]; int in7 = (short)src[sAt + 7 * sStep];
        int in8 = (short)src[sAt + 8 * sStep]; int in9 = (short)src[sAt + 9 * sStep]; int in10 = (short)src[sAt + 10 * sStep]; int in11 = (short)src[sAt + 11 * sStep];
        int in12 = (short)src[sAt + 12 * sStep]; int in13 = (short)src[sAt + 13 * sStep]; int in14 = (short)src[sAt + 14 * sStep]; int in15 = (short)src[sAt + 15 * sStep];
        int in16 = (short)src[sAt + 16 * sStep]; int in17 = (short)src[sAt + 17 * sStep]; int in18 = (short)src[sAt + 18 * sStep]; int in19 = (short)src[sAt + 19 * sStep];
        int in20 = (short)src[sAt + 20 * sStep]; int in21 = (short)src[sAt + 21 * sStep]; int in22 = (short)src[sAt + 22 * sStep]; int in23 = (short)src[sAt + 23 * sStep];
        int in24 = (short)src[sAt + 24 * sStep]; int in25 = (short)src[sAt + 25 * sStep]; int in26 = (short)src[sAt + 26 * sStep]; int in27 = (short)src[sAt + 27 * sStep];
        int in28 = (short)src[sAt + 28 * sStep]; int in29 = (short)src[sAt + 29 * sStep]; int in30 = (short)src[sAt + 30 * sStep]; int in31 = (short)src[sAt + 31 * sStep];
        int a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, b16, b17, b18, b19, b20, b21, b22, b23, b24, b25, b26, b27, b28, b29, b30, b31, t1, t2;
        a0 = in0;
        a1 = in16;
        a2 = in8;
        a3 = in24;
        a4 = in4;
        a5 = in20;
        a6 = in12;
        a7 = in28;
        a8 = in2;
        a9 = in18;
        a10 = in10;
        a11 = in26;
        a12 = in6;
        a13 = in22;
        a14 = in14;
        a15 = in30;
        t1 = in1 * C31 - in31 * C1;
        t2 = in1 * C1 + in31 * C31;
        a16 = (short)(R14(t1));
        a31 = (short)(R14(t2));
        t1 = in17 * C15 - in15 * C17;
        t2 = in17 * C17 + in15 * C15;
        a17 = (short)(R14(t1));
        a30 = (short)(R14(t2));
        t1 = in9 * C23 - in23 * C9;
        t2 = in9 * C9 + in23 * C23;
        a18 = (short)(R14(t1));
        a29 = (short)(R14(t2));
        t1 = in25 * C7 - in7 * C25;
        t2 = in25 * C25 + in7 * C7;
        a19 = (short)(R14(t1));
        a28 = (short)(R14(t2));
        t1 = in5 * C27 - in27 * C5;
        t2 = in5 * C5 + in27 * C27;
        a20 = (short)(R14(t1));
        a27 = (short)(R14(t2));
        t1 = in21 * C11 - in11 * C21;
        t2 = in21 * C21 + in11 * C11;
        a21 = (short)(R14(t1));
        a26 = (short)(R14(t2));
        t1 = in13 * C19 - in19 * C13;
        t2 = in13 * C13 + in19 * C19;
        a22 = (short)(R14(t1));
        a25 = (short)(R14(t2));
        t1 = in29 * C3 - in3 * C29;
        t2 = in29 * C29 + in3 * C3;
        a23 = (short)(R14(t1));
        a24 = (short)(R14(t2));
        b0 = a0;
        b1 = a1;
        b2 = a2;
        b3 = a3;
        b4 = a4;
        b5 = a5;
        b6 = a6;
        b7 = a7;
        t1 = a8 * C30 - a15 * C2;
        t2 = a8 * C2 + a15 * C30;
        b8 = (short)(R14(t1));
        b15 = (short)(R14(t2));
        t1 = a9 * C14 - a14 * C18;
        t2 = a9 * C18 + a14 * C14;
        b9 = (short)(R14(t1));
        b14 = (short)(R14(t2));
        t1 = a10 * C22 - a13 * C10;
        t2 = a10 * C10 + a13 * C22;
        b10 = (short)(R14(t1));
        b13 = (short)(R14(t2));
        t1 = a11 * C6 - a12 * C26;
        t2 = a11 * C26 + a12 * C6;
        b11 = (short)(R14(t1));
        b12 = (short)(R14(t2));
        b16 = (short)(a16 + a17);
        b17 = (short)(a16 - a17);
        b18 = (short)(-a18 + a19);
        b19 = (short)(a18 + a19);
        b20 = (short)(a20 + a21);
        b21 = (short)(a20 - a21);
        b22 = (short)(-a22 + a23);
        b23 = (short)(a22 + a23);
        b24 = (short)(a24 + a25);
        b25 = (short)(a24 - a25);
        b26 = (short)(-a26 + a27);
        b27 = (short)(a26 + a27);
        b28 = (short)(a28 + a29);
        b29 = (short)(a28 - a29);
        b30 = (short)(-a30 + a31);
        b31 = (short)(a30 + a31);
        a0 = b0;
        a1 = b1;
        a2 = b2;
        a3 = b3;
        t1 = b4 * C28 - b7 * C4;
        t2 = b4 * C4 + b7 * C28;
        a4 = (short)(R14(t1));
        a7 = (short)(R14(t2));
        t1 = b5 * C12 - b6 * C20;
        t2 = b5 * C20 + b6 * C12;
        a5 = (short)(R14(t1));
        a6 = (short)(R14(t2));
        a8 = (short)(b8 + b9);
        a9 = (short)(b8 - b9);
        a10 = (short)(-b10 + b11);
        a11 = (short)(b10 + b11);
        a12 = (short)(b12 + b13);
        a13 = (short)(b12 - b13);
        a14 = (short)(-b14 + b15);
        a15 = (short)(b14 + b15);
        a16 = b16;
        a31 = b31;
        t1 = -b17 * C4 + b30 * C28;
        t2 = b17 * C28 + b30 * C4;
        a17 = (short)(R14(t1));
        a30 = (short)(R14(t2));
        t1 = -b18 * C28 - b29 * C4;
        t2 = -b18 * C4 + b29 * C28;
        a18 = (short)(R14(t1));
        a29 = (short)(R14(t2));
        a19 = b19;
        a20 = b20;
        t1 = -b21 * C20 + b26 * C12;
        t2 = b21 * C12 + b26 * C20;
        a21 = (short)(R14(t1));
        a26 = (short)(R14(t2));
        t1 = -b22 * C12 - b25 * C20;
        t2 = -b22 * C20 + b25 * C12;
        a22 = (short)(R14(t1));
        a25 = (short)(R14(t2));
        a23 = b23;
        a24 = b24;
        a27 = b27;
        a28 = b28;
        t1 = (a0 + a1) * C16;
        t2 = (a0 - a1) * C16;
        b0 = (short)(R14(t1));
        b1 = (short)(R14(t2));
        t1 = a2 * C24 - a3 * C8;
        t2 = a2 * C8 + a3 * C24;
        b2 = (short)(R14(t1));
        b3 = (short)(R14(t2));
        b4 = (short)(a4 + a5);
        b5 = (short)(a4 - a5);
        b6 = (short)(-a6 + a7);
        b7 = (short)(a6 + a7);
        b8 = a8;
        b15 = a15;
        t1 = -a9 * C8 + a14 * C24;
        t2 = a9 * C24 + a14 * C8;
        b9 = (short)(R14(t1));
        b14 = (short)(R14(t2));
        t1 = -a10 * C24 - a13 * C8;
        t2 = -a10 * C8 + a13 * C24;
        b10 = (short)(R14(t1));
        b13 = (short)(R14(t2));
        b11 = a11;
        b12 = a12;
        b16 = (short)(a16 + a19);
        b17 = (short)(a17 + a18);
        b18 = (short)(a17 - a18);
        b19 = (short)(a16 - a19);
        b20 = (short)(-a20 + a23);
        b21 = (short)(-a21 + a22);
        b22 = (short)(a21 + a22);
        b23 = (short)(a20 + a23);
        b24 = (short)(a24 + a27);
        b25 = (short)(a25 + a26);
        b26 = (short)(a25 - a26);
        b27 = (short)(a24 - a27);
        b28 = (short)(-a28 + a31);
        b29 = (short)(-a29 + a30);
        b30 = (short)(a29 + a30);
        b31 = (short)(a28 + a31);
        a0 = (short)(b0 + b3);
        a1 = (short)(b1 + b2);
        a2 = (short)(b1 - b2);
        a3 = (short)(b0 - b3);
        a4 = b4;
        t1 = (b6 - b5) * C16;
        t2 = (b5 + b6) * C16;
        a5 = (short)(R14(t1));
        a6 = (short)(R14(t2));
        a7 = b7;
        a8 = (short)(b8 + b11);
        a9 = (short)(b9 + b10);
        a10 = (short)(b9 - b10);
        a11 = (short)(b8 - b11);
        a12 = (short)(-b12 + b15);
        a13 = (short)(-b13 + b14);
        a14 = (short)(b13 + b14);
        a15 = (short)(b12 + b15);
        a16 = b16;
        a17 = b17;
        t1 = -b18 * C8 + b29 * C24;
        t2 = b18 * C24 + b29 * C8;
        a18 = (short)(R14(t1));
        a29 = (short)(R14(t2));
        t1 = -b19 * C8 + b28 * C24;
        t2 = b19 * C24 + b28 * C8;
        a19 = (short)(R14(t1));
        a28 = (short)(R14(t2));
        t1 = -b20 * C24 - b27 * C8;
        t2 = -b20 * C8 + b27 * C24;
        a20 = (short)(R14(t1));
        a27 = (short)(R14(t2));
        t1 = -b21 * C24 - b26 * C8;
        t2 = -b21 * C8 + b26 * C24;
        a21 = (short)(R14(t1));
        a26 = (short)(R14(t2));
        a22 = b22;
        a23 = b23;
        a24 = b24;
        a25 = b25;
        a30 = b30;
        a31 = b31;
        b0 = (short)(a0 + a7);
        b1 = (short)(a1 + a6);
        b2 = (short)(a2 + a5);
        b3 = (short)(a3 + a4);
        b4 = (short)(a3 - a4);
        b5 = (short)(a2 - a5);
        b6 = (short)(a1 - a6);
        b7 = (short)(a0 - a7);
        b8 = a8;
        b9 = a9;
        t1 = (-a10 + a13) * C16;
        t2 = (a10 + a13) * C16;
        b10 = (short)(R14(t1));
        b13 = (short)(R14(t2));
        t1 = (-a11 + a12) * C16;
        t2 = (a11 + a12) * C16;
        b11 = (short)(R14(t1));
        b12 = (short)(R14(t2));
        b14 = a14;
        b15 = a15;
        b16 = (short)(a16 + a23);
        b17 = (short)(a17 + a22);
        b18 = (short)(a18 + a21);
        b19 = (short)(a19 + a20);
        b20 = (short)(a19 - a20);
        b21 = (short)(a18 - a21);
        b22 = (short)(a17 - a22);
        b23 = (short)(a16 - a23);
        b24 = (short)(-a24 + a31);
        b25 = (short)(-a25 + a30);
        b26 = (short)(-a26 + a29);
        b27 = (short)(-a27 + a28);
        b28 = (short)(a27 + a28);
        b29 = (short)(a26 + a29);
        b30 = (short)(a25 + a30);
        b31 = (short)(a24 + a31);
        a0 = (short)(b0 + b15);
        a1 = (short)(b1 + b14);
        a2 = (short)(b2 + b13);
        a3 = (short)(b3 + b12);
        a4 = (short)(b4 + b11);
        a5 = (short)(b5 + b10);
        a6 = (short)(b6 + b9);
        a7 = (short)(b7 + b8);
        a8 = (short)(b7 - b8);
        a9 = (short)(b6 - b9);
        a10 = (short)(b5 - b10);
        a11 = (short)(b4 - b11);
        a12 = (short)(b3 - b12);
        a13 = (short)(b2 - b13);
        a14 = (short)(b1 - b14);
        a15 = (short)(b0 - b15);
        a16 = b16;
        a17 = b17;
        a18 = b18;
        a19 = b19;
        t1 = (-b20 + b27) * C16;
        t2 = (b20 + b27) * C16;
        a20 = (short)(R14(t1));
        a27 = (short)(R14(t2));
        t1 = (-b21 + b26) * C16;
        t2 = (b21 + b26) * C16;
        a21 = (short)(R14(t1));
        a26 = (short)(R14(t2));
        t1 = (-b22 + b25) * C16;
        t2 = (b22 + b25) * C16;
        a22 = (short)(R14(t1));
        a25 = (short)(R14(t2));
        t1 = (-b23 + b24) * C16;
        t2 = (b23 + b24) * C16;
        a23 = (short)(R14(t1));
        a24 = (short)(R14(t2));
        a28 = b28;
        a29 = b29;
        a30 = b30;
        a31 = b31;
        dst[dAt + 0 * dStep] = (short)(a0 + a31);
        dst[dAt + 1 * dStep] = (short)(a1 + a30);
        dst[dAt + 2 * dStep] = (short)(a2 + a29);
        dst[dAt + 3 * dStep] = (short)(a3 + a28);
        dst[dAt + 4 * dStep] = (short)(a4 + a27);
        dst[dAt + 5 * dStep] = (short)(a5 + a26);
        dst[dAt + 6 * dStep] = (short)(a6 + a25);
        dst[dAt + 7 * dStep] = (short)(a7 + a24);
        dst[dAt + 8 * dStep] = (short)(a8 + a23);
        dst[dAt + 9 * dStep] = (short)(a9 + a22);
        dst[dAt + 10 * dStep] = (short)(a10 + a21);
        dst[dAt + 11 * dStep] = (short)(a11 + a20);
        dst[dAt + 12 * dStep] = (short)(a12 + a19);
        dst[dAt + 13 * dStep] = (short)(a13 + a18);
        dst[dAt + 14 * dStep] = (short)(a14 + a17);
        dst[dAt + 15 * dStep] = (short)(a15 + a16);
        dst[dAt + 16 * dStep] = (short)(a15 - a16);
        dst[dAt + 17 * dStep] = (short)(a14 - a17);
        dst[dAt + 18 * dStep] = (short)(a13 - a18);
        dst[dAt + 19 * dStep] = (short)(a12 - a19);
        dst[dAt + 20 * dStep] = (short)(a11 - a20);
        dst[dAt + 21 * dStep] = (short)(a10 - a21);
        dst[dAt + 22 * dStep] = (short)(a9 - a22);
        dst[dAt + 23 * dStep] = (short)(a8 - a23);
        dst[dAt + 24 * dStep] = (short)(a7 - a24);
        dst[dAt + 25 * dStep] = (short)(a6 - a25);
        dst[dAt + 26 * dStep] = (short)(a5 - a26);
        dst[dAt + 27 * dStep] = (short)(a4 - a27);
        dst[dAt + 28 * dStep] = (short)(a3 - a28);
        dst[dAt + 29 * dStep] = (short)(a2 - a29);
        dst[dAt + 30 * dStep] = (short)(a1 - a30);
        dst[dAt + 31 * dStep] = (short)(a0 - a31);
    }
}
