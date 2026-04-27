// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP8 inverse transforms. Structural port of libvpx vp8/common/idctllm.c
// to clean C#. RFC 6386 sec 14.
//
// Three primitives:
//   1. ShortIdct4x4Llm - 4x4 IDCT for AC coefficients (predict + clamp+add)
//   2. DcOnlyIdctAdd   - 4x4 fast path when only DC is non-zero
//   3. ShortInvWalsh4x4 - 4x4 inverse Walsh-Hadamard for the Y2 second-order
//                         transform (decoded 16 DCs put through this then
//                         distributed back to the Y4 block DC slots)
//
// Two fixed-point constants, libvpx names retained for traceability:
//   cospi8sqrt2minus1 = 20091  // sqrt(2)*cos(pi/8) - 1
//   sinpi8sqrt2       = 35468  // sqrt(2)*sin(pi/8)
// The "minus 1" trick keeps the multiply in 16-bit fixed-point precision.

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>VP8 4x4 inverse DCT and Walsh-Hadamard transforms.</summary>
public static class Vp8InverseTransform
{
    private const int CospiSqrt2Minus1 = 20091;
    private const int SinpiSqrt2 = 35468;

    /// <summary>
    /// 4x4 IDCT with predict-and-add. Mirrors libvpx <c>vp8_short_idct4x4llm_c</c>:
    /// performs column then row 1D IDCT on <paramref name="input"/>, then for
    /// each output pixel adds the corresponding pred byte and clamps to [0, 255].
    /// </summary>
    /// <param name="input">16 short coefficients in raster order (row-major 4x4).</param>
    /// <param name="pred">Prediction bytes (4x4) at <paramref name="predStride"/>.</param>
    /// <param name="predStride">Stride of pred buffer in bytes.</param>
    /// <param name="dst">Destination bytes (4x4) at <paramref name="dstStride"/>.</param>
    /// <param name="dstStride">Stride of dst buffer in bytes.</param>
    public static void ShortIdct4x4Llm(
        ReadOnlySpan<short> input,
        ReadOnlySpan<byte> pred, int predStride,
        Span<byte> dst, int dstStride)
    {
        if (input.Length < 16) throw new ArgumentException("input must have 16 entries", nameof(input));
        Span<short> output = stackalloc short[16];

        // First 1D IDCT pass: per column. ip[0], ip[4], ip[8], ip[12] are one column.
        for (int i = 0; i < 4; i++)
        {
            int a1 = input[i + 0] + input[i + 8];
            int b1 = input[i + 0] - input[i + 8];

            int temp1 = (input[i + 4] * SinpiSqrt2) >> 16;
            int temp2 = input[i + 12] + ((input[i + 12] * CospiSqrt2Minus1) >> 16);
            int c1 = temp1 - temp2;

            temp1 = input[i + 4] + ((input[i + 4] * CospiSqrt2Minus1) >> 16);
            temp2 = (input[i + 12] * SinpiSqrt2) >> 16;
            int d1 = temp1 + temp2;

            output[i + 0]  = (short)(a1 + d1);
            output[i + 12] = (short)(a1 - d1);
            output[i + 4]  = (short)(b1 + c1);
            output[i + 8]  = (short)(b1 - c1);
        }

        // Second 1D IDCT pass: per row. ip[0], ip[1], ip[2], ip[3] are one row.
        // Includes the round + 3-bit-right-shift normalization.
        Span<short> stage2 = stackalloc short[16];
        for (int i = 0; i < 4; i++)
        {
            int row = i * 4;
            int a1 = output[row + 0] + output[row + 2];
            int b1 = output[row + 0] - output[row + 2];

            int temp1 = (output[row + 1] * SinpiSqrt2) >> 16;
            int temp2 = output[row + 3] + ((output[row + 3] * CospiSqrt2Minus1) >> 16);
            int c1 = temp1 - temp2;

            temp1 = output[row + 1] + ((output[row + 1] * CospiSqrt2Minus1) >> 16);
            temp2 = (output[row + 3] * SinpiSqrt2) >> 16;
            int d1 = temp1 + temp2;

            stage2[row + 0] = (short)((a1 + d1 + 4) >> 3);
            stage2[row + 3] = (short)((a1 - d1 + 4) >> 3);
            stage2[row + 1] = (short)((b1 + c1 + 4) >> 3);
            stage2[row + 2] = (short)((b1 - c1 + 4) >> 3);
        }

        // Predict + add + clamp.
        for (int r = 0; r < 4; r++)
        {
            int rowBase = r * 4;
            int dstRow = r * dstStride;
            int predRow = r * predStride;
            for (int c = 0; c < 4; c++)
            {
                int a = stage2[rowBase + c] + pred[predRow + c];
                if (a < 0) a = 0;
                else if (a > 255) a = 255;
                dst[dstRow + c] = (byte)a;
            }
        }
    }

    /// <summary>
    /// 4x4 DC-only IDCT fast path. Mirrors libvpx <c>vp8_dc_only_idct_add_c</c>:
    /// when only DC is non-zero, the 4x4 result is constant a1 = (input_dc + 4) >> 3.
    /// </summary>
    public static void DcOnlyIdctAdd(
        short inputDc,
        ReadOnlySpan<byte> pred, int predStride,
        Span<byte> dst, int dstStride)
    {
        int a1 = (inputDc + 4) >> 3;
        for (int r = 0; r < 4; r++)
        {
            int dstRow = r * dstStride;
            int predRow = r * predStride;
            for (int c = 0; c < 4; c++)
            {
                int a = a1 + pred[predRow + c];
                if (a < 0) a = 0;
                else if (a > 255) a = 255;
                dst[dstRow + c] = (byte)a;
            }
        }
    }

    /// <summary>
    /// 4x4 inverse Walsh-Hadamard transform for the VP8 Y2 second-order block.
    /// Mirrors libvpx <c>vp8_short_inv_walsh4x4_c</c>.
    /// </summary>
    /// <param name="input">16 input coefficients (row-major 4x4).</param>
    /// <param name="mbDqCoeff">
    /// 16-element output. The decoded DCs are written at strided offsets
    /// (mbDqCoeff[0], [16], [32], ...) but this method matches libvpx
    /// behavior and writes into mbDqCoeff[0..15] at the index pattern below.
    /// libvpx callers typically distribute these into the 16 Y4 block DC
    /// slots after the call.
    /// </param>
    public static void ShortInvWalsh4x4(ReadOnlySpan<short> input, Span<short> mbDqCoeff)
    {
        if (input.Length < 16) throw new ArgumentException("input must have 16 entries", nameof(input));
        if (mbDqCoeff.Length < 16) throw new ArgumentException("mbDqCoeff must hold 16 entries", nameof(mbDqCoeff));

        Span<short> output = stackalloc short[16];

        // Column pass. ip[0], ip[4], ip[8], ip[12] are one column.
        for (int i = 0; i < 4; i++)
        {
            int a1 = input[i + 0] + input[i + 12];
            int b1 = input[i + 4] + input[i + 8];
            int c1 = input[i + 4] - input[i + 8];
            int d1 = input[i + 0] - input[i + 12];

            output[i + 0]  = (short)(a1 + b1);
            output[i + 4]  = (short)(c1 + d1);
            output[i + 8]  = (short)(a1 - b1);
            output[i + 12] = (short)(d1 - c1);
        }

        // Row pass with +3 round, >>3 shift.
        for (int i = 0; i < 4; i++)
        {
            int row = i * 4;
            int a1 = output[row + 0] + output[row + 3];
            int b1 = output[row + 1] + output[row + 2];
            int c1 = output[row + 1] - output[row + 2];
            int d1 = output[row + 0] - output[row + 3];

            int a2 = a1 + b1;
            int b2 = c1 + d1;
            int c2 = a1 - b1;
            int d2 = d1 - c1;

            mbDqCoeff[row + 0] = (short)((a2 + 3) >> 3);
            mbDqCoeff[row + 1] = (short)((b2 + 3) >> 3);
            mbDqCoeff[row + 2] = (short)((c2 + 3) >> 3);
            mbDqCoeff[row + 3] = (short)((d2 + 3) >> 3);
        }
    }
}
