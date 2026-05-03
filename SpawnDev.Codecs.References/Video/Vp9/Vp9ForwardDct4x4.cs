// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 4x4 forward DCT. Bit-exact port of libvpx vpx_dsp/fwd_txfm.c
// vpx_fdct4x4_c. RFC 6386 / VP9 spec section 8 (transforms).
//
// Two-pass DCT with butterfly + cospi multiplications. cospi constants
// from vpx_dsp/txfm_common.h:
//   cospi_8_64  = 15137
//   cospi_16_64 = 11585
//   cospi_24_64 = 6270
// fdct_round_shift = (input + (1 << 13)) >> 14 (DCT_CONST_BITS = 14)
//
// Pairs with the existing Vp9Idct4x4 in the codebase. Top-level
// Vp9Encoder will use this as the per-block forward transform.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 4x4 forward DCT (encoder side).</summary>
public static class Vp9ForwardDct4x4
{
    private const int DctConstBits = 14;
    private const int DctConstRounding = 1 << (DctConstBits - 1);

    private const int Cospi8_64  = 15137;
    private const int Cospi16_64 = 11585;
    private const int Cospi24_64 = 6270;

    private static int FdctRoundShift(long input) =>
        (int)((input + DctConstRounding) >> DctConstBits);

    /// <summary>
    /// Forward 4x4 DCT. Mirrors libvpx <c>vpx_fdct4x4_c</c>.
    /// </summary>
    /// <param name="input">Input samples (rowStride * 4 entries).</param>
    /// <param name="rowStrideShorts">Row stride in shorts.</param>
    /// <param name="output">16 output coefficients (raster 4x4).</param>
    public static void Transform(ReadOnlySpan<short> input, int rowStrideShorts, Span<int> output)
    {
        if (input.Length < rowStrideShorts * 4)
            throw new ArgumentException($"input must have at least {rowStrideShorts * 4} entries", nameof(input));
        if (output.Length < 16)
            throw new ArgumentException("output must have 16 entries", nameof(output));

        Span<int> intermediate = stackalloc int[16];

        // Pass 1: column DCT, results transposed.
        int inputOffset = 0;
        int outOffset = 0;
        for (int i = 0; i < 4; i++)
        {
            int[] inHigh = new int[4];
            inHigh[0] = input[inputOffset + 0 * rowStrideShorts] * 16;
            inHigh[1] = input[inputOffset + 1 * rowStrideShorts] * 16;
            inHigh[2] = input[inputOffset + 2 * rowStrideShorts] * 16;
            inHigh[3] = input[inputOffset + 3 * rowStrideShorts] * 16;
            // libvpx adds 1 to the (0,0) DC if non-zero (rounding bias).
            if (i == 0 && inHigh[0] != 0) inHigh[0]++;

            int s0 = inHigh[0] + inHigh[3];
            int s1 = inHigh[1] + inHigh[2];
            int s2 = inHigh[1] - inHigh[2];
            int s3 = inHigh[0] - inHigh[3];

            intermediate[outOffset + 0] = FdctRoundShift((long)(s0 + s1) * Cospi16_64);
            intermediate[outOffset + 2] = FdctRoundShift((long)(s0 - s1) * Cospi16_64);
            intermediate[outOffset + 1] = FdctRoundShift((long)s2 * Cospi24_64 + (long)s3 * Cospi8_64);
            intermediate[outOffset + 3] = FdctRoundShift((long)(-s2) * Cospi8_64 + (long)s3 * Cospi24_64);

            inputOffset++;
            outOffset += 4;
        }

        // Pass 2: row DCT (rows are now columns of intermediate).
        outOffset = 0;
        for (int i = 0; i < 4; i++)
        {
            int[] inHigh = new int[4];
            inHigh[0] = intermediate[i + 0 * 4];
            inHigh[1] = intermediate[i + 1 * 4];
            inHigh[2] = intermediate[i + 2 * 4];
            inHigh[3] = intermediate[i + 3 * 4];

            int s0 = inHigh[0] + inHigh[3];
            int s1 = inHigh[1] + inHigh[2];
            int s2 = inHigh[1] - inHigh[2];
            int s3 = inHigh[0] - inHigh[3];

            output[outOffset + 0] = FdctRoundShift((long)(s0 + s1) * Cospi16_64);
            output[outOffset + 2] = FdctRoundShift((long)(s0 - s1) * Cospi16_64);
            output[outOffset + 1] = FdctRoundShift((long)s2 * Cospi24_64 + (long)s3 * Cospi8_64);
            output[outOffset + 3] = FdctRoundShift((long)(-s2) * Cospi8_64 + (long)s3 * Cospi24_64);

            outOffset += 4;
        }

        // Final post-pass: divide by 4 with +1 rounding.
        for (int i = 0; i < 16; i++)
            output[i] = (output[i] + 1) >> 2;
    }
}
