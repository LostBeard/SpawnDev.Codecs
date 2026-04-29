// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 4x4 forward DCT, GPU-callable form for in-kernel reuse. Bit-exact
// mirror of Vp9ForwardDct4x4.Transform (libvpx vpx_fdct4x4_c port).
//
// Pairs with the existing Vp9Idct4x4Gpu (decoder side) - now both
// directions of the VP9 4x4 DCT have GPU primitives. Combined with the
// existing Vp9ForwardDct8x8Gpu and Vp9ForwardDct16x16Gpu, the VP9
// forward DCT family at 4x4 / 8x8 / 16x16 is now in-kernel callable.
//
// Two-pass shape (matches CPU):
//   Pass 1: column DCT with results transposed into scratch.
//   Pass 2: row DCT on transposed scratch.
//   Post:   divide by 4 with +1 rounding bias.
//
// libvpx oddity: (0,0) DC of the first column gets +1 bias if non-zero.

using ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// GPU-callable VP9 4x4 forward DCT helper. Bit-exact mirror of
/// <see cref="Vp9ForwardDct4x4"/> for in-kernel use.
/// </summary>
public static class Vp9ForwardDct4x4Gpu
{
    private const int DctConstBits = 14;
    private const int DctConstRounding = 1 << (DctConstBits - 1);

    private const int Cospi8_64 = 15137;
    private const int Cospi16_64 = 11585;
    private const int Cospi24_64 = 6270;

    /// <summary>
    /// Forward 4x4 DCT. Reads <paramref name="input"/> (row-major shorts
    /// with the given row stride) starting at <paramref name="inBase"/>;
    /// writes 16 output coefficients to <paramref name="output"/>
    /// starting at <paramref name="outBase"/> (raster 4x4).
    /// </summary>
    /// <param name="scratch">Per-call scratch (length &gt;= 16 ints).</param>
    public static void Transform(
        ArrayView<short> input, long inBase, int rowStrideShorts,
        ArrayView<int> output, long outBase,
        ArrayView<int> scratch, long scratchBase)
    {
        // Pass 1: column DCT into scratch, transposed.
        long inputOffset = inBase;
        long outOffset = scratchBase;
        for (int i = 0; i < 4; i++)
        {
            int h0 = input[inputOffset + 0 * rowStrideShorts] * 16;
            int h1 = input[inputOffset + 1 * rowStrideShorts] * 16;
            int h2 = input[inputOffset + 2 * rowStrideShorts] * 16;
            int h3 = input[inputOffset + 3 * rowStrideShorts] * 16;

            // libvpx +1 bias on the (0,0) DC if non-zero.
            if (i == 0 && h0 != 0) h0++;

            int s0 = h0 + h3;
            int s1 = h1 + h2;
            int s2 = h1 - h2;
            int s3 = h0 - h3;

            scratch[outOffset + 0] = FdctRoundShift((long)(s0 + s1) * Cospi16_64);
            scratch[outOffset + 2] = FdctRoundShift((long)(s0 - s1) * Cospi16_64);
            scratch[outOffset + 1] = FdctRoundShift((long)s2 * Cospi24_64 + (long)s3 * Cospi8_64);
            scratch[outOffset + 3] = FdctRoundShift((long)(-s2) * Cospi8_64 + (long)s3 * Cospi24_64);

            inputOffset++;
            outOffset += 4;
        }

        // Pass 2: row DCT (rows are now columns of scratch).
        outOffset = outBase;
        for (int i = 0; i < 4; i++)
        {
            int h0 = scratch[scratchBase + i + 0 * 4];
            int h1 = scratch[scratchBase + i + 1 * 4];
            int h2 = scratch[scratchBase + i + 2 * 4];
            int h3 = scratch[scratchBase + i + 3 * 4];

            int s0 = h0 + h3;
            int s1 = h1 + h2;
            int s2 = h1 - h2;
            int s3 = h0 - h3;

            output[outOffset + 0] = FdctRoundShift((long)(s0 + s1) * Cospi16_64);
            output[outOffset + 2] = FdctRoundShift((long)(s0 - s1) * Cospi16_64);
            output[outOffset + 1] = FdctRoundShift((long)s2 * Cospi24_64 + (long)s3 * Cospi8_64);
            output[outOffset + 3] = FdctRoundShift((long)(-s2) * Cospi8_64 + (long)s3 * Cospi24_64);

            outOffset += 4;
        }

        // Post-pass: divide by 4 with +1 rounding.
        for (int i = 0; i < 16; i++)
        {
            output[outBase + i] = (output[outBase + i] + 1) >> 2;
        }
    }

    /// <summary>libvpx fdct_round_shift: (input + 1 &lt;&lt; 13) &gt;&gt; 14.</summary>
    private static int FdctRoundShift(long input) =>
        (int)((input + DctConstRounding) >> DctConstBits);
}
