// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP8 forward transforms - encoder-side counterparts of Vp8InverseTransform.
// Structural port of libvpx vp8/encoder/dct.c to clean C#. RFC 6386 sec 14
// (transforms; encoder side).
//
// Two primitives:
//   1. ShortFdct4x4 - 4x4 forward DCT (sin/cos pi/8 fixed-point integer)
//   2. ShortWalsh4x4 - 4x4 forward Walsh-Hadamard for the Y2 second-order
//                       transform (encoder takes the 16 Y4 DCs through this
//                       to produce the Y2 coefficient block)
//
// libvpx uses pitch/2 as a row-stride argument because the C source is
// indexed via short* with byte-pitch. This port uses an integer row-stride
// in shorts.

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>VP8 forward transforms (encoder side).</summary>
public static class Vp8ForwardTransform
{
    /// <summary>
    /// 4x4 forward DCT with sin/cos pi/8 fixed-point integer arithmetic.
    /// Mirrors libvpx <c>vp8_short_fdct4x4_c</c> bit-for-bit.
    /// </summary>
    /// <param name="input">16 input samples (rowStride * 4 entries minimum).</param>
    /// <param name="rowStrideShorts">Row stride in shorts (libvpx pitch/2; commonly 4 for a packed 4x4).</param>
    /// <param name="output">16 output coefficients (raster 4x4).</param>
    public static void ShortFdct4x4(ReadOnlySpan<short> input, int rowStrideShorts, Span<short> output)
    {
        if (input.Length < rowStrideShorts * 4)
            throw new ArgumentException($"input must have at least {rowStrideShorts * 4} entries", nameof(input));
        if (output.Length < 16)
            throw new ArgumentException("output must have 16 entries", nameof(output));

        // First pass: rows. Output is 4x4 packed (op += 4 per row).
        for (int i = 0; i < 4; i++)
        {
            int rowBase = i * rowStrideShorts;
            int a1 = (input[rowBase + 0] + input[rowBase + 3]) * 8;
            int b1 = (input[rowBase + 1] + input[rowBase + 2]) * 8;
            int c1 = (input[rowBase + 1] - input[rowBase + 2]) * 8;
            int d1 = (input[rowBase + 0] - input[rowBase + 3]) * 8;

            int outBase = i * 4;
            output[outBase + 0] = (short)(a1 + b1);
            output[outBase + 2] = (short)(a1 - b1);
            output[outBase + 1] = (short)((c1 * 2217 + d1 * 5352 + 14500) >> 12);
            output[outBase + 3] = (short)((d1 * 2217 - c1 * 5352 + 7500) >> 12);
        }

        // Second pass: columns. Indexes [0]/[4]/[8]/[12] are one column.
        // Apply in-place on the output buffer (libvpx ip = op pattern).
        Span<short> stage1 = stackalloc short[16];
        for (int i = 0; i < 16; i++) stage1[i] = output[i];

        for (int i = 0; i < 4; i++)
        {
            int a1 = stage1[i + 0] + stage1[i + 12];
            int b1 = stage1[i + 4] + stage1[i + 8];
            int c1 = stage1[i + 4] - stage1[i + 8];
            int d1 = stage1[i + 0] - stage1[i + 12];

            output[i + 0]  = (short)((a1 + b1 + 7) >> 4);
            output[i + 8]  = (short)((a1 - b1 + 7) >> 4);
            // libvpx: ((c*2217 + d*5352 + 12000) >> 16) + (d != 0)
            output[i + 4]  = (short)(((c1 * 2217 + d1 * 5352 + 12000) >> 16) + (d1 != 0 ? 1 : 0));
            output[i + 12] = (short)((d1 * 2217 - c1 * 5352 + 51000) >> 16);
        }
    }

    /// <summary>
    /// 4x4 forward Walsh-Hadamard transform for the Y2 block. Mirrors
    /// libvpx <c>vp8_short_walsh4x4_c</c>. Used to encode the 16 Y4 DC
    /// values as the Y2 coefficient block.
    /// </summary>
    public static void ShortWalsh4x4(ReadOnlySpan<short> input, int rowStrideShorts, Span<short> output)
    {
        if (input.Length < rowStrideShorts * 4)
            throw new ArgumentException($"input must have at least {rowStrideShorts * 4} entries", nameof(input));
        if (output.Length < 16)
            throw new ArgumentException("output must have 16 entries", nameof(output));

        // First pass: rows.
        Span<short> stage1 = stackalloc short[16];
        for (int i = 0; i < 4; i++)
        {
            int rowBase = i * rowStrideShorts;
            int a1 = (input[rowBase + 0] + input[rowBase + 2]) * 4;
            int d1 = (input[rowBase + 1] + input[rowBase + 3]) * 4;
            int c1 = (input[rowBase + 1] - input[rowBase + 3]) * 4;
            int b1 = (input[rowBase + 0] - input[rowBase + 2]) * 4;

            int outBase = i * 4;
            stage1[outBase + 0] = (short)(a1 + d1 + (a1 != 0 ? 1 : 0));
            stage1[outBase + 1] = (short)(b1 + c1);
            stage1[outBase + 2] = (short)(b1 - c1);
            stage1[outBase + 3] = (short)(a1 - d1);
        }

        // Second pass: columns. Indexes [0]/[4]/[8]/[12] one column.
        for (int i = 0; i < 4; i++)
        {
            int a1 = stage1[i + 0] + stage1[i + 8];
            int d1 = stage1[i + 4] + stage1[i + 12];
            int c1 = stage1[i + 4] - stage1[i + 12];
            int b1 = stage1[i + 0] - stage1[i + 8];

            int a2 = a1 + d1;
            int b2 = b1 + c1;
            int c2 = b1 - c1;
            int d2 = a1 - d1;

            a2 += a2 < 0 ? 1 : 0;
            b2 += b2 < 0 ? 1 : 0;
            c2 += c2 < 0 ? 1 : 0;
            d2 += d2 < 0 ? 1 : 0;

            output[i + 0]  = (short)((a2 + 3) >> 3);
            output[i + 4]  = (short)((b2 + 3) >> 3);
            output[i + 8]  = (short)((c2 + 3) >> 3);
            output[i + 12] = (short)((d2 + 3) >> 3);
        }
    }
}
