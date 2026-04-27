// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 8x8 forward DCT. Bit-exact port of libvpx vpx_dsp/fwd_txfm.c
// vpx_fdct8x8_c.
//
// Two-pass: column DCT then row DCT. Pass 1 multiplies inputs by 4
// (input *= 4); pass 2 reads from intermediate buffer. Final post-pass
// divides by 2.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 8x8 forward DCT (encoder side).</summary>
public static class Vp9ForwardDct8x8
{
    /// <summary>
    /// 8x8 forward DCT. Mirrors libvpx <c>vpx_fdct8x8_c</c>.
    /// </summary>
    /// <param name="input">Input samples (rowStride * 8 entries minimum).</param>
    /// <param name="rowStrideShorts">Row stride in shorts.</param>
    /// <param name="output">64 output coefficients (raster 8x8).</param>
    public static void Transform(ReadOnlySpan<short> input, int rowStrideShorts, Span<int> output)
    {
        if (input.Length < rowStrideShorts * 8)
            throw new ArgumentException($"input must hold at least {rowStrideShorts * 8} entries", nameof(input));
        if (output.Length < 64)
            throw new ArgumentException("output must hold 64 entries", nameof(output));

        Span<int> intermediate = stackalloc int[64];
        Span<int> intermediateSpan = intermediate;

        // Pass 1: column DCT, write to intermediate.
        DoPass(input, rowStrideShorts, intermediate, isFirstPass: true);
        // Pass 2: row DCT (reading from intermediate as columns).
        DoPassFromIntermediate(intermediate, output);

        // Final scale: divide by 2.
        for (int i = 0; i < 64; i++) output[i] /= 2;
    }

    private static void DoPass(ReadOnlySpan<short> input, int stride, Span<int> output, bool isFirstPass)
    {
        int outOffset = 0;
        for (int col = 0; col < 8; col++)
        {
            int s0, s1, s2, s3, s4, s5, s6, s7;
            // Pass 1: input *= 4
            s0 = (input[col + 0 * stride] + input[col + 7 * stride]) * 4;
            s1 = (input[col + 1 * stride] + input[col + 6 * stride]) * 4;
            s2 = (input[col + 2 * stride] + input[col + 5 * stride]) * 4;
            s3 = (input[col + 3 * stride] + input[col + 4 * stride]) * 4;
            s4 = (input[col + 3 * stride] - input[col + 4 * stride]) * 4;
            s5 = (input[col + 2 * stride] - input[col + 5 * stride]) * 4;
            s6 = (input[col + 1 * stride] - input[col + 6 * stride]) * 4;
            s7 = (input[col + 0 * stride] - input[col + 7 * stride]) * 4;

            ButterflyAndStore(s0, s1, s2, s3, s4, s5, s6, s7, output.Slice(outOffset, 8));
            outOffset += 8;
        }
    }

    private static void DoPassFromIntermediate(ReadOnlySpan<int> input, Span<int> output)
    {
        int outOffset = 0;
        for (int col = 0; col < 8; col++)
        {
            // After pass 1 we wrote 8 cols x 8 rows transposed. Reading
            // input[col + row * 8] gives one column of intermediate.
            int s0 = input[col + 0 * 8] + input[col + 7 * 8];
            int s1 = input[col + 1 * 8] + input[col + 6 * 8];
            int s2 = input[col + 2 * 8] + input[col + 5 * 8];
            int s3 = input[col + 3 * 8] + input[col + 4 * 8];
            int s4 = input[col + 3 * 8] - input[col + 4 * 8];
            int s5 = input[col + 2 * 8] - input[col + 5 * 8];
            int s6 = input[col + 1 * 8] - input[col + 6 * 8];
            int s7 = input[col + 0 * 8] - input[col + 7 * 8];

            ButterflyAndStore(s0, s1, s2, s3, s4, s5, s6, s7, output.Slice(outOffset, 8));
            outOffset += 8;
        }
    }

    private static void ButterflyAndStore(int s0, int s1, int s2, int s3, int s4, int s5, int s6, int s7, Span<int> outRow)
    {
        // fdct4(step, step) for s0..s3
        int x0 = s0 + s3;
        int x1 = s1 + s2;
        int x2 = s1 - s2;
        int x3 = s0 - s3;
        long t0 = (long)(x0 + x1) * Vp9CospiConstants.Cospi16_64;
        long t1 = (long)(x0 - x1) * Vp9CospiConstants.Cospi16_64;
        long t2 = (long)x2 * Vp9CospiConstants.Cospi24_64 + (long)x3 * Vp9CospiConstants.Cospi8_64;
        long t3 = (long)(-x2) * Vp9CospiConstants.Cospi8_64 + (long)x3 * Vp9CospiConstants.Cospi24_64;
        outRow[0] = Vp9CospiConstants.RoundShift(t0);
        outRow[2] = Vp9CospiConstants.RoundShift(t2);
        outRow[4] = Vp9CospiConstants.RoundShift(t1);
        outRow[6] = Vp9CospiConstants.RoundShift(t3);

        // Stage 2 - 4 for s4..s7
        long u0 = (long)(s6 - s5) * Vp9CospiConstants.Cospi16_64;
        long u1 = (long)(s6 + s5) * Vp9CospiConstants.Cospi16_64;
        int v2 = Vp9CospiConstants.RoundShift(u0);
        int v3 = Vp9CospiConstants.RoundShift(u1);

        int y0 = s4 + v2;
        int y1 = s4 - v2;
        int y2 = s7 - v3;
        int y3 = s7 + v3;

        long w0 = (long)y0 * Vp9CospiConstants.Cospi28_64 + (long)y3 * Vp9CospiConstants.Cospi4_64;
        long w1 = (long)y1 * Vp9CospiConstants.Cospi12_64 + (long)y2 * Vp9CospiConstants.Cospi20_64;
        long w2 = (long)y2 * Vp9CospiConstants.Cospi12_64 + (long)y1 * (-Vp9CospiConstants.Cospi20_64);
        long w3 = (long)y3 * Vp9CospiConstants.Cospi28_64 + (long)y0 * (-Vp9CospiConstants.Cospi4_64);
        outRow[1] = Vp9CospiConstants.RoundShift(w0);
        outRow[3] = Vp9CospiConstants.RoundShift(w2);
        outRow[5] = Vp9CospiConstants.RoundShift(w1);
        outRow[7] = Vp9CospiConstants.RoundShift(w3);
    }
}
