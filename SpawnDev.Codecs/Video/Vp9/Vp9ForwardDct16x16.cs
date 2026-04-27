// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 16x16 forward DCT. Bit-exact port of libvpx vpx_dsp/fwd_txfm.c
// vpx_fdct16x16_c.
//
// Two-pass transform: pass 0 transforms columns and transposes results
// (input multiplied by 4); pass 1 transforms rows from intermediate
// buffer (input rounded by ((x + 1) >> 2) - "half_round_shift" inputs).
//
// Each pass internally splits inputs into:
//   - "even" 8-element fdct8 (handles low frequencies)
//   - "odd" 8-element transform (handles high frequencies via stages 2-6
//     using cospi_2/6/10/14/18/22/26/30 constants)

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 16x16 forward DCT (encoder side).</summary>
public static class Vp9ForwardDct16x16
{
    /// <summary>16x16 forward DCT. Mirrors libvpx <c>vpx_fdct16x16_c</c>.</summary>
    public static void Transform(ReadOnlySpan<short> input, int rowStrideShorts, Span<int> output)
    {
        if (input.Length < rowStrideShorts * 16)
            throw new ArgumentException($"input must hold at least {rowStrideShorts * 16} entries", nameof(input));
        if (output.Length < 256)
            throw new ArgumentException("output must hold 256 entries", nameof(output));

        Span<int> intermediate = stackalloc int[256];

        // Pass 1: columns.
        DoPass(input, rowStrideShorts, intermediate, isFirstPass: true);
        // Pass 2: rows (treating intermediate as columns of transposed buffer).
        DoPass(intermediate, output);
    }

    private static void DoPass(ReadOnlySpan<short> input, int stride, Span<int> output, bool isFirstPass)
    {
        Span<int> step1 = stackalloc int[8];
        Span<int> step2 = stackalloc int[8];
        Span<int> step3 = stackalloc int[8];
        Span<int> inHigh = stackalloc int[8];
        Span<int> outRow = stackalloc int[16];

        for (int col = 0; col < 16; col++)
        {
            // Pass 1: input *= 4
            inHigh[0] = (input[col + 0 * stride] + input[col + 15 * stride]) * 4;
            inHigh[1] = (input[col + 1 * stride] + input[col + 14 * stride]) * 4;
            inHigh[2] = (input[col + 2 * stride] + input[col + 13 * stride]) * 4;
            inHigh[3] = (input[col + 3 * stride] + input[col + 12 * stride]) * 4;
            inHigh[4] = (input[col + 4 * stride] + input[col + 11 * stride]) * 4;
            inHigh[5] = (input[col + 5 * stride] + input[col + 10 * stride]) * 4;
            inHigh[6] = (input[col + 6 * stride] + input[col + 9 * stride]) * 4;
            inHigh[7] = (input[col + 7 * stride] + input[col + 8 * stride]) * 4;
            step1[0] = (input[col + 7 * stride] - input[col + 8 * stride]) * 4;
            step1[1] = (input[col + 6 * stride] - input[col + 9 * stride]) * 4;
            step1[2] = (input[col + 5 * stride] - input[col + 10 * stride]) * 4;
            step1[3] = (input[col + 4 * stride] - input[col + 11 * stride]) * 4;
            step1[4] = (input[col + 3 * stride] - input[col + 12 * stride]) * 4;
            step1[5] = (input[col + 2 * stride] - input[col + 13 * stride]) * 4;
            step1[6] = (input[col + 1 * stride] - input[col + 14 * stride]) * 4;
            step1[7] = (input[col + 0 * stride] - input[col + 15 * stride]) * 4;

            ButterflyAndStore(inHigh, step1, step2, step3, outRow);
            for (int j = 0; j < 16; j++) output[col * 16 + j] = outRow[j];
        }
    }

    private static void DoPass(ReadOnlySpan<int> input, Span<int> output)
    {
        Span<int> step1 = stackalloc int[8];
        Span<int> step2 = stackalloc int[8];
        Span<int> step3 = stackalloc int[8];
        Span<int> inHigh = stackalloc int[8];
        Span<int> outRow = stackalloc int[16];

        for (int col = 0; col < 16; col++)
        {
            // Pass 2: input rounded via (x + 1) >> 2.
            inHigh[0] = ((input[col + 0 * 16] + 1) >> 2) + ((input[col + 15 * 16] + 1) >> 2);
            inHigh[1] = ((input[col + 1 * 16] + 1) >> 2) + ((input[col + 14 * 16] + 1) >> 2);
            inHigh[2] = ((input[col + 2 * 16] + 1) >> 2) + ((input[col + 13 * 16] + 1) >> 2);
            inHigh[3] = ((input[col + 3 * 16] + 1) >> 2) + ((input[col + 12 * 16] + 1) >> 2);
            inHigh[4] = ((input[col + 4 * 16] + 1) >> 2) + ((input[col + 11 * 16] + 1) >> 2);
            inHigh[5] = ((input[col + 5 * 16] + 1) >> 2) + ((input[col + 10 * 16] + 1) >> 2);
            inHigh[6] = ((input[col + 6 * 16] + 1) >> 2) + ((input[col + 9 * 16] + 1) >> 2);
            inHigh[7] = ((input[col + 7 * 16] + 1) >> 2) + ((input[col + 8 * 16] + 1) >> 2);
            step1[0] = ((input[col + 7 * 16] + 1) >> 2) - ((input[col + 8 * 16] + 1) >> 2);
            step1[1] = ((input[col + 6 * 16] + 1) >> 2) - ((input[col + 9 * 16] + 1) >> 2);
            step1[2] = ((input[col + 5 * 16] + 1) >> 2) - ((input[col + 10 * 16] + 1) >> 2);
            step1[3] = ((input[col + 4 * 16] + 1) >> 2) - ((input[col + 11 * 16] + 1) >> 2);
            step1[4] = ((input[col + 3 * 16] + 1) >> 2) - ((input[col + 12 * 16] + 1) >> 2);
            step1[5] = ((input[col + 2 * 16] + 1) >> 2) - ((input[col + 13 * 16] + 1) >> 2);
            step1[6] = ((input[col + 1 * 16] + 1) >> 2) - ((input[col + 14 * 16] + 1) >> 2);
            step1[7] = ((input[col + 0 * 16] + 1) >> 2) - ((input[col + 15 * 16] + 1) >> 2);

            ButterflyAndStore(inHigh, step1, step2, step3, outRow);
            for (int j = 0; j < 16; j++) output[col * 16 + j] = outRow[j];
        }
    }

    private static void ButterflyAndStore(Span<int> inHigh, Span<int> step1, Span<int> step2, Span<int> step3, Span<int> outRow)
    {
        // Even half: fdct8 on inHigh[0..7]
        int s0 = inHigh[0] + inHigh[7];
        int s1 = inHigh[1] + inHigh[6];
        int s2 = inHigh[2] + inHigh[5];
        int s3 = inHigh[3] + inHigh[4];
        int s4 = inHigh[3] - inHigh[4];
        int s5 = inHigh[2] - inHigh[5];
        int s6 = inHigh[1] - inHigh[6];
        int s7 = inHigh[0] - inHigh[7];

        int x0 = s0 + s3, x1 = s1 + s2, x2 = s1 - s2, x3 = s0 - s3;
        long t0 = (long)(x0 + x1) * Vp9CospiConstants.Cospi16_64;
        long t1 = (long)(x0 - x1) * Vp9CospiConstants.Cospi16_64;
        long t2 = (long)x3 * Vp9CospiConstants.Cospi8_64 + (long)x2 * Vp9CospiConstants.Cospi24_64;
        long t3 = (long)x3 * Vp9CospiConstants.Cospi24_64 - (long)x2 * Vp9CospiConstants.Cospi8_64;
        outRow[0] = Vp9CospiConstants.RoundShift(t0);
        outRow[4] = Vp9CospiConstants.RoundShift(t2);
        outRow[8] = Vp9CospiConstants.RoundShift(t1);
        outRow[12] = Vp9CospiConstants.RoundShift(t3);

        long u0 = (long)(s6 - s5) * Vp9CospiConstants.Cospi16_64;
        long u1 = (long)(s6 + s5) * Vp9CospiConstants.Cospi16_64;
        int v2 = Vp9CospiConstants.RoundShift(u0);
        int v3 = Vp9CospiConstants.RoundShift(u1);

        int y0 = s4 + v2, y1 = s4 - v2, y2 = s7 - v3, y3 = s7 + v3;
        long w0 = (long)y0 * Vp9CospiConstants.Cospi28_64 + (long)y3 * Vp9CospiConstants.Cospi4_64;
        long w1 = (long)y1 * Vp9CospiConstants.Cospi12_64 + (long)y2 * Vp9CospiConstants.Cospi20_64;
        long w2 = (long)y2 * Vp9CospiConstants.Cospi12_64 + (long)y1 * (-Vp9CospiConstants.Cospi20_64);
        long w3 = (long)y3 * Vp9CospiConstants.Cospi28_64 + (long)y0 * (-Vp9CospiConstants.Cospi4_64);
        outRow[2] = Vp9CospiConstants.RoundShift(w0);
        outRow[6] = Vp9CospiConstants.RoundShift(w2);
        outRow[10] = Vp9CospiConstants.RoundShift(w1);
        outRow[14] = Vp9CospiConstants.RoundShift(w3);

        // Odd half: high-frequency butterfly using step1[0..7].
        // Stage 2
        long temp1 = (long)(step1[5] - step1[2]) * Vp9CospiConstants.Cospi16_64;
        long temp2 = (long)(step1[4] - step1[3]) * Vp9CospiConstants.Cospi16_64;
        step2[2] = Vp9CospiConstants.RoundShift(temp1);
        step2[3] = Vp9CospiConstants.RoundShift(temp2);
        temp1 = (long)(step1[4] + step1[3]) * Vp9CospiConstants.Cospi16_64;
        temp2 = (long)(step1[5] + step1[2]) * Vp9CospiConstants.Cospi16_64;
        step2[4] = Vp9CospiConstants.RoundShift(temp1);
        step2[5] = Vp9CospiConstants.RoundShift(temp2);

        // Stage 3
        step3[0] = step1[0] + step2[3];
        step3[1] = step1[1] + step2[2];
        step3[2] = step1[1] - step2[2];
        step3[3] = step1[0] - step2[3];
        step3[4] = step1[7] - step2[4];
        step3[5] = step1[6] - step2[5];
        step3[6] = step1[6] + step2[5];
        step3[7] = step1[7] + step2[4];

        // Stage 4
        temp1 = (long)step3[1] * (-Vp9CospiConstants.Cospi8_64) + (long)step3[6] * Vp9CospiConstants.Cospi24_64;
        temp2 = (long)step3[2] * Vp9CospiConstants.Cospi24_64 + (long)step3[5] * Vp9CospiConstants.Cospi8_64;
        step2[1] = Vp9CospiConstants.RoundShift(temp1);
        step2[2] = Vp9CospiConstants.RoundShift(temp2);
        temp1 = (long)step3[2] * Vp9CospiConstants.Cospi8_64 - (long)step3[5] * Vp9CospiConstants.Cospi24_64;
        temp2 = (long)step3[1] * Vp9CospiConstants.Cospi24_64 + (long)step3[6] * Vp9CospiConstants.Cospi8_64;
        step2[5] = Vp9CospiConstants.RoundShift(temp1);
        step2[6] = Vp9CospiConstants.RoundShift(temp2);

        // Stage 5
        step1[0] = step3[0] + step2[1];
        step1[1] = step3[0] - step2[1];
        step1[2] = step3[3] + step2[2];
        step1[3] = step3[3] - step2[2];
        step1[4] = step3[4] - step2[5];
        step1[5] = step3[4] + step2[5];
        step1[6] = step3[7] - step2[6];
        step1[7] = step3[7] + step2[6];

        // Stage 6
        temp1 = (long)step1[0] * Vp9CospiConstants.Cospi30_64 + (long)step1[7] * Vp9CospiConstants.Cospi2_64;
        temp2 = (long)step1[1] * Vp9CospiConstants.Cospi14_64 + (long)step1[6] * Vp9CospiConstants.Cospi18_64;
        outRow[1] = Vp9CospiConstants.RoundShift(temp1);
        outRow[9] = Vp9CospiConstants.RoundShift(temp2);
        temp1 = (long)step1[2] * Vp9CospiConstants.Cospi22_64 + (long)step1[5] * Vp9CospiConstants.Cospi10_64;
        temp2 = (long)step1[3] * Vp9CospiConstants.Cospi6_64 + (long)step1[4] * Vp9CospiConstants.Cospi26_64;
        outRow[5] = Vp9CospiConstants.RoundShift(temp1);
        outRow[13] = Vp9CospiConstants.RoundShift(temp2);
        temp1 = (long)step1[3] * (-Vp9CospiConstants.Cospi26_64) + (long)step1[4] * Vp9CospiConstants.Cospi6_64;
        temp2 = (long)step1[2] * (-Vp9CospiConstants.Cospi10_64) + (long)step1[5] * Vp9CospiConstants.Cospi22_64;
        outRow[3] = Vp9CospiConstants.RoundShift(temp1);
        outRow[11] = Vp9CospiConstants.RoundShift(temp2);
        temp1 = (long)step1[1] * (-Vp9CospiConstants.Cospi18_64) + (long)step1[6] * Vp9CospiConstants.Cospi14_64;
        temp2 = (long)step1[0] * (-Vp9CospiConstants.Cospi2_64) + (long)step1[7] * Vp9CospiConstants.Cospi30_64;
        outRow[7] = Vp9CospiConstants.RoundShift(temp1);
        outRow[15] = Vp9CospiConstants.RoundShift(temp2);
    }
}
