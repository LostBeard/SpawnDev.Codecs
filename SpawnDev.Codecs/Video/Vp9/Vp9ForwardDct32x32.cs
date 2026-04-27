// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 32x32 forward DCT. Bit-exact port of libvpx vpx_dsp/fwd_txfm.c
// vpx_fdct32x32_c (high-precision variant, round=0 between passes,
// round=0 stays in line with the encoder default; the wrapper itself
// applies a separate half_round_shift between passes).
//
// Two-pass transform, 7 stages each pass, with cospi multiplications
// at stages 2, 3, 4, 5, 6 and 7 plus a bit-reversed final permutation.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 32x32 forward DCT (encoder side).</summary>
public static class Vp9ForwardDct32x32
{
    /// <summary>32x32 forward DCT. Mirrors libvpx <c>vpx_fdct32x32_c</c>.</summary>
    public static void Transform(ReadOnlySpan<short> input, int rowStrideShorts, Span<int> output)
    {
        if (input.Length < rowStrideShorts * 32)
            throw new ArgumentException($"input must hold at least {rowStrideShorts * 32} entries", nameof(input));
        if (output.Length < 1024)
            throw new ArgumentException("output must hold 1024 entries", nameof(output));

        Span<long> intermediate = stackalloc long[1024];

        // Pass 1 (columns), input *4 then half_round_shift between passes.
        Span<long> tempIn = stackalloc long[32];
        Span<long> tempOut = stackalloc long[32];
        for (int i = 0; i < 32; i++)
        {
            for (int j = 0; j < 32; j++) tempIn[j] = input[j * rowStrideShorts + i] * 4L;
            Fdct32(tempIn, tempOut, round: false);
            for (int j = 0; j < 32; j++)
                intermediate[j * 32 + i] = HalfRoundShift(tempOut[j]);
        }

        // Pass 2 (rows), no extra round between final pass and output.
        for (int i = 0; i < 32; i++)
        {
            for (int j = 0; j < 32; j++) tempIn[j] = intermediate[j + i * 32];
            Fdct32(tempIn, tempOut, round: false);
            for (int j = 0; j < 32; j++) output[j + i * 32] = (int)tempOut[j];
        }
    }

    /// <summary>libvpx <c>half_round_shift</c>: <c>(input + 1 + (input&lt;0)) &gt;&gt; 2</c>.</summary>
    private static long HalfRoundShift(long input) =>
        (input + 1 + (input < 0 ? 1 : 0)) >> 2;

    /// <summary>libvpx <c>dct_32_round</c>: same as <c>fdct_round_shift</c> (rounded right shift by 14).</summary>
    private static long DctRound(long input) =>
        (input + Vp9CospiConstants.DctConstRounding) >> Vp9CospiConstants.DctConstBits;

    /// <summary>1D 32-point forward DCT. Mirrors libvpx <c>vpx_fdct32</c>.</summary>
    private static void Fdct32(ReadOnlySpan<long> input, Span<long> output, bool round)
    {
        Span<long> step = stackalloc long[32];

        // Stage 1
        step[0]  = input[0]  + input[31];
        step[1]  = input[1]  + input[30];
        step[2]  = input[2]  + input[29];
        step[3]  = input[3]  + input[28];
        step[4]  = input[4]  + input[27];
        step[5]  = input[5]  + input[26];
        step[6]  = input[6]  + input[25];
        step[7]  = input[7]  + input[24];
        step[8]  = input[8]  + input[23];
        step[9]  = input[9]  + input[22];
        step[10] = input[10] + input[21];
        step[11] = input[11] + input[20];
        step[12] = input[12] + input[19];
        step[13] = input[13] + input[18];
        step[14] = input[14] + input[17];
        step[15] = input[15] + input[16];
        step[16] = -input[16] + input[15];
        step[17] = -input[17] + input[14];
        step[18] = -input[18] + input[13];
        step[19] = -input[19] + input[12];
        step[20] = -input[20] + input[11];
        step[21] = -input[21] + input[10];
        step[22] = -input[22] + input[9];
        step[23] = -input[23] + input[8];
        step[24] = -input[24] + input[7];
        step[25] = -input[25] + input[6];
        step[26] = -input[26] + input[5];
        step[27] = -input[27] + input[4];
        step[28] = -input[28] + input[3];
        step[29] = -input[29] + input[2];
        step[30] = -input[30] + input[1];
        step[31] = -input[31] + input[0];

        // Stage 2
        output[0]  = step[0]  + step[15];
        output[1]  = step[1]  + step[14];
        output[2]  = step[2]  + step[13];
        output[3]  = step[3]  + step[12];
        output[4]  = step[4]  + step[11];
        output[5]  = step[5]  + step[10];
        output[6]  = step[6]  + step[9];
        output[7]  = step[7]  + step[8];
        output[8]  = -step[8]  + step[7];
        output[9]  = -step[9]  + step[6];
        output[10] = -step[10] + step[5];
        output[11] = -step[11] + step[4];
        output[12] = -step[12] + step[3];
        output[13] = -step[13] + step[2];
        output[14] = -step[14] + step[1];
        output[15] = -step[15] + step[0];

        output[16] = step[16];
        output[17] = step[17];
        output[18] = step[18];
        output[19] = step[19];

        long c16 = Vp9CospiConstants.Cospi16_64;
        output[20] = DctRound((-step[20] + step[27]) * c16);
        output[21] = DctRound((-step[21] + step[26]) * c16);
        output[22] = DctRound((-step[22] + step[25]) * c16);
        output[23] = DctRound((-step[23] + step[24]) * c16);
        output[24] = DctRound((step[24] + step[23]) * c16);
        output[25] = DctRound((step[25] + step[22]) * c16);
        output[26] = DctRound((step[26] + step[21]) * c16);
        output[27] = DctRound((step[27] + step[20]) * c16);

        output[28] = step[28];
        output[29] = step[29];
        output[30] = step[30];
        output[31] = step[31];

        if (round)
        {
            for (int k = 0; k < 32; k++) output[k] = HalfRoundShift(output[k]);
        }

        // Stage 3
        step[0] = output[0] + output[7];
        step[1] = output[1] + output[6];
        step[2] = output[2] + output[5];
        step[3] = output[3] + output[4];
        step[4] = -output[4] + output[3];
        step[5] = -output[5] + output[2];
        step[6] = -output[6] + output[1];
        step[7] = -output[7] + output[0];
        step[8] = output[8];
        step[9] = output[9];
        step[10] = DctRound((-output[10] + output[13]) * c16);
        step[11] = DctRound((-output[11] + output[12]) * c16);
        step[12] = DctRound((output[12] + output[11]) * c16);
        step[13] = DctRound((output[13] + output[10]) * c16);
        step[14] = output[14];
        step[15] = output[15];

        step[16] = output[16] + output[23];
        step[17] = output[17] + output[22];
        step[18] = output[18] + output[21];
        step[19] = output[19] + output[20];
        step[20] = -output[20] + output[19];
        step[21] = -output[21] + output[18];
        step[22] = -output[22] + output[17];
        step[23] = -output[23] + output[16];
        step[24] = -output[24] + output[31];
        step[25] = -output[25] + output[30];
        step[26] = -output[26] + output[29];
        step[27] = -output[27] + output[28];
        step[28] = output[28] + output[27];
        step[29] = output[29] + output[26];
        step[30] = output[30] + output[25];
        step[31] = output[31] + output[24];

        // Stage 4
        long c8  = Vp9CospiConstants.Cospi8_64;
        long c24 = Vp9CospiConstants.Cospi24_64;

        output[0] = step[0] + step[3];
        output[1] = step[1] + step[2];
        output[2] = -step[2] + step[1];
        output[3] = -step[3] + step[0];
        output[4] = step[4];
        output[5] = DctRound((-step[5] + step[6]) * c16);
        output[6] = DctRound((step[6] + step[5]) * c16);
        output[7] = step[7];
        output[8] = step[8] + step[11];
        output[9] = step[9] + step[10];
        output[10] = -step[10] + step[9];
        output[11] = -step[11] + step[8];
        output[12] = -step[12] + step[15];
        output[13] = -step[13] + step[14];
        output[14] = step[14] + step[13];
        output[15] = step[15] + step[12];

        output[16] = step[16];
        output[17] = step[17];
        output[18] = DctRound(step[18] * -c8 + step[29] * c24);
        output[19] = DctRound(step[19] * -c8 + step[28] * c24);
        output[20] = DctRound(step[20] * -c24 + step[27] * -c8);
        output[21] = DctRound(step[21] * -c24 + step[26] * -c8);
        output[22] = step[22];
        output[23] = step[23];
        output[24] = step[24];
        output[25] = step[25];
        output[26] = DctRound(step[26] * c24 + step[21] * -c8);
        output[27] = DctRound(step[27] * c24 + step[20] * -c8);
        output[28] = DctRound(step[28] * c8 + step[19] * c24);
        output[29] = DctRound(step[29] * c8 + step[18] * c24);
        output[30] = step[30];
        output[31] = step[31];

        // Stage 5
        step[0] = DctRound((output[0] + output[1]) * c16);
        step[1] = DctRound((-output[1] + output[0]) * c16);
        step[2] = DctRound(output[2] * c24 + output[3] * c8);
        step[3] = DctRound(output[3] * c24 - output[2] * c8);
        step[4] = output[4] + output[5];
        step[5] = -output[5] + output[4];
        step[6] = -output[6] + output[7];
        step[7] = output[7] + output[6];
        step[8] = output[8];
        step[9] = DctRound(output[9] * -c8 + output[14] * c24);
        step[10] = DctRound(output[10] * -c24 + output[13] * -c8);
        step[11] = output[11];
        step[12] = output[12];
        step[13] = DctRound(output[13] * c24 + output[10] * -c8);
        step[14] = DctRound(output[14] * c8 + output[9] * c24);
        step[15] = output[15];

        step[16] = output[16] + output[19];
        step[17] = output[17] + output[18];
        step[18] = -output[18] + output[17];
        step[19] = -output[19] + output[16];
        step[20] = -output[20] + output[23];
        step[21] = -output[21] + output[22];
        step[22] = output[22] + output[21];
        step[23] = output[23] + output[20];
        step[24] = output[24] + output[27];
        step[25] = output[25] + output[26];
        step[26] = -output[26] + output[25];
        step[27] = -output[27] + output[24];
        step[28] = -output[28] + output[31];
        step[29] = -output[29] + output[30];
        step[30] = output[30] + output[29];
        step[31] = output[31] + output[28];

        // Stage 6
        long c4  = Vp9CospiConstants.Cospi4_64;
        long c28 = Vp9CospiConstants.Cospi28_64;
        long c12 = Vp9CospiConstants.Cospi12_64;
        long c20 = Vp9CospiConstants.Cospi20_64;

        output[0] = step[0];
        output[1] = step[1];
        output[2] = step[2];
        output[3] = step[3];
        output[4] = DctRound(step[4] * c28 + step[7] * c4);
        output[5] = DctRound(step[5] * c12 + step[6] * c20);
        output[6] = DctRound(step[6] * c12 + step[5] * -c20);
        output[7] = DctRound(step[7] * c28 + step[4] * -c4);
        output[8] = step[8] + step[9];
        output[9] = -step[9] + step[8];
        output[10] = -step[10] + step[11];
        output[11] = step[11] + step[10];
        output[12] = step[12] + step[13];
        output[13] = -step[13] + step[12];
        output[14] = -step[14] + step[15];
        output[15] = step[15] + step[14];

        output[16] = step[16];
        output[17] = DctRound(step[17] * -c4 + step[30] * c28);
        output[18] = DctRound(step[18] * -c28 + step[29] * -c4);
        output[19] = step[19];
        output[20] = step[20];
        output[21] = DctRound(step[21] * -c20 + step[26] * c12);
        output[22] = DctRound(step[22] * -c12 + step[25] * -c20);
        output[23] = step[23];
        output[24] = step[24];
        output[25] = DctRound(step[25] * c12 + step[22] * -c20);
        output[26] = DctRound(step[26] * c20 + step[21] * c12);
        output[27] = step[27];
        output[28] = step[28];
        output[29] = DctRound(step[29] * c28 + step[18] * -c4);
        output[30] = DctRound(step[30] * c4 + step[17] * c28);
        output[31] = step[31];

        // Stage 7
        long c2  = Vp9CospiConstants.Cospi2_64;
        long c30 = Vp9CospiConstants.Cospi30_64;
        long c14 = Vp9CospiConstants.Cospi14_64;
        long c18 = Vp9CospiConstants.Cospi18_64;
        long c10 = Vp9CospiConstants.Cospi10_64;
        long c22 = Vp9CospiConstants.Cospi22_64;
        long c26 = Vp9CospiConstants.Cospi26_64;
        long c6  = Vp9CospiConstants.Cospi6_64;

        step[0] = output[0];
        step[1] = output[1];
        step[2] = output[2];
        step[3] = output[3];
        step[4] = output[4];
        step[5] = output[5];
        step[6] = output[6];
        step[7] = output[7];
        step[8]  = DctRound(output[8]  * c30 + output[15] * c2);
        step[9]  = DctRound(output[9]  * c14 + output[14] * c18);
        step[10] = DctRound(output[10] * c22 + output[13] * c10);
        step[11] = DctRound(output[11] * c6  + output[12] * c26);
        step[12] = DctRound(output[12] * c6  + output[11] * -c26);
        step[13] = DctRound(output[13] * c22 + output[10] * -c10);
        step[14] = DctRound(output[14] * c14 + output[9]  * -c18);
        step[15] = DctRound(output[15] * c30 + output[8]  * -c2);

        step[16] = output[16] + output[17];
        step[17] = -output[17] + output[16];
        step[18] = -output[18] + output[19];
        step[19] = output[19] + output[18];
        step[20] = output[20] + output[21];
        step[21] = -output[21] + output[20];
        step[22] = -output[22] + output[23];
        step[23] = output[23] + output[22];
        step[24] = output[24] + output[25];
        step[25] = -output[25] + output[24];
        step[26] = -output[26] + output[27];
        step[27] = output[27] + output[26];
        step[28] = output[28] + output[29];
        step[29] = -output[29] + output[28];
        step[30] = -output[30] + output[31];
        step[31] = output[31] + output[30];

        // Final stage --- bit-reversed output indices
        long c1  = Vp9CospiConstants.Cospi1_64;
        long c31 = Vp9CospiConstants.Cospi31_64;
        long c15 = Vp9CospiConstants.Cospi15_64;
        long c17 = Vp9CospiConstants.Cospi17_64;
        long c23 = Vp9CospiConstants.Cospi23_64;
        long c9  = Vp9CospiConstants.Cospi9_64;
        long c7  = Vp9CospiConstants.Cospi7_64;
        long c25 = Vp9CospiConstants.Cospi25_64;
        long c27 = Vp9CospiConstants.Cospi27_64;
        long c5  = Vp9CospiConstants.Cospi5_64;
        long c11 = Vp9CospiConstants.Cospi11_64;
        long c21 = Vp9CospiConstants.Cospi21_64;
        long c19 = Vp9CospiConstants.Cospi19_64;
        long c13 = Vp9CospiConstants.Cospi13_64;
        long c3  = Vp9CospiConstants.Cospi3_64;
        long c29 = Vp9CospiConstants.Cospi29_64;

        output[0]  = step[0];
        output[16] = step[1];
        output[8]  = step[2];
        output[24] = step[3];
        output[4]  = step[4];
        output[20] = step[5];
        output[12] = step[6];
        output[28] = step[7];
        output[2]  = step[8];
        output[18] = step[9];
        output[10] = step[10];
        output[26] = step[11];
        output[6]  = step[12];
        output[22] = step[13];
        output[14] = step[14];
        output[30] = step[15];

        output[1]  = DctRound(step[16] * c31 + step[31] * c1);
        output[17] = DctRound(step[17] * c15 + step[30] * c17);
        output[9]  = DctRound(step[18] * c23 + step[29] * c9);
        output[25] = DctRound(step[19] * c7  + step[28] * c25);
        output[5]  = DctRound(step[20] * c27 + step[27] * c5);
        output[21] = DctRound(step[21] * c11 + step[26] * c21);
        output[13] = DctRound(step[22] * c19 + step[25] * c13);
        output[29] = DctRound(step[23] * c3  + step[24] * c29);
        output[3]  = DctRound(step[24] * c3  + step[23] * -c29);
        output[19] = DctRound(step[25] * c19 + step[22] * -c13);
        output[11] = DctRound(step[26] * c11 + step[21] * -c21);
        output[27] = DctRound(step[27] * c27 + step[20] * -c5);
        output[7]  = DctRound(step[28] * c7  + step[19] * -c25);
        output[23] = DctRound(step[29] * c23 + step[18] * -c9);
        output[15] = DctRound(step[30] * c15 + step[17] * -c17);
        output[31] = DctRound(step[31] * c31 + step[16] * -c1);
    }
}
