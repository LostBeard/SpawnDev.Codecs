// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 inverse DCT 32x32 - bit-exact CPU oracle for libvpx idct32_c +
// vp9_idct32x32_1024_add_c. Largest square transform VP9 uses for luma.
//
// Spec: VP9 Bitstream Specification sec 8.7.1.?? (32x32 DCT)
// libvpx: vpx_dsp/inv_txfm.c (idct32_c) + vp9/common/vp9_idct.c.
// Both lifted verbatim for bit-exactness; stage-by-stage layout kept
// identical to upstream so a reviewer can line up the two side-by-side.
//
// Structure
//   - 8-stage butterfly: 7 named stages + a final stage that combines
//     into the 32-element output.
//   - Every Q14 cosine constant from the 64-sample table is used
//     (cospi_1_64 through cospi_31_64, all odd + even positions).
//   - Final round is (x + 32) >> 6 - same as the 16x16 transform.
//   - Input reordering on stage 1 mirrors libvpx:
//     0,16,8,24,4,20,12,28,2,18,10,26,6,22,14,30 for the "even half"
//     (slots 0..15 of step1). The 16 odd inputs go through pre-rotation
//     into slots 16..31 before stage 2.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// CPU oracle for VP9 inverse DCT 32x32. Kernel equivalent lands once
/// SpawnDev.ILGPU's LocalMemory IR fix rc ships.
/// </summary>
public static class Vp9Idct32x32Reference
{
    // Q14 cosine table, exact integer values per libvpx vpx_dsp/txfm_common.h.
    private const int CosPi1_64 = 16364;
    private const int CosPi2_64 = 16305;
    private const int CosPi3_64 = 16207;
    private const int CosPi4_64 = 16069;
    private const int CosPi5_64 = 15893;
    private const int CosPi6_64 = 15679;
    private const int CosPi7_64 = 15426;
    private const int CosPi8_64 = 15137;
    private const int CosPi9_64 = 14811;
    private const int CosPi10_64 = 14449;
    private const int CosPi11_64 = 14053;
    private const int CosPi12_64 = 13623;
    private const int CosPi13_64 = 13160;
    private const int CosPi14_64 = 12665;
    private const int CosPi15_64 = 12140;
    private const int CosPi16_64 = 11585;
    private const int CosPi17_64 = 11003;
    private const int CosPi18_64 = 10394;
    private const int CosPi19_64 = 9760;
    private const int CosPi20_64 = 9102;
    private const int CosPi21_64 = 8423;
    private const int CosPi22_64 = 7723;
    private const int CosPi23_64 = 7005;
    private const int CosPi24_64 = 6270;
    private const int CosPi25_64 = 5520;
    private const int CosPi26_64 = 4756;
    private const int CosPi27_64 = 3981;
    private const int CosPi28_64 = 3196;
    private const int CosPi29_64 = 2404;
    private const int CosPi30_64 = 1606;
    private const int CosPi31_64 = 804;

    /// <summary>
    /// Apply <paramref name="input"/> (1024 coefficients, row-major 32x32)
    /// as a residual to <paramref name="dest"/> (32x32 block of 8-bit
    /// pixels with <paramref name="stride"/> bytes per row).
    /// Matches libvpx <c>vp9_idct32x32_1024_add</c> bit-exactly.
    /// </summary>
    public static void Idct32x32_1024_Add(
        ReadOnlySpan<short> input, Span<byte> dest, int stride)
    {
        if (input.Length < 1024)
            throw new ArgumentException("input must have >= 1024 coefficients", nameof(input));
        if (stride < 32)
            throw new ArgumentException("stride must be >= 32", nameof(stride));
        if (dest.Length < 31 * stride + 32)
            throw new ArgumentException("dest too small for 32 rows at the given stride", nameof(dest));

        // 32x32 int16 intermediate is 2 KiB - heap-allocated rather than
        // stackalloc to stay well under the 1 MiB stack budget consumers
        // may have in constrained hosts (Blazor WASM has a small stack).
        var tmp = new short[1024];

        Span<short> rowScratch = stackalloc short[32];
        Span<short> colScratch = stackalloc short[32];
        Span<short> colOut = stackalloc short[32];

        // Row pass.
        for (int row = 0; row < 32; row++)
        {
            Idct32_1d(input.Slice(row * 32, 32), tmp.AsSpan(row * 32, 32));
        }

        // Column pass + final round + residual-add + pixel clip.
        for (int col = 0; col < 32; col++)
        {
            for (int j = 0; j < 32; j++) colScratch[j] = tmp[j * 32 + col];
            Idct32_1d(colScratch, colOut);
            for (int j = 0; j < 32; j++)
            {
                int residual = (colOut[j] + 32) >> 6;
                int predictor = dest[j * stride + col];
                int sum = predictor + residual;
                if (sum < 0) sum = 0;
                else if (sum > 255) sum = 255;
                dest[j * stride + col] = (byte)sum;
            }
        }
        _ = rowScratch; // kept for symmetry with smaller transforms; unused here
    }

    /// <summary>
    /// One-dimensional 32-point iDCT butterfly, bit-exact against
    /// libvpx <c>idct32_c</c>. 8 effective stages.
    /// </summary>
    private static void Idct32_1d(ReadOnlySpan<short> input, Span<short> output)
    {
        Span<short> step1 = stackalloc short[32];
        Span<short> step2 = stackalloc short[32];

        // Stage 1: reorder even-indexed inputs into slots 0..15;
        // pre-rotate odd-indexed inputs into slots 16..31.
        step1[0] = input[0];
        step1[1] = input[16];
        step1[2] = input[8];
        step1[3] = input[24];
        step1[4] = input[4];
        step1[5] = input[20];
        step1[6] = input[12];
        step1[7] = input[28];
        step1[8] = input[2];
        step1[9] = input[18];
        step1[10] = input[10];
        step1[11] = input[26];
        step1[12] = input[6];
        step1[13] = input[22];
        step1[14] = input[14];
        step1[15] = input[30];

        Rotate(input[1], input[31], CosPi31_64, CosPi1_64, out step1[16], out step1[31]);
        Rotate(input[17], input[15], CosPi15_64, CosPi17_64, out step1[17], out step1[30]);
        Rotate(input[9], input[23], CosPi23_64, CosPi9_64, out step1[18], out step1[29]);
        Rotate(input[25], input[7], CosPi7_64, CosPi25_64, out step1[19], out step1[28]);
        Rotate(input[5], input[27], CosPi27_64, CosPi5_64, out step1[20], out step1[27]);
        Rotate(input[21], input[11], CosPi11_64, CosPi21_64, out step1[21], out step1[26]);
        Rotate(input[13], input[19], CosPi19_64, CosPi13_64, out step1[22], out step1[25]);
        Rotate(input[29], input[3], CosPi3_64, CosPi29_64, out step1[23], out step1[24]);

        // Stage 2.
        for (int i = 0; i < 8; i++) step2[i] = step1[i];
        Rotate(step1[8], step1[15], CosPi30_64, CosPi2_64, out step2[8], out step2[15]);
        Rotate(step1[9], step1[14], CosPi14_64, CosPi18_64, out step2[9], out step2[14]);
        Rotate(step1[10], step1[13], CosPi22_64, CosPi10_64, out step2[10], out step2[13]);
        Rotate(step1[11], step1[12], CosPi6_64, CosPi26_64, out step2[11], out step2[12]);

        step2[16] = (short)(step1[16] + step1[17]);
        step2[17] = (short)(step1[16] - step1[17]);
        step2[18] = (short)(-step1[18] + step1[19]);
        step2[19] = (short)(step1[18] + step1[19]);
        step2[20] = (short)(step1[20] + step1[21]);
        step2[21] = (short)(step1[20] - step1[21]);
        step2[22] = (short)(-step1[22] + step1[23]);
        step2[23] = (short)(step1[22] + step1[23]);
        step2[24] = (short)(step1[24] + step1[25]);
        step2[25] = (short)(step1[24] - step1[25]);
        step2[26] = (short)(-step1[26] + step1[27]);
        step2[27] = (short)(step1[26] + step1[27]);
        step2[28] = (short)(step1[28] + step1[29]);
        step2[29] = (short)(step1[28] - step1[29]);
        step2[30] = (short)(-step1[30] + step1[31]);
        step2[31] = (short)(step1[30] + step1[31]);

        // Stage 3.
        for (int i = 0; i < 4; i++) step1[i] = step2[i];
        Rotate(step2[4], step2[7], CosPi28_64, CosPi4_64, out step1[4], out step1[7]);
        Rotate(step2[5], step2[6], CosPi12_64, CosPi20_64, out step1[5], out step1[6]);
        step1[8] = (short)(step2[8] + step2[9]);
        step1[9] = (short)(step2[8] - step2[9]);
        step1[10] = (short)(-step2[10] + step2[11]);
        step1[11] = (short)(step2[10] + step2[11]);
        step1[12] = (short)(step2[12] + step2[13]);
        step1[13] = (short)(step2[12] - step2[13]);
        step1[14] = (short)(-step2[14] + step2[15]);
        step1[15] = (short)(step2[14] + step2[15]);

        step1[16] = step2[16];
        step1[31] = step2[31];
        {
            int t1 = -step2[17] * CosPi4_64 + step2[30] * CosPi28_64;
            int t2 = step2[17] * CosPi28_64 + step2[30] * CosPi4_64;
            step1[17] = Rs14(t1);
            step1[30] = Rs14(t2);
        }
        {
            int t1 = -step2[18] * CosPi28_64 - step2[29] * CosPi4_64;
            int t2 = -step2[18] * CosPi4_64 + step2[29] * CosPi28_64;
            step1[18] = Rs14(t1);
            step1[29] = Rs14(t2);
        }
        step1[19] = step2[19];
        step1[20] = step2[20];
        {
            int t1 = -step2[21] * CosPi20_64 + step2[26] * CosPi12_64;
            int t2 = step2[21] * CosPi12_64 + step2[26] * CosPi20_64;
            step1[21] = Rs14(t1);
            step1[26] = Rs14(t2);
        }
        {
            int t1 = -step2[22] * CosPi12_64 - step2[25] * CosPi20_64;
            int t2 = -step2[22] * CosPi20_64 + step2[25] * CosPi12_64;
            step1[22] = Rs14(t1);
            step1[25] = Rs14(t2);
        }
        step1[23] = step2[23];
        step1[24] = step2[24];
        step1[27] = step2[27];
        step1[28] = step2[28];

        // Stage 4.
        {
            int t1 = (step1[0] + step1[1]) * CosPi16_64;
            int t2 = (step1[0] - step1[1]) * CosPi16_64;
            step2[0] = Rs14(t1);
            step2[1] = Rs14(t2);
        }
        Rotate(step1[2], step1[3], CosPi24_64, CosPi8_64, out step2[2], out step2[3]);
        step2[4] = (short)(step1[4] + step1[5]);
        step2[5] = (short)(step1[4] - step1[5]);
        step2[6] = (short)(-step1[6] + step1[7]);
        step2[7] = (short)(step1[6] + step1[7]);

        step2[8] = step1[8];
        step2[15] = step1[15];
        {
            int t1 = -step1[9] * CosPi8_64 + step1[14] * CosPi24_64;
            int t2 = step1[9] * CosPi24_64 + step1[14] * CosPi8_64;
            step2[9] = Rs14(t1);
            step2[14] = Rs14(t2);
        }
        {
            int t1 = -step1[10] * CosPi24_64 - step1[13] * CosPi8_64;
            int t2 = -step1[10] * CosPi8_64 + step1[13] * CosPi24_64;
            step2[10] = Rs14(t1);
            step2[13] = Rs14(t2);
        }
        step2[11] = step1[11];
        step2[12] = step1[12];

        step2[16] = (short)(step1[16] + step1[19]);
        step2[17] = (short)(step1[17] + step1[18]);
        step2[18] = (short)(step1[17] - step1[18]);
        step2[19] = (short)(step1[16] - step1[19]);
        step2[20] = (short)(-step1[20] + step1[23]);
        step2[21] = (short)(-step1[21] + step1[22]);
        step2[22] = (short)(step1[21] + step1[22]);
        step2[23] = (short)(step1[20] + step1[23]);
        step2[24] = (short)(step1[24] + step1[27]);
        step2[25] = (short)(step1[25] + step1[26]);
        step2[26] = (short)(step1[25] - step1[26]);
        step2[27] = (short)(step1[24] - step1[27]);
        step2[28] = (short)(-step1[28] + step1[31]);
        step2[29] = (short)(-step1[29] + step1[30]);
        step2[30] = (short)(step1[29] + step1[30]);
        step2[31] = (short)(step1[28] + step1[31]);

        // Stage 5.
        step1[0] = (short)(step2[0] + step2[3]);
        step1[1] = (short)(step2[1] + step2[2]);
        step1[2] = (short)(step2[1] - step2[2]);
        step1[3] = (short)(step2[0] - step2[3]);
        step1[4] = step2[4];
        {
            int t1 = (step2[6] - step2[5]) * CosPi16_64;
            int t2 = (step2[5] + step2[6]) * CosPi16_64;
            step1[5] = Rs14(t1);
            step1[6] = Rs14(t2);
        }
        step1[7] = step2[7];

        step1[8] = (short)(step2[8] + step2[11]);
        step1[9] = (short)(step2[9] + step2[10]);
        step1[10] = (short)(step2[9] - step2[10]);
        step1[11] = (short)(step2[8] - step2[11]);
        step1[12] = (short)(-step2[12] + step2[15]);
        step1[13] = (short)(-step2[13] + step2[14]);
        step1[14] = (short)(step2[13] + step2[14]);
        step1[15] = (short)(step2[12] + step2[15]);

        step1[16] = step2[16];
        step1[17] = step2[17];
        {
            int t1 = -step2[18] * CosPi8_64 + step2[29] * CosPi24_64;
            int t2 = step2[18] * CosPi24_64 + step2[29] * CosPi8_64;
            step1[18] = Rs14(t1);
            step1[29] = Rs14(t2);
        }
        {
            int t1 = -step2[19] * CosPi8_64 + step2[28] * CosPi24_64;
            int t2 = step2[19] * CosPi24_64 + step2[28] * CosPi8_64;
            step1[19] = Rs14(t1);
            step1[28] = Rs14(t2);
        }
        {
            int t1 = -step2[20] * CosPi24_64 - step2[27] * CosPi8_64;
            int t2 = -step2[20] * CosPi8_64 + step2[27] * CosPi24_64;
            step1[20] = Rs14(t1);
            step1[27] = Rs14(t2);
        }
        {
            int t1 = -step2[21] * CosPi24_64 - step2[26] * CosPi8_64;
            int t2 = -step2[21] * CosPi8_64 + step2[26] * CosPi24_64;
            step1[21] = Rs14(t1);
            step1[26] = Rs14(t2);
        }
        step1[22] = step2[22];
        step1[23] = step2[23];
        step1[24] = step2[24];
        step1[25] = step2[25];
        step1[30] = step2[30];
        step1[31] = step2[31];

        // Stage 6.
        step2[0] = (short)(step1[0] + step1[7]);
        step2[1] = (short)(step1[1] + step1[6]);
        step2[2] = (short)(step1[2] + step1[5]);
        step2[3] = (short)(step1[3] + step1[4]);
        step2[4] = (short)(step1[3] - step1[4]);
        step2[5] = (short)(step1[2] - step1[5]);
        step2[6] = (short)(step1[1] - step1[6]);
        step2[7] = (short)(step1[0] - step1[7]);
        step2[8] = step1[8];
        step2[9] = step1[9];
        {
            int t1 = (-step1[10] + step1[13]) * CosPi16_64;
            int t2 = (step1[10] + step1[13]) * CosPi16_64;
            step2[10] = Rs14(t1);
            step2[13] = Rs14(t2);
        }
        {
            int t1 = (-step1[11] + step1[12]) * CosPi16_64;
            int t2 = (step1[11] + step1[12]) * CosPi16_64;
            step2[11] = Rs14(t1);
            step2[12] = Rs14(t2);
        }
        step2[14] = step1[14];
        step2[15] = step1[15];

        step2[16] = (short)(step1[16] + step1[23]);
        step2[17] = (short)(step1[17] + step1[22]);
        step2[18] = (short)(step1[18] + step1[21]);
        step2[19] = (short)(step1[19] + step1[20]);
        step2[20] = (short)(step1[19] - step1[20]);
        step2[21] = (short)(step1[18] - step1[21]);
        step2[22] = (short)(step1[17] - step1[22]);
        step2[23] = (short)(step1[16] - step1[23]);
        step2[24] = (short)(-step1[24] + step1[31]);
        step2[25] = (short)(-step1[25] + step1[30]);
        step2[26] = (short)(-step1[26] + step1[29]);
        step2[27] = (short)(-step1[27] + step1[28]);
        step2[28] = (short)(step1[27] + step1[28]);
        step2[29] = (short)(step1[26] + step1[29]);
        step2[30] = (short)(step1[25] + step1[30]);
        step2[31] = (short)(step1[24] + step1[31]);

        // Stage 7.
        step1[0] = (short)(step2[0] + step2[15]);
        step1[1] = (short)(step2[1] + step2[14]);
        step1[2] = (short)(step2[2] + step2[13]);
        step1[3] = (short)(step2[3] + step2[12]);
        step1[4] = (short)(step2[4] + step2[11]);
        step1[5] = (short)(step2[5] + step2[10]);
        step1[6] = (short)(step2[6] + step2[9]);
        step1[7] = (short)(step2[7] + step2[8]);
        step1[8] = (short)(step2[7] - step2[8]);
        step1[9] = (short)(step2[6] - step2[9]);
        step1[10] = (short)(step2[5] - step2[10]);
        step1[11] = (short)(step2[4] - step2[11]);
        step1[12] = (short)(step2[3] - step2[12]);
        step1[13] = (short)(step2[2] - step2[13]);
        step1[14] = (short)(step2[1] - step2[14]);
        step1[15] = (short)(step2[0] - step2[15]);

        step1[16] = step2[16];
        step1[17] = step2[17];
        step1[18] = step2[18];
        step1[19] = step2[19];
        {
            int t1 = (-step2[20] + step2[27]) * CosPi16_64;
            int t2 = (step2[20] + step2[27]) * CosPi16_64;
            step1[20] = Rs14(t1);
            step1[27] = Rs14(t2);
        }
        {
            int t1 = (-step2[21] + step2[26]) * CosPi16_64;
            int t2 = (step2[21] + step2[26]) * CosPi16_64;
            step1[21] = Rs14(t1);
            step1[26] = Rs14(t2);
        }
        {
            int t1 = (-step2[22] + step2[25]) * CosPi16_64;
            int t2 = (step2[22] + step2[25]) * CosPi16_64;
            step1[22] = Rs14(t1);
            step1[25] = Rs14(t2);
        }
        {
            int t1 = (-step2[23] + step2[24]) * CosPi16_64;
            int t2 = (step2[23] + step2[24]) * CosPi16_64;
            step1[23] = Rs14(t1);
            step1[24] = Rs14(t2);
        }
        step1[28] = step2[28];
        step1[29] = step2[29];
        step1[30] = step2[30];
        step1[31] = step2[31];

        // Final stage: 32-way combining butterfly into the output.
        output[0] = (short)(step1[0] + step1[31]);
        output[1] = (short)(step1[1] + step1[30]);
        output[2] = (short)(step1[2] + step1[29]);
        output[3] = (short)(step1[3] + step1[28]);
        output[4] = (short)(step1[4] + step1[27]);
        output[5] = (short)(step1[5] + step1[26]);
        output[6] = (short)(step1[6] + step1[25]);
        output[7] = (short)(step1[7] + step1[24]);
        output[8] = (short)(step1[8] + step1[23]);
        output[9] = (short)(step1[9] + step1[22]);
        output[10] = (short)(step1[10] + step1[21]);
        output[11] = (short)(step1[11] + step1[20]);
        output[12] = (short)(step1[12] + step1[19]);
        output[13] = (short)(step1[13] + step1[18]);
        output[14] = (short)(step1[14] + step1[17]);
        output[15] = (short)(step1[15] + step1[16]);
        output[16] = (short)(step1[15] - step1[16]);
        output[17] = (short)(step1[14] - step1[17]);
        output[18] = (short)(step1[13] - step1[18]);
        output[19] = (short)(step1[12] - step1[19]);
        output[20] = (short)(step1[11] - step1[20]);
        output[21] = (short)(step1[10] - step1[21]);
        output[22] = (short)(step1[9] - step1[22]);
        output[23] = (short)(step1[8] - step1[23]);
        output[24] = (short)(step1[7] - step1[24]);
        output[25] = (short)(step1[6] - step1[25]);
        output[26] = (short)(step1[5] - step1[26]);
        output[27] = (short)(step1[4] - step1[27]);
        output[28] = (short)(step1[3] - step1[28]);
        output[29] = (short)(step1[2] - step1[29]);
        output[30] = (short)(step1[1] - step1[30]);
        output[31] = (short)(step1[0] - step1[31]);
    }

    /// <summary>
    /// Standard 2x2 rotation butterfly: output a = a*c1 - b*c2 (rounded);
    /// output b = a*c2 + b*c1 (rounded). Q14 rounding. Matches the libvpx
    /// `temp1/temp2/dct_const_round_shift` pair pattern.
    /// </summary>
    private static void Rotate(
        short a, short b, int cosA, int cosB,
        out short outA, out short outB)
    {
        int t1 = a * cosA - b * cosB;
        int t2 = a * cosB + b * cosA;
        outA = Rs14(t1);
        outB = Rs14(t2);
    }

    private static short Rs14(int value) => (short)((value + (1 << 13)) >> 14);
}
