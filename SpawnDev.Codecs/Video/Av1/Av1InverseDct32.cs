// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 32-point inverse DCT (1D). Bit-exact port of libaom
// av1/common/av1_inv_txfm1d.c av1_idct32.
//
// Upstream Copyright (c) 2016, Alliance for Open Media.
// Upstream license: BSD 2-Clause + AV1 Patent License 1.0. See NOTICE.md.
//
// Nine stages, each a butterfly + cospi-rotation network.
// half_btf and cospi_arr come from Av1ForwardDct4 (shared cospi table).

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>AV1 32-point inverse DCT (1D building block).</summary>
public static class Av1InverseDct32
{
    /// <summary>libaom default cos_bit for the inverse 32-point DCT.</summary>
    public const int DefaultCosBit = 12;

    /// <summary>
    /// 32-point inverse DCT. Mirrors libaom <c>av1_idct32</c>.
    /// </summary>
    public static void Transform(ReadOnlySpan<int> input, Span<int> output, int cosBit = DefaultCosBit)
    {
        if (input.Length < 32) throw new ArgumentException("input must have 32 entries", nameof(input));
        if (output.Length < 32) throw new ArgumentException("output must have 32 entries", nameof(output));
        if (cosBit < 10 || cosBit > 13) throw new ArgumentOutOfRangeException(nameof(cosBit), "must be in [10, 13]");

        int Btf(int w0, int in0, int w1, int in1) =>
            Av1ForwardDct4.HalfBtf(w0, in0, w1, in1, cosBit);
        var cospi = Av1ForwardDct4.CospiArr(cosBit);

        Span<int> bf0 = stackalloc int[32];
        Span<int> bf1 = stackalloc int[32];

        // Stage 1: input bit-reverse permutation.
        bf1[0] = input[0];
        bf1[1] = input[16];
        bf1[2] = input[8];
        bf1[3] = input[24];
        bf1[4] = input[4];
        bf1[5] = input[20];
        bf1[6] = input[12];
        bf1[7] = input[28];
        bf1[8] = input[2];
        bf1[9] = input[18];
        bf1[10] = input[10];
        bf1[11] = input[26];
        bf1[12] = input[6];
        bf1[13] = input[22];
        bf1[14] = input[14];
        bf1[15] = input[30];
        bf1[16] = input[1];
        bf1[17] = input[17];
        bf1[18] = input[9];
        bf1[19] = input[25];
        bf1[20] = input[5];
        bf1[21] = input[21];
        bf1[22] = input[13];
        bf1[23] = input[29];
        bf1[24] = input[3];
        bf1[25] = input[19];
        bf1[26] = input[11];
        bf1[27] = input[27];
        bf1[28] = input[7];
        bf1[29] = input[23];
        bf1[30] = input[15];
        bf1[31] = input[31];

        // Stage 2: rotate upper 16 (16..31).
        for (int i = 0; i < 16; i++) bf0[i] = bf1[i];
        bf0[16] = Btf(cospi[62], bf1[16], -cospi[2], bf1[31]);
        bf0[17] = Btf(cospi[30], bf1[17], -cospi[34], bf1[30]);
        bf0[18] = Btf(cospi[46], bf1[18], -cospi[18], bf1[29]);
        bf0[19] = Btf(cospi[14], bf1[19], -cospi[50], bf1[28]);
        bf0[20] = Btf(cospi[54], bf1[20], -cospi[10], bf1[27]);
        bf0[21] = Btf(cospi[22], bf1[21], -cospi[42], bf1[26]);
        bf0[22] = Btf(cospi[38], bf1[22], -cospi[26], bf1[25]);
        bf0[23] = Btf(cospi[6], bf1[23], -cospi[58], bf1[24]);
        bf0[24] = Btf(cospi[58], bf1[23], cospi[6], bf1[24]);
        bf0[25] = Btf(cospi[26], bf1[22], cospi[38], bf1[25]);
        bf0[26] = Btf(cospi[42], bf1[21], cospi[22], bf1[26]);
        bf0[27] = Btf(cospi[10], bf1[20], cospi[54], bf1[27]);
        bf0[28] = Btf(cospi[50], bf1[19], cospi[14], bf1[28]);
        bf0[29] = Btf(cospi[18], bf1[18], cospi[46], bf1[29]);
        bf0[30] = Btf(cospi[34], bf1[17], cospi[30], bf1[30]);
        bf0[31] = Btf(cospi[2], bf1[16], cospi[62], bf1[31]);

        // Stage 3: rotate middle (8..15), butterfly upper (16..31).
        for (int i = 0; i < 8; i++) bf1[i] = bf0[i];
        bf1[8] = Btf(cospi[60], bf0[8], -cospi[4], bf0[15]);
        bf1[9] = Btf(cospi[28], bf0[9], -cospi[36], bf0[14]);
        bf1[10] = Btf(cospi[44], bf0[10], -cospi[20], bf0[13]);
        bf1[11] = Btf(cospi[12], bf0[11], -cospi[52], bf0[12]);
        bf1[12] = Btf(cospi[52], bf0[11], cospi[12], bf0[12]);
        bf1[13] = Btf(cospi[20], bf0[10], cospi[44], bf0[13]);
        bf1[14] = Btf(cospi[36], bf0[9], cospi[28], bf0[14]);
        bf1[15] = Btf(cospi[4], bf0[8], cospi[60], bf0[15]);
        bf1[16] = bf0[16] + bf0[17];
        bf1[17] = bf0[16] - bf0[17];
        bf1[18] = -bf0[18] + bf0[19];
        bf1[19] = bf0[18] + bf0[19];
        bf1[20] = bf0[20] + bf0[21];
        bf1[21] = bf0[20] - bf0[21];
        bf1[22] = -bf0[22] + bf0[23];
        bf1[23] = bf0[22] + bf0[23];
        bf1[24] = bf0[24] + bf0[25];
        bf1[25] = bf0[24] - bf0[25];
        bf1[26] = -bf0[26] + bf0[27];
        bf1[27] = bf0[26] + bf0[27];
        bf1[28] = bf0[28] + bf0[29];
        bf1[29] = bf0[28] - bf0[29];
        bf1[30] = -bf0[30] + bf0[31];
        bf1[31] = bf0[30] + bf0[31];

        // Stage 4.
        for (int i = 0; i < 4; i++) bf0[i] = bf1[i];
        bf0[4] = Btf(cospi[56], bf1[4], -cospi[8], bf1[7]);
        bf0[5] = Btf(cospi[24], bf1[5], -cospi[40], bf1[6]);
        bf0[6] = Btf(cospi[40], bf1[5], cospi[24], bf1[6]);
        bf0[7] = Btf(cospi[8], bf1[4], cospi[56], bf1[7]);
        bf0[8] = bf1[8] + bf1[9];
        bf0[9] = bf1[8] - bf1[9];
        bf0[10] = -bf1[10] + bf1[11];
        bf0[11] = bf1[10] + bf1[11];
        bf0[12] = bf1[12] + bf1[13];
        bf0[13] = bf1[12] - bf1[13];
        bf0[14] = -bf1[14] + bf1[15];
        bf0[15] = bf1[14] + bf1[15];
        bf0[16] = bf1[16];
        bf0[17] = Btf(-cospi[8], bf1[17], cospi[56], bf1[30]);
        bf0[18] = Btf(-cospi[56], bf1[18], -cospi[8], bf1[29]);
        bf0[19] = bf1[19];
        bf0[20] = bf1[20];
        bf0[21] = Btf(-cospi[40], bf1[21], cospi[24], bf1[26]);
        bf0[22] = Btf(-cospi[24], bf1[22], -cospi[40], bf1[25]);
        bf0[23] = bf1[23];
        bf0[24] = bf1[24];
        bf0[25] = Btf(-cospi[40], bf1[22], cospi[24], bf1[25]);
        bf0[26] = Btf(cospi[24], bf1[21], cospi[40], bf1[26]);
        bf0[27] = bf1[27];
        bf0[28] = bf1[28];
        bf0[29] = Btf(-cospi[8], bf1[18], cospi[56], bf1[29]);
        bf0[30] = Btf(cospi[56], bf1[17], cospi[8], bf1[30]);
        bf0[31] = bf1[31];

        // Stage 5.
        bf1[0] = Btf(cospi[32], bf0[0], cospi[32], bf0[1]);
        bf1[1] = Btf(cospi[32], bf0[0], -cospi[32], bf0[1]);
        bf1[2] = Btf(cospi[48], bf0[2], -cospi[16], bf0[3]);
        bf1[3] = Btf(cospi[16], bf0[2], cospi[48], bf0[3]);
        bf1[4] = bf0[4] + bf0[5];
        bf1[5] = bf0[4] - bf0[5];
        bf1[6] = -bf0[6] + bf0[7];
        bf1[7] = bf0[6] + bf0[7];
        bf1[8] = bf0[8];
        bf1[9] = Btf(-cospi[16], bf0[9], cospi[48], bf0[14]);
        bf1[10] = Btf(-cospi[48], bf0[10], -cospi[16], bf0[13]);
        bf1[11] = bf0[11];
        bf1[12] = bf0[12];
        bf1[13] = Btf(-cospi[16], bf0[10], cospi[48], bf0[13]);
        bf1[14] = Btf(cospi[48], bf0[9], cospi[16], bf0[14]);
        bf1[15] = bf0[15];
        bf1[16] = bf0[16] + bf0[19];
        bf1[17] = bf0[17] + bf0[18];
        bf1[18] = bf0[17] - bf0[18];
        bf1[19] = bf0[16] - bf0[19];
        bf1[20] = -bf0[20] + bf0[23];
        bf1[21] = -bf0[21] + bf0[22];
        bf1[22] = bf0[21] + bf0[22];
        bf1[23] = bf0[20] + bf0[23];
        bf1[24] = bf0[24] + bf0[27];
        bf1[25] = bf0[25] + bf0[26];
        bf1[26] = bf0[25] - bf0[26];
        bf1[27] = bf0[24] - bf0[27];
        bf1[28] = -bf0[28] + bf0[31];
        bf1[29] = -bf0[29] + bf0[30];
        bf1[30] = bf0[29] + bf0[30];
        bf1[31] = bf0[28] + bf0[31];

        // Stage 6.
        bf0[0] = bf1[0] + bf1[3];
        bf0[1] = bf1[1] + bf1[2];
        bf0[2] = bf1[1] - bf1[2];
        bf0[3] = bf1[0] - bf1[3];
        bf0[4] = bf1[4];
        bf0[5] = Btf(-cospi[32], bf1[5], cospi[32], bf1[6]);
        bf0[6] = Btf(cospi[32], bf1[5], cospi[32], bf1[6]);
        bf0[7] = bf1[7];
        bf0[8] = bf1[8] + bf1[11];
        bf0[9] = bf1[9] + bf1[10];
        bf0[10] = bf1[9] - bf1[10];
        bf0[11] = bf1[8] - bf1[11];
        bf0[12] = -bf1[12] + bf1[15];
        bf0[13] = -bf1[13] + bf1[14];
        bf0[14] = bf1[13] + bf1[14];
        bf0[15] = bf1[12] + bf1[15];
        bf0[16] = bf1[16];
        bf0[17] = bf1[17];
        bf0[18] = Btf(-cospi[16], bf1[18], cospi[48], bf1[29]);
        bf0[19] = Btf(-cospi[16], bf1[19], cospi[48], bf1[28]);
        bf0[20] = Btf(-cospi[48], bf1[20], -cospi[16], bf1[27]);
        bf0[21] = Btf(-cospi[48], bf1[21], -cospi[16], bf1[26]);
        bf0[22] = bf1[22];
        bf0[23] = bf1[23];
        bf0[24] = bf1[24];
        bf0[25] = bf1[25];
        bf0[26] = Btf(-cospi[16], bf1[21], cospi[48], bf1[26]);
        bf0[27] = Btf(-cospi[16], bf1[20], cospi[48], bf1[27]);
        bf0[28] = Btf(cospi[48], bf1[19], cospi[16], bf1[28]);
        bf0[29] = Btf(cospi[48], bf1[18], cospi[16], bf1[29]);
        bf0[30] = bf1[30];
        bf0[31] = bf1[31];

        // Stage 7.
        bf1[0] = bf0[0] + bf0[7];
        bf1[1] = bf0[1] + bf0[6];
        bf1[2] = bf0[2] + bf0[5];
        bf1[3] = bf0[3] + bf0[4];
        bf1[4] = bf0[3] - bf0[4];
        bf1[5] = bf0[2] - bf0[5];
        bf1[6] = bf0[1] - bf0[6];
        bf1[7] = bf0[0] - bf0[7];
        bf1[8] = bf0[8];
        bf1[9] = bf0[9];
        bf1[10] = Btf(-cospi[32], bf0[10], cospi[32], bf0[13]);
        bf1[11] = Btf(-cospi[32], bf0[11], cospi[32], bf0[12]);
        bf1[12] = Btf(cospi[32], bf0[11], cospi[32], bf0[12]);
        bf1[13] = Btf(cospi[32], bf0[10], cospi[32], bf0[13]);
        bf1[14] = bf0[14];
        bf1[15] = bf0[15];
        bf1[16] = bf0[16] + bf0[23];
        bf1[17] = bf0[17] + bf0[22];
        bf1[18] = bf0[18] + bf0[21];
        bf1[19] = bf0[19] + bf0[20];
        bf1[20] = bf0[19] - bf0[20];
        bf1[21] = bf0[18] - bf0[21];
        bf1[22] = bf0[17] - bf0[22];
        bf1[23] = bf0[16] - bf0[23];
        bf1[24] = -bf0[24] + bf0[31];
        bf1[25] = -bf0[25] + bf0[30];
        bf1[26] = -bf0[26] + bf0[29];
        bf1[27] = -bf0[27] + bf0[28];
        bf1[28] = bf0[27] + bf0[28];
        bf1[29] = bf0[26] + bf0[29];
        bf1[30] = bf0[25] + bf0[30];
        bf1[31] = bf0[24] + bf0[31];

        // Stage 8.
        bf0[0] = bf1[0] + bf1[15];
        bf0[1] = bf1[1] + bf1[14];
        bf0[2] = bf1[2] + bf1[13];
        bf0[3] = bf1[3] + bf1[12];
        bf0[4] = bf1[4] + bf1[11];
        bf0[5] = bf1[5] + bf1[10];
        bf0[6] = bf1[6] + bf1[9];
        bf0[7] = bf1[7] + bf1[8];
        bf0[8] = bf1[7] - bf1[8];
        bf0[9] = bf1[6] - bf1[9];
        bf0[10] = bf1[5] - bf1[10];
        bf0[11] = bf1[4] - bf1[11];
        bf0[12] = bf1[3] - bf1[12];
        bf0[13] = bf1[2] - bf1[13];
        bf0[14] = bf1[1] - bf1[14];
        bf0[15] = bf1[0] - bf1[15];
        for (int i = 16; i < 20; i++) bf0[i] = bf1[i];
        bf0[20] = Btf(-cospi[32], bf1[20], cospi[32], bf1[27]);
        bf0[21] = Btf(-cospi[32], bf1[21], cospi[32], bf1[26]);
        bf0[22] = Btf(-cospi[32], bf1[22], cospi[32], bf1[25]);
        bf0[23] = Btf(-cospi[32], bf1[23], cospi[32], bf1[24]);
        bf0[24] = Btf(cospi[32], bf1[23], cospi[32], bf1[24]);
        bf0[25] = Btf(cospi[32], bf1[22], cospi[32], bf1[25]);
        bf0[26] = Btf(cospi[32], bf1[21], cospi[32], bf1[26]);
        bf0[27] = Btf(cospi[32], bf1[20], cospi[32], bf1[27]);
        for (int i = 28; i < 32; i++) bf0[i] = bf1[i];

        // Stage 9: outer butterfly (write to output).
        output[0] = bf0[0] + bf0[31];
        output[1] = bf0[1] + bf0[30];
        output[2] = bf0[2] + bf0[29];
        output[3] = bf0[3] + bf0[28];
        output[4] = bf0[4] + bf0[27];
        output[5] = bf0[5] + bf0[26];
        output[6] = bf0[6] + bf0[25];
        output[7] = bf0[7] + bf0[24];
        output[8] = bf0[8] + bf0[23];
        output[9] = bf0[9] + bf0[22];
        output[10] = bf0[10] + bf0[21];
        output[11] = bf0[11] + bf0[20];
        output[12] = bf0[12] + bf0[19];
        output[13] = bf0[13] + bf0[18];
        output[14] = bf0[14] + bf0[17];
        output[15] = bf0[15] + bf0[16];
        output[16] = bf0[15] - bf0[16];
        output[17] = bf0[14] - bf0[17];
        output[18] = bf0[13] - bf0[18];
        output[19] = bf0[12] - bf0[19];
        output[20] = bf0[11] - bf0[20];
        output[21] = bf0[10] - bf0[21];
        output[22] = bf0[9] - bf0[22];
        output[23] = bf0[8] - bf0[23];
        output[24] = bf0[7] - bf0[24];
        output[25] = bf0[6] - bf0[25];
        output[26] = bf0[5] - bf0[26];
        output[27] = bf0[4] - bf0[27];
        output[28] = bf0[3] - bf0[28];
        output[29] = bf0[2] - bf0[29];
        output[30] = bf0[1] - bf0[30];
        output[31] = bf0[0] - bf0[31];
    }
}
