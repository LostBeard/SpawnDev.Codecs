// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 32-point forward DCT (1D). Bit-exact port of libaom
// av1/encoder/av1_fwd_txfm1d.c av1_fdct32.
//
// 9 stages with cospi_arr-driven half_btf rotations + final
// bit-reversed scatter permutation. The largest 1D transform AV1 uses
// (64-point fdct decomposes into 32-point fdct + extra bit-reversal).

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>AV1 32-point forward DCT (1D).</summary>
public static class Av1ForwardDct32
{
    /// <summary>Default cosine-precision bits (libaom).</summary>
    public const int DefaultCosBit = 12;

    /// <summary>32-point forward DCT. Mirrors libaom <c>av1_fdct32</c>.</summary>
    public static void Transform(ReadOnlySpan<int> input, Span<int> output, int cosBit = DefaultCosBit)
    {
        if (input.Length < 32) throw new ArgumentException("input must have 32 entries", nameof(input));
        if (output.Length < 32) throw new ArgumentException("output must have 32 entries", nameof(output));
        if (cosBit < 10 || cosBit > 13) throw new ArgumentOutOfRangeException(nameof(cosBit));

        var cospi = Av1ForwardDct4.CospiArr(cosBit);
        Span<int> bf0 = stackalloc int[32];
        Span<int> bf1 = stackalloc int[32];

        // Stage 1: butterfly add/sub across the full 32-element width.
        bf0[0]  = input[0]  + input[31];
        bf0[1]  = input[1]  + input[30];
        bf0[2]  = input[2]  + input[29];
        bf0[3]  = input[3]  + input[28];
        bf0[4]  = input[4]  + input[27];
        bf0[5]  = input[5]  + input[26];
        bf0[6]  = input[6]  + input[25];
        bf0[7]  = input[7]  + input[24];
        bf0[8]  = input[8]  + input[23];
        bf0[9]  = input[9]  + input[22];
        bf0[10] = input[10] + input[21];
        bf0[11] = input[11] + input[20];
        bf0[12] = input[12] + input[19];
        bf0[13] = input[13] + input[18];
        bf0[14] = input[14] + input[17];
        bf0[15] = input[15] + input[16];
        bf0[16] = -input[16] + input[15];
        bf0[17] = -input[17] + input[14];
        bf0[18] = -input[18] + input[13];
        bf0[19] = -input[19] + input[12];
        bf0[20] = -input[20] + input[11];
        bf0[21] = -input[21] + input[10];
        bf0[22] = -input[22] + input[9];
        bf0[23] = -input[23] + input[8];
        bf0[24] = -input[24] + input[7];
        bf0[25] = -input[25] + input[6];
        bf0[26] = -input[26] + input[5];
        bf0[27] = -input[27] + input[4];
        bf0[28] = -input[28] + input[3];
        bf0[29] = -input[29] + input[2];
        bf0[30] = -input[30] + input[1];
        bf0[31] = -input[31] + input[0];

        // Stage 2.
        bf1[0]  = bf0[0]  + bf0[15];
        bf1[1]  = bf0[1]  + bf0[14];
        bf1[2]  = bf0[2]  + bf0[13];
        bf1[3]  = bf0[3]  + bf0[12];
        bf1[4]  = bf0[4]  + bf0[11];
        bf1[5]  = bf0[5]  + bf0[10];
        bf1[6]  = bf0[6]  + bf0[9];
        bf1[7]  = bf0[7]  + bf0[8];
        bf1[8]  = -bf0[8]  + bf0[7];
        bf1[9]  = -bf0[9]  + bf0[6];
        bf1[10] = -bf0[10] + bf0[5];
        bf1[11] = -bf0[11] + bf0[4];
        bf1[12] = -bf0[12] + bf0[3];
        bf1[13] = -bf0[13] + bf0[2];
        bf1[14] = -bf0[14] + bf0[1];
        bf1[15] = -bf0[15] + bf0[0];
        bf1[16] = bf0[16];
        bf1[17] = bf0[17];
        bf1[18] = bf0[18];
        bf1[19] = bf0[19];
        bf1[20] = Av1ForwardDct4.HalfBtf(-cospi[32], bf0[20], cospi[32], bf0[27], cosBit);
        bf1[21] = Av1ForwardDct4.HalfBtf(-cospi[32], bf0[21], cospi[32], bf0[26], cosBit);
        bf1[22] = Av1ForwardDct4.HalfBtf(-cospi[32], bf0[22], cospi[32], bf0[25], cosBit);
        bf1[23] = Av1ForwardDct4.HalfBtf(-cospi[32], bf0[23], cospi[32], bf0[24], cosBit);
        bf1[24] = Av1ForwardDct4.HalfBtf( cospi[32], bf0[24], cospi[32], bf0[23], cosBit);
        bf1[25] = Av1ForwardDct4.HalfBtf( cospi[32], bf0[25], cospi[32], bf0[22], cosBit);
        bf1[26] = Av1ForwardDct4.HalfBtf( cospi[32], bf0[26], cospi[32], bf0[21], cosBit);
        bf1[27] = Av1ForwardDct4.HalfBtf( cospi[32], bf0[27], cospi[32], bf0[20], cosBit);
        bf1[28] = bf0[28];
        bf1[29] = bf0[29];
        bf1[30] = bf0[30];
        bf1[31] = bf0[31];

        // Stage 3.
        bf0[0]  = bf1[0] + bf1[7];
        bf0[1]  = bf1[1] + bf1[6];
        bf0[2]  = bf1[2] + bf1[5];
        bf0[3]  = bf1[3] + bf1[4];
        bf0[4]  = -bf1[4] + bf1[3];
        bf0[5]  = -bf1[5] + bf1[2];
        bf0[6]  = -bf1[6] + bf1[1];
        bf0[7]  = -bf1[7] + bf1[0];
        bf0[8]  = bf1[8];
        bf0[9]  = bf1[9];
        bf0[10] = Av1ForwardDct4.HalfBtf(-cospi[32], bf1[10], cospi[32], bf1[13], cosBit);
        bf0[11] = Av1ForwardDct4.HalfBtf(-cospi[32], bf1[11], cospi[32], bf1[12], cosBit);
        bf0[12] = Av1ForwardDct4.HalfBtf( cospi[32], bf1[12], cospi[32], bf1[11], cosBit);
        bf0[13] = Av1ForwardDct4.HalfBtf( cospi[32], bf1[13], cospi[32], bf1[10], cosBit);
        bf0[14] = bf1[14];
        bf0[15] = bf1[15];
        bf0[16] = bf1[16] + bf1[23];
        bf0[17] = bf1[17] + bf1[22];
        bf0[18] = bf1[18] + bf1[21];
        bf0[19] = bf1[19] + bf1[20];
        bf0[20] = -bf1[20] + bf1[19];
        bf0[21] = -bf1[21] + bf1[18];
        bf0[22] = -bf1[22] + bf1[17];
        bf0[23] = -bf1[23] + bf1[16];
        bf0[24] = -bf1[24] + bf1[31];
        bf0[25] = -bf1[25] + bf1[30];
        bf0[26] = -bf1[26] + bf1[29];
        bf0[27] = -bf1[27] + bf1[28];
        bf0[28] = bf1[28] + bf1[27];
        bf0[29] = bf1[29] + bf1[26];
        bf0[30] = bf1[30] + bf1[25];
        bf0[31] = bf1[31] + bf1[24];

        // Stage 4.
        bf1[0]  = bf0[0] + bf0[3];
        bf1[1]  = bf0[1] + bf0[2];
        bf1[2]  = -bf0[2] + bf0[1];
        bf1[3]  = -bf0[3] + bf0[0];
        bf1[4]  = bf0[4];
        bf1[5]  = Av1ForwardDct4.HalfBtf(-cospi[32], bf0[5], cospi[32], bf0[6], cosBit);
        bf1[6]  = Av1ForwardDct4.HalfBtf( cospi[32], bf0[6], cospi[32], bf0[5], cosBit);
        bf1[7]  = bf0[7];
        bf1[8]  = bf0[8] + bf0[11];
        bf1[9]  = bf0[9] + bf0[10];
        bf1[10] = -bf0[10] + bf0[9];
        bf1[11] = -bf0[11] + bf0[8];
        bf1[12] = -bf0[12] + bf0[15];
        bf1[13] = -bf0[13] + bf0[14];
        bf1[14] = bf0[14] + bf0[13];
        bf1[15] = bf0[15] + bf0[12];
        bf1[16] = bf0[16];
        bf1[17] = bf0[17];
        bf1[18] = Av1ForwardDct4.HalfBtf(-cospi[16], bf0[18],  cospi[48], bf0[29], cosBit);
        bf1[19] = Av1ForwardDct4.HalfBtf(-cospi[16], bf0[19],  cospi[48], bf0[28], cosBit);
        bf1[20] = Av1ForwardDct4.HalfBtf(-cospi[48], bf0[20], -cospi[16], bf0[27], cosBit);
        bf1[21] = Av1ForwardDct4.HalfBtf(-cospi[48], bf0[21], -cospi[16], bf0[26], cosBit);
        bf1[22] = bf0[22];
        bf1[23] = bf0[23];
        bf1[24] = bf0[24];
        bf1[25] = bf0[25];
        bf1[26] = Av1ForwardDct4.HalfBtf( cospi[48], bf0[26], -cospi[16], bf0[21], cosBit);
        bf1[27] = Av1ForwardDct4.HalfBtf( cospi[48], bf0[27], -cospi[16], bf0[20], cosBit);
        bf1[28] = Av1ForwardDct4.HalfBtf( cospi[16], bf0[28],  cospi[48], bf0[19], cosBit);
        bf1[29] = Av1ForwardDct4.HalfBtf( cospi[16], bf0[29],  cospi[48], bf0[18], cosBit);
        bf1[30] = bf0[30];
        bf1[31] = bf0[31];

        // Stage 5.
        bf0[0]  = Av1ForwardDct4.HalfBtf( cospi[32], bf1[0], cospi[32], bf1[1], cosBit);
        bf0[1]  = Av1ForwardDct4.HalfBtf(-cospi[32], bf1[1], cospi[32], bf1[0], cosBit);
        bf0[2]  = Av1ForwardDct4.HalfBtf( cospi[48], bf1[2], cospi[16], bf1[3], cosBit);
        bf0[3]  = Av1ForwardDct4.HalfBtf( cospi[48], bf1[3], -cospi[16], bf1[2], cosBit);
        bf0[4]  = bf1[4] + bf1[5];
        bf0[5]  = -bf1[5] + bf1[4];
        bf0[6]  = -bf1[6] + bf1[7];
        bf0[7]  = bf1[7] + bf1[6];
        bf0[8]  = bf1[8];
        bf0[9]  = Av1ForwardDct4.HalfBtf(-cospi[16], bf1[9],  cospi[48], bf1[14], cosBit);
        bf0[10] = Av1ForwardDct4.HalfBtf(-cospi[48], bf1[10], -cospi[16], bf1[13], cosBit);
        bf0[11] = bf1[11];
        bf0[12] = bf1[12];
        bf0[13] = Av1ForwardDct4.HalfBtf( cospi[48], bf1[13], -cospi[16], bf1[10], cosBit);
        bf0[14] = Av1ForwardDct4.HalfBtf( cospi[16], bf1[14],  cospi[48], bf1[9],  cosBit);
        bf0[15] = bf1[15];
        bf0[16] = bf1[16] + bf1[19];
        bf0[17] = bf1[17] + bf1[18];
        bf0[18] = -bf1[18] + bf1[17];
        bf0[19] = -bf1[19] + bf1[16];
        bf0[20] = -bf1[20] + bf1[23];
        bf0[21] = -bf1[21] + bf1[22];
        bf0[22] = bf1[22] + bf1[21];
        bf0[23] = bf1[23] + bf1[20];
        bf0[24] = bf1[24] + bf1[27];
        bf0[25] = bf1[25] + bf1[26];
        bf0[26] = -bf1[26] + bf1[25];
        bf0[27] = -bf1[27] + bf1[24];
        bf0[28] = -bf1[28] + bf1[31];
        bf0[29] = -bf1[29] + bf1[30];
        bf0[30] = bf1[30] + bf1[29];
        bf0[31] = bf1[31] + bf1[28];

        // Stage 6.
        bf1[0]  = bf0[0];
        bf1[1]  = bf0[1];
        bf1[2]  = bf0[2];
        bf1[3]  = bf0[3];
        bf1[4]  = Av1ForwardDct4.HalfBtf( cospi[56], bf0[4],  cospi[ 8], bf0[7], cosBit);
        bf1[5]  = Av1ForwardDct4.HalfBtf( cospi[24], bf0[5],  cospi[40], bf0[6], cosBit);
        bf1[6]  = Av1ForwardDct4.HalfBtf( cospi[24], bf0[6], -cospi[40], bf0[5], cosBit);
        bf1[7]  = Av1ForwardDct4.HalfBtf( cospi[56], bf0[7], -cospi[ 8], bf0[4], cosBit);
        bf1[8]  = bf0[8] + bf0[9];
        bf1[9]  = -bf0[9] + bf0[8];
        bf1[10] = -bf0[10] + bf0[11];
        bf1[11] = bf0[11] + bf0[10];
        bf1[12] = bf0[12] + bf0[13];
        bf1[13] = -bf0[13] + bf0[12];
        bf1[14] = -bf0[14] + bf0[15];
        bf1[15] = bf0[15] + bf0[14];
        bf1[16] = bf0[16];
        bf1[17] = Av1ForwardDct4.HalfBtf(-cospi[ 8], bf0[17],  cospi[56], bf0[30], cosBit);
        bf1[18] = Av1ForwardDct4.HalfBtf(-cospi[56], bf0[18], -cospi[ 8], bf0[29], cosBit);
        bf1[19] = bf0[19];
        bf1[20] = bf0[20];
        bf1[21] = Av1ForwardDct4.HalfBtf(-cospi[40], bf0[21],  cospi[24], bf0[26], cosBit);
        bf1[22] = Av1ForwardDct4.HalfBtf(-cospi[24], bf0[22], -cospi[40], bf0[25], cosBit);
        bf1[23] = bf0[23];
        bf1[24] = bf0[24];
        bf1[25] = Av1ForwardDct4.HalfBtf( cospi[24], bf0[25], -cospi[40], bf0[22], cosBit);
        bf1[26] = Av1ForwardDct4.HalfBtf( cospi[40], bf0[26],  cospi[24], bf0[21], cosBit);
        bf1[27] = bf0[27];
        bf1[28] = bf0[28];
        bf1[29] = Av1ForwardDct4.HalfBtf( cospi[56], bf0[29], -cospi[ 8], bf0[18], cosBit);
        bf1[30] = Av1ForwardDct4.HalfBtf( cospi[ 8], bf0[30],  cospi[56], bf0[17], cosBit);
        bf1[31] = bf0[31];

        // Stage 7.
        bf0[0]  = bf1[0];
        bf0[1]  = bf1[1];
        bf0[2]  = bf1[2];
        bf0[3]  = bf1[3];
        bf0[4]  = bf1[4];
        bf0[5]  = bf1[5];
        bf0[6]  = bf1[6];
        bf0[7]  = bf1[7];
        bf0[8]  = Av1ForwardDct4.HalfBtf( cospi[60], bf1[ 8],  cospi[ 4], bf1[15], cosBit);
        bf0[9]  = Av1ForwardDct4.HalfBtf( cospi[28], bf1[ 9],  cospi[36], bf1[14], cosBit);
        bf0[10] = Av1ForwardDct4.HalfBtf( cospi[44], bf1[10],  cospi[20], bf1[13], cosBit);
        bf0[11] = Av1ForwardDct4.HalfBtf( cospi[12], bf1[11],  cospi[52], bf1[12], cosBit);
        bf0[12] = Av1ForwardDct4.HalfBtf( cospi[12], bf1[12], -cospi[52], bf1[11], cosBit);
        bf0[13] = Av1ForwardDct4.HalfBtf( cospi[44], bf1[13], -cospi[20], bf1[10], cosBit);
        bf0[14] = Av1ForwardDct4.HalfBtf( cospi[28], bf1[14], -cospi[36], bf1[ 9], cosBit);
        bf0[15] = Av1ForwardDct4.HalfBtf( cospi[60], bf1[15], -cospi[ 4], bf1[ 8], cosBit);
        bf0[16] = bf1[16] + bf1[17];
        bf0[17] = -bf1[17] + bf1[16];
        bf0[18] = -bf1[18] + bf1[19];
        bf0[19] = bf1[19] + bf1[18];
        bf0[20] = bf1[20] + bf1[21];
        bf0[21] = -bf1[21] + bf1[20];
        bf0[22] = -bf1[22] + bf1[23];
        bf0[23] = bf1[23] + bf1[22];
        bf0[24] = bf1[24] + bf1[25];
        bf0[25] = -bf1[25] + bf1[24];
        bf0[26] = -bf1[26] + bf1[27];
        bf0[27] = bf1[27] + bf1[26];
        bf0[28] = bf1[28] + bf1[29];
        bf0[29] = -bf1[29] + bf1[28];
        bf0[30] = -bf1[30] + bf1[31];
        bf0[31] = bf1[31] + bf1[30];

        // Stage 8.
        bf1[0]  = bf0[0];
        bf1[1]  = bf0[1];
        bf1[2]  = bf0[2];
        bf1[3]  = bf0[3];
        bf1[4]  = bf0[4];
        bf1[5]  = bf0[5];
        bf1[6]  = bf0[6];
        bf1[7]  = bf0[7];
        bf1[8]  = bf0[8];
        bf1[9]  = bf0[9];
        bf1[10] = bf0[10];
        bf1[11] = bf0[11];
        bf1[12] = bf0[12];
        bf1[13] = bf0[13];
        bf1[14] = bf0[14];
        bf1[15] = bf0[15];
        bf1[16] = Av1ForwardDct4.HalfBtf(cospi[62], bf0[16], cospi[ 2], bf0[31], cosBit);
        bf1[17] = Av1ForwardDct4.HalfBtf(cospi[30], bf0[17], cospi[34], bf0[30], cosBit);
        bf1[18] = Av1ForwardDct4.HalfBtf(cospi[46], bf0[18], cospi[18], bf0[29], cosBit);
        bf1[19] = Av1ForwardDct4.HalfBtf(cospi[14], bf0[19], cospi[50], bf0[28], cosBit);
        bf1[20] = Av1ForwardDct4.HalfBtf(cospi[54], bf0[20], cospi[10], bf0[27], cosBit);
        bf1[21] = Av1ForwardDct4.HalfBtf(cospi[22], bf0[21], cospi[42], bf0[26], cosBit);
        bf1[22] = Av1ForwardDct4.HalfBtf(cospi[38], bf0[22], cospi[26], bf0[25], cosBit);
        bf1[23] = Av1ForwardDct4.HalfBtf(cospi[ 6], bf0[23], cospi[58], bf0[24], cosBit);
        bf1[24] = Av1ForwardDct4.HalfBtf(cospi[ 6], bf0[24], -cospi[58], bf0[23], cosBit);
        bf1[25] = Av1ForwardDct4.HalfBtf(cospi[38], bf0[25], -cospi[26], bf0[22], cosBit);
        bf1[26] = Av1ForwardDct4.HalfBtf(cospi[22], bf0[26], -cospi[42], bf0[21], cosBit);
        bf1[27] = Av1ForwardDct4.HalfBtf(cospi[54], bf0[27], -cospi[10], bf0[20], cosBit);
        bf1[28] = Av1ForwardDct4.HalfBtf(cospi[14], bf0[28], -cospi[50], bf0[19], cosBit);
        bf1[29] = Av1ForwardDct4.HalfBtf(cospi[46], bf0[29], -cospi[18], bf0[18], cosBit);
        bf1[30] = Av1ForwardDct4.HalfBtf(cospi[30], bf0[30], -cospi[34], bf0[17], cosBit);
        bf1[31] = Av1ForwardDct4.HalfBtf(cospi[62], bf0[31], -cospi[ 2], bf0[16], cosBit);

        // Stage 9: bit-reversed scatter to output.
        output[0]  = bf1[0];
        output[1]  = bf1[16];
        output[2]  = bf1[8];
        output[3]  = bf1[24];
        output[4]  = bf1[4];
        output[5]  = bf1[20];
        output[6]  = bf1[12];
        output[7]  = bf1[28];
        output[8]  = bf1[2];
        output[9]  = bf1[18];
        output[10] = bf1[10];
        output[11] = bf1[26];
        output[12] = bf1[6];
        output[13] = bf1[22];
        output[14] = bf1[14];
        output[15] = bf1[30];
        output[16] = bf1[1];
        output[17] = bf1[17];
        output[18] = bf1[9];
        output[19] = bf1[25];
        output[20] = bf1[5];
        output[21] = bf1[21];
        output[22] = bf1[13];
        output[23] = bf1[29];
        output[24] = bf1[3];
        output[25] = bf1[19];
        output[26] = bf1[11];
        output[27] = bf1[27];
        output[28] = bf1[7];
        output[29] = bf1[23];
        output[30] = bf1[15];
        output[31] = bf1[31];
    }
}
