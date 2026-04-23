// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Structural port of the silk_shell_code_table[0..3] iCDFs from libopus
// silk/tables_pulses_per_block.c to clean C#. These are the binary-split iCDFs
// used by the shell coder's decode_split helper for four different block sizes
// (1-pulse, 2-pulse, 4-pulse, 8-pulse subblocks).
//
// Upstream Copyright (c) 2006-2011 Skype Limited. BSD 3-Clause. See NOTICE.md.

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// Shell coder iCDF tables. The shell coder recursively splits a pulse count
/// across a 16-sample block via a balanced binary tree; at each split depth it
/// uses a different table (Table0 for the final 2-leaf splits, Table3 for the
/// 2-way split of the whole block). <see cref="Offsets"/> gives the start
/// offset into a given table for each possible input pulse count
/// <c>p in [0, 16]</c>.
/// </summary>
internal static class SilkShellCodeTables
{
    /// <summary>
    /// <c>silk_shell_code_table_offsets[p]</c>: byte offset into each shell table
    /// at which the iCDF sub-table for pulse count <c>p</c> begins. Length is
    /// <c>SILK_MAX_PULSES + 1 = 17</c>.
    /// </summary>
    internal static readonly byte[] Offsets =
    {
          0,   0,   2,   5,   9,  14,  20,  27,
         35,  44,  54,  65,  77,  90, 104, 119,
        135,
    };

    /// <summary><c>silk_shell_code_table0</c>: 152-byte flat table for the leaf-level binary split.</summary>
    internal static readonly byte[] Table0 =
    {
        128,   0, 214,  42,   0, 235, 128,  21,
          0, 244, 184,  72,  11,   0, 248, 214,
        128,  42,   7,   0, 248, 225, 170,  80,
         25,   5,   0, 251, 236, 198, 126,  54,
         18,   3,   0, 250, 238, 211, 159,  82,
         35,  15,   5,   0, 250, 231, 203, 168,
        128,  88,  53,  25,   6,   0, 252, 238,
        216, 185, 148, 108,  71,  40,  18,   4,
          0, 253, 243, 225, 199, 166, 128,  90,
         57,  31,  13,   3,   0, 254, 246, 233,
        212, 183, 147, 109,  73,  44,  23,  10,
          2,   0, 255, 250, 240, 223, 198, 166,
        128,  90,  58,  33,  16,   6,   1,   0,
        255, 251, 244, 231, 210, 181, 146, 110,
         75,  46,  25,  12,   5,   1,   0, 255,
        253, 248, 238, 221, 196, 164, 128,  92,
         60,  35,  18,   8,   3,   1,   0, 255,
        253, 249, 242, 229, 208, 180, 146, 110,
         76,  48,  27,  14,   7,   3,   1,   0,
    };

    /// <summary><c>silk_shell_code_table1</c>: 152-byte flat table for the next split level up.</summary>
    internal static readonly byte[] Table1 =
    {
        129,   0, 207,  50,   0, 236, 129,  20,
          0, 245, 185,  72,  10,   0, 249, 213,
        129,  42,   6,   0, 250, 226, 169,  87,
         27,   4,   0, 251, 233, 194, 130,  62,
         20,   4,   0, 250, 236, 207, 160,  99,
         47,  17,   3,   0, 255, 240, 217, 182,
        131,  81,  41,  11,   1,   0, 255, 254,
        233, 201, 159, 107,  61,  20,   2,   1,
          0, 255, 249, 233, 206, 170, 128,  86,
         50,  23,   7,   1,   0, 255, 250, 238,
        217, 186, 148, 108,  70,  39,  18,   6,
          1,   0, 255, 252, 243, 226, 200, 166,
        128,  90,  56,  30,  13,   4,   1,   0,
        255, 252, 245, 231, 209, 180, 146, 110,
         76,  47,  25,  11,   4,   1,   0, 255,
        253, 248, 237, 219, 194, 163, 128,  93,
         62,  37,  19,   8,   3,   1,   0, 255,
        254, 250, 241, 226, 205, 177, 145, 111,
         79,  51,  30,  15,   6,   2,   1,   0,
    };

    /// <summary><c>silk_shell_code_table2</c>: 152-byte flat table for the next split level up.</summary>
    internal static readonly byte[] Table2 =
    {
        129,   0, 203,  54,   0, 234, 129,  23,
          0, 245, 184,  73,  10,   0, 250, 215,
        129,  41,   5,   0, 252, 232, 173,  86,
         24,   3,   0, 253, 240, 200, 129,  56,
         15,   2,   0, 253, 244, 217, 164,  94,
         38,  10,   1,   0, 253, 245, 226, 189,
        132,  71,  27,   7,   1,   0, 253, 246,
        231, 203, 159, 105,  56,  23,   6,   1,
          0, 255, 248, 235, 213, 179, 133,  85,
         47,  19,   5,   1,   0, 255, 254, 243,
        221, 194, 159, 117,  70,  37,  12,   2,
          1,   0, 255, 254, 248, 234, 208, 171,
        128,  85,  48,  22,   8,   2,   1,   0,
        255, 254, 250, 240, 220, 189, 149, 107,
         67,  36,  16,   6,   2,   1,   0, 255,
        254, 251, 243, 227, 201, 166, 128,  90,
         55,  29,  13,   5,   2,   1,   0, 255,
        254, 252, 246, 234, 213, 183, 147, 109,
         73,  43,  22,  10,   4,   2,   1,   0,
    };

    /// <summary><c>silk_shell_code_table3</c>: 152-byte flat table for the top split (splits the full 16-sample block into two 8-sample halves).</summary>
    internal static readonly byte[] Table3 =
    {
        130,   0, 200,  58,   0, 231, 130,  26,
          0, 244, 184,  76,  12,   0, 249, 214,
        130,  43,   6,   0, 252, 232, 173,  87,
         24,   3,   0, 253, 241, 203, 131,  56,
         14,   2,   0, 254, 246, 221, 167,  94,
         35,   8,   1,   0, 254, 249, 232, 193,
        130,  65,  23,   5,   1,   0, 255, 251,
        239, 211, 162,  99,  45,  15,   4,   1,
          0, 255, 251, 243, 223, 186, 131,  74,
         33,  11,   3,   1,   0, 255, 252, 245,
        230, 202, 158, 105,  57,  24,   8,   2,
          1,   0, 255, 253, 247, 235, 214, 179,
        132,  84,  44,  19,   7,   2,   1,   0,
        255, 254, 250, 240, 223, 196, 159, 112,
         69,  36,  15,   6,   2,   1,   0, 255,
        254, 253, 245, 231, 209, 176, 136,  93,
         55,  27,  11,   3,   2,   1,   0, 255,
        254, 253, 252, 239, 221, 194, 158, 117,
         76,  42,  18,   4,   3,   2,   1,   0,
    };
}
