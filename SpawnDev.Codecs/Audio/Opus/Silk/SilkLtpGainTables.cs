// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Structural port of libopus silk/tables_LTP.c LTP gain vector codebooks.
// Three codebooks (8, 16, 32 entries) selected by PERIndex; each entry is a
// 5-tap Q7 filter that defines the LTP (long-term prediction) response for
// one subframe.
//
// Upstream Copyright (c) 2006-2011 Skype Limited. BSD 3-Clause. See NOTICE.md.

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// LTP gain vector codebooks. Flat row-major storage: entry for
/// <c>(perIndex, ltpIndex)</c> starts at offset
/// <c>ltpIndex * <see cref="LtpVecSize"/></c> within the selected per-codebook array.
/// Values are signed Q7 gain taps.
/// </summary>
internal static class SilkLtpGainTables
{
    /// <summary>Number of taps in a single LTP gain vector. Libopus <c>LTP_ORDER = 5</c>.</summary>
    internal const int LtpVecSize = 5;

    /// <summary>Codebook 0: 8 entries x 5 taps.</summary>
    internal static readonly sbyte[] Vq0 =
    {
          4,   6,  24,   7,   5,
          0,   0,   2,   0,   0,
         12,  28,  41,  13,  -4,
         -9,  15,  42,  25,  14,
          1,  -2,  62,  41,  -9,
        -10,  37,  65,  -4,   3,
         -6,   4,  66,   7,  -8,
         16,  14,  38,  -3,  33,
    };

    /// <summary>Codebook 1: 16 entries x 5 taps.</summary>
    internal static readonly sbyte[] Vq1 =
    {
         13,  22,  39,  23,  12,
         -1,  36,  64,  27,  -6,
         -7,  10,  55,  43,  17,
          1,   1,   8,   1,   1,
          6, -11,  74,  53,  -9,
        -12,  55,  76, -12,   8,
         -3,   3,  93,  27,  -4,
         26,  39,  59,   3,  -8,
          2,   0,  77,  11,   9,
         -8,  22,  44,  -6,   7,
         40,   9,  26,   3,   9,
         -7,  20, 101,  -7,   4,
          3,  -8,  42,  26,   0,
        -15,  33,  68,   2,  23,
         -2,  55,  46,  -2,  15,
          3,  -1,  21,  16,  41,
    };

    /// <summary>Codebook 2: 32 entries x 5 taps.</summary>
    internal static readonly sbyte[] Vq2 =
    {
         -6,  27,  61,  39,   5,
        -11,  42,  88,   4,   1,
         -2,  60,  65,   6,  -4,
         -1,  -5,  73,  56,   1,
         -9,  19,  94,  29,  -9,
          0,  12,  99,   6,   4,
          8, -19, 102,  46, -13,
          3,   2,  13,   3,   2,
          9, -21,  84,  72, -18,
        -11,  46, 104, -22,   8,
         18,  38,  48,  23,   0,
        -16,  70,  83, -21,  11,
          5, -11, 117,  22,  -8,
         -6,  23, 117, -12,   3,
          3,  -8,  95,  28,   4,
        -10,  15,  77,  60, -15,
         -1,   4, 124,   2,  -4,
          3,  38,  84,  24, -25,
          2,  13,  42,  13,  31,
         21,  -4,  56,  46,  -1,
         -1,  35,  79, -13,  19,
         -7,  65,  88,  -9, -14,
         20,   4,  81,  49, -29,
         20,   0,  75,   3, -17,
          5,  -9,  44,  92,  -8,
          1,  -3,  22,  69,  31,
         -6,  95,  41, -12,   5,
         39,  67,  16,  -4,   1,
          0,  -6, 120,  55, -36,
        -13,  44, 122,   4, -24,
         81,   5,  11,   3,   7,
          2,   0,   9,  10,  88,
    };

    /// <summary>Select the LTP gain codebook for the given <paramref name="perIndex"/>.</summary>
    internal static sbyte[] Select(int perIndex)
    {
        return perIndex switch
        {
            0 => Vq0,
            1 => Vq1,
            2 => Vq2,
            _ => throw new ArgumentOutOfRangeException(nameof(perIndex), $"perIndex must be 0, 1, or 2 (got {perIndex})."),
        };
    }
}
