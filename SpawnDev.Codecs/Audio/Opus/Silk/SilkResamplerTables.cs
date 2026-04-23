// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Structural port of libopus silk/resampler_rom.c coefficient tables for the
// SILK downsample filter. Each table begins with 2 AR2 IIR coefficients in
// Q14, followed by the polyphase FIR coefficients.
//
// Upstream Copyright (c) 2006-2011 Skype Limited. BSD 3-Clause. See NOTICE.md.

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// Downsample FIR filter coefficient tables. Layout: <c>[AR2_Q14[2], FIR_Coefs[...]]</c>.
/// The AR2 coefficients pre-whiten the input; the FIR section does the actual
/// polyphase interpolated downsampling at the target rate.
/// </summary>
internal static class SilkResamplerTables
{
    /// <summary>3/4 downsample (e.g. 16 -&gt; 12 kHz). FIR order 18, 3 polyphase fractions.</summary>
    internal static readonly short[] Coefs3To4 =
    {
        // AR2 Q14 pre-filter
        -20694, -13867,
        // FIR polyphase rows (3 fractions x 9 taps each)
            -49,     64,     17,   -157,    353,   -496,    163,  11047,  22205,
            -39,      6,     91,   -170,    186,     23,   -896,   6336,  19928,
            -19,    -36,    102,    -89,    -24,    328,   -951,   2568,  15909,
    };

    /// <summary>2/3 downsample (e.g. 12 -&gt; 8 kHz). FIR order 18, 2 polyphase fractions.</summary>
    internal static readonly short[] Coefs2To3 =
    {
        -14457, -14019,
            64,    128,   -122,     36,    310,   -768,    584,   9267,  17733,
            12,    128,     18,   -142,    288,   -117,   -865,   4123,  14459,
    };

    /// <summary>1/2 downsample (e.g. 16 -&gt; 8 kHz). FIR order 24, 1 polyphase fraction (symmetric).</summary>
    internal static readonly short[] Coefs1To2 =
    {
            616, -14323,
            -10,     39,     58,    -46,    -84,    120,    184,   -315,   -541,   1284,   5380,   9024,
    };

    /// <summary>1/3 downsample. FIR order 36, 1 polyphase fraction (symmetric).</summary>
    internal static readonly short[] Coefs1To3 =
    {
         16102, -15162,
            -13,      0,     20,     26,      5,    -31,    -43,     -4,     65,     90,      7,   -157,   -248,    -44,    593,   1583,   2612,   3271,
    };

    /// <summary>1/4 downsample. FIR order 36, 1 polyphase fraction (symmetric).</summary>
    internal static readonly short[] Coefs1To4 =
    {
         22500, -15099,
             3,    -14,    -20,    -15,      2,     25,     37,     25,    -16,    -71,   -107,    -79,     50,    292,    623,    982,   1288,   1464,
    };

    /// <summary>1/6 downsample. FIR order 36, 1 polyphase fraction (symmetric).</summary>
    internal static readonly short[] Coefs1To6 =
    {
         27540, -15257,
            17,     12,      8,      1,    -10,    -22,    -30,    -32,    -22,      3,     44,    100,    168,    243,    317,    381,    429,    455,
    };
}
