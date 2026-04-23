// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Structural port of libopus silk/tables_gain.c + silk/tables_other.c to clean C#.
// These small entropy tables are read by the SILK bitstream indices decoder
// (silk_decode_indices) to recover signal type / offset, gain indices, NLSF
// interpolation factor, LTP scaling, seeds, and other scalar fields.
//
// Upstream Copyright (c) 2006-2011 Skype Limited. BSD 3-Clause. See NOTICE.md.

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// Small iCDF (inverse cumulative distribution function) lookup tables referenced
/// by the SILK indices decoder. Each entry matches libopus bit-exactly. The
/// per-table probability resolution (FTB) is 8 bits unless otherwise documented.
/// </summary>
internal static class SilkIcdfTables
{
    // ----- Uniform iCDFs (uniform over N symbols, ftb = 8) -----

    /// <summary><c>silk_uniform3_iCDF</c>: uniform over 3 symbols.</summary>
    internal static readonly byte[] Uniform3 = { 171, 85, 0 };

    /// <summary><c>silk_uniform4_iCDF</c>: uniform over 4 symbols.</summary>
    internal static readonly byte[] Uniform4 = { 192, 128, 64, 0 };

    /// <summary><c>silk_uniform5_iCDF</c>: uniform over 5 symbols.</summary>
    internal static readonly byte[] Uniform5 = { 205, 154, 102, 51, 0 };

    /// <summary><c>silk_uniform6_iCDF</c>: uniform over 6 symbols.</summary>
    internal static readonly byte[] Uniform6 = { 213, 171, 128, 85, 43, 0 };

    /// <summary><c>silk_uniform8_iCDF</c>: uniform over 8 symbols (used for gain LSBs).</summary>
    internal static readonly byte[] Uniform8 = { 224, 192, 160, 128, 96, 64, 32, 0 };

    // ----- Signal-type / quantizer-offset iCDFs -----

    /// <summary><c>silk_type_offset_VAD_iCDF</c>: signalType+offset when VAD flag is set (4 symbols).</summary>
    internal static readonly byte[] TypeOffsetVad = { 232, 158, 10, 0 };

    /// <summary><c>silk_type_offset_no_VAD_iCDF</c>: signalType+offset when VAD flag is clear (2 symbols).</summary>
    internal static readonly byte[] TypeOffsetNoVad = { 230, 0 };

    // ----- NLSF decode auxiliaries -----

    /// <summary><c>silk_NLSF_EXT_iCDF</c>: extension iCDF for NLSF residual indices at the rail limits.</summary>
    internal static readonly byte[] NlsfExt = { 100, 40, 16, 7, 3, 1, 0 };

    /// <summary><c>silk_NLSF_interpolation_factor_iCDF</c>: 5-symbol iCDF for the Q2 interpolation coefficient.</summary>
    internal static readonly byte[] NlsfInterpolationFactor = { 243, 221, 192, 181, 0 };

    // ----- LTP scaling -----

    /// <summary><c>silk_LTPscale_iCDF</c>: 3-symbol iCDF for LTP scale index (used only on independently-coded voiced frames).</summary>
    internal static readonly byte[] LtpScale = { 128, 64, 0 };

    // ----- Gain decode iCDFs (silk/tables_gain.c) -----

    /// <summary>
    /// <c>silk_gain_iCDF[signalType]</c>: 8-symbol iCDF used for the MSBs of an
    /// independently-coded gain index. Index 0 = inactive, 1 = unvoiced, 2 = voiced.
    /// Laid out as a flat 24-entry array; use <see cref="GainIcdfOffset"/> to locate each row.
    /// </summary>
    internal static readonly byte[] Gain =
    {
        224, 112,  44,  15,   3,   2,   1,   0,  // inactive
        254, 237, 192, 132,  70,  23,   4,   0,  // unvoiced
        255, 252, 226, 155,  61,  11,   2,   0,  // voiced
    };

    /// <summary>Number of rows in <see cref="Gain"/> (one per SILK signal type).</summary>
    internal const int GainIcdfNumTypes = 3;

    /// <summary>Entries per row in <see cref="Gain"/>.</summary>
    internal const int GainIcdfEntriesPerType = 8;

    /// <summary>Byte offset into <see cref="Gain"/> for signal type <paramref name="signalType"/>.</summary>
    internal static int GainIcdfOffset(int signalType) => signalType * GainIcdfEntriesPerType;

    /// <summary>
    /// <c>silk_delta_gain_iCDF</c>: 41-symbol iCDF for delta-coded gain indices. Used for
    /// subframes after the first, and for the first subframe when conditional coding is in effect.
    /// </summary>
    internal static readonly byte[] DeltaGain =
    {
        250, 245, 234, 203,  71,  50,  42,  38,
         35,  33,  31,  29,  28,  27,  26,  25,
         24,  23,  22,  21,  20,  19,  18,  17,
         16,  15,  14,  13,  12,  11,  10,   9,
          8,   7,   6,   5,   4,   3,   2,   1,
          0,
    };
}
