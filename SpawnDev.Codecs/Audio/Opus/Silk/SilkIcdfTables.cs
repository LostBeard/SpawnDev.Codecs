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

    // ----- Pitch decode iCDFs (silk/tables_pitch_lag.c) -----

    /// <summary>
    /// <c>silk_pitch_lag_iCDF</c>: 32-symbol iCDF for the coarse pitch lag index.
    /// Dimension is <c>2 * (PITCH_EST_MAX_LAG_MS - PITCH_EST_MIN_LAG_MS)</c>.
    /// </summary>
    internal static readonly byte[] PitchLag =
    {
        253, 250, 244, 233, 212, 182, 150, 131,
        120, 110,  98,  85,  72,  60,  49,  40,
         32,  25,  19,  15,  13,  11,   9,   8,
          7,   6,   5,   4,   3,   2,   1,   0,
    };

    /// <summary>
    /// <c>silk_pitch_delta_iCDF</c>: 21-symbol iCDF for delta-coded pitch lags.
    /// Raw symbol 0 signals "use absolute coding"; symbols 1..20 encode delta = raw - 9
    /// applied to the previous frame's lag.
    /// </summary>
    internal static readonly byte[] PitchDelta =
    {
        210, 208, 206, 203, 199, 193, 183, 168,
        142, 104,  74,  52,  37,  27,  20,  14,
         10,   6,   4,   2,   0,
    };

    /// <summary>
    /// <c>silk_pitch_contour_iCDF</c>: 34-symbol pitch-contour iCDF for 20 ms non-NB frames.
    /// </summary>
    internal static readonly byte[] PitchContour =
    {
        223, 201, 183, 167, 152, 138, 124, 111,
         98,  88,  79,  70,  62,  56,  50,  44,
         39,  35,  31,  27,  24,  21,  18,  16,
         14,  12,  10,   8,   6,   4,   3,   2,
          1,   0,
    };

    /// <summary>
    /// <c>silk_pitch_contour_NB_iCDF</c>: 11-symbol pitch-contour iCDF for 20 ms NB frames.
    /// </summary>
    internal static readonly byte[] PitchContourNb =
    {
        188, 176, 155, 138, 119,  97,  67,  43,
         26,  10,   0,
    };

    /// <summary>
    /// <c>silk_pitch_contour_10_ms_iCDF</c>: 12-symbol pitch-contour iCDF for 10 ms non-NB frames.
    /// </summary>
    internal static readonly byte[] PitchContour10Ms =
    {
        165, 119,  80,  61,  47,  35,  27,  20,
         14,   9,   4,   0,
    };

    /// <summary>
    /// <c>silk_pitch_contour_10_ms_NB_iCDF</c>: 3-symbol pitch-contour iCDF for 10 ms NB frames.
    /// </summary>
    internal static readonly byte[] PitchContour10MsNb =
    {
        113,  63,   0,
    };

    /// <summary>
    /// Select the pitch-contour iCDF for a given sample rate and subframe count.
    /// NB uses smaller codebooks; 10 ms frames use shorter variants.
    /// </summary>
    /// <param name="fsKHz">Internal SILK sample rate in kHz (8, 12, or 16).</param>
    /// <param name="nbSubfr">Subframe count - 2 for 10 ms frames, 4 for 20 ms frames.</param>
    internal static byte[] SelectPitchContour(int fsKHz, int nbSubfr)
    {
        if (fsKHz == 8)
        {
            return nbSubfr == 4 ? PitchContourNb : PitchContour10MsNb;
        }
        return nbSubfr == 4 ? PitchContour : PitchContour10Ms;
    }

    /// <summary>
    /// Select the pitch-lag LSB iCDF for a given sample rate. NB uses 2 bits (Uniform4),
    /// MB uses ~2.58 bits (Uniform6), WB uses 3 bits (Uniform8).
    /// </summary>
    /// <param name="fsKHz">Internal SILK sample rate (8, 12, or 16).</param>
    internal static byte[] SelectPitchLagLowBits(int fsKHz)
    {
        return fsKHz switch
        {
            16 => Uniform8,
            12 => Uniform6,
            8 => Uniform4,
            _ => throw new ArgumentException($"Unsupported SILK fs_kHz: {fsKHz}.", nameof(fsKHz)),
        };
    }

    // ----- LTP decode iCDFs (silk/tables_LTP.c) -----

    /// <summary>
    /// <c>silk_LTP_per_index_iCDF</c>: 3-symbol iCDF for the LTP periodicity index
    /// (selects which LTP gain codebook the current frame uses).
    /// </summary>
    internal static readonly byte[] LtpPerIndex = { 179, 99, 0 };

    /// <summary>
    /// <c>silk_LTP_gain_iCDF_0</c>: 8-symbol iCDF for LTP gain index, codebook 0.
    /// </summary>
    internal static readonly byte[] LtpGain0 =
    {
        71, 56, 43, 30, 21, 12,  6,  0,
    };

    /// <summary>
    /// <c>silk_LTP_gain_iCDF_1</c>: 16-symbol iCDF for LTP gain index, codebook 1.
    /// </summary>
    internal static readonly byte[] LtpGain1 =
    {
        199, 165, 144, 124, 109,  96,  84,  71,
         61,  51,  42,  32,  23,  15,   8,   0,
    };

    /// <summary>
    /// <c>silk_LTP_gain_iCDF_2</c>: 32-symbol iCDF for LTP gain index, codebook 2.
    /// </summary>
    internal static readonly byte[] LtpGain2 =
    {
        241, 225, 211, 199, 187, 175, 164, 153,
        142, 132, 123, 114, 105,  96,  88,  80,
         72,  64,  57,  50,  44,  38,  33,  29,
         24,  20,  16,  12,   9,   5,   2,   0,
    };

    /// <summary>
    /// Select the LTP gain iCDF for the given PERIndex (0, 1, or 2).
    /// Matches libopus <c>silk_LTP_gain_iCDF_ptrs</c>.
    /// </summary>
    internal static byte[] SelectLtpGain(int perIndex)
    {
        return perIndex switch
        {
            0 => LtpGain0,
            1 => LtpGain1,
            2 => LtpGain2,
            _ => throw new ArgumentOutOfRangeException(nameof(perIndex), $"perIndex must be 0, 1, or 2 (got {perIndex}).")
        };
    }
}
