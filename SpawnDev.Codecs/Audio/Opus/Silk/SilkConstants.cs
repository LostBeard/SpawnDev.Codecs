// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Structural port of selected constants from libopus silk/define.h.
// Upstream Copyright (c) 2006-2011 Skype Limited. BSD 3-Clause. See NOTICE.md.

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// SILK constants ported from libopus silk/define.h. Only the constants referenced
/// by currently-implemented subsystems live here; additional constants are added
/// in the slices that need them.
/// </summary>
internal static class SilkConstants
{
    // ----- Subframe structure -----

    /// <summary>Maximum number of subframes per SILK frame.</summary>
    internal const int MAX_NB_SUBFR = 4;

    /// <summary>Minimum number of subframes per SILK frame (used for 10ms frames).</summary>
    internal const int MIN_NB_SUBFR = 2;

    // ----- Gain quantization -----

    /// <summary>Minimum quantized gain in dB.</summary>
    internal const int MIN_QGAIN_DB = 2;

    /// <summary>Maximum quantized gain in dB.</summary>
    internal const int MAX_QGAIN_DB = 88;

    /// <summary>Number of scalar quantizer levels for the first subframe's gain.</summary>
    internal const int N_LEVELS_QGAIN = 64;

    /// <summary>Maximum number of delta-gain steps allowed per subframe.</summary>
    internal const int MAX_DELTA_GAIN_QUANT = 36;

    /// <summary>Minimum number of delta-gain steps allowed per subframe (value is negative).</summary>
    internal const int MIN_DELTA_GAIN_QUANT = -4;

    // ----- Derived gain constants (matches libopus gain_quant.c preprocessor math) -----

    /// <summary>
    /// Gain quantizer offset in Q7: <c>(MIN_QGAIN_DB * 128) / 6 + 16 * 128</c>.
    /// Evaluates to <c>2090</c>.
    /// </summary>
    internal const int GAIN_OFFSET_Q7 = (MIN_QGAIN_DB * 128) / 6 + 16 * 128;

    /// <summary>
    /// Gain quantizer scale in Q16: <c>(65536 * (N_LEVELS_QGAIN - 1)) / (((MAX_QGAIN_DB - MIN_QGAIN_DB) * 128) / 6)</c>.
    /// Evaluates to <c>2251</c>.
    /// </summary>
    internal const int GAIN_SCALE_Q16 =
        (65536 * (N_LEVELS_QGAIN - 1)) / (((MAX_QGAIN_DB - MIN_QGAIN_DB) * 128) / 6);

    /// <summary>
    /// Gain quantizer inverse scale in Q16: <c>(65536 * (((MAX_QGAIN_DB - MIN_QGAIN_DB) * 128) / 6)) / (N_LEVELS_QGAIN - 1)</c>.
    /// Evaluates to <c>1907825</c>.
    /// </summary>
    internal const int GAIN_INV_SCALE_Q16 =
        (65536 * (((MAX_QGAIN_DB - MIN_QGAIN_DB) * 128) / 6)) / (N_LEVELS_QGAIN - 1);

    /// <summary>Upper bound on <c>silk_log2lin</c> input (clamped in <c>silk_gains_dequant</c>; 3967 = 31 in Q7).</summary>
    internal const int GAIN_LOG_CLAMP_HIGH_Q7 = 3967;

    // ----- NLSF -----

    /// <summary>Maximum LPC filter order (SILK uses 10 for NB/MB and 16 for WB).</summary>
    internal const int MAX_LPC_ORDER = 16;

    /// <summary>Maximum absolute amplitude of an NLSF quantization residual.</summary>
    internal const int NLSF_QUANT_MAX_AMPLITUDE = 4;

    /// <summary>Cosine lookup table size for NLSF-to-LPC conversion.</summary>
    internal const int LSF_COS_TAB_SZ_FIX = 128;

    /// <summary>
    /// NLSF quantization level adjustment in Q10 format, matching libopus
    /// <c>SILK_FIX_CONST(0.1, 10)</c>. Computed as <c>(int)(0.1 * 1024 + 0.5) = 102</c>.
    /// </summary>
    internal const int NLSF_QUANT_LEVEL_ADJ_Q10 = 102;

    // ----- LPC stability / inverse prediction gain -----

    /// <summary>
    /// Maximum inverse prediction power gain accepted for a stable LPC filter.
    /// Matches libopus <c>MAX_PREDICTION_POWER_GAIN = 1e4f</c> in silk/define.h.
    /// </summary>
    internal const float MAX_PREDICTION_POWER_GAIN = 1e4f;

    /// <summary>
    /// Q30 threshold below which an LPC filter's inverse prediction gain is considered unstable:
    /// <c>SILK_FIX_CONST(1.0f / MAX_PREDICTION_POWER_GAIN, 30)</c>.
    /// Computed as <c>(int)(1e-4 * 2^30 + 0.5) = 107374</c>.
    /// </summary>
    internal const int INV_GAIN_Q30_MIN = 107374;

    /// <summary>
    /// Maximum number of bandwidth-expansion iterations applied in NLSF2A when the
    /// produced LPC filter is unstable. Matches libopus <c>MAX_LPC_STABILIZE_ITERATIONS = 16</c>.
    /// </summary>
    internal const int MAX_LPC_STABILIZE_ITERATIONS = 16;

    // ----- Pulses / shell coder -----

    /// <summary>Log2 of <see cref="SHELL_CODEC_FRAME_LENGTH"/>. Libopus: 4.</summary>
    internal const int LOG2_SHELL_CODEC_FRAME_LENGTH = 4;

    /// <summary>Shell coder block size in samples. Libopus: <c>1 &lt;&lt; LOG2_SHELL_CODEC_FRAME_LENGTH = 16</c>.</summary>
    internal const int SHELL_CODEC_FRAME_LENGTH = 16;

    /// <summary>Maximum number of shell-coder blocks in a frame (for the longest supported SILK frame). Libopus: 20.</summary>
    internal const int MAX_NB_SHELL_BLOCKS = 20;

    /// <summary>Maximum pulse magnitude encodable via a single rate-level iCDF before the LSB-extension escape fires. Libopus: 16.</summary>
    internal const int SILK_MAX_PULSES = 16;

    /// <summary>Number of rate levels in the pulses-per-block iCDF. Libopus: 10.</summary>
    internal const int N_RATE_LEVELS = 10;

    // ----- Pitch estimator constants (silk/pitch_est_defines.h) -----

    /// <summary>Maximum number of subframes used by pitch estimation. Libopus <c>PE_MAX_NB_SUBFR = 4</c>.</summary>
    internal const int PE_MAX_NB_SUBFR = 4;

    /// <summary>Minimum pitch lag in milliseconds. Libopus <c>PE_MIN_LAG_MS = 2</c>.</summary>
    internal const int PE_MIN_LAG_MS = 2;

    /// <summary>Maximum pitch lag in milliseconds. Libopus <c>PE_MAX_LAG_MS = 18</c>.</summary>
    internal const int PE_MAX_LAG_MS = 18;

    /// <summary>Pitch contour codebook size, stage-2 / 20 ms / NB. Libopus <c>PE_NB_CBKS_STAGE2_EXT = 11</c>.</summary>
    internal const int PE_NB_CBKS_STAGE2_EXT = 11;

    /// <summary>Pitch contour codebook size, stage-2 / 10 ms / NB. Libopus <c>PE_NB_CBKS_STAGE2_10MS = 3</c>.</summary>
    internal const int PE_NB_CBKS_STAGE2_10MS = 3;

    /// <summary>Pitch contour codebook size, stage-3 / 20 ms / non-NB. Libopus <c>PE_NB_CBKS_STAGE3_MAX = 34</c>.</summary>
    internal const int PE_NB_CBKS_STAGE3_MAX = 34;

    /// <summary>Pitch contour codebook size, stage-3 / 10 ms / non-NB. Libopus <c>PE_NB_CBKS_STAGE3_10MS = 12</c>.</summary>
    internal const int PE_NB_CBKS_STAGE3_10MS = 12;

    // ----- Excitation decode + PRNG -----

    /// <summary>Constant added to the dequantized pulse amplitude in silk_decode_core. Libopus <c>QUANT_LEVEL_ADJUST_Q10 = 80</c>.</summary>
    internal const int QUANT_LEVEL_ADJUST_Q10 = 80;

    /// <summary>PRNG increment (additive constant) used by <c>silk_RAND</c>. Libopus <c>RAND_INCREMENT = 907633515</c>.</summary>
    internal const int RAND_INCREMENT = 907633515;

    /// <summary>PRNG multiplier used by <c>silk_RAND</c>. Libopus <c>RAND_MULTIPLIER = 196314165</c>.</summary>
    internal const int RAND_MULTIPLIER = 196314165;

    /// <summary>
    /// Quantization offsets in Q10 from libopus <c>silk_Quantization_Offsets_Q10[2][2]</c>.
    /// Indexed by <c>[signalType &gt;&gt; 1][quantOffsetType]</c>: row 0 covers non-voiced
    /// (UV), row 1 covers voiced (V). Columns are LOW / HIGH.
    /// </summary>
    internal static readonly short[,] QUANTIZATION_OFFSETS_Q10 =
    {
        { 100, 240 }, // UVL, UVH (non-voiced)
        {  32, 100 }, // VL,  VH  (voiced)
    };

    // ----- Frame geometry + post-loss / init constants -----

    /// <summary>Maximum SILK internal sample rate in kHz. Libopus <c>MAX_FS_KHZ = 16</c>.</summary>
    internal const int MAX_FS_KHZ = 16;

    /// <summary>Maximum SILK frame length in milliseconds. Libopus <c>MAX_FRAME_LENGTH_MS = 20</c>.</summary>
    internal const int MAX_FRAME_LENGTH_MS = 20;

    /// <summary>SILK subframe length in milliseconds. Libopus <c>SUB_FRAME_LENGTH_MS = 5</c>.</summary>
    internal const int SUB_FRAME_LENGTH_MS = 5;

    /// <summary>LTP buffer length in milliseconds. Libopus <c>LTP_MEM_LENGTH_MS = 20</c>.</summary>
    internal const int LTP_MEM_LENGTH_MS = 20;

    /// <summary>Maximum SILK frame length in samples: <c>MAX_FRAME_LENGTH_MS * MAX_FS_KHZ = 320</c>.</summary>
    internal const int MAX_FRAME_LENGTH = MAX_FRAME_LENGTH_MS * MAX_FS_KHZ;

    /// <summary>Maximum SILK subframe length in samples: <c>SUB_FRAME_LENGTH_MS * MAX_FS_KHZ = 80</c>.</summary>
    internal const int MAX_SUB_FRAME_LENGTH = SUB_FRAME_LENGTH_MS * MAX_FS_KHZ;

    /// <summary>Maximum LTP buffer length in samples: <c>LTP_MEM_LENGTH_MS * MAX_FS_KHZ = 320</c>.</summary>
    internal const int MAX_LTP_MEM_LENGTH = LTP_MEM_LENGTH_MS * MAX_FS_KHZ;

    /// <summary>
    /// Q16 chirp factor applied to LPC coefficients after a lost-packet reset to widen the
    /// filter and improve packet-loss robustness. Matches libopus <c>BWE_AFTER_LOSS_Q16 = 63570</c>
    /// (~= 0.970 in Q16).
    /// </summary>
    internal const int BWE_AFTER_LOSS_Q16 = 63570;

    /// <summary>Signal type: no voice activity. Libopus <c>TYPE_NO_VOICE_ACTIVITY = 0</c>.</summary>
    internal const int TYPE_NO_VOICE_ACTIVITY = 0;

    /// <summary>Signal type: unvoiced. Libopus <c>TYPE_UNVOICED = 1</c>.</summary>
    internal const int TYPE_UNVOICED = 1;

    /// <summary>Signal type: voiced. Libopus <c>TYPE_VOICED = 2</c>.</summary>
    internal const int TYPE_VOICED = 2;
}
