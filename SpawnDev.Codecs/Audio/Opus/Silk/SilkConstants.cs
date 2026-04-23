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
}
