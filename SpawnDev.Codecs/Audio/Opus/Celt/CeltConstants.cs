// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Structural port of libopus celt/celt.h + celt/modes.c + celt/mdct.h selected
// constants to clean C#. CELT (Constrained Energy Lapped Transform) is the
// transform-coded half of Opus, complementing SILK. Used for music, wideband
// audio, and low-delay coding.
//
// Upstream Copyright (c) 2007-2008 CSIRO, 2007-2009 Xiph.Org Foundation,
// 2008 Gregory Maxwell. BSD 3-Clause. See NOTICE.md.

namespace SpawnDev.Codecs.Audio.Opus.Celt;

/// <summary>
/// CELT codec constants. Values match the libopus 48 kHz / 5 ms-frame
/// configuration used for all of Opus' CELT modes.
/// </summary>
internal static class CeltConstants
{
    // ----- Frame geometry (48 kHz) -----

    /// <summary>Maximum supported sample rate in Hz. Opus runs CELT at 48 kHz internally.</summary>
    internal const int MAX_SAMPLE_RATE_HZ = 48000;

    /// <summary>
    /// Smallest CELT frame size in samples at 48 kHz. 2.5 ms = 120 samples.
    /// </summary>
    internal const int FRAME_SIZE_2_5MS = 120;

    /// <summary>5 ms frame (240 samples at 48 kHz).</summary>
    internal const int FRAME_SIZE_5MS = 240;

    /// <summary>10 ms frame (480 samples at 48 kHz).</summary>
    internal const int FRAME_SIZE_10MS = 480;

    /// <summary>20 ms frame (960 samples at 48 kHz).</summary>
    internal const int FRAME_SIZE_20MS = 960;

    // ----- Band structure -----

    /// <summary>
    /// Total number of frequency bands at 5 ms / 48 kHz (fullband coverage).
    /// Libopus <c>eband5ms[]</c> has <c>nbEBands + 1 = 22</c> entries, so nbEBands = 21.
    /// </summary>
    internal const int NB_BANDS_FULLBAND = 21;

    /// <summary>Number of critical bands included for Narrowband (4 kHz cutoff).</summary>
    internal const int NB_BANDS_NB = 13;

    /// <summary>Number of critical bands included for Wideband (8 kHz cutoff).</summary>
    internal const int NB_BANDS_WB = 17;

    /// <summary>Number of critical bands included for Superwideband (12 kHz cutoff).</summary>
    internal const int NB_BANDS_SWB = 19;

    /// <summary>
    /// Critical-band boundary table at 5 ms / 48 kHz. Each entry is an MDCT-bin index;
    /// band <c>k</c> spans bins <c>[eband5ms[k], eband5ms[k+1])</c>.
    ///
    /// At 48 kHz / 5 ms the MDCT has 120 bins, so each bin is 200 Hz wide. The
    /// corresponding Hz values for the boundaries (from libopus) are:
    /// <code>
    ///   0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 2000, 2400, 2800,
    ///   3200, 4000, 4800, 5600, 6800, 8000, 9600, 12000, 15600, 20000
    /// </code>
    /// </summary>
    internal static readonly short[] Eband5Ms =
    {
        0,  1,  2,  3,  4,  5,  6,  7,  8, 10, 12,
       14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100
    };

    /// <summary>
    /// Total number of valid (bandwidth, frame-size) configurations. Configs 0-15 are
    /// SILK or Hybrid; configs 16-31 are CELT. At 48 kHz output, CELT covers
    /// 2.5/5/10/20 ms at NB/WB/SWB/FB.
    /// </summary>
    internal const int NB_CONFIGS = 32;

    // ----- Fixed-point shifts and Q-format constants (libopus celt/celt.h) -----

    /// <summary>Q15 unity. <c>Q15ONE = 0x7FFF</c>.</summary>
    internal const int Q15ONE = 32767;

    /// <summary>Float scale used to convert PCM samples between [-1, +1] and the int16 range used internally.</summary>
    internal const float CELT_SIG_SCALE = 32768.0f;

    /// <summary>Right-shift applied when rounding sig samples back to int16.</summary>
    internal const int SIG_SHIFT = 12;

    /// <summary>Norm scaling used by PVQ codec.</summary>
    internal const int NORM_SCALING = 16384;

    /// <summary>Right-shift used by the dB-domain energy quantization.</summary>
    internal const int DB_SHIFT = 10;

    /// <summary>Smallest positive value treated as non-zero by the energy quantizer.</summary>
    internal const int EPSILON = 1;

    // ----- Comb filter / post-filter (libopus celt/celt.h) -----

    /// <summary>Maximum pitch period in samples. Defines the comb-filter / post-filter window upper bound.</summary>
    internal const int COMBFILTER_MAXPERIOD = 1024;

    /// <summary>Minimum allowed pitch period in samples for the comb-filter / post-filter.</summary>
    internal const int COMBFILTER_MINPERIOD = 15;

    // ----- Decode buffer + PLC (libopus celt/celt.h, opus_decoder.c) -----

    /// <summary>
    /// Size of the running CELT synthesis history per channel, in samples at 48 kHz.
    /// Defined large enough to hold the longest possible MDCT overlap-add window
    /// plus pitch-based PLC lookahead. From libopus opus_decoder.c.
    /// </summary>
    internal const int DECODE_BUFFER_SIZE = 2048;

    /// <summary>Maximum pitch lag in samples used by the pitch-based PLC; corresponds to ~67 Hz at 48 kHz.</summary>
    internal const int PLC_PITCH_LAG_MAX = 720;

    /// <summary>Minimum pitch lag in samples used by the pitch-based PLC; corresponds to ~480 Hz at 48 kHz.</summary>
    internal const int PLC_PITCH_LAG_MIN = 100;

    /// <summary>Order of the LPC filter used by the pitch-based PLC.</summary>
    internal const int LPC_ORDER = 24;

    /// <summary>Maximum pitch period in samples used by the pitch search.</summary>
    internal const int MAX_PERIOD = 1024;

    // ----- Bit allocation (libopus celt/rate.h) -----

    /// <summary>Number of pre-computed allocation rows per CELT mode.</summary>
    internal const int BITALLOC_SIZE = 11;

    /// <summary>Maximum pseudo-band count used by the bit allocator.</summary>
    internal const int MAX_PSEUDO = 40;

    /// <summary>log2 of <see cref="MAX_PSEUDO"/>.</summary>
    internal const int LOG_MAX_PSEUDO = 6;

    /// <summary>Maximum pulse count used by PVQ.</summary>
    internal const int CELT_MAX_PULSES = 128;

    /// <summary>Maximum bits allocated to fine energy refinement per band.</summary>
    internal const int MAX_FINE_BITS = 8;

    /// <summary>Bias offset applied to fine-energy bit allocations per band.</summary>
    internal const int FINE_OFFSET = 21;

    /// <summary>Bias offset applied to PVQ shape-quant theta angle.</summary>
    internal const int QTHETA_OFFSET = 4;

    /// <summary>Two-phase variant of QTHETA_OFFSET.</summary>
    internal const int QTHETA_OFFSET_TWOPHASE = 16;
}
