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
}
