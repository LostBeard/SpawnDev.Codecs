// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 default delta_q + delta_lf CDF tables.
//
// Upstream Copyright (c) 2016, Alliance for Open Media. All rights reserved.
// Upstream license: BSD 2-Clause + AV1 Patent License 1.0. See NOTICE.md.
// Upstream source: aomedia.googlesource.com/aom av1/common/entropymode.c
//   default_delta_q_cdf       (line 840)
//   default_delta_lf_cdf      (line 849)
//   default_delta_lf_multi_cdf (lines 844-848)
//
// Constants (av1/common/enums.h):
//   DELTA_Q_SMALL  = 3
//   DELTA_Q_PROBS  = DELTA_Q_SMALL = 3   (so delta_q_cdf has 4 active syms)
//   DELTA_LF_SMALL = 3
//   DELTA_LF_PROBS = DELTA_LF_SMALL = 3
//   FRAME_LF_COUNT = 4
//
// Stored as inverse CDF (ICDF: CDF_PROB_TOP - cumprob), padded to libaom's
// CDF_SIZE() row width per AOM_CDFn macro semantics. Compatible directly
// with Av1RangeDecoder.DecodeCdfQ15.

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>
/// AV1 default CDFs for delta_q + delta_lf entropy decode (per-superblock
/// quant + loop filter delta signaling).
/// </summary>
internal static class Av1DefaultDeltaCdfs
{
    /// <summary>
    /// <c>default_delta_q_cdf[CDF_SIZE(DELTA_Q_PROBS + 1)]</c>.
    /// 4 active symbols. AOM_CDF4(28160, 32120, 32677) -> ICDF [4608, 648, 91, 0, 0].
    /// </summary>
    public static readonly ushort[] DefaultDeltaQCdf =
        new ushort[] { 4608, 648, 91, 0, 0 };

    /// <summary>
    /// <c>default_delta_lf_cdf[CDF_SIZE(DELTA_LF_PROBS + 1)]</c>.
    /// 4 active symbols. AOM_CDF4(28160, 32120, 32677) -> ICDF [4608, 648, 91, 0, 0].
    /// </summary>
    public static readonly ushort[] DefaultDeltaLfCdf =
        new ushort[] { 4608, 648, 91, 0, 0 };

    /// <summary>
    /// <c>default_delta_lf_multi_cdf[FRAME_LF_COUNT][CDF_SIZE(DELTA_LF_PROBS + 1)]</c>.
    /// 4 rows; identical CDF per LF id by upstream default.
    /// </summary>
    public static readonly ushort[][] DefaultDeltaLfMultiCdf = new ushort[][]
    {
        new ushort[] { 4608, 648, 91, 0, 0 },
        new ushort[] { 4608, 648, 91, 0, 0 },
        new ushort[] { 4608, 648, 91, 0, 0 },
        new ushort[] { 4608, 648, 91, 0, 0 },
    };

    /// <summary>libaom <c>DELTA_Q_SMALL</c>.</summary>
    public const int DeltaQSmall = 3;

    /// <summary>libaom <c>DELTA_Q_PROBS</c>.</summary>
    public const int DeltaQProbs = 3;

    /// <summary>libaom <c>DELTA_LF_SMALL</c>.</summary>
    public const int DeltaLfSmall = 3;

    /// <summary>libaom <c>DELTA_LF_PROBS</c>.</summary>
    public const int DeltaLfProbs = 3;
}
