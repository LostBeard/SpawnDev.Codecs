// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 default block-level binary CDF tables (skip, intrabc, txfm_partition,
// skip_mode, etc).
//
// Upstream Copyright (c) 2016, Alliance for Open Media. All rights reserved.
// Upstream license: BSD 2-Clause + AV1 Patent License 1.0. See NOTICE.md.
// Upstream source: aomedia.googlesource.com/aom av1/common/entropymode.c
//   default_skip_txfm_cdfs   (lines 796-798)
//   default_skip_mode_cdfs   (lines 800-801)
//   default_intrabc_cdf      (line  815-816)
//   default_txfm_partition_cdf (lines 785-794)
//
// Constants (av1/common/enums.h):
//   SKIP_CONTEXTS                 = 3
//   SKIP_MODE_CONTEXTS            = 3
//   TXFM_PARTITION_CONTEXTS       = (TX_SIZES - TX_8X8) * 6 - 3 = 21  (TX_SIZES = 5)
//
// Stored as inverse CDF (ICDF: CDF_PROB_TOP - cumprob), padded to CDF_SIZE(2)=3 per row.

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>
/// AV1 default CDFs for block-level binary symbols (skip flag, intrabc flag,
/// txfm partition split, etc).
/// </summary>
internal static class Av1DefaultBlockCdfs
{
    /// <summary>
    /// <c>default_skip_txfm_cdfs[SKIP_CONTEXTS][CDF_SIZE(2)]</c>.
    /// Skip-coefficient flag per skip-context (above + left skip count).
    /// </summary>
    public static readonly ushort[][] DefaultSkipTxfmCdf = new ushort[][]
    {
        new ushort[] { 1097, 0, 0 },
        new ushort[] { 16253, 0, 0 },
        new ushort[] { 28192, 0, 0 },
    };

    /// <summary>
    /// <c>default_skip_mode_cdfs[SKIP_MODE_CONTEXTS][CDF_SIZE(2)]</c>.
    /// Skip-mode flag (inter-frame fast path).
    /// </summary>
    public static readonly ushort[][] DefaultSkipModeCdf = new ushort[][]
    {
        new ushort[] { 147, 0, 0 },   // 32768 - 32621
        new ushort[] { 12060, 0, 0 }, // 32768 - 20708
        new ushort[] { 24641, 0, 0 }, // 32768 - 8127
    };

    /// <summary>
    /// <c>default_intrabc_cdf[CDF_SIZE(2)]</c>.
    /// Intra-block-copy flag (used in intra-frame screen content path).
    /// </summary>
    public static readonly ushort[] DefaultIntrabcCdf =
        new ushort[] { 2237, 0, 0 };

    /// <summary>
    /// <c>default_txfm_partition_cdf[TXFM_PARTITION_CONTEXTS][CDF_SIZE(2)]</c>.
    /// Per-context transform-partition split flag (whether a TX block is
    /// further subdivided).
    /// </summary>
    public static readonly ushort[][] DefaultTxfmPartitionCdf = new ushort[][]
    {
        new ushort[] { 4187, 0, 0 },
        new ushort[] { 8922, 0, 0 },
        new ushort[] { 11921, 0, 0 },
        new ushort[] { 8453, 0, 0 },
        new ushort[] { 14572, 0, 0 },
        new ushort[] { 20635, 0, 0 },
        new ushort[] { 13977, 0, 0 },
        new ushort[] { 21881, 0, 0 },
        new ushort[] { 21763, 0, 0 },
        new ushort[] { 5589, 0, 0 },
        new ushort[] { 12764, 0, 0 },
        new ushort[] { 21487, 0, 0 },
        new ushort[] { 6219, 0, 0 },
        new ushort[] { 13460, 0, 0 },
        new ushort[] { 18544, 0, 0 },
        new ushort[] { 4753, 0, 0 },
        new ushort[] { 11222, 0, 0 },
        new ushort[] { 18368, 0, 0 },
        new ushort[] { 4603, 0, 0 },
        new ushort[] { 10367, 0, 0 },
        new ushort[] { 16680, 0, 0 },
    };
}
