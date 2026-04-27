// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 default partition CDF tables.
//
// Upstream Copyright (c) 2016, Alliance for Open Media. All rights reserved.
// Upstream license: BSD 2-Clause + AV1 Patent License 1.0. See NOTICE.md.
// Upstream source: aomedia.googlesource.com/aom av1/common/entropymode.c
//   default_partition_cdf[PARTITION_CONTEXTS][CDF_SIZE(EXT_PARTITION_TYPES)]
//   (lines 154-176).
//
// Layout: 20 contexts (PARTITION_CONTEXTS = PARTITION_BLOCK_SIZES *
// PARTITION_PLOFFSET = 5*4). Context = partition_plane_context(blockSize, row, col)
// in libaom blockd.h:partition_plane_context(); spec sec 9.3.
//
// Per-context symbol set varies by block size:
//   - Contexts 0..3  (8x8 splits)    : 4 symbols  - PARTITION_TYPES (NONE, HORZ, VERT, SPLIT only)
//   - Contexts 4..15 (16/32/64 sb)   : 10 symbols - all EXT_PARTITION_TYPES
//   - Contexts 16..19 (128x128 sb)   : 8 symbols  - HORZ_4 / VERT_4 not allowed
//
// Each row is sized CDF_SIZE(EXT_PARTITION_TYPES) = 11 ushorts. Trailing
// entries past the active symbol set are zero (matches the 0-padded C struct
// shape produced by the AOM_CDFn macros + zero-init for unused fields).
//
// Stored as inverse CDF (icdf[i] = CDF_PROB_TOP - cumulative_prob(i)) per
// AOM_ICDF macro. Compatible directly with Av1RangeDecoder.DecodeCdfQ15.

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>
/// AV1 default partition CDFs. Indexed by partition plane context (0..19).
/// Pass the appropriate row + active symbol count to
/// <see cref="EntropyCoders.Av1RangeDecoder.DecodeCdfQ15"/>.
/// </summary>
internal static class Av1DefaultPartitionCdfs
{
    /// <summary>Active symbol count for a given partition context.</summary>
    public static int SymbolCount(int ctx) => ctx switch
    {
        >= 0 and <= 3 => Av1PartitionConstants.PartitionTypes,            // 4
        >= 4 and <= 15 => Av1PartitionConstants.ExtPartitionTypes,        // 10
        >= 16 and <= 19 => Av1PartitionConstants.ExtPartitionTypes - 2,   // 8 (no HORZ_4 / VERT_4)
        _ => throw new ArgumentOutOfRangeException(nameof(ctx)),
    };

    /// <summary>
    /// Default partition CDFs. <c>DefaultPartitionCdf[ctx]</c> is the inverse-CDF
    /// row for partition context <c>ctx</c> in [0, 19].
    /// </summary>
    public static readonly ushort[][] DefaultPartitionCdf = new ushort[][]
    {
        // PARTITION_TYPES = 4 (smallest blocks)
        new ushort[] { 13636, 7258, 2376, 0, 0 }, // ctx 0
        new ushort[] { 18840, 12913, 4228, 0, 0 }, // ctx 1
        new ushort[] { 20246, 9089, 4139, 0, 0 }, // ctx 2
        new ushort[] { 22872, 13985, 6915, 0, 0 }, // ctx 3
        // EXT_PARTITION_TYPES = 10 (mid sizes)
        new ushort[] { 17171, 11839, 8197, 6062, 5104, 3947, 3167, 2197, 866, 0, 0 }, // ctx 4
        new ushort[] { 24843, 21725, 15983, 10298, 8797, 7725, 6117, 4067, 2934, 0, 0 }, // ctx 5
        new ushort[] { 27354, 19499, 17657, 12280, 10408, 8268, 7231, 6432, 651, 0, 0 }, // ctx 6
        new ushort[] { 30106, 26406, 24154, 11908, 9715, 7990, 6332, 4939, 1597, 0, 0 }, // ctx 7
        new ushort[] { 14306, 11848, 9644, 5121, 4541, 3719, 3249, 2590, 1224, 0, 0 }, // ctx 8
        new ushort[] { 25079, 23708, 20712, 7776, 7108, 6586, 5817, 4727, 3716, 0, 0 }, // ctx 9
        new ushort[] { 26753, 23759, 22706, 8224, 7359, 6223, 5697, 5242, 721, 0, 0 }, // ctx 10
        new ushort[] { 31374, 30560, 29972, 4154, 3707, 3302, 2928, 2583, 869, 0, 0 }, // ctx 11
        new ushort[] { 12631, 11221, 9690, 3202, 2931, 2507, 2244, 1876, 1044, 0, 0 }, // ctx 12
        new ushort[] { 26036, 25278, 23271, 4824, 4518, 4253, 3799, 3138, 2664, 0, 0 }, // ctx 13
        new ushort[] { 26823, 25105, 24420, 4085, 3651, 3019, 2704, 2470, 530, 0, 0 }, // ctx 14
        new ushort[] { 31898, 31556, 31281, 1570, 1374, 1194, 1025, 887, 436, 0, 0 }, // ctx 15
        // 8 symbols (128x128 superblock; no HORZ_4 / VERT_4)
        new ushort[] { 4869, 4549, 4239, 284, 229, 149, 129, 0, 0, 0, 0 }, // ctx 16
        new ushort[] { 26161, 25778, 24500, 708, 549, 430, 397, 0, 0, 0, 0 }, // ctx 17
        new ushort[] { 27339, 26092, 25646, 741, 541, 237, 186, 0, 0, 0, 0 }, // ctx 18
        new ushort[] { 32057, 31802, 31596, 320, 230, 151, 104, 0, 0, 0, 0 }, // ctx 19
    };
}
