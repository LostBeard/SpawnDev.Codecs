// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 partition context tracker. Mirrors libaom MACROBLOCKD's
// above_partition_context + left_partition_context bitmasks plus the
// partition_plane_context() and partition_cdf_length() helpers.
//
// Upstream Copyright (c) 2016, Alliance for Open Media. All rights reserved.
// Upstream license: BSD 2-Clause + AV1 Patent License 1.0. See NOTICE.md.
// Upstream source: aomedia.googlesource.com/aom
//   av1/common/av1_common_int.h:partition_plane_context (line 1541)
//   av1/common/av1_common_int.h:partition_cdf_length    (line 1558)
//   av1/common/av1_common_int.h:update_partition_context (line 1443)
//   av1/common/av1_common_int.h:update_ext_partition_context (line 1502)
//   av1/common/common_data.h:partition_context_lookup   (line 385)
//   av1/common/common_data.h:mi_size_wide_log2          (line 25)
//   av1/common/common_data.h:subsize_lookup             (line 71)
//
// AV1 spec sec 9.3 Partition Context.

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>
/// AV1 partition context tracker for one tile. Maintains per-column-of-mi
/// "above" partition context bytes and per-row-of-mi "left" partition context
/// bytes. The bits in each byte encode whether a same-row/column block at a
/// given log2 size is split.
/// </summary>
internal sealed class Av1PartitionContext
{
    /// <summary>Maximum mode-info block edge in 4x4 units within a 128x128 superblock (libaom MAX_MIB_SIZE).</summary>
    public const int MaxMibSize = 32;
    /// <summary>libaom MAX_MIB_MASK = MAX_MIB_SIZE - 1.</summary>
    public const int MaxMibMask = MaxMibSize - 1;

    /// <summary>Mi_Width_Log2 in 4x4 units, indexed by BLOCK_SIZE (BLOCK_SIZES_ALL = 22).
    /// From av1/common/common_data.h:mi_size_wide_log2.</summary>
    public static readonly byte[] MiSizeWideLog2 = new byte[]
    {
        0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 0, 2, 1, 3, 2, 4
    };

    /// <summary>Mi_Height_Log2 in 4x4 units. From av1/common/common_data.h:mi_size_high_log2.</summary>
    public static readonly byte[] MiSizeHighLog2 = new byte[]
    {
        0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 2, 0, 3, 1, 4, 2
    };

    /// <summary>Block width in 4x4 units. From av1/common/common_data.h:mi_size_wide.</summary>
    public static readonly byte[] MiSizeWide = new byte[]
    {
        1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 1, 4, 2, 8, 4, 16
    };

    /// <summary>Block height in 4x4 units. From av1/common/common_data.h:mi_size_high.</summary>
    public static readonly byte[] MiSizeHigh = new byte[]
    {
        1, 2, 1, 2, 4, 2, 4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 4, 1, 8, 2, 16, 4
    };

    /// <summary>BLOCK_8X8 index in libaom enum (mi_size_wide_log2[BLOCK_8X8] = 1).</summary>
    public const int Block8x8 = 3;
    /// <summary>BLOCK_8X8 log2-mi-size in libaom (mi_size_wide_log2[BLOCK_8X8] = 1).</summary>
    public const int Block8x8Log2 = 1;
    /// <summary>BLOCK_128X128 index in libaom enum.</summary>
    public const int Block128x128 = 15;

    /// <summary>partition_context_lookup[BLOCK_SIZES_ALL].above (libaom common_data.h line 385).</summary>
    public static readonly byte[] PartitionContextLookupAbove = new byte[]
    {
        31, 31, 30, 30, 30, 28, 28, 28, 24, 24, 24, 16, 16, 16, 0, 0,
        31, 28, 30, 24, 28, 16
    };
    /// <summary>partition_context_lookup[BLOCK_SIZES_ALL].left (libaom common_data.h line 385).</summary>
    public static readonly byte[] PartitionContextLookupLeft = new byte[]
    {
        31, 30, 31, 30, 28, 30, 28, 24, 28, 24, 16, 24, 16, 0, 16, 0,
        28, 31, 24, 30, 16, 28
    };

    private readonly byte[] _aboveCtx;
    private readonly byte[] _leftCtx;
    private readonly int _tileMiCols;

    /// <summary>
    /// Construct a context tracker for a tile of the given mi-coord size.
    /// <paramref name="tileMiCols"/> is the tile width in 4x4 units (one byte per column).
    /// </summary>
    public Av1PartitionContext(int tileMiCols)
    {
        if (tileMiCols < 0) throw new ArgumentOutOfRangeException(nameof(tileMiCols));
        _tileMiCols = tileMiCols;
        _aboveCtx = new byte[Math.Max(tileMiCols, 1)];
        _leftCtx = new byte[MaxMibSize];
    }

    /// <summary>Reset all above contexts (call at the start of each tile / superblock row).</summary>
    public void ResetAbove()
    {
        Array.Fill(_aboveCtx, (byte)0);
    }

    /// <summary>Reset all left contexts (call at the start of each tile / superblock column).</summary>
    public void ResetLeft()
    {
        Array.Fill(_leftCtx, (byte)0);
    }

    /// <summary>
    /// Compute the partition_plane_context for a block at (miRow, miCol) of size <paramref name="bsize"/>.
    /// Mirrors libaom <c>partition_plane_context()</c> in av1/common/av1_common_int.h.
    /// </summary>
    public int GetContext(int miRow, int miCol, int bsize)
    {
        if (bsize < 0 || bsize >= MiSizeWideLog2.Length)
            throw new ArgumentOutOfRangeException(nameof(bsize));
        if (miCol < 0 || miCol >= _aboveCtx.Length)
            throw new ArgumentOutOfRangeException(nameof(miCol));

        int bsl = MiSizeWideLog2[bsize] - Block8x8Log2;
        if (bsl < 0) throw new ArgumentException("Block must be >= 8x8 for partition CDF lookup.", nameof(bsize));

        int above = (_aboveCtx[miCol] >> bsl) & 1;
        int left = (_leftCtx[miRow & MaxMibMask] >> bsl) & 1;
        return (left * 2 + above) + bsl * Av1PartitionConstants.PartitionPlaneOffset;
    }

    /// <summary>
    /// After-decoding update for the above + left context arrays. Mirrors libaom
    /// <c>update_partition_context()</c> for the simple case (NONE, HORZ, VERT,
    /// HORZ_4, VERT_4, and SPLIT-of-8x8). For HORZ_A/B, VERT_A/B the caller
    /// should invoke this twice with the appropriate sub-block sizes per
    /// libaom <c>update_ext_partition_context()</c>.
    /// </summary>
    public void UpdateContext(int miRow, int miCol, int subsize)
    {
        if (subsize < 0 || subsize >= PartitionContextLookupAbove.Length)
            throw new ArgumentOutOfRangeException(nameof(subsize));

        byte aboveVal = PartitionContextLookupAbove[subsize];
        byte leftVal = PartitionContextLookupLeft[subsize];

        // bw / bh are in 4x4 units (mi units).
        int bw = MiSizeWide[subsize];
        int bh = MiSizeHigh[subsize];

        int colEnd = Math.Min(_aboveCtx.Length, miCol + bw);
        for (int c = miCol; c < colEnd; c++)
            _aboveCtx[c] = aboveVal;

        int leftStart = miRow & MaxMibMask;
        int rowEnd = Math.Min(MaxMibSize, leftStart + bh);
        for (int r = leftStart; r < rowEnd; r++)
            _leftCtx[r] = leftVal;
    }

    /// <summary>
    /// Number of active partition CDF symbols for a block size. Mirrors libaom
    /// <c>partition_cdf_length()</c>.
    /// </summary>
    public static int PartitionCdfLength(int bsize)
    {
        if (bsize <= Block8x8) return Av1PartitionConstants.PartitionTypes;
        if (bsize == Block128x128) return Av1PartitionConstants.ExtPartitionTypes - 2;
        return Av1PartitionConstants.ExtPartitionTypes;
    }
}
