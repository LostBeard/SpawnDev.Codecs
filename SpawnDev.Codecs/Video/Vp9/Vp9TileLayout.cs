// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 tile layout calculator. Given frame dimensions in mi (8x8)
// units and the log2 tile counts from the uncompressed header,
// computes the (mi_row_start, mi_row_end, mi_col_start, mi_col_end)
// boundaries of any tile at index (row, col). Mirror of libvpx
// vp9/common/vp9_tile_common.c.
//
// Tiles partition the frame into independently-decodable rectangles
// for parallelism. The layout aligns to 8x8 mi boundaries for rows
// and to 64x64 super-block boundaries for columns. The tile widths
// are quantized to 64 px (8 mi); the tile heights are not (any 8 mi
// boundary works).

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 tile bounds in mi units.</summary>
public readonly record struct Vp9TileBounds(int MiRowStart, int MiRowEnd, int MiColStart, int MiColEnd)
{
    /// <summary>Width of the tile in mi units (8x8 blocks).</summary>
    public int MiWidth => MiColEnd - MiColStart;

    /// <summary>Height of the tile in mi units (8x8 blocks).</summary>
    public int MiHeight => MiRowEnd - MiRowStart;
}

/// <summary>VP9 tile layout calculator.</summary>
public static class Vp9TileLayout
{
    /// <summary>libvpx <c>MI_BLOCK_SIZE_LOG2</c> = 3 (8 mi per superblock side).</summary>
    public const int MiBlockSizeLog2 = 3;

    /// <summary>libvpx <c>MI_BLOCK_SIZE</c> = 8 (mi units per superblock side).</summary>
    public const int MiBlockSize = 1 << MiBlockSizeLog2;

    /// <summary>
    /// Round <paramref name="mis"/> up to the next multiple of
    /// <see cref="MiBlockSize"/> = 8. libvpx
    /// <c>mi_cols_aligned_to_sb</c>.
    /// </summary>
    public static int MiAlignedToSb(int mis)
    {
        return (mis + MiBlockSize - 1) & ~(MiBlockSize - 1);
    }

    /// <summary>
    /// Compute the mi-unit offset of tile column / row
    /// <paramref name="idx"/>. libvpx <c>get_tile_offset</c>:
    ///   sb_cols = mi_aligned_to_sb(mis) / MI_BLOCK_SIZE
    ///   offset = ((idx * sb_cols) &gt;&gt; log2) * MI_BLOCK_SIZE
    ///   return min(offset, mis)
    /// </summary>
    public static int GetTileOffset(int idx, int mis, int log2Tiles)
    {
        if (mis < 0)
            throw new ArgumentOutOfRangeException(nameof(mis), mis, "mis must be >= 0.");
        if (log2Tiles < 0)
            throw new ArgumentOutOfRangeException(nameof(log2Tiles), log2Tiles, "log2Tiles must be >= 0.");
        int sbCols = MiAlignedToSb(mis) >> MiBlockSizeLog2;
        int offset = ((idx * sbCols) >> log2Tiles) << MiBlockSizeLog2;
        return Math.Min(offset, mis);
    }

    /// <summary>
    /// Compute the bounds of tile (<paramref name="tileRow"/>,
    /// <paramref name="tileCol"/>) given the frame's mi dimensions
    /// and log2 tile counts. Mirror of libvpx <c>vp9_tile_init</c>.
    /// </summary>
    public static Vp9TileBounds Compute(
        int tileRow, int tileCol,
        int miRows, int miCols,
        int log2TileRows, int log2TileCols)
    {
        int rowStart = GetTileOffset(tileRow, miRows, log2TileRows);
        int rowEnd = GetTileOffset(tileRow + 1, miRows, log2TileRows);
        int colStart = GetTileOffset(tileCol, miCols, log2TileCols);
        int colEnd = GetTileOffset(tileCol + 1, miCols, log2TileCols);
        return new Vp9TileBounds(rowStart, rowEnd, colStart, colEnd);
    }

    /// <summary>libvpx <c>MIN_TILE_WIDTH_B64</c>: minimum tile width in SBs.</summary>
    public const int MinTileWidthSb = 4;

    /// <summary>libvpx <c>MAX_TILE_WIDTH_B64</c>: maximum tile width in SBs.</summary>
    public const int MaxTileWidthSb = 64;

    /// <summary>
    /// Compute (min, max) bounds for the <c>log2_tile_cols</c> field
    /// given the frame's mi-column width. The decoder validates that
    /// the bitstream's value lies in this range. Mirror of libvpx
    /// <c>vp9_get_tile_n_bits</c>.
    /// </summary>
    public static (int Min, int Max) GetLog2TileColsBounds(int miCols)
    {
        if (miCols < 0)
            throw new ArgumentOutOfRangeException(nameof(miCols), miCols, "miCols must be >= 0.");
        int sbCols = MiAlignedToSb(miCols) >> MiBlockSizeLog2;

        int maxLog2 = 1;
        while ((sbCols >> maxLog2) >= MinTileWidthSb) maxLog2++;
        maxLog2--;
        if (maxLog2 < 0) maxLog2 = 0;

        int minLog2 = 0;
        while ((MaxTileWidthSb << minLog2) < sbCols) minLog2++;

        return (minLog2, maxLog2);
    }
}
