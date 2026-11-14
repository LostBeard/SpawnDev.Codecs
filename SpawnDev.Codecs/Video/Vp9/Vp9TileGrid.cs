// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// The encoder's VP9 tile grid: how many tile columns / rows a frame is coded in and the superblocks each tile covers.
// Boundaries are libvpx get_tile_offset (vp9/common/vp9_tile_common.c): offset = ((idx * sb_cols) >> log2) << 3 mode-info
// units, i.e. superblock (idx * sbCols) >> log2 - the formula the decoder here uses (Vp9ParseKernels.Tiles.cs,
// Vp9ReconKernels.cs, References Vp9TileLayout). The column count is limited by calc_min / calc_max_log2_tile_cols
// (a tile is 4 to 64 superblocks wide), rows by 0..2.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>A frame's tile grid (log2 tile columns / rows, in the legal range for its width).</summary>
public readonly struct Vp9TileGrid
{
    /// <summary>Log2 of the tile columns and rows.</summary>
    public readonly int Log2Cols, Log2Rows;
    /// <summary>Superblock columns / rows of the frame.</summary>
    public readonly int SbCols, SbRows;

    private Vp9TileGrid(int log2Cols, int log2Rows, int sbCols, int sbRows)
    {
        Log2Cols = log2Cols; Log2Rows = log2Rows; SbCols = sbCols; SbRows = sbRows;
    }

    /// <summary>
    /// The grid of a <paramref name="width"/> x <paramref name="height"/> frame with the requested log2 tile columns
    /// (clamped to what the width allows; -1 = the most it allows) and rows (clamped to 0..2).
    /// </summary>
    public static Vp9TileGrid For(int width, int height, int log2Cols, int log2Rows)
    {
        int miCols = (width + 7) >> 3, miRows = (height + 7) >> 3;
        int sbCols = (miCols + 7) >> 3, sbRows = (miRows + 7) >> 3;
        var (min, max) = Vp9TileInfoParser.GetTileNBits(miCols);
        int cols = log2Cols < 0 ? max : Math.Clamp(log2Cols, min, max);
        int rows = Math.Clamp(log2Rows, 0, 2);
        while (rows > 0 && (1 << rows) > sbRows) rows--;   // no tile row without a superblock row
        return new Vp9TileGrid(cols, rows, sbCols, sbRows);
    }

    /// <summary>Tile columns, rows, and tiles.</summary>
    public int TileCols => 1 << Log2Cols;
    /// <summary>See <see cref="TileCols"/>.</summary>
    public int TileRows => 1 << Log2Rows;
    /// <summary>See <see cref="TileCols"/>.</summary>
    public int Count => TileCols * TileRows;

    /// <summary>The header kernels' tile argument: log2 columns | log2 rows &lt;&lt; 8.</summary>
    public int HeaderArg => Log2Cols | (Log2Rows << 8);

    /// <summary>First superblock column of tile column <paramref name="c"/> (c = TileCols: the frame's width in superblocks).</summary>
    public int SbColStart(int c) => (c * SbCols) >> Log2Cols;
    /// <summary>First superblock row of tile row <paramref name="r"/> (r = TileRows: the frame's height in superblocks).</summary>
    public int SbRowStart(int r) => (r * SbRows) >> Log2Rows;

    /// <summary>Tile <paramref name="t"/> (row-major: t = row * TileCols + col): its superblock rectangle.</summary>
    public (int SbCol0, int SbCol1, int SbRow0, int SbRow1) Tile(int t)
    {
        int r = t >> Log2Cols, c = t & (TileCols - 1);
        return (SbColStart(c), SbColStart(c + 1), SbRowStart(r), SbRowStart(r + 1));
    }
}
