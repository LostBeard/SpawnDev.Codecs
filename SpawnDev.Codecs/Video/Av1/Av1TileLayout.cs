// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// The tile grid of an AV1 frame with uniform tile spacing (AV1 spec 5.9.15 tile_info, 64x64 superblocks).
//
// Tiles are the encoder's unit of parallelism (one GPU thread per tile, see Av1FrameSequentialEncodeKernel), so the
// layout is chosen from ONE number: the largest tile size wanted, in superblocks per side. 1 = one superblock per
// tile (the most threads, the most bits lost at tile edges); a large value = the fewest tiles the frame size allows
// (a single tile up to 4096 pixels wide and 4096 x 2304 pixels of area).
//
// With uniform spacing a frame header codes only TileColsLog2 and TileRowsLog2; the decoder derives
//     tileWidthSb = (sbCols + (1 << TileColsLog2) - 1) >> TileColsLog2,   TileCols = ceil(sbCols / tileWidthSb)
// and the same for rows. TileColsLog2 lies between minLog2TileCols (tiles at most 4096 pixels wide) and
// maxLog2TileCols (tile_log2(1, min(sbCols, 64)): at most 64 tile columns), rows between
// max(minLog2Tiles - TileColsLog2, 0) and maxLog2TileRows.
//
// How many tiles a decoder takes. The syntax allows 64 x 64. Measured 2026-11-13 with FFmpeg 8.1 on an RTX 4070
// (keyframes of this encoder, one superblock per tile): libdav1d and libaom decode every count tried (2040 tiles at
// 3840x2160); the hardware decoders (NVDEC av1_cuvid, D3D11VA, D3D12VA) decode up to 256 tiles byte for byte like
// libdav1d (also as 64 x 4 and 4 x 64) and FAIL from 272 tiles on. So a frame gets at most maxTiles tiles,
// HardwareDecoderMaxTiles (256) unless the caller raises it: when the wanted tile size gives more, the tiles grow
// (columns or rows are halved) to the layout with the smallest tiles that stays within the limit.
//
// AV1 level limits (MaxTiles 8 .. 128, MaxTileCols 4 .. 16 by level, cropped tiles of at least 8 pixels) are not
// applied by Compute unless the caller passes them (maxTileCols, minEdgeTilePixels): the decoders above do not
// enforce them (104 tiles in 13 columns at 832x480 decode on all six). Av1Level.TileLayout passes the limits of a
// level; the sequence header signals the level the grid really needs (Av1Level.Select), 31 when it fits none.

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>The tile grid of an AV1 frame (uniform tile spacing, 64x64 superblocks).</summary>
public readonly struct Av1TileLayout
{
    /// <summary>Superblock columns of the frame.</summary>
    public int SbCols { get; init; }
    /// <summary>Superblock rows of the frame.</summary>
    public int SbRows { get; init; }
    /// <summary>The coded <c>TileColsLog2</c>.</summary>
    public int TileColsLog2 { get; init; }
    /// <summary>The coded <c>TileRowsLog2</c>.</summary>
    public int TileRowsLog2 { get; init; }
    /// <summary>Smallest codable <c>TileColsLog2</c> for the frame size.</summary>
    public int MinLog2TileCols { get; init; }
    /// <summary>Largest codable <c>TileColsLog2</c> for the frame size.</summary>
    public int MaxLog2TileCols { get; init; }
    /// <summary>Smallest codable <c>TileRowsLog2</c> given <see cref="TileColsLog2"/>.</summary>
    public int MinLog2TileRows { get; init; }
    /// <summary>Largest codable <c>TileRowsLog2</c> for the frame size.</summary>
    public int MaxLog2TileRows { get; init; }
    /// <summary>Tile width in superblocks (the last tile column may be narrower).</summary>
    public int TileWidthSb { get; init; }
    /// <summary>Tile height in superblocks (the last tile row may be shorter).</summary>
    public int TileHeightSb { get; init; }
    /// <summary>Tile columns.</summary>
    public int TileCols { get; init; }
    /// <summary>Tile rows.</summary>
    public int TileRows { get; init; }

    /// <summary>Tiles of the frame, in raster order (<c>row * TileCols + column</c>).</summary>
    public int TileCount => TileCols * TileRows;

    /// <summary>
    /// The most tiles per frame the hardware AV1 decoders measured decode (NVDEC, D3D11VA, D3D12VA on an RTX 4070:
    /// 256 decode, 272 fail). The default limit of <see cref="Compute"/>.
    /// </summary>
    public const int HardwareDecoderMaxTiles = 256;

    /// <summary>
    /// The layout of a <paramref name="width"/> x <paramref name="height"/> frame whose tiles are at most
    /// <paramref name="tileSizeSuperblocks"/> superblocks wide and high where the frame size allows it (at most 64
    /// tile columns and 64 tile rows), and the fewest tiles possible when it is large. When that gives more than
    /// <paramref name="maxTiles"/> tiles, the tiles grow to the layout with the smallest tiles (in superblocks) that
    /// has at most <paramref name="maxTiles"/>; a picture too large for that gets the fewest tiles its size allows.
    /// The same holds for the two limits an AV1 level adds: at most <paramref name="maxTileCols"/> tile columns, and
    /// a last tile column and row that keep at least <paramref name="minEdgeTilePixels"/> pixels of the picture
    /// (a level needs 8; the defaults, 64 and 1, limit nothing).
    /// </summary>
    public static Av1TileLayout Compute(int width, int height, int tileSizeSuperblocks, int maxTiles = HardwareDecoderMaxTiles,
        int maxTileCols = 64, int minEdgeTilePixels = 1)
    {
        if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width));
        if (height <= 0) throw new ArgumentOutOfRangeException(nameof(height));
        if (tileSizeSuperblocks < 1) throw new ArgumentOutOfRangeException(nameof(tileSizeSuperblocks), "tileSizeSuperblocks must be at least 1.");
        if (maxTiles < 1) throw new ArgumentOutOfRangeException(nameof(maxTiles), "maxTiles must be at least 1.");
        if (maxTileCols < 1) throw new ArgumentOutOfRangeException(nameof(maxTileCols), "maxTileCols must be at least 1.");
        if (minEdgeTilePixels < 1) throw new ArgumentOutOfRangeException(nameof(minEdgeTilePixels), "minEdgeTilePixels must be at least 1.");

        int miCols = ((width + 7) >> 3) << 1, miRows = ((height + 7) >> 3) << 1;
        int sbCols = (miCols + 15) >> 4, sbRows = (miRows + 15) >> 4;
        const int maxTileWidthSb = 4096 >> 6;                   // 64 superblocks
        const int maxTileAreaSb = (4096 * 2304) >> 12;          // 2304 superblocks
        int maxLog2Cols = TileLog2(1, Math.Min(sbCols, 64));
        int maxLog2Rows = TileLog2(1, Math.Min(sbRows, 64));
        int minLog2Cols = TileLog2(maxTileWidthSb, sbCols);
        int minLog2Tiles = Math.Max(minLog2Cols, TileLog2(maxTileAreaSb, sbRows * sbCols));

        int colsLog2 = minLog2Cols;
        while (colsLog2 < maxLog2Cols && ((sbCols + (1 << colsLog2) - 1) >> colsLog2) > tileSizeSuperblocks) colsLog2++;
        int tileWidthSb = (sbCols + (1 << colsLog2) - 1) >> colsLog2;
        int tileCols = (sbCols + tileWidthSb - 1) / tileWidthSb;

        int minLog2Rows = Math.Max(minLog2Tiles - colsLog2, 0);
        int rowsLog2 = minLog2Rows;
        while (rowsLog2 < maxLog2Rows && ((sbRows + (1 << rowsLog2) - 1) >> rowsLog2) > tileSizeSuperblocks) rowsLog2++;
        int tileHeightSb = (sbRows + (1 << rowsLog2) - 1) >> rowsLog2;
        int tileRows = (sbRows + tileHeightSb - 1) / tileHeightSb;

        // Within the limits: tile count, tile columns, and the picture pixels left to the last tile column / row.
        bool Fits(int w, int cols, int h, int rows) =>
            (long)cols * rows <= maxTiles && cols <= maxTileCols
            && width - (cols - 1) * (long)w * 64 >= minEdgeTilePixels && height - (rows - 1) * (long)h * 64 >= minEdgeTilePixels;

        if (!Fits(tileWidthSb, tileCols, tileHeightSb, tileRows))
        {
            // Outside a limit: among the layouts with tiles no smaller than the wanted ones, the one with the smallest
            // tile (width x height in superblocks) that fits; equal sizes: the most tiles, then the most columns.
            // When none fits (a picture that needs more than maxTiles tiles of the largest size): the fewest tiles.
            int wantCols = colsLog2, wantRows = rowsLog2;
            long bestArea = long.MaxValue, bestCount = -1;
            colsLog2 = minLog2Cols;
            rowsLog2 = Math.Max(minLog2Tiles - minLog2Cols, 0);
            for (int c = wantCols; c >= minLog2Cols; c--)
            {
                int w = (sbCols + (1 << c) - 1) >> c;
                int cols = (sbCols + w - 1) / w;
                for (int r = wantRows; r >= Math.Max(minLog2Tiles - c, 0); r--)
                {
                    int h = (sbRows + (1 << r) - 1) >> r;
                    int rows = (sbRows + h - 1) / h;
                    long count = (long)cols * rows, area = (long)w * h;
                    if (!Fits(w, cols, h, rows)) continue;
                    if (area < bestArea || (area == bestArea && count > bestCount))
                    {
                        bestArea = area; bestCount = count; colsLog2 = c; rowsLog2 = r;
                    }
                }
            }
            minLog2Rows = Math.Max(minLog2Tiles - colsLog2, 0);
            tileWidthSb = (sbCols + (1 << colsLog2) - 1) >> colsLog2;
            tileCols = (sbCols + tileWidthSb - 1) / tileWidthSb;
            tileHeightSb = (sbRows + (1 << rowsLog2) - 1) >> rowsLog2;
            tileRows = (sbRows + tileHeightSb - 1) / tileHeightSb;
        }

        return new Av1TileLayout
        {
            SbCols = sbCols, SbRows = sbRows,
            TileColsLog2 = colsLog2, TileRowsLog2 = rowsLog2,
            MinLog2TileCols = minLog2Cols, MaxLog2TileCols = maxLog2Cols,
            MinLog2TileRows = minLog2Rows, MaxLog2TileRows = maxLog2Rows,
            TileWidthSb = tileWidthSb, TileHeightSb = tileHeightSb,
            TileCols = tileCols, TileRows = tileRows,
        };
    }

    /// <summary>The fewest tiles the frame size allows (one tile for pictures up to 4096 wide and 4096 x 2304 in area).</summary>
    public static Av1TileLayout SingleTile(int width, int height) => Compute(width, height, int.MaxValue, int.MaxValue);

    /// <summary>AV1 <c>tile_log2(blkSize, target)</c>: the smallest k with <c>(blkSize &lt;&lt; k) &gt;= target</c>.</summary>
    public static int TileLog2(int blkSize, int target)
    {
        int k = 0;
        while ((blkSize << k) < target) k++;
        return k;
    }
}
