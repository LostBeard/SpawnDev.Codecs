// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 tile info parser - the tile dimensions section of the
// uncompressed frame header. Mirror of libvpx
// vp9/decoder/vp9_decodeframe.c setup_tile_info() and the helper
// vp9_get_tile_n_bits in vp9/common/vp9_tile_common.c.
//
// Bitstream layout (VP9 spec sec 6.2.13):
//   tile_cols_log2 in [min_log2_tile_cols, max_log2_tile_cols] is
//     encoded as (max - min) increment bits. The decoder reads up to
//     (max - min) bits; each 1 increments tile_cols_log2 starting
//     from min, until either (a) max - min bits read or (b) a 0 bit.
//   tile_rows_log2 in [0, 2]:
//     0 -> read 1 bit -> 0; if 1, read another bit and add to it.
//
// vp9_get_tile_n_bits() derives min / max log2 from mi_cols (the
// frame width in 8x8 mode info units). Constants in libvpx:
//   MI_BLOCK_SIZE_LOG2 = 3   (SB64 = 64 pix = 8x8 mi units)
//   MIN_TILE_WIDTH_B64 = 4   (smallest tile width = 4 SB64 = 256 pix)
//   MAX_TILE_WIDTH_B64 = 64  (largest tile width = 64 SB64 = 4096 pix)
//
// log2_tile_cols also has a hard cap of 6 (validated post-parse).

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// Parsed VP9 tile-info fields. Bit-exact against libvpx
/// <c>setup_tile_info</c>.
/// </summary>
public sealed record Vp9TileInfo
{
    /// <summary>libvpx <c>MI_BLOCK_SIZE_LOG2</c> (SB64 = 8x8 mi units).</summary>
    public const int MiBlockSizeLog2 = 3;

    /// <summary>libvpx <c>MIN_TILE_WIDTH_B64</c>.</summary>
    public const int MinTileWidthSb64 = 4;

    /// <summary>libvpx <c>MAX_TILE_WIDTH_B64</c>.</summary>
    public const int MaxTileWidthSb64 = 64;

    /// <summary>Log2 of the number of tile columns. 0..6.</summary>
    public required int Log2TileCols { get; init; }

    /// <summary>Log2 of the number of tile rows. 0..2.</summary>
    public required int Log2TileRows { get; init; }

    /// <summary>Min log2 tile columns derived from mi_cols.</summary>
    public required int MinLog2TileCols { get; init; }

    /// <summary>Max log2 tile columns derived from mi_cols.</summary>
    public required int MaxLog2TileCols { get; init; }

    /// <summary>Number of tile columns = 1 &lt;&lt; <see cref="Log2TileCols"/>.</summary>
    public int TileCols => 1 << Log2TileCols;

    /// <summary>Number of tile rows = 1 &lt;&lt; <see cref="Log2TileRows"/>.</summary>
    public int TileRows => 1 << Log2TileRows;
}

/// <summary>Parser for VP9 tile info in the uncompressed header.</summary>
public static class Vp9TileInfoParser
{
    /// <summary>
    /// Compute the min and max <c>log2_tile_cols</c> bounds from
    /// <paramref name="miCols"/>. Mirror of libvpx
    /// <c>vp9_get_tile_n_bits</c>.
    /// </summary>
    /// <param name="miCols">Frame width in 8x8 mode info units.</param>
    public static (int MinLog2, int MaxLog2) GetTileNBits(int miCols)
    {
        if (miCols < 0)
            throw new ArgumentOutOfRangeException(nameof(miCols), "mi_cols must be non-negative");
        // sb_cols = mi_cols rounded up to multiple of 8 (SB64), divided by 8.
        int sbCols = AlignUp(miCols, 1 << Vp9TileInfo.MiBlockSizeLog2) >> Vp9TileInfo.MiBlockSizeLog2;

        // VP9 spec sec 6.2.14 calc_min/max_log2_tile_cols. min_log2 is the
        // forced-multi-tile floor (driven by MAX_TILE_WIDTH); max_log2 is
        // the splitting ceiling (driven by MIN_TILE_WIDTH). Previously
        // these formulas were transposed which gave min > max for any
        // sb_cols >= MIN_TILE_WIDTH (broke widths > 320 vs ffmpeg).
        int minLog2 = 0;
        while ((Vp9TileInfo.MaxTileWidthSb64 << minLog2) < sbCols)
            minLog2++;

        int maxLog2 = 1;
        while ((sbCols >> maxLog2) >= Vp9TileInfo.MinTileWidthSb64)
            maxLog2++;
        maxLog2--;

        return (minLog2, maxLog2);
    }

    /// <summary>
    /// Parse tile info from <paramref name="reader"/> given
    /// <paramref name="miCols"/>.
    /// </summary>
    internal static Vp9TileInfo Parse(ref Vp9BitReader reader, int miCols)
    {
        var (minLog2, maxLog2) = GetTileNBits(miCols);

        int log2TileCols = minLog2;
        int increments = maxLog2 - minLog2;
        while (increments-- > 0 && reader.ReadFlag())
            log2TileCols++;
        if (log2TileCols > 6)
            throw new InvalidDataException(
                $"VP9 tile_cols_log2 = {log2TileCols} exceeds the 6-bit cap.");

        int log2TileRows = (int)reader.ReadBits(1);
        if (log2TileRows != 0)
            log2TileRows += (int)reader.ReadBits(1);

        return new Vp9TileInfo
        {
            Log2TileCols = log2TileCols,
            Log2TileRows = log2TileRows,
            MinLog2TileCols = minLog2,
            MaxLog2TileCols = maxLog2,
        };
    }

    /// <summary>Convenience overload for unit tests.</summary>
    public static Vp9TileInfo Parse(ReadOnlySpan<byte> data, int miCols)
    {
        var r = new Vp9BitReader(data);
        return Parse(ref r, miCols);
    }

    /// <summary>
    /// Compute the SB64-column boundary for tile <paramref name="tileIdx"/>
    /// in [0, <paramref name="tileCount"/>). Mirror of libvpx
    /// <c>get_tile_offset</c> from <c>vp9_tile_common.c</c>:
    /// <code>
    /// int offset = ((idx * mi_count) >> log2_tile_count);
    /// return ALIGN_POWER_OF_TWO(offset, MI_BLOCK_SIZE_LOG2) >> MI_BLOCK_SIZE_LOG2;
    /// </code>
    /// Returns the starting SB64 column for the tile (0-based). Combined with
    /// the next tile's start (or <paramref name="sbCount"/> for the last tile),
    /// gives the half-open SB-column range <c>[start, end)</c> the tile
    /// occupies.
    /// </summary>
    /// <param name="tileIdx">Tile column index, [0, tileCount).</param>
    /// <param name="tileCount">Total tile columns = 1 &lt;&lt; log2TileCols.</param>
    /// <param name="sbCount">Total SB64 columns in the frame (mi_cols &gt;&gt; 3, rounded up).</param>
    public static int GetTileOffsetSb(int tileIdx, int tileCount, int sbCount)
    {
        if ((uint)tileIdx > (uint)tileCount)
            throw new ArgumentOutOfRangeException(nameof(tileIdx),
                "tileIdx must be in [0, tileCount].");
        if (tileCount <= 0 || (tileCount & (tileCount - 1)) != 0)
            throw new ArgumentOutOfRangeException(nameof(tileCount),
                "tileCount must be a positive power of 2.");
        if (sbCount < 0)
            throw new ArgumentOutOfRangeException(nameof(sbCount),
                "sbCount must be non-negative.");
        // Per libvpx: offset = (idx * mi_count) >> log2_tile_count, where
        // mi_count is mi_cols (not sb_cols). The mi-aligned offset is then
        // SB64-aligned (round up) and converted back to SB units.
        int log2TileCount = 0;
        while ((1 << log2TileCount) < tileCount) log2TileCount++;
        int miCount = sbCount << Vp9TileInfo.MiBlockSizeLog2;
        long miOffset = ((long)tileIdx * miCount) >> log2TileCount;
        long miAligned = AlignUp((int)miOffset, 1 << Vp9TileInfo.MiBlockSizeLog2);
        return (int)(miAligned >> Vp9TileInfo.MiBlockSizeLog2);
    }

    /// <summary>
    /// Compute the half-open SB64-column range
    /// <c>[<see cref="ValueTuple{Int32,Int32}.Item1">SbColStart</see>,
    /// <see cref="ValueTuple{Int32,Int32}.Item2">SbColEnd</see>)</c> that
    /// tile <paramref name="tileColIdx"/> occupies in a frame with
    /// <paramref name="tileCols"/> total tile columns and
    /// <paramref name="sbCols"/> total SB64 columns.
    /// </summary>
    public static (int SbColStart, int SbColEnd) GetTileColRange(
        int tileColIdx, int tileCols, int sbCols)
    {
        int start = GetTileOffsetSb(tileColIdx, tileCols, sbCols);
        int end = (tileColIdx + 1 == tileCols)
            ? sbCols
            : GetTileOffsetSb(tileColIdx + 1, tileCols, sbCols);
        return (start, end);
    }

    /// <summary>
    /// Compute the half-open SB64-row range for tile row
    /// <paramref name="tileRowIdx"/> in a frame with
    /// <paramref name="tileRows"/> total tile rows and
    /// <paramref name="sbRows"/> total SB64 rows.
    /// </summary>
    public static (int SbRowStart, int SbRowEnd) GetTileRowRange(
        int tileRowIdx, int tileRows, int sbRows)
    {
        int start = GetTileOffsetSb(tileRowIdx, tileRows, sbRows);
        int end = (tileRowIdx + 1 == tileRows)
            ? sbRows
            : GetTileOffsetSb(tileRowIdx + 1, tileRows, sbRows);
        return (start, end);
    }

    private static int AlignUp(int value, int alignment)
        => (value + alignment - 1) & ~(alignment - 1);
}
