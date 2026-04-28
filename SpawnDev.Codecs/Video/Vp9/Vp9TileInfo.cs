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

    private static int AlignUp(int value, int alignment)
        => (value + alignment - 1) & ~(alignment - 1);
}
