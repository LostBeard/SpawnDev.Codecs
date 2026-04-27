// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 tile group structure - byte ranges of each tile inside the
// Frame OBU payload. Mirrors libaom's tile_buffer offsets parsed
// out of read_tile_group_data.

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>One tile's compressed bytes inside a Frame OBU payload.</summary>
public readonly record struct Av1TileBuffer(int TileRow, int TileCol, int Offset, int Length);

/// <summary>A parsed AV1 tile group (per-tile compressed-byte ranges).</summary>
public sealed record Av1TileGroup
{
    /// <summary>Tile group start tile index.</summary>
    public required int StartTile { get; init; }
    /// <summary>Tile group end tile index (inclusive).</summary>
    public required int EndTile { get; init; }
    /// <summary>One entry per tile in this group.</summary>
    public required IReadOnlyList<Av1TileBuffer> Tiles { get; init; }
}

/// <summary>Stateless AV1 tile group extractor.</summary>
public static class Av1TileGroupExtractor
{
    /// <summary>
    /// Extract the byte ranges for each tile from a Frame OBU payload,
    /// given the parsed complete frame header. Mirrors libaom's
    /// <c>read_tile_group_data</c>.
    /// </summary>
    public static Av1TileGroup Extract(ReadOnlySpan<byte> framePayload, Av1CompleteFrameHeader header)
    {
        int totalTiles = header.TileInfo.TileCols * header.TileInfo.TileRows;
        int headerBytes = header.HeaderSizeBytes;
        int pos = headerBytes;

        int startTile = 0;
        int endTile = totalTiles - 1;

        if (totalTiles > 1)
        {
            // tile_start_and_end_present_flag (single bit at byte boundary)
            byte b = framePayload[pos];
            bool present = (b & 0x80) != 0;
            // For a single tile group covering everything, this bit is 0.
            // If 1 follows, parse start/end tile via tile_log2 bits.
            if (present)
            {
                throw new NotImplementedException(
                    "AV1 multi-tile-group bitstreams are not yet supported.");
            }
            pos += 1; // skip the byte-aligned flag
        }

        var tiles = new List<Av1TileBuffer>(totalTiles);
        int tileSizeBytes = header.TileInfo.TileSizeBytes;
        for (int idx = 0; idx <= endTile - startTile; idx++)
        {
            int tileRow = (startTile + idx) / header.TileInfo.TileCols;
            int tileCol = (startTile + idx) % header.TileInfo.TileCols;
            int tileLength;
            if (idx == endTile - startTile)
            {
                // Last tile: takes the rest of the payload.
                tileLength = framePayload.Length - pos;
            }
            else
            {
                tileLength = (int)ReadLeBytes(framePayload.Slice(pos, tileSizeBytes)) + 1;
                pos += tileSizeBytes;
            }
            tiles.Add(new Av1TileBuffer(tileRow, tileCol, pos, tileLength));
            pos += tileLength;
        }

        return new Av1TileGroup
        {
            StartTile = startTile,
            EndTile = endTile,
            Tiles = tiles,
        };
    }

    private static long ReadLeBytes(ReadOnlySpan<byte> data)
    {
        long v = 0;
        for (int i = 0; i < data.Length; i++)
        {
            v |= ((long)data[i]) << (i * 8);
        }
        return v;
    }
}
