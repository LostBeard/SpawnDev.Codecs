// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 tile group extraction. After the byte-aligned uncompressed
// header (see Vp9CompleteUncompressedHeaderParser) and the
// first_partition_size compressed header, the remaining frame
// bytes are tile data: tile_cols * tile_rows tiles, each preceded
// by a 4-byte big-endian tile_size for all but the LAST tile.
//
// libvpx reference: vp9/decoder/vp9_decodeframe.c decode_tiles +
// init_tile_data.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>One tile's byte range within a VP9 frame.</summary>
public readonly record struct Vp9TileSlice(int Offset, int Length);

/// <summary>VP9 tile group - the per-tile byte ranges of a frame's tile data.</summary>
public sealed record Vp9TileGroup
{
    /// <summary>
    /// Per-tile byte slices into the original frame buffer, in
    /// row-major order (row 0 left to right, then row 1, ...).
    /// Length = tile_cols * tile_rows.
    /// </summary>
    public required IReadOnlyList<Vp9TileSlice> Tiles { get; init; }
}

/// <summary>VP9 tile group extractor.</summary>
public static class Vp9TileGroupExtractor
{
    /// <summary>
    /// Extract per-tile byte ranges from a frame given its complete
    /// uncompressed header (provides tile dimensions + offsets).
    /// </summary>
    /// <param name="frame">A single VP9 frame's bytes.</param>
    /// <param name="header">Complete uncompressed header for this frame.</param>
    public static Vp9TileGroup Extract(ReadOnlySpan<byte> frame, Vp9UncompressedHeader header)
    {
        ArgumentNullException.ThrowIfNull(header);

        int tileCols = header.TileInfo.TileCols;
        int tileRows = header.TileInfo.TileRows;
        int totalTiles = tileCols * tileRows;
        if (totalTiles == 0)
            throw new InvalidDataException("VP9 tile_cols * tile_rows = 0; expected >= 1.");

        // Tile data starts after the byte-aligned uncompressed header
        // + the first_partition_size compressed header.
        int dataStart = header.UncompressedHeaderSizeBytes + header.FirstPartitionSize;
        int dataEnd = frame.Length;
        if (dataStart > dataEnd)
            throw new InvalidDataException(
                $"VP9 tile data start {dataStart} exceeds frame length {frame.Length}.");

        var tiles = new Vp9TileSlice[totalTiles];
        int pos = dataStart;
        for (int i = 0; i < totalTiles; i++)
        {
            bool isLastTile = i == totalTiles - 1;
            int tileLen;
            int tileStart;
            if (isLastTile)
            {
                tileLen = dataEnd - pos;
                tileStart = pos;
                pos = dataEnd;
            }
            else
            {
                if (pos + 4 > dataEnd)
                    throw new InvalidDataException(
                        $"VP9 tile {i} size header (4B) starts at {pos} but frame only has {dataEnd}B.");
                // Big-endian uint32.
                int sz = (frame[pos] << 24) | (frame[pos + 1] << 16)
                       | (frame[pos + 2] << 8) | frame[pos + 3];
                if (sz < 0)
                    throw new InvalidDataException($"VP9 tile {i} declared negative size.");
                pos += 4;
                tileStart = pos;
                tileLen = sz;
                if (pos + tileLen > dataEnd)
                    throw new InvalidDataException(
                        $"VP9 tile {i} ({tileLen}B starting at {pos}) overruns frame end {dataEnd}.");
                pos += tileLen;
            }
            tiles[i] = new Vp9TileSlice(tileStart, tileLen);
        }

        return new Vp9TileGroup { Tiles = tiles };
    }
}
