// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 keyframe decode walker - top-level orchestrator that:
//   - Validates input is a keyframe with parsed complete header
//   - Allocates output Y/U/V planes at the correct dimensions
//   - Walks superblocks within each tile (skeleton)
//   - Recursively decodes the partition tree (skeleton)
//   - For each leaf block: decodes intra mode + coefficients +
//     applies inverse transform + applies intra prediction +
//     reconstructs into the output
//
// Block-level decode requires the full AV1 CDF tables for entropy
// decode of mode info / partitioning / coefficients - the largest
// missing piece (libaom token_cdfs.h alone is ~5000 lines of static
// data). Until those land, the walker throws NotImplementedException
// at the entropy decode boundary so callers fail loud rather than
// emit wrong pixels.
//
// What IS wired up:
//   - End-to-end skeleton from header parse to output framebuffer alloc
//   - Per-tile range decoder construction (uses Av1RangeDecoder)
//   - Superblock grid iteration in spec order
//   - Output stride / plane sizes matched to ffmpeg layout
//
// What is NOT wired up (NotImplementedException):
//   - CDF tables (entropy normative data)
//   - Mode info decode (partition / intra mode / tx size / segment / skip)
//   - Coefficient decode + inverse quant
//   - Per-block intra prediction selection / edge buffer assembly
//   - Per-block inverse transform dispatch
//   - Per-block reconstruction
//
// This file ships the architecture skeleton + clear NotImplemented
// boundaries so the next session has a complete frame to plug entropy
// + reconstruction into without re-deriving the geometry.

using SpawnDev.Codecs.EntropyCoders;

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>AV1 keyframe walker (top-level orchestrator).</summary>
public sealed class Av1KeyframeWalker
{
    /// <summary>
    /// Walk a single AV1 keyframe and produce a planar 8-bit YUV
    /// frame buffer. Throws NotImplementedException for portions of
    /// the pipeline that depend on the not-yet-ported CDF tables.
    /// </summary>
    public Av1FrameBuffer DecodeFrame(
        ReadOnlyMemory<byte> framePayload,
        Av1SequenceHeader sh,
        Av1CompleteFrameHeader header,
        Av1TileGroup tileGroup)
    {
        ArgumentNullException.ThrowIfNull(sh);
        ArgumentNullException.ThrowIfNull(header);
        ArgumentNullException.ThrowIfNull(tileGroup);

        if (!header.Prefix.FrameIsIntra)
        {
            throw new NotImplementedException(
                "Av1KeyframeWalker only handles key / intra-only frames.");
        }

        // Allocate output planes at the parsed dimensions.
        int yW = header.Prefix.FrameWidth;
        int yH = header.Prefix.FrameHeight;
        int subX = sh.SubsamplingX;
        int subY = sh.SubsamplingY;
        int cW = subX != 0 ? (yW + 1) >> 1 : yW;
        int cH = subY != 0 ? (yH + 1) >> 1 : yH;
        var y = new byte[yW * yH];
        var u = new byte[cW * cH];
        var v = new byte[cW * cH];

        // Iterate tiles, construct a per-tile range decoder, walk the
        // superblock grid. The per-block decode is what's NotImplemented.
        foreach (var tile in tileGroup.Tiles)
        {
            DecodeTile(framePayload, sh, header, tile, y, u, v);
        }

        return new Av1FrameBuffer
        {
            Y = y,
            U = u,
            V = v,
            LumaWidth = yW,
            LumaHeight = yH,
            ChromaWidth = cW,
            ChromaHeight = cH,
        };
    }

    private void DecodeTile(
        ReadOnlyMemory<byte> framePayload,
        Av1SequenceHeader sh,
        Av1CompleteFrameHeader header,
        Av1TileBuffer tile,
        byte[] y, byte[] u, byte[] v)
    {
        // Per-tile range decoder. Uses the production Av1RangeDecoder.
        var tileBytes = framePayload.Slice(tile.Offset, tile.Length).ToArray();
        var rangeDec = new Av1RangeDecoder(tileBytes);

        // Compute the superblock geometry for this tile.
        int sbSize = sh.Use128x128Superblock ? 128 : 64;
        int rowStart = header.TileInfo.RowStartSb[tile.TileRow];
        int rowEnd = header.TileInfo.RowStartSb[tile.TileRow + 1];
        int colStart = header.TileInfo.ColStartSb[tile.TileCol];
        int colEnd = header.TileInfo.ColStartSb[tile.TileCol + 1];

        // Walk superblocks in raster scan within the tile.
        for (int sbRow = rowStart; sbRow < rowEnd; sbRow++)
        {
            for (int sbCol = colStart; sbCol < colEnd; sbCol++)
            {
                DecodeSuperblock(rangeDec, sh, header, sbRow, sbCol, sbSize, y, u, v);
            }
        }
    }

    private void DecodeSuperblock(
        Av1RangeDecoder rd,
        Av1SequenceHeader sh,
        Av1CompleteFrameHeader header,
        int sbRow, int sbCol, int sbSize,
        byte[] y, byte[] u, byte[] v)
    {
        // Recursive partition decode: starts at sbSize x sbSize, decodes
        // a partition symbol, then recurses on the children based on
        // PARTITION_NONE / HORZ / VERT / SPLIT / HORZ_A / HORZ_B /
        // VERT_A / VERT_B / HORZ_4 / VERT_4.
        DecodePartition(rd, sh, header, sbRow * sbSize, sbCol * sbSize, sbSize, y, u, v);
    }

    private void DecodePartition(
        Av1RangeDecoder rd,
        Av1SequenceHeader sh,
        Av1CompleteFrameHeader header,
        int rowPx, int colPx, int blockSize,
        byte[] y, byte[] u, byte[] v)
    {
        // Decoding the partition symbol requires the partition CDF for
        // the current ctx. The CDFs are not yet ported (token_cdfs.h +
        // entropymode.c are the missing pieces). Without a partition
        // symbol we cannot recurse correctly, so this is the boundary.
        throw new NotImplementedException(
            "AV1 partition tree decode requires the partition CDF tables " +
            "from libaom av1/common/entropymode.c default_partition_cdf[]. " +
            "Porting that table set + the partition_ctx() function is the " +
            "next step in the AV1 decoder pipeline.");
    }
}
