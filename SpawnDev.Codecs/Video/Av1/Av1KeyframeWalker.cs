// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 keyframe decode walker - top-level orchestrator that:
//   - Validates input is a keyframe with parsed complete header
//   - Allocates output Y/U/V planes at the correct dimensions
//   - Walks superblocks within each tile (skeleton)
//   - Recursively decodes the partition tree (now using libaom's default
//     partition CDFs from Av1DefaultPartitionCdfs)
//   - For each leaf block: decodes intra mode + coefficients +
//     applies inverse transform + applies intra prediction +
//     reconstructs into the output
//
// As of this revision the partition CDF + partition_plane_context() are
// wired up, AND every default CDF table required for block decode is now
// ported (Av1Default{Partition,Block,IntraMode,Txfm,Segment,Coef}Cdfs).
// Per-block decode itself (mode_info read + decode_coefs() + inverse
// transform + intra predict + reconstruct) is the next step and currently
// throws NotImplementedException so callers can still observe the entropy
// stream consuming partition bits before hitting the next missing piece.
//
// What IS wired up:
//   - End-to-end skeleton from header parse to output framebuffer alloc
//   - Per-tile range decoder construction (uses Av1RangeDecoder)
//   - Superblock grid iteration in spec order
//   - Partition tree decode using default partition CDFs
//   - partition_plane_context()-equivalent tracking via Av1PartitionContext
//   - subsize_lookup[] for partition -> child block size mapping
//   - Output stride / plane sizes matched to ffmpeg layout
//
// What is NOT wired up (NotImplementedException):
//   - Mode info decode (intra mode / tx size / segment / skip / dc_sign etc)
//   - decode_coefs() loop (libaom av1/decoder/decodetxb.c) - reads txb_skip,
//     eob, base, br, dc_sign using the now-available Av1DefaultCoefCdfs
//   - Inverse quant pipeline glue
//   - Per-block intra prediction selection / edge buffer assembly
//   - Per-block inverse transform dispatch
//   - Per-block reconstruction
//   - Adaptive CDF updates per AV1 spec sec 9.4
//
// All default CDF tables required to drive the above are present:
//   Av1DefaultPartitionCdfs / Av1DefaultBlockCdfs / Av1DefaultIntraModeCdfs /
//   Av1DefaultTxfmCdfs / Av1DefaultSegmentCdfs / Av1DefaultCoefCdfs.
//
// Spec references: AV1 Bitstream and Decoding Process Specification
//   sec 5.11.4 Partition syntax
//   sec 6.4.4  Partition semantics
//   sec 9.3    Conversion tables (Partition_Subsize, Mi_Width_Log2, etc)

using SpawnDev.Codecs.EntropyCoders;

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>AV1 keyframe walker (top-level orchestrator).</summary>
public sealed class Av1KeyframeWalker
{
    /// <summary>BLOCK_SIZE enum index for BLOCK_64X64 (libaom enums.h).</summary>
    private const int Block64x64 = 12;
    /// <summary>BLOCK_SIZE enum index for BLOCK_128X128 (libaom enums.h).</summary>
    private const int Block128x128 = 15;

    /// <summary>
    /// Subsize lookup table from av1/common/common_data.h:subsize_lookup[EXT_PARTITION_TYPES][SQR_BLOCK_SIZES].
    /// Indexed by [partition][sqr_bsize_idx] where sqr_bsize_idx = 0..5 for BLOCK_4X4..BLOCK_128X128.
    /// Returns -1 (BLOCK_INVALID) for invalid combinations.
    /// </summary>
    private static readonly int[,] s_subsizeLookup = new int[10, 6]
    {
        // PARTITION_NONE
        { 0, 3, 6, 9, 12, 15 }, // BLOCK_4X4, BLOCK_8X8, BLOCK_16X16, BLOCK_32X32, BLOCK_64X64, BLOCK_128X128
        // PARTITION_HORZ
        { -1, 2, 5, 8, 11, 14 }, // -, BLOCK_8X4, BLOCK_16X8, BLOCK_32X16, BLOCK_64X32, BLOCK_128X64
        // PARTITION_VERT
        { -1, 1, 4, 7, 10, 13 }, // -, BLOCK_4X8, BLOCK_8X16, BLOCK_16X32, BLOCK_32X64, BLOCK_64X128
        // PARTITION_SPLIT
        { -1, 0, 3, 6, 9, 12 },  // -, BLOCK_4X4, BLOCK_8X8, BLOCK_16X16, BLOCK_32X32, BLOCK_64X64
        // PARTITION_HORZ_A
        { -1, -1, 5, 8, 11, 14 }, // -, -, BLOCK_16X8, BLOCK_32X16, BLOCK_64X32, BLOCK_128X64
        // PARTITION_HORZ_B
        { -1, -1, 5, 8, 11, 14 },
        // PARTITION_VERT_A
        { -1, -1, 4, 7, 10, 13 }, // -, -, BLOCK_8X16, BLOCK_16X32, BLOCK_32X64, BLOCK_64X128
        // PARTITION_VERT_B
        { -1, -1, 4, 7, 10, 13 },
        // PARTITION_HORZ_4
        { -1, -1, 17, 19, 21, -1 }, // -, -, BLOCK_16X4, BLOCK_32X8, BLOCK_64X16, -
        // PARTITION_VERT_4
        { -1, -1, 16, 18, 20, -1 }, // -, -, BLOCK_4X16, BLOCK_8X32, BLOCK_16X64, -
    };

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
        int sbSizePx = sh.Use128x128Superblock ? 128 : 64;
        int sbBlockIdx = sh.Use128x128Superblock ? Block128x128 : Block64x64;
        int rowStart = header.TileInfo.RowStartSb[tile.TileRow];
        int rowEnd = header.TileInfo.RowStartSb[tile.TileRow + 1];
        int colStart = header.TileInfo.ColStartSb[tile.TileCol];
        int colEnd = header.TileInfo.ColStartSb[tile.TileCol + 1];

        // Compute the tile's mi-grid width to size the partition context.
        int tileMiCols = (colEnd - colStart) * (sbSizePx >> 2); // sbSizePx / 4
        int frameMiRows = (header.Prefix.FrameHeight + 7) >> 3 << 1; // round up to 8 then to mi units
        int frameMiCols = (header.Prefix.FrameWidth + 7) >> 3 << 1;
        var pctx = new Av1PartitionContext(Math.Max(tileMiCols, frameMiCols));

        // Walk superblocks in raster scan within the tile.
        for (int sbRow = rowStart; sbRow < rowEnd; sbRow++)
        {
            pctx.ResetLeft();
            for (int sbCol = colStart; sbCol < colEnd; sbCol++)
            {
                int miRow = sbRow * (sbSizePx >> 2);
                int miCol = sbCol * (sbSizePx >> 2);
                DecodeSuperblock(rangeDec, sh, header, pctx, miRow, miCol, sbBlockIdx, y, u, v);
            }
        }
    }

    private void DecodeSuperblock(
        Av1RangeDecoder rd,
        Av1SequenceHeader sh,
        Av1CompleteFrameHeader header,
        Av1PartitionContext pctx,
        int miRow, int miCol, int sbBlockIdx,
        byte[] y, byte[] u, byte[] v)
    {
        // Recursive partition decode: starts at the superblock size,
        // decodes a partition symbol, then recurses on the children.
        DecodePartition(rd, sh, header, pctx, miRow, miCol, sbBlockIdx, y, u, v);
    }

    private void DecodePartition(
        Av1RangeDecoder rd,
        Av1SequenceHeader sh,
        Av1CompleteFrameHeader header,
        Av1PartitionContext pctx,
        int miRow, int miCol, int bsize,
        byte[] y, byte[] u, byte[] v)
    {
        // Per AV1 spec sec 5.11.4: minimum partition size is BLOCK_8X8;
        // smaller blocks have an implicit PARTITION_NONE.
        if (bsize < Av1PartitionContext.Block8x8)
        {
            DecodeBlock(rd, sh, header, pctx, miRow, miCol, bsize, Av1PartitionType.None, y, u, v);
            return;
        }

        // Compute the partition context (combination of above + left split bits).
        int ctx = pctx.GetContext(miRow, miCol, bsize);
        int nsyms = Av1PartitionContext.PartitionCdfLength(bsize);

        // Decode the partition symbol from the appropriate CDF row.
        var cdf = Av1DefaultPartitionCdfs.DefaultPartitionCdf[ctx];
        Av1PartitionType partition = (Av1PartitionType)rd.DecodeCdfQ15(cdf, nsyms);

        // Map (bsize, partition) -> sub-block size via subsize_lookup.
        int sqrIdx = SqrBlockSizeIndex(bsize);
        int subsize = s_subsizeLookup[(int)partition, sqrIdx];
        if (subsize < 0)
        {
            throw new InvalidDataException(
                $"AV1 invalid partition: bsize={bsize}, partition={partition}");
        }

        int hbs = Av1PartitionContext.MiSizeWide[bsize] >> 1; // half block size in mi units
        int qbs = hbs >> 1; // quarter block size

        // Recurse / leaf-decode per partition type. Mirrors libaom's
        // decode_partition() switch (av1/decoder/decodeframe.c line 1296).
        switch (partition)
        {
            case Av1PartitionType.None:
                DecodeBlock(rd, sh, header, pctx, miRow, miCol, subsize, partition, y, u, v);
                pctx.UpdateContext(miRow, miCol, subsize);
                break;

            case Av1PartitionType.Horz:
                DecodeBlock(rd, sh, header, pctx, miRow, miCol, subsize, partition, y, u, v);
                if (miRow + hbs < FrameMiRows(header))
                    DecodeBlock(rd, sh, header, pctx, miRow + hbs, miCol, subsize, partition, y, u, v);
                pctx.UpdateContext(miRow, miCol, subsize);
                break;

            case Av1PartitionType.Vert:
                DecodeBlock(rd, sh, header, pctx, miRow, miCol, subsize, partition, y, u, v);
                if (miCol + hbs < FrameMiCols(header))
                    DecodeBlock(rd, sh, header, pctx, miRow, miCol + hbs, subsize, partition, y, u, v);
                pctx.UpdateContext(miRow, miCol, subsize);
                break;

            case Av1PartitionType.Split:
                DecodePartition(rd, sh, header, pctx, miRow, miCol, subsize, y, u, v);
                DecodePartition(rd, sh, header, pctx, miRow, miCol + hbs, subsize, y, u, v);
                DecodePartition(rd, sh, header, pctx, miRow + hbs, miCol, subsize, y, u, v);
                DecodePartition(rd, sh, header, pctx, miRow + hbs, miCol + hbs, subsize, y, u, v);
                break;

            case Av1PartitionType.Horz4:
                for (int i = 0; i < 4; i++)
                {
                    int r = miRow + i * qbs;
                    if (i > 0 && r >= FrameMiRows(header)) break;
                    DecodeBlock(rd, sh, header, pctx, r, miCol, subsize, partition, y, u, v);
                }
                pctx.UpdateContext(miRow, miCol, subsize);
                break;

            case Av1PartitionType.Vert4:
                for (int i = 0; i < 4; i++)
                {
                    int c = miCol + i * qbs;
                    if (i > 0 && c >= FrameMiCols(header)) break;
                    DecodeBlock(rd, sh, header, pctx, miRow, c, subsize, partition, y, u, v);
                }
                pctx.UpdateContext(miRow, miCol, subsize);
                break;

            case Av1PartitionType.HorzA:
            case Av1PartitionType.HorzB:
            case Av1PartitionType.VertA:
            case Av1PartitionType.VertB:
                // These mixed partitions decode 3 sub-blocks. Matches libaom
                // decode_partition() lines 1357-1378.
                throw new NotImplementedException(
                    $"AV1 partition {partition} (mixed split) requires per-block decode " +
                    "which depends on the remaining mode-info CDFs (intra mode, skip, " +
                    "tx size, coef CDFs from token_cdfs.h). See Av1KeyframeWalker.DecodeBlock.");

            default:
                throw new InvalidDataException(
                    $"AV1 unknown partition type: {(int)partition}");
        }
    }

    private void DecodeBlock(
        Av1RangeDecoder rd,
        Av1SequenceHeader sh,
        Av1CompleteFrameHeader header,
        Av1PartitionContext pctx,
        int miRow, int miCol, int bsize, Av1PartitionType partition,
        byte[] y, byte[] u, byte[] v)
    {
        // Per-block decode dependencies. Status reflects PORTED tables; the
        // execution wiring (read_modes_b -> read_inter_block_mode_info ->
        // decode_tokens -> reconstruct) is the next step.
        //   - partition CDF                  -> Av1DefaultPartitionCdfs        PORTED
        //   - skip flag CDF                  -> Av1DefaultBlockCdfs.DefaultSkipTxfmCdf  PORTED
        //   - intrabc + skip_mode + txfm_partition -> Av1DefaultBlockCdfs              PORTED
        //   - intra mode CDFs (kf_y / y / uv) -> Av1DefaultIntraModeCdfs                PORTED
        //   - segmentation CDFs              -> Av1DefaultSegmentCdfs           PORTED
        //   - tx size + intra/inter ext_tx   -> Av1DefaultTxfmCdfs              PORTED
        //   - coefficient CDFs (token_cdfs.h) -> Av1DefaultCoefCdfs              PORTED
        //   - inverse quant tables           -> Av1DequantTables                PORTED
        //   - inverse transforms             -> Av1Inverse{Dct,Adst,Identity}*  PORTED
        //   - intra predictor                -> Av1IntraPredictor               PORTED
        //
        // Remaining wiring (this method's body):
        //   - mode_info decode (intra mode + uv mode + skip + tx_size + segment)
        //   - decode_coefs() loop per plane / per tx block (libaom
        //     av1/decoder/decodetxb.c) - reads txb_skip, eob, base, br, dc_sign
        //     using the Av1DefaultCoefCdfs tables; builds dequantized coeff array
        //   - inverse transform dispatch
        //   - intra predictor edge buffer assembly + invocation
        //   - reconstruct sample array (predictor + residual) -> y/u/v planes
        //   - adaptive CDF updates per AV1 spec sec 9.4 (defer until decode works)
        throw new NotImplementedException(
            $"AV1 per-block decode at miRow={miRow}, miCol={miCol}, bsize={bsize}, " +
            $"partition={partition}: all default CDF tables are ported (partition, " +
            "block-level binary, intra mode, txfm size/type, segment, coefficient). " +
            "Remaining: wire mode_info decode + decode_coefs() + inverse transform + " +
            "intra predict + reconstruct. See libaom av1/decoder/decodeframe.c " +
            "decode_partition() and av1/decoder/decodetxb.c decode_coefs() for the " +
            "execution order. The data tables are now sufficient to drive the decoder.");
    }

    private static int FrameMiRows(Av1CompleteFrameHeader header)
    {
        // mi_rows = (FrameHeight + 7) >> 3 << 1. AV1 mi units are 4 luma samples.
        // Ceiling-div to 8-px alignment, then *2 to convert 8-px -> 4-px (mi).
        return ((header.Prefix.FrameHeight + 7) >> 3) << 1;
    }

    private static int FrameMiCols(Av1CompleteFrameHeader header)
    {
        return ((header.Prefix.FrameWidth + 7) >> 3) << 1;
    }

    /// <summary>
    /// Map a square BLOCK_SIZE enum value to its 0..5 sqr_bsize index used by
    /// subsize_lookup. Mirrors libaom <c>get_sqr_bsize_idx()</c>.
    /// </summary>
    private static int SqrBlockSizeIndex(int bsize)
    {
        // BLOCK_4X4=0, BLOCK_8X8=3, BLOCK_16X16=6, BLOCK_32X32=9, BLOCK_64X64=12, BLOCK_128X128=15
        return bsize switch
        {
            0 => 0,
            3 => 1,
            6 => 2,
            9 => 3,
            12 => 4,
            15 => 5,
            _ => throw new ArgumentException($"Not a square block size: {bsize}", nameof(bsize)),
        };
    }
}
