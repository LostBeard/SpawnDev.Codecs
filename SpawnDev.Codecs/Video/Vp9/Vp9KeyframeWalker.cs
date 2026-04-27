// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 keyframe walker. Drives the per-frame block decode after the
// uncompressed header + compressed header have been parsed. For
// every tile in the tile group:
//
//   1. Reset per-tile-row left contexts.
//   2. For each 64x64 superblock in tile order:
//      - Recurse partition tree starting at Block64x64.
//      - At each leaf: read mode info, read coefficients, predict +
//        invert + add into the frame buffer.
//
// libvpx reference: vp9/decoder/vp9_decodeframe.c decode_tiles +
// decode_partition + decode_block. This walker covers the keyframe
// (intra-only) path; inter prediction is out of scope.
//
// Loop filter is OUT OF SCOPE for this slice - the output will look
// blocky vs ffmpeg's loop-filtered reference but should be the
// recognizable BBB scene at correct mean / variance.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// VP9 keyframe walker - top-level integrator that walks every tile,
/// every superblock, every partition tree leaf, and reconstructs the
/// pixels into a <see cref="Vp9FrameBuffer"/>.
/// </summary>
public sealed class Vp9KeyframeWalker
{
    /// <summary>
    /// Decode an intra-only frame (keyframe or intra_only) to a fully
    /// reconstructed YUV plane buffer. Skips loop filter.
    /// </summary>
    /// <param name="frameBytes">
    /// One VP9 frame's bytes (post-superframe-split). Must contain the
    /// uncompressed header, compressed header, and tile data.
    /// </param>
    /// <param name="header">
    /// Already-parsed complete uncompressed header for this frame.
    /// </param>
    /// <param name="compressedState">
    /// Compressed-header probability state (post-update). Provides the
    /// CoefProbs + SkipProbs + TxModeProbs the walker reads through.
    /// </param>
    /// <param name="compressedResult">
    /// Frame-level tx_mode + reference_mode from compressed header.
    /// </param>
    /// <param name="tileGroup">
    /// Per-tile byte ranges from <see cref="Vp9TileGroupExtractor"/>.
    /// </param>
    /// <returns>
    /// The reconstructed YUV frame buffer. Pre-loopfilter pixels.
    /// </returns>
    public Vp9FrameBuffer DecodeFrame(
        ReadOnlyMemory<byte> frameBytes,
        Vp9UncompressedHeader header,
        Vp9CompressedHeaderState compressedState,
        Vp9CompressedHeaderResult compressedResult,
        Vp9TileGroup tileGroup)
    {
        ArgumentNullException.ThrowIfNull(header);
        ArgumentNullException.ThrowIfNull(compressedState);
        ArgumentNullException.ThrowIfNull(compressedResult);
        ArgumentNullException.ThrowIfNull(tileGroup);

        var fh = header.FrameHeader;
        bool isIntraOnly = fh.FrameType == Vp9FrameType.Key || fh.IntraOnly;
        if (!isIntraOnly)
            throw new NotImplementedException(
                "Vp9KeyframeWalker decodes intra-only frames; inter prediction is out of scope.");

        // Subsampling: derived from header (default 4:2:0 for Profile 0).
        var subsampling = new Vp9SubsamplingPair(
            SubsamplingX: fh.SubsamplingX ? 1 : 0,
            SubsamplingY: fh.SubsamplingY ? 1 : 0);

        int frameW = fh.FrameWidth;
        int frameH = fh.FrameHeight;
        var fb = new Vp9FrameBuffer(frameW, frameH, subsampling);

        // mi grid dimensions (8x8 mode info units, rounded up).
        int miCols = (frameW + 7) >> 3;
        int miRows = (frameH + 7) >> 3;
        // Aligned to 8 mi (= 64 px = 1 SB) for context allocation.
        int miColsAligned = (miCols + 7) & ~7;
        int miRowsAligned = (miRows + 7) & ~7;

        // 4x4-grid intra mode contexts. Frame-wide above row + per-tile
        // left column. Indexed in 4x4 b-units (= 2 per mi).
        int b4Cols = miColsAligned * 2;
        var aboveYMode = new Vp9IntraMode[b4Cols];
        Array.Fill(aboveYMode, Vp9IntraMode.DcPred);

        // Skip context: 1 bit per mi, above is frame-wide, left per tile-row.
        var aboveSkip = new byte[miColsAligned];
        // Partition context: 1 bit per mi at each bsl level, packed.
        // libvpx stores 8 bits per mi (1 per bsl level). Above frame-wide,
        // left per tile-row. For sub-tile-row reset + simple decode it is
        // enough to keep one byte per mi column.
        var abovePartCtx = new byte[miColsAligned];
        // tx_size per mi (frame-wide above, tile-row left). 0..3 enum
        // value of the chosen tx_size for each block; missing = max_tx.
        var aboveTxSize = new byte[miColsAligned];

        var ctx = new WalkerContext
        {
            FrameBytes = frameBytes,
            Header = header,
            CompressedState = compressedState,
            CompressedResult = compressedResult,
            FrameBuffer = fb,
            MiCols = miCols,
            MiRows = miRows,
            MiColsAligned = miColsAligned,
            Subsampling = subsampling,
            AboveYMode = aboveYMode,
            AboveSkip = aboveSkip,
            AbovePartCtx = abovePartCtx,
            AboveTxSize = aboveTxSize,
            Trace = Trace,
        };

        // Walk every tile.
        int tileCols = header.TileInfo.TileCols;
        int tileRows = header.TileInfo.TileRows;
        for (int tileRow = 0; tileRow < tileRows; tileRow++)
        {
            for (int tileCol = 0; tileCol < tileCols; tileCol++)
            {
                int tileIdx = tileRow * tileCols + tileCol;
                var slice = tileGroup.Tiles[tileIdx];
                var bounds = Vp9TileLayout.Compute(
                    tileRow, tileCol, miRows, miCols,
                    header.TileInfo.Log2TileRows, header.TileInfo.Log2TileCols);

                // Materialize tile bytes for the bool decoder (it takes byte[]).
                var tileBytes = new byte[slice.Length];
                frameBytes.Span.Slice(slice.Offset, slice.Length).CopyTo(tileBytes);
                var br = new Vp9BoolDecoder(tileBytes, 0, tileBytes.Length);

                // libvpx left contexts are 8 mi (16 b4) wide and reset
                // at the start of every SB row within a tile.
                ctx.LeftYMode = new Vp9IntraMode[16];
                Array.Fill(ctx.LeftYMode, Vp9IntraMode.DcPred);
                ctx.LeftSkip = new byte[8];
                ctx.LeftPartCtx = new byte[8];
                ctx.LeftTxSize = new byte[8];
                ctx.TileBounds = bounds;

                // Walk SBs in tile order (row-major). libvpx zeroes the
                // left context arrays at the start of every SB row.
                for (int miRow = bounds.MiRowStart; miRow < bounds.MiRowEnd; miRow += 8)
                {
                    Array.Fill(ctx.LeftYMode, Vp9IntraMode.DcPred);
                    Array.Clear(ctx.LeftSkip);
                    Array.Clear(ctx.LeftPartCtx);
                    Array.Clear(ctx.LeftTxSize);
                    for (int miCol = bounds.MiColStart; miCol < bounds.MiColEnd; miCol += 8)
                    {
                        DecodePartition(ctx, br, miRow, miCol, Vp9BlockSize.Block64x64);
                    }
                }
            }
        }

        return fb;
    }

    /// <summary>
    /// Walker mutable state passed by ref through partition + block
    /// decoders. Eliminates repeated argument plumbing.
    /// </summary>
    /// <summary>Per-block decode trace records (debug only).</summary>
    public List<DecodedBlockTrace>? Trace { get; set; }

    /// <summary>Trace record for a single decoded leaf block.</summary>
    public sealed record DecodedBlockTrace(
        int MiRow, int MiCol, Vp9BlockSize Bsize, Vp9TxSize TxSize,
        Vp9IntraMode YMode, Vp9IntraMode UvMode, int Skip, int SegmentId,
        int SkipContext, int TxSizeContext);

    private sealed class WalkerContext
    {
        public required ReadOnlyMemory<byte> FrameBytes { get; init; }
        public required Vp9UncompressedHeader Header { get; init; }
        public required Vp9CompressedHeaderState CompressedState { get; init; }
        public required Vp9CompressedHeaderResult CompressedResult { get; init; }
        public required Vp9FrameBuffer FrameBuffer { get; init; }
        public required int MiCols { get; init; }
        public required int MiRows { get; init; }
        public required int MiColsAligned { get; init; }
        public required Vp9SubsamplingPair Subsampling { get; init; }

        // Frame-wide above contexts (column-indexed).
        public required Vp9IntraMode[] AboveYMode { get; init; }
        public required byte[] AboveSkip { get; init; }
        public required byte[] AbovePartCtx { get; init; }
        public required byte[] AboveTxSize { get; init; }

        // Per-tile-row left contexts (mutable, replaced per tile).
        public Vp9IntraMode[] LeftYMode { get; set; } = Array.Empty<Vp9IntraMode>();
        public byte[] LeftSkip { get; set; } = Array.Empty<byte>();
        public byte[] LeftPartCtx { get; set; } = Array.Empty<byte>();
        public byte[] LeftTxSize { get; set; } = Array.Empty<byte>();

        public Vp9TileBounds TileBounds { get; set; }
        public List<DecodedBlockTrace>? Trace { get; set; }
    }

    /// <summary>
    /// Recursively walk the partition tree at the given (mi_row, mi_col)
    /// for the given block size. Reads the partition decision and
    /// either recurses or decodes a leaf.
    /// </summary>
    private static void DecodePartition(
        WalkerContext ctx, Vp9BoolDecoder br,
        int miRow, int miCol, Vp9BlockSize bsize)
    {
        if (miRow >= ctx.MiRows || miCol >= ctx.MiCols) return;

        // Block must be at least 8x8 to have a partition decision.
        // Sub-8x8 (Block4x4 etc.) are leaves with no partition.
        Vp9PartitionType partition;
        int bsl = Vp9BlockSizes.MiWidthLog2[(int)bsize];
        // libvpx hbs = num_8x8_blocks_wide_lookup[bsize] / 2 = (1 << bsl) / 2.
        // For Block64x64 (bsl=3): hbs=4; Block32x32: 2; Block16x16: 1.
        // Block8x8 has bsl=0 -> hbs=0; partition splits stay at 4x4 sub-blocks
        // (sub-mi), handled as leaves below.
        int hbs = (bsl > 0) ? (1 << (bsl - 1)) : 0;

        bool hasRows = (miRow + hbs) < ctx.MiRows;
        bool hasCols = (miCol + hbs) < ctx.MiCols;

        if (bsize >= Vp9BlockSize.Block8x8)
        {
            // libvpx partition-context derivation:
            // sizeIdx = MiWidthLog2 (0..3 for 8x8..64x64)
            // splitState = (left_split * 2 + above_split) where each is 1
            //   if the corresponding side is partition-split at this bsl.
            // Bit at level (3 - bsl) from the partition context byte
            // (libvpx: partition_plane_context computes it differently;
            // using simplified per-bsl bit indexing).
            int leftIdx = (miRow & 7);
            int aboveIdx = miCol;
            int leftBit = (ctx.LeftPartCtx[leftIdx] >> bsl) & 1;
            int aboveBit = (ctx.AbovePartCtx[aboveIdx] >> bsl) & 1;
            int splitState = leftBit * 2 + aboveBit;
            int sizeIdx = bsl;
            var probs = Vp9PartitionProbs.KeyframeProbs(sizeIdx, splitState);

            if (hasRows && hasCols)
            {
                partition = Vp9PartitionTree.Decode(p => br.Read(p), probs);
            }
            else if (!hasRows && hasCols)
            {
                // Forced split or horz at bottom edge. libvpx reads only
                // Split vs Horz with prob[1] (Horz vs not).
                int bit = br.Read(probs[1]);
                partition = bit != 0 ? Vp9PartitionType.Split : Vp9PartitionType.Horz;
            }
            else if (hasRows && !hasCols)
            {
                int bit = br.Read(probs[2]);
                partition = bit != 0 ? Vp9PartitionType.Split : Vp9PartitionType.Vert;
            }
            else
            {
                partition = Vp9PartitionType.Split;
            }
        }
        else
        {
            partition = Vp9PartitionType.None;
        }

        var subsize = Vp9SubsizeLookup.Subsize(bsize, partition);
        switch (partition)
        {
            case Vp9PartitionType.None:
                DecodeLeafBlock(ctx, br, miRow, miCol, subsize);
                break;
            case Vp9PartitionType.Horz:
                DecodeLeafBlock(ctx, br, miRow, miCol, subsize);
                if (hbs > 0 && miRow + hbs < ctx.MiRows)
                    DecodeLeafBlock(ctx, br, miRow + hbs, miCol, subsize);
                break;
            case Vp9PartitionType.Vert:
                DecodeLeafBlock(ctx, br, miRow, miCol, subsize);
                if (hbs > 0 && miCol + hbs < ctx.MiCols)
                    DecodeLeafBlock(ctx, br, miRow, miCol + hbs, subsize);
                break;
            case Vp9PartitionType.Split:
                if (bsize == Vp9BlockSize.Block8x8)
                {
                    // libvpx: 8x8 + Split is handled as a single leaf
                    // block where the 4 4x4 sub-blocks each have their
                    // own Y mode but share one skip + tx_size + UV mode
                    // + UV coef block. The walker passes the SUBSIZE
                    // (Block4x4) so the leaf reads 4 Y modes (one per
                    // 4x4 cell) but tracks the parent block's geometry
                    // for shared mode-info reads.
                    DecodeLeafBlock(ctx, br, miRow, miCol, subsize, parentBsize: bsize);
                }
                else
                {
                    DecodePartition(ctx, br, miRow,       miCol,       subsize);
                    DecodePartition(ctx, br, miRow,       miCol + hbs, subsize);
                    DecodePartition(ctx, br, miRow + hbs, miCol,       subsize);
                    DecodePartition(ctx, br, miRow + hbs, miCol + hbs, subsize);
                }
                break;
        }

        // Update partition contexts. libvpx only writes for partitions
        // that recursed at most once (not SPLIT on bsize > 8x8) - the
        // children of a SPLIT have already updated their own slots.
        if (bsize >= Vp9BlockSize.Block8x8 &&
            (bsize == Vp9BlockSize.Block8x8 || partition != Vp9PartitionType.Split))
        {
            UpdatePartitionContext(ctx, miRow, miCol, bsize, subsize);
        }
    }

    /// <summary>
    /// libvpx <c>partition_context_lookup</c>: 13-entry table of
    /// (above, left) partition-context byte payloads keyed by the
    /// CHILD block size (subsize after the partition decision). The
    /// values encode "is this block split at level bsl?" as a packed
    /// bitfield read by <c>partition_plane_context</c>.
    /// </summary>
    private static readonly (byte Above, byte Left)[] PartitionContextLookup = new (byte, byte)[]
    {
        (15, 15),  // 4x4
        (15, 14),  // 4x8
        (14, 15),  // 8x4
        (14, 14),  // 8x8
        (14, 12),  // 8x16
        (12, 14),  // 16x8
        (12, 12),  // 16x16
        (12,  8),  // 16x32
        ( 8, 12),  // 32x16
        ( 8,  8),  // 32x32
        ( 8,  0),  // 32x64
        ( 0,  8),  // 64x32
        ( 0,  0),  // 64x64
    };

    /// <summary>
    /// libvpx <c>update_partition_context</c>. Writes the
    /// per-subsize-encoded above/left context bytes across the bs
    /// cells covered by this partition decision.
    /// </summary>
    private static void UpdatePartitionContext(
        WalkerContext ctx, int miRow, int miCol,
        Vp9BlockSize bsize, Vp9BlockSize subsize)
    {
        int bsl = Vp9BlockSizes.MiWidthLog2[(int)bsize];
        int bs = 1 << bsl; // num_8x8_blocks_wide_lookup[bsize]
        int subIdx = (int)subsize < (int)Vp9BlockSize.Invalid ? (int)subsize : (int)bsize;
        var (above, left) = PartitionContextLookup[subIdx];

        for (int i = 0; i < bs; i++)
        {
            int c = miCol + i;
            int r = (miRow + i) & 7;
            if (c < ctx.MiColsAligned)
                ctx.AbovePartCtx[c] = above;
            if (r < ctx.LeftPartCtx.Length)
                ctx.LeftPartCtx[r] = left;
        }
    }

    /// <summary>
    /// Decode a single leaf block: read mode info, read coefficients,
    /// predict + invert + add for each plane. Updates above/left mode
    /// context arrays.
    ///
    /// <paramref name="parentBsize"/> handles the special-case Block8x8
    /// + partition Split: the 4 4x4 children share one mode-info reads
    /// (skip, tx_size, UV mode, UV coef) but get 4 individual Y modes.
    /// When <paramref name="parentBsize"/> is null the bsize is used
    /// for both purposes.
    /// </summary>
    private static void DecodeLeafBlock(
        WalkerContext ctx, Vp9BoolDecoder br,
        int miRow, int miCol, Vp9BlockSize bsize,
        Vp9BlockSize? parentBsize = null)
    {
        if (miRow >= ctx.MiRows || miCol >= ctx.MiCols) return;

        var header = ctx.Header;
        var seg = header.Segmentation;

        // For block-level mode info (skip, tx_size, UV mode, plane
        // geometry, coefficient reads) use the PARENT bsize when set
        // (Block8x8+Split case). For Y-mode iteration use the leaf
        // bsize so the per-4x4 mode reads happen.
        var blockBsize = parentBsize ?? bsize;

        // 1. segment_id (always 0 for BBB - segmentation disabled).
        int segmentId = Vp9KeyframeModeInfo.ReadIntraSegmentId(seg, br);

        // 2. skip flag. skip_context = above_skip + left_skip.
        // libvpx left_context is 8 mi wide indexed by miRow & 7.
        int leftIdxMi = miRow & 7;
        int leftSkipBit = ctx.LeftSkip[leftIdxMi];
        int aboveSkipBit = miCol < ctx.AboveSkip.Length
            ? ctx.AboveSkip[miCol] : 0;
        int skipContext = aboveSkipBit + leftSkipBit;
        int skipFlag = Vp9KeyframeModeInfo.ReadSkip(
            seg, segmentId, ctx.CompressedState.SkipProbs, skipContext, br);

        // 3. tx_size. Context: get_tx_size_context with above + left
        // (we store skip-aware context implicit via skip flag).
        // For simplicity at this slice, use tx_size_context = 0 when no
        // neighbors; it lines up with libvpx's "missing -> max_tx" which
        // keeps both ctx pieces at max for an interior all-fresh block.
        // The vp9_first_partition demo showed ctx=1 at top-left. Match
        // libvpx by deriving from above/left tx_size below.
        int txSizeContext = ComputeTxSizeContext(ctx, miRow, miCol, blockBsize);
        var txSize = Vp9KeyframeModeInfo.ReadTxSize(
            ctx.CompressedResult.TxMode, blockBsize, txSizeContext,
            ctx.CompressedState.TxModeProbs, br);

        // 4. Y intra mode(s). For 8x8+ block, one Y mode for whole
        // block (above/left = 4x4 corner-cell modes in our context).
        // For sub-8x8 (Block4x4/Block4x8/Block8x4) libvpx reads up to
        // 4 separate Y modes per 4x4 sub-block. This walker handles
        // 8x8+ first and treats sub-8x8 as a single mode (good enough
        // for BBB which uses at least 8x8 throughout the first frame
        // per the existing first-partition trace).
        int b4Row = miRow * 2; // 4x4 grid units
        int b4Col = miCol * 2;
        // libvpx left_4x4 is 16-wide indexed by (mi_row & 7) * 2 + ...
        int leftB4Idx = (miRow & 7) * 2;
        var aboveMode = b4Col < ctx.AboveYMode.Length
            ? ctx.AboveYMode[b4Col] : Vp9IntraMode.DcPred;
        var leftMode = ctx.LeftYMode[leftB4Idx];

        // For sub-8x8 blocks libvpx reads one Y mode per 4x4 sub-block
        // using the (above_4x4, left_4x4) cells specific to that
        // sub-block. We track those per-4x4 modes; the block's overall
        // "mode" stored back into the above/left context arrays is the
        // bottom-right sub-block's mode.
        // Use blockBsize (parent) for the cell grid since 8x8+Split is
        // a single leaf with 4 4x4 sub-cells.
        int b4WideForRead = Vp9BlockSizes.B4x4Width(blockBsize);
        int b4HighForRead = Vp9BlockSizes.B4x4Height(blockBsize);
        var sub4x4Modes = new Vp9IntraMode[b4HighForRead, b4WideForRead];
        int leftBaseRow = leftB4Idx;
        // Stride = 1 when leaf bsize is sub-8x8 (one Y mode per 4x4
        // cell). Otherwise stride = full block (one Y mode for whole
        // block).
        int strideY = (bsize >= Vp9BlockSize.Block8x8) ? b4HighForRead : Vp9BlockSizes.B4x4Height(bsize);
        int strideX = (bsize >= Vp9BlockSize.Block8x8) ? b4WideForRead : Vp9BlockSizes.B4x4Width(bsize);
        for (int idy = 0; idy < b4HighForRead; idy += strideY)
        {
            for (int idx = 0; idx < b4WideForRead; idx += strideX)
            {
                // (above, left) for this 4x4 sub-block.
                int colCell = b4Col + idx;
                int rowCell = leftBaseRow + idy;
                Vp9IntraMode aboveCell = (idy > 0)
                    ? sub4x4Modes[idy - 1, idx]
                    : (colCell < ctx.AboveYMode.Length ? ctx.AboveYMode[colCell] : Vp9IntraMode.DcPred);
                Vp9IntraMode leftCell = (idx > 0)
                    ? sub4x4Modes[idy, idx - 1]
                    : (rowCell >= 0 && rowCell < ctx.LeftYMode.Length ? ctx.LeftYMode[rowCell] : Vp9IntraMode.DcPred);
                var yProbs = Vp9IntraModeProbs.KeyframeYProbs(aboveCell, leftCell);
                var subMode = Vp9IntraModeTree.Decode(p => br.Read(p), yProbs);
                // Replicate to all 4x4 cells covered by this read (1 for
                // sub-8x8, full block for 8x8+).
                for (int dy = 0; dy < strideY; dy++)
                    for (int dx = 0; dx < strideX; dx++)
                        sub4x4Modes[idy + dy, idx + dx] = subMode;
            }
        }
        // Block's "overall" Y mode is the bottom-right 4x4's mode -
        // matches libvpx storing bmi[3].as_mode for sub-8x8.
        Vp9IntraMode yMode = sub4x4Modes[b4HighForRead - 1, b4WideForRead - 1];

        // 5. uv_mode keyed by yMode.
        var uvProbs = Vp9IntraModeProbs.KeyframeUvProbs(yMode);
        var uvMode = Vp9IntraModeTree.Decode(p => br.Read(p), uvProbs);

        ctx.Trace?.Add(new DecodedBlockTrace(
            miRow, miCol, blockBsize, txSize, yMode, uvMode, skipFlag, segmentId,
            skipContext, ComputeTxSizeContext(ctx, miRow, miCol, blockBsize)));

        // Update mode-info contexts (per-4x4 cells across the block).
        // Use blockBsize for context updates so 8x8+Split correctly
        // covers all 4 4x4 cells.
        int b4Wide = Vp9BlockSizes.B4x4Width(blockBsize);
        int b4High = Vp9BlockSizes.B4x4Height(blockBsize);
        for (int i = 0; i < b4Wide; i++)
        {
            int c = b4Col + i;
            if (c < ctx.AboveYMode.Length) ctx.AboveYMode[c] = yMode;
        }
        for (int i = 0; i < b4High; i++)
        {
            int r = (leftB4Idx + i) & 15;
            ctx.LeftYMode[r] = yMode;
        }
        // skip + tx_size context update.
        int miWide = Vp9BlockSizes.MiWidth(blockBsize);
        int miHigh = Vp9BlockSizes.MiHeight(blockBsize);
        for (int i = 0; i < miWide; i++)
        {
            int c = miCol + i;
            if (c < ctx.AboveSkip.Length) ctx.AboveSkip[c] = (byte)skipFlag;
            if (c < ctx.AboveTxSize.Length) ctx.AboveTxSize[c] = (byte)txSize;
        }
        for (int i = 0; i < miHigh; i++)
        {
            int r = (leftIdxMi + i) & 7;
            ctx.LeftSkip[r] = (byte)skipFlag;
            ctx.LeftTxSize[r] = (byte)txSize;
        }

        // Decode pixels: Y plane then UV planes. Pass the per-4x4 mode
        // grid so sub-8x8 blocks can use the right mode per tx-block.
        // blockBsize is the parent block (= bsize except for 8x8+Split).
        DecodePlanePixels(ctx, br, miRow, miCol, blockBsize, txSize, sub4x4Modes, uvMode,
            skipFlag, segmentId, plane: 0);
        if (ctx.Subsampling.SubsamplingX == 1 && ctx.Subsampling.SubsamplingY == 1)
        {
            DecodePlanePixels(ctx, br, miRow, miCol, blockBsize, txSize, sub4x4Modes, uvMode,
                skipFlag, segmentId, plane: 1);
            DecodePlanePixels(ctx, br, miRow, miCol, blockBsize, txSize, sub4x4Modes, uvMode,
                skipFlag, segmentId, plane: 2);
        }
        else
        {
            throw new NotImplementedException("only 4:2:0 subsampling supported");
        }
    }

    /// <summary>
    /// libvpx <c>get_tx_size_context</c>. Returns 0 or 1 based on
    /// whether the (above + left) tx_sizes sum exceeds max_tx_size.
    /// Missing neighbors are treated as max_tx (libvpx default), so
    /// top-left blocks see ctx = 1.
    ///
    /// Walker stores per-mi neighbor tx_size + skip in the
    /// AboveTxSize / LeftTxSize / AboveSkip / LeftSkip context arrays;
    /// this helper consumes them.
    /// </summary>
    private static int ComputeTxSizeContext(
        WalkerContext ctx, int miRow, int miCol, Vp9BlockSize bsize)
    {
        var maxTxSize = Vp9MaxTxSize.ForBlockSize(bsize);
        int maxIdx = (int)maxTxSize;

        bool hasAbove = miRow > 0;
        bool hasLeft = miCol > ctx.TileBounds.MiColStart;

        int aboveCtx = (hasAbove && miCol < ctx.AboveTxSize.Length && ctx.AboveSkip[miCol] == 0)
            ? ctx.AboveTxSize[miCol] : maxIdx;
        int leftIdx = miRow & 7;
        int leftCtx = (hasLeft && ctx.LeftSkip[leftIdx] == 0)
            ? ctx.LeftTxSize[leftIdx] : maxIdx;
        if (!hasAbove) aboveCtx = leftCtx;
        if (!hasLeft) leftCtx = aboveCtx;
        return (aboveCtx + leftCtx) > maxIdx ? 1 : 0;
    }

    /// <summary>
    /// Decode pixels for one plane of a leaf block: walk every tx-block
    /// in raster order, predict from already-reconstructed neighbors,
    /// then read coefficients (if !skip), dequantize, inverse transform
    /// and add to the predicted block.
    /// </summary>
    private static void DecodePlanePixels(
        WalkerContext ctx, Vp9BoolDecoder br,
        int miRow, int miCol, Vp9BlockSize bsize, Vp9TxSize lumaTxSize,
        Vp9IntraMode[,] sub4x4Modes, Vp9IntraMode uvMode,
        int skipFlag, int segmentId, int plane)
    {
        bool isUv = plane != 0;
        int ssX = isUv ? ctx.Subsampling.SubsamplingX : 0;
        int ssY = isUv ? ctx.Subsampling.SubsamplingY : 0;

        // Block size + tx size for this plane.
        Vp9BlockSize planeBsize = isUv
            ? Vp9ChromaBlockSize.ForLumaBlock(bsize)
            : bsize;
        Vp9TxSize txSize = isUv
            ? Vp9ChromaBlockSize.GetChromaTxSize(lumaTxSize, bsize)
            : lumaTxSize;
        // Block-wide mode for chroma; for luma we look up per-4x4 below.
        Vp9IntraMode blockMode = isUv ? uvMode : sub4x4Modes[sub4x4Modes.GetLength(0) - 1, sub4x4Modes.GetLength(1) - 1];

        int blockWidthPx = Vp9BlockSizes.Width(planeBsize);
        int blockHeightPx = Vp9BlockSizes.Height(planeBsize);
        int txN = Vp9IntraBlockDecode.TxSizeToN(txSize);
        int txCols = Math.Max(1, blockWidthPx / txN);
        int txRows = Math.Max(1, blockHeightPx / txN);

        // Plane buffer + dimensions.
        byte[] planeBuf = plane switch
        {
            0 => ctx.FrameBuffer.Y,
            1 => ctx.FrameBuffer.U,
            _ => ctx.FrameBuffer.V,
        };
        int planeStride = plane == 0 ? ctx.FrameBuffer.LumaWidth : ctx.FrameBuffer.ChromaWidth;
        int planeHeight = plane == 0 ? ctx.FrameBuffer.LumaHeight : ctx.FrameBuffer.ChromaHeight;

        // Top-left pixel coordinate of the block within this plane.
        int blockX0 = (miCol << 3) >> ssX;
        int blockY0 = (miRow << 3) >> ssY;

        // Plane-type for coef-prob indexing.
        var planeType = isUv
            ? Vp9BlockCoefDecoder.PlaneType.Uv
            : Vp9BlockCoefDecoder.PlaneType.Y;

        // Per-plane quantizer.
        int qindex = Vp9SegmentationLookup.ResolveQIndex(
            ctx.Header.Segmentation, segmentId,
            ctx.Header.Quantization.BaseQIndex);
        // libvpx applies y_dc_delta to Y_DC and 0 to Y_AC; uv_dc_delta
        // and uv_ac_delta apply to U/V (V uses the same UV deltas as U).
        Vp9PlaneQuantizer planeQuant = isUv
            ? new Vp9PlaneQuantizer(
                Dc: Vp9Dequantizer.DcQuant(qindex, ctx.Header.Quantization.UvDcDeltaQ),
                Ac: Vp9Dequantizer.AcQuant(qindex, ctx.Header.Quantization.UvAcDeltaQ))
            : new Vp9PlaneQuantizer(
                Dc: Vp9Dequantizer.DcQuant(qindex, ctx.Header.Quantization.YDcDeltaQ),
                Ac: Vp9Dequantizer.AcQuant(qindex, 0));

        // tx_type / scan derived per tx-block (sub-8x8 luma can vary).
        var coefProbs = ctx.CompressedState.CoefProbs[(int)txSize];

        // Per-tx-block buffers.
        Span<short> coeffs = stackalloc short[1024];
        var aboveBuf = new byte[txN * 2];
        var leftBuf = new byte[txN];
        // dstLocal hoisted out of the loop to avoid CA2014; sized for
        // 32x32 max which fits comfortably on the stack at 1 KB.
        Span<byte> dstLocal = stackalloc byte[32 * 32];

        for (int ty = 0; ty < txRows; ty++)
        {
            int yPx = blockY0 + ty * txN;
            if (yPx >= planeHeight) continue;
            for (int tx = 0; tx < txCols; tx++)
            {
                int xPx = blockX0 + tx * txN;
                if (xPx >= planeStride) continue;

                bool hasAbove = yPx > 0;
                bool hasLeft = xPx > 0;

                // Pick per-tx-block intra mode + transform type.
                Vp9IntraMode mode;
                if (isUv)
                {
                    mode = blockMode;
                }
                else
                {
                    int gridDim0 = sub4x4Modes.GetLength(0);
                    int gridDim1 = sub4x4Modes.GetLength(1);
                    int per4x4 = Math.Max(1, txN / 4);
                    int gridY = Math.Min(ty * per4x4, gridDim0 - 1);
                    int gridX = Math.Min(tx * per4x4, gridDim1 - 1);
                    mode = sub4x4Modes[gridY, gridX];
                }
                var txType = (txSize == Vp9TxSize.Tx32x32 || isUv)
                    ? Vp9TxType.DctDct
                    : Vp9IntraTxType.ForMode(mode);
                var scanType = Vp9ScanTables.ScanTypeForTxType(txType);

                // Build above row.
                if (hasAbove)
                {
                    int aboveCount = Math.Min(2 * txN, planeStride - xPx);
                    int aboveRowOff = (yPx - 1) * planeStride + xPx;
                    for (int i = 0; i < aboveCount; i++)
                        aboveBuf[i] = planeBuf[aboveRowOff + i];
                    // If less than 2N available (right-edge extension),
                    // replicate last sample.
                    for (int i = aboveCount; i < 2 * txN; i++)
                        aboveBuf[i] = (aboveCount > 0) ? aboveBuf[aboveCount - 1]
                            : Vp9IntraEdgeFill.AboveFill;
                }
                else
                {
                    for (int i = 0; i < 2 * txN; i++) aboveBuf[i] = Vp9IntraEdgeFill.AboveFill;
                }

                // Build left column.
                if (hasLeft)
                {
                    int leftCount = Math.Min(txN, planeHeight - yPx);
                    int leftColOff = yPx * planeStride + (xPx - 1);
                    for (int i = 0; i < leftCount; i++)
                        leftBuf[i] = planeBuf[leftColOff + i * planeStride];
                    for (int i = leftCount; i < txN; i++)
                        leftBuf[i] = leftCount > 0 ? leftBuf[leftCount - 1]
                            : Vp9IntraEdgeFill.LeftFill;
                }
                else
                {
                    for (int i = 0; i < txN; i++) leftBuf[i] = Vp9IntraEdgeFill.LeftFill;
                }

                byte topLeft = (hasAbove && hasLeft)
                    ? planeBuf[(yPx - 1) * planeStride + (xPx - 1)]
                    : Vp9IntraEdgeFill.ResolveCorner(hasAbove, hasLeft, refValue: 128);

                // Run predictor into the hoisted local buffer.
                dstLocal[..(txN * txN)].Clear();
                Vp9IntraPredictor.Predict(
                    mode, topLeft, aboveBuf.AsSpan(0, 2 * txN), leftBuf.AsSpan(0, txN),
                    dstLocal[..(txN * txN)], txN, txN,
                    haveAbove: hasAbove, haveLeft: hasLeft);

                if (skipFlag == 0)
                {
                    // Read this tx-block's coefficients then dequant + iDCT.
                    coeffs.Clear();
                    int eob = Vp9BlockCoefDecoder.DecodeBlockCoefficients(
                        p => br.Read(p),
                        txSize, scanType, planeType,
                        Vp9BlockCoefDecoder.RefType.Intra,
                        coeffs, isHighBitDepth: false,
                        coefProbs: coefProbs);
                    if (eob > 0)
                    {
                        Vp9Dequantizer.DequantizeInPlace(coeffs[..(txN * txN)], planeQuant);
                        Vp9InverseTransform.Apply(
                            txType, txSize, coeffs[..(txN * txN)],
                            dstLocal[..(txN * txN)], stride: txN);
                    }
                }

                // Copy local prediction (post-residual) into frame buffer.
                int copyW = Math.Min(txN, planeStride - xPx);
                int copyH = Math.Min(txN, planeHeight - yPx);
                for (int r = 0; r < copyH; r++)
                {
                    int dstOff = (yPx + r) * planeStride + xPx;
                    int srcOff = r * txN;
                    for (int c = 0; c < copyW; c++)
                        planeBuf[dstOff + c] = dstLocal[srcOff + c];
                }
            }
        }
    }
}
