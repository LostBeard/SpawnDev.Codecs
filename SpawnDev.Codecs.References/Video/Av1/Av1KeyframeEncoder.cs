// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 keyframe encoder. v1 minimal-but-correct emit path that produces a
// decodable AV1 keyframe bitstream for a picture of any size.
//
// v1 simplifications (mirrors Vp9KeyframeEncoder shape):
//   - Profile 0 (8-bit 4:2:0); chroma planes are ((width + 1) / 2) x ((height + 1) / 2)
//   - Every BLOCK_16X16 uses Y intra mode = DC_PRED, transform = TX_16X16 + DCT_DCT
//   - Every chroma block uses UV intra mode = DC_PRED, transform = TX_8X8 + DCT_DCT
//   - A 16x16 that crosses the bottom / right edge of the picture is split into BLOCK_8X8 leaves (Y TX_8X8,
//     chroma TX_4X4); the coded size is each dimension rounded up to 8, and source pixels past the picture
//     repeat its last column / row
//   - tx_mode = LARGEST (no per-block tx_size signaling)
//   - Uniform tile spacing; tiles of at most tileSizeSuperblocks 64x64 superblocks per side (Av1TileLayout).
//     Each tile has its own range coder and contexts and predicts only from inside itself, so the GPU encoder
//     runs one thread per tile; this reference encodes them one after another
//   - 64x64 superblocks (Use128x128Superblock = false)
//   - Loop filter disabled
//   - Segmentation disabled
//   - CDEF / loop restoration disabled
//   - Default coefficient probabilities (no compressed-header CDF updates)
//   - Default skip probabilities
//   - error_resilient_mode = true (implicit for visible KeyFrame)
//
// Bitstream shape:
//   1. Temporal Delimiter OBU
//   2. Sequence Header OBU (Av1SequenceHeaderWriter)
//   3. Frame OBU containing uncompressed_header + tile_group payload
//
// More sophisticated mode selection / RD-optimized quantization / loop
// filtering / CDEF layers on top; the produced bitstream is a fully-valid
// AV1 keyframe accepted by libaom + libdav1d + ffmpeg.

using SpawnDev.Codecs.EntropyCoders;

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>AV1 keyframe encoder (DC-prediction-only, single tile, no LF/CDEF/LR).</summary>
public static class Av1KeyframeEncoder
{
    /// <summary>libaom <c>BLOCK_16X16</c> enum index.</summary>
    public const int Block16x16 = 6;

    /// <summary>libaom <c>BLOCK_32X32</c> enum index.</summary>
    public const int Block32x32 = 9;

    /// <summary>libaom <c>BLOCK_64X64</c> enum index.</summary>
    public const int Block64x64 = 12;

    /// <summary>
    /// Encode a single AV1 keyframe from YUV420 source. Returns the full
    /// keyframe bitstream (TD + SH + Frame OBU) ready to wrap in IVF / WebM.
    /// </summary>
    /// <param name="ySrc">Y plane bytes, length = ySrcStride * height.</param>
    /// <param name="ySrcStride">Y plane row stride in bytes (&gt;= width).</param>
    /// <param name="uSrc">U plane bytes (length &gt;= uvSrcStride * ((height + 1) / 2)).</param>
    /// <param name="uvSrcStride">UV plane row stride in bytes (&gt;= (width + 1) / 2).</param>
    /// <param name="vSrc">V plane bytes (length &gt;= uvSrcStride * ((height + 1) / 2)).</param>
    /// <param name="width">Frame width in pixels.</param>
    /// <param name="height">Frame height in pixels.</param>
    /// <param name="baseQIndex">Base quantizer index 1..255 (lower = higher quality).</param>
    /// <param name="tileSizeSuperblocks">Largest tile size in 64x64 superblocks per side (1 = one superblock per
    /// tile, the GPU encoder's default; <see cref="int.MaxValue"/> = the fewest tiles the picture allows).</param>
    /// <param name="maxTiles">The most tiles of the frame (<see cref="Av1TileLayout.Compute"/>; the GPU encoder's
    /// <c>MaxTiles</c>).</param>
    /// <param name="frameRateNum">Frames per second the stream is shown at, numerator (the GPU encoder's
    /// <c>SetFrameRate</c>): with the picture size, the tile grid and the coded bytes it decides the level the
    /// sequence header signals (<see cref="Av1Level"/>).</param>
    /// <param name="frameRateDen">Frames per second, denominator.</param>
    /// <param name="levelLimit">Keeps the tile grid inside an AV1 level (the GPU encoder's <c>LevelLimit</c>;
    /// <see cref="Av1Level.TileLayout"/>). 31 = no limit.</param>
    public static byte[] EncodeKeyFrame(
        ReadOnlySpan<byte> ySrc, int ySrcStride,
        ReadOnlySpan<byte> uSrc, int uvSrcStride,
        ReadOnlySpan<byte> vSrc,
        int width, int height,
        int baseQIndex = 32,
        int tileSizeSuperblocks = 1,
        int maxTiles = Av1TileLayout.HardwareDecoderMaxTiles,
        int frameRateNum = 30,
        int frameRateDen = 1,
        int levelLimit = Av1Level.MaximumParameters)
    {
        if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width));
        if (height <= 0) throw new ArgumentOutOfRangeException(nameof(height));
        if (baseQIndex < 1 || baseQIndex > 255)
            throw new ArgumentOutOfRangeException(nameof(baseQIndex),
                "baseQIndex must be in [1, 255]; lossless not supported in v1.");

        // ---- 1. The tile grid, and the levels a stream with it can signal (each with its TD + SH OBUs) ----
        var tiles = Av1Level.TileLayout(width, height, tileSizeSuperblocks, maxTiles, frameRateNum, frameRateDen, levelLimit);
        var candidates = Av1KeyframeFraming.BuildLevelCandidates(width, height, tiles, frameRateNum, frameRateDen);
        // Reify a SH so downstream writers consume a parsed view without re-parsing (the level changes nothing below).
        var sh = Av1KeyframeFraming.SynthesizeSequenceHeader(
            Av1KeyframeFraming.BuildSequenceHeaderConfig(width, height, candidates[0].Signal));

        // ---- 2. Build the Frame OBU payload (uncompressed_header + tile_group) ----

        // The tiles' entropy streams first: the tile group needs every tile's byte length (a size prefix before
        // every tile but the last).
        byte[][] tileBytes = EncodeTiles(
            ySrc, ySrcStride, uSrc, uvSrcStride, vSrc,
            width, height, baseQIndex, tiles);

        byte[] uncompressedHeader = Av1KeyframeFraming.BuildUncompressedHeader(sh, width, height, baseQIndex, tiles);
        byte[] frameOuter = BuildFramePayload(uncompressedHeader, tileBytes);

        byte[] frameObu = Av1ObuWriter.EmitObu(
            Av1ObuType.Frame, frameOuter, hasSizeField: true);

        // ---- 3. The level this frame fits, then its TD + SH OBUs and the Frame OBU ----
        return WithLevelPrefix(candidates, frameObu);
    }

    /// <summary>
    /// The temporal unit of <paramref name="frameObu"/>: the Temporal Delimiter and Sequence Header OBUs of the first
    /// of <paramref name="candidates"/> whose level a Frame OBU of this length conforms to, then the Frame OBU.
    /// </summary>
    private static byte[] WithLevelPrefix(List<Av1KeyframeFraming.LevelCandidate> candidates, byte[] frameObu)
    {
        byte[] prefix = Av1KeyframeFraming.SelectLevelCandidate(candidates, frameObu.Length).Prefix;
        var output = new byte[prefix.Length + frameObu.Length];
        Buffer.BlockCopy(prefix, 0, output, 0, prefix.Length);
        Buffer.BlockCopy(frameObu, 0, output, prefix.Length, frameObu.Length);
        return output;
    }

    /// <summary>
    /// The Frame OBU payload: the uncompressed header, then the tile group: one zero byte when there is more than
    /// one tile (tile_start_and_end_present_flag = 0 and the byte alignment), then each tile in raster order, every
    /// tile but the last preceded by its tile_size_minus_1 (4 bytes, little-endian).
    /// </summary>
    private static byte[] BuildFramePayload(byte[] uncompressedHeader, byte[][] tileBytes)
    {
        int total = uncompressedHeader.Length + (tileBytes.Length > 1 ? 1 : 0);
        for (int t = 0; t < tileBytes.Length; t++)
            total += tileBytes[t].Length + (t < tileBytes.Length - 1 ? Av1KeyframeFraming.TileSizeBytes : 0);
        var payload = new byte[total];
        Buffer.BlockCopy(uncompressedHeader, 0, payload, 0, uncompressedHeader.Length);
        int at = uncompressedHeader.Length;
        if (tileBytes.Length > 1) payload[at++] = 0;
        for (int t = 0; t < tileBytes.Length; t++)
        {
            if (t < tileBytes.Length - 1)
            {
                int sizeMinus1 = tileBytes[t].Length - 1;
                for (int b = 0; b < Av1KeyframeFraming.TileSizeBytes; b++) payload[at++] = (byte)(sizeMinus1 >> (8 * b));
            }
            Buffer.BlockCopy(tileBytes[t], 0, payload, at, tileBytes[t].Length);
            at += tileBytes[t].Length;
        }
        return payload;
    }

    /// <summary>
    /// V3 GPU encoder integration helper. Same as <see cref="EncodeKeyFrame"/>
    /// but accepts pre-computed tile bytes (produced by the GPU walker) and
    /// only performs OBU framing (TD + SH + Frame OBU wrap) on host. The
    /// framing is metadata struct setup + bit-packing of fixed config, not
    /// codec-data math, so it stays inside the CARDINAL rule's "metadata
    /// struct setup" allowance.
    /// </summary>
    internal static byte[] EncodeKeyFrameWithExternalTile(
        int width, int height, int baseQIndex, byte[] tileBytes)
    {
        if (tileBytes is null) throw new ArgumentNullException(nameof(tileBytes));
        if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width));
        if (height <= 0) throw new ArgumentOutOfRangeException(nameof(height));
        // No alignment check - the OBU framing signals display dims directly
        // and AV1 decoders reconstruct working dims internally as
        // alignUp(display_dim, 64), matching the encoder's tile bytes.

        // ---- The levels a single-tile stream at 30 frames per second can signal (TD + SH OBUs of each) ----
        var tiles = Av1TileLayout.SingleTile(width, height);
        var candidates = Av1KeyframeFraming.BuildLevelCandidates(width, height, tiles, 30, 1);
        var sh = Av1KeyframeFraming.SynthesizeSequenceHeader(
            Av1KeyframeFraming.BuildSequenceHeaderConfig(width, height, candidates[0].Signal));

        // ---- Frame OBU = uncompressed header + (GPU-produced) tile bytes ----
        byte[] uncompressedHeader = Av1KeyframeFraming.BuildUncompressedHeader(sh, width, height, baseQIndex, tiles);
        var frameOuter = new byte[uncompressedHeader.Length + tileBytes.Length];
        Buffer.BlockCopy(uncompressedHeader, 0, frameOuter, 0, uncompressedHeader.Length);
        Buffer.BlockCopy(tileBytes, 0, frameOuter, uncompressedHeader.Length, tileBytes.Length);
        byte[] frameObu = Av1ObuWriter.EmitObu(
            Av1ObuType.Frame, frameOuter, hasSizeField: true);

        return WithLevelPrefix(candidates, frameObu);
    }

    // ------------------------------------------------------------------
    // Tile data: SB walk + per-block emit
    // ------------------------------------------------------------------

    private sealed class TileEncodeState
    {
        public Av1RangeEncoder Re = null!;
        public Av1PartitionContext Pctx = null!;
        public Av1EntropyContext EntropyCtx = null!;

        // Per-tile mode info grid for above/left intra-mode + skip lookups.
        public Av1IntraMode[] AboveYMode = null!;
        public bool[] AboveSkip = null!;
        public Av1IntraMode[] LeftYMode = null!;
        public bool[] LeftSkip = null!;

        // Reconstruction planes (used by subsequent blocks for intra prediction).
        public byte[] ReconY = null!;
        public byte[] ReconU = null!;
        public byte[] ReconV = null!;
        // Recon plane sizes = the CODED size (each picture dimension rounded up to 8).
        public int LumaW;
        public int LumaH;
        public int ChromaW;
        public int ChromaH;
        // Source plane sizes (the picture); a block past them reads the last column / row.
        public int SrcLumaW;
        public int SrcLumaH;
        public int SrcChromaW;
        public int SrcChromaH;
        // The tile's first mi row / column: a block there has no above / left neighbour.
        public int TileMiRowStart;
        public int TileMiColStart;
        // One past the tile's last mi row / column (clipped to the frame): no pixel past it is read.
        public int TileMiRowEnd;
        public int TileMiColEnd;
    }

    /// <summary>
    /// Encode the whole keyframe as ONE tile and return the raw range-coder
    /// byte stream. Internal so the GPU walker test can compare
    /// bit-exact against this CPU reference without going through the
    /// full OBU framing.
    /// </summary>
    internal static byte[] EncodeSingleTile(
        ReadOnlySpan<byte> ySrc, int ySrcStride,
        ReadOnlySpan<byte> uSrc, int uvSrcStride,
        ReadOnlySpan<byte> vSrc,
        int width, int height, int baseQIndex)
    {
        var tiles = Av1TileLayout.SingleTile(width, height);
        if (tiles.TileCount != 1)
            throw new NotSupportedException($"A {width}x{height} picture cannot be a single AV1 tile (it needs {tiles.TileCount}).");
        return EncodeTiles(ySrc, ySrcStride, uSrc, uvSrcStride, vSrc, width, height, baseQIndex, tiles)[0];
    }

    /// <summary>
    /// Encode every tile of the grid <paramref name="tiles"/> (raster order) and return each tile's raw range-coder
    /// byte stream. A tile starts with a fresh range coder and zeroed contexts and never reads a pixel or a context
    /// outside itself; the reconstruction planes are shared (each tile writes its own rectangle).
    /// </summary>
    internal static byte[][] EncodeTiles(
        ReadOnlySpan<byte> ySrc, int ySrcStride,
        ReadOnlySpan<byte> uSrc, int uvSrcStride,
        ReadOnlySpan<byte> vSrc,
        int width, int height, int baseQIndex, Av1TileLayout tiles)
    {
        int frameMiCols = ((width + 7) >> 3) << 1;     // mi rows in 4-px units
        int frameMiRows = ((height + 7) >> 3) << 1;
        int sbSize = 64;
        int sbMi = sbSize >> 2; // 16 mi units per SB

        var reconY = new byte[(frameMiCols * 4) * (frameMiRows * 4)];
        var reconU = new byte[(frameMiCols * 2) * (frameMiRows * 2)];
        var reconV = new byte[(frameMiCols * 2) * (frameMiRows * 2)];
        var result = new byte[tiles.TileCount][];

        for (int tileRow = 0; tileRow < tiles.TileRows; tileRow++)
            for (int tileCol = 0; tileCol < tiles.TileCols; tileCol++)
            {
                int miRowStart = tileRow * tiles.TileHeightSb * sbMi;
                int miColStart = tileCol * tiles.TileWidthSb * sbMi;
                int miRowEnd = Math.Min(miRowStart + tiles.TileHeightSb * sbMi, frameMiRows);
                int miColEnd = Math.Min(miColStart + tiles.TileWidthSb * sbMi, frameMiCols);

                var state = new TileEncodeState
                {
                    Re = new Av1RangeEncoder(),
                    Pctx = new Av1PartitionContext(frameMiCols),
                    EntropyCtx = new Av1EntropyContext(frameMiCols),
                    AboveYMode = new Av1IntraMode[frameMiCols],
                    AboveSkip = new bool[frameMiCols],
                    LeftYMode = new Av1IntraMode[Av1PartitionContext.MaxMibSize],
                    LeftSkip = new bool[Av1PartitionContext.MaxMibSize],
                    ReconY = reconY,
                    ReconU = reconU,
                    ReconV = reconV,
                    LumaW = frameMiCols * 4,
                    LumaH = frameMiRows * 4,
                    ChromaW = frameMiCols * 2,
                    ChromaH = frameMiRows * 2,
                    SrcLumaW = width,
                    SrcLumaH = height,
                    SrcChromaW = (width + 1) / 2,
                    SrcChromaH = (height + 1) / 2,
                    TileMiRowStart = miRowStart,
                    TileMiColStart = miColStart,
                    TileMiRowEnd = miRowEnd,
                    TileMiColEnd = miColEnd,
                };
                Array.Fill(state.AboveYMode, Av1IntraMode.Dc);
                Array.Fill(state.LeftYMode, Av1IntraMode.Dc);

                for (int miRow = miRowStart; miRow < miRowEnd; miRow += sbMi)
                {
                    state.Pctx.ResetLeft();
                    state.EntropyCtx.ResetLeft();
                    Array.Fill(state.LeftYMode, Av1IntraMode.Dc);
                    Array.Clear(state.LeftSkip);
                    for (int miCol = miColStart; miCol < miColEnd; miCol += sbMi)
                    {
                        EncodePartition(
                            state, miRow, miCol, Block64x64,
                            ySrc, ySrcStride, uSrc, uvSrcStride, vSrc,
                            width, height, frameMiRows, frameMiCols, baseQIndex);
                    }
                }

                result[tileRow * tiles.TileCols + tileCol] = state.Re.Done();
            }
        return result;
    }

    /// <summary>
    /// Recursively descend the partition tree, emitting partition symbols
    /// where required and per-leaf block coefficients at BLOCK_16X16 (BLOCK_8X8
    /// where a 16x16 crosses the picture edge).
    /// </summary>
    private static void EncodePartition(
        TileEncodeState st,
        int miRow, int miCol, int bsize,
        ReadOnlySpan<byte> ySrc, int ySrcStride,
        ReadOnlySpan<byte> uSrc, int uvSrcStride,
        ReadOnlySpan<byte> vSrc,
        int width, int height,
        int frameMiRows, int frameMiCols,
        int baseQIndex)
    {
        if (miRow >= frameMiRows || miCol >= frameMiCols) return;

        int bsl = Av1PartitionContext.MiSizeWideLog2[bsize];
        int bs = 1 << bsl; // mi units
        int hbs = bs >> 1;

        // For our v1 encoder every leaf is BLOCK_16X16 where the whole block is inside the picture.
        // BLOCK_16X16 has bsl = 2 (4 mi). Larger sizes recurse to SPLIT; a whole BLOCK_16X16 issues
        // PARTITION_NONE. A BLOCK_16X16 crossing the bottom / right edge cannot be PARTITION_NONE (the
        // syntax only allows SPLIT or HORZ / VERT there): it splits into BLOCK_8X8 leaves, which are
        // always whole (the picture is a whole number of 8x8 blocks) and issue PARTITION_NONE.
        Av1PartitionType partition;
        if (bsize >= Av1PartitionContext.Block8x8)
        {
            bool hasRows = (miRow + hbs) < frameMiRows;
            bool hasCols = (miCol + hbs) < frameMiCols;

            if (bsize == Av1PartitionContext.Block8x8)
            {
                partition = Av1PartitionType.None;
            }
            else if (bsize == Block16x16)
            {
                partition = hasRows && hasCols ? Av1PartitionType.None : Av1PartitionType.Split;
            }
            else
            {
                partition = Av1PartitionType.Split;
            }

            if (hasRows && hasCols)
            {
                // Standard partition CDF write.
                int ctx = st.Pctx.GetContext(miRow, miCol, bsize);
                int nsyms = Av1PartitionContext.PartitionCdfLength(bsize);
                var cdf = Av1DefaultPartitionCdfs.DefaultPartitionCdf[ctx];
                st.Re.EncodeCdfQ15((int)partition, cdf, nsyms);
            }
            else if (!hasRows && !hasCols)
            {
                // Forced split, no bit emitted.
                partition = Av1PartitionType.Split;
            }
            else if (!hasRows /* &amp;&amp; hasCols */)
            {
                // Bottom edge (!hasRows && hasCols): partition is SPLIT or HORZ.
                // libaom uses partition_gather_vert_alike on the per-context CDF
                // to derive a 2-symbol CDF, then writes (p == SPLIT) as sym 1.
                int ctx = st.Pctx.GetContext(miRow, miCol, bsize);
                var rowCdf = Av1DefaultPartitionCdfs.DefaultPartitionCdf[ctx];
                ushort[] cdf2 = GatherVertAlike(rowCdf, bsize);
                st.Re.EncodeCdfQ15(1 /* SPLIT */, cdf2, 2);
                partition = Av1PartitionType.Split;
            }
            else
            {
                // Right edge (hasRows && !hasCols): partition is SPLIT or VERT.
                // libaom uses partition_gather_horz_alike to derive the 2-sym CDF.
                int ctx = st.Pctx.GetContext(miRow, miCol, bsize);
                var rowCdf = Av1DefaultPartitionCdfs.DefaultPartitionCdf[ctx];
                ushort[] cdf2 = GatherHorzAlike(rowCdf, bsize);
                st.Re.EncodeCdfQ15(1 /* SPLIT */, cdf2, 2);
                partition = Av1PartitionType.Split;
            }
        }
        else
        {
            // Sub-8x8 block: implicit NONE.
            partition = Av1PartitionType.None;
        }

        // Map (bsize, partition) -&gt; subsize to recurse / leaf-decode.
        int subsize = Av1KeyframeWalkerSubsize(bsize, partition);

        switch (partition)
        {
            case Av1PartitionType.None:
                EncodeLeafBlock(
                    st, miRow, miCol, subsize,
                    ySrc, ySrcStride, uSrc, uvSrcStride, vSrc,
                    width, height, baseQIndex);
                st.Pctx.UpdateContext(miRow, miCol, subsize);
                break;
            case Av1PartitionType.Split:
                EncodePartition(st, miRow, miCol, subsize,
                    ySrc, ySrcStride, uSrc, uvSrcStride, vSrc,
                    width, height, frameMiRows, frameMiCols, baseQIndex);
                EncodePartition(st, miRow, miCol + hbs, subsize,
                    ySrc, ySrcStride, uSrc, uvSrcStride, vSrc,
                    width, height, frameMiRows, frameMiCols, baseQIndex);
                EncodePartition(st, miRow + hbs, miCol, subsize,
                    ySrc, ySrcStride, uSrc, uvSrcStride, vSrc,
                    width, height, frameMiRows, frameMiCols, baseQIndex);
                EncodePartition(st, miRow + hbs, miCol + hbs, subsize,
                    ySrc, ySrcStride, uSrc, uvSrcStride, vSrc,
                    width, height, frameMiRows, frameMiCols, baseQIndex);
                break;
            default:
                throw new NotSupportedException(
                    $"v1 encoder doesn't emit partition type {partition}");
        }
    }

    /// <summary>
    /// Mirrors libaom <c>partition_gather_vert_alike</c>: derive the 2-symbol
    /// CDF used at the bottom edge (!hasRows &amp;&amp; hasCols) where the only valid
    /// partitions are SPLIT and HORZ.
    /// </summary>
    private static ushort[] GatherVertAlike(ushort[] cdf, int bsize)
    {
        // libaom: out[0] = CDF_PROB_TOP - sum_of_probs(VERT, SPLIT, HORZ_A,
        //   VERT_A, VERT_B [, VERT_4 if bsize != 128x128]).
        // Then out[0] = AOM_ICDF(out[0]).
        // The 2-symbol CDF: sym 0 = HORZ, sym 1 = SPLIT. The encoder writes
        // (p == SPLIT) so sym 1 fires for SPLIT.
        const int CdfProbTop = 1 << 15;
        int outVal = CdfProbTop;
        outVal -= CdfElementProb(cdf, (int)Av1PartitionType.Vert);
        outVal -= CdfElementProb(cdf, (int)Av1PartitionType.Split);
        outVal -= CdfElementProb(cdf, (int)Av1PartitionType.HorzA);
        outVal -= CdfElementProb(cdf, (int)Av1PartitionType.VertA);
        outVal -= CdfElementProb(cdf, (int)Av1PartitionType.VertB);
        if (bsize != Av1PartitionContext.Block128x128)
            outVal -= CdfElementProb(cdf, (int)Av1PartitionType.Vert4);
        outVal = CdfProbTop - outVal; // AOM_ICDF
        return new ushort[] { (ushort)outVal, 0, 0 };
    }

    /// <summary>
    /// Mirrors libaom <c>partition_gather_horz_alike</c>: derive the 2-symbol
    /// CDF used at the right edge (hasRows &amp;&amp; !hasCols) where the only valid
    /// partitions are SPLIT and VERT.
    /// </summary>
    private static ushort[] GatherHorzAlike(ushort[] cdf, int bsize)
    {
        // libaom: out[0] = CDF_PROB_TOP - sum_of_probs(HORZ, SPLIT, HORZ_A,
        //   HORZ_B, VERT_A [, HORZ_4 if bsize != 128x128]).
        // Then out[0] = AOM_ICDF(out[0]).
        // The 2-symbol CDF: sym 0 = VERT, sym 1 = SPLIT.
        const int CdfProbTop = 1 << 15;
        int outVal = CdfProbTop;
        outVal -= CdfElementProb(cdf, (int)Av1PartitionType.Horz);
        outVal -= CdfElementProb(cdf, (int)Av1PartitionType.Split);
        outVal -= CdfElementProb(cdf, (int)Av1PartitionType.HorzA);
        outVal -= CdfElementProb(cdf, (int)Av1PartitionType.HorzB);
        outVal -= CdfElementProb(cdf, (int)Av1PartitionType.VertA);
        if (bsize != Av1PartitionContext.Block128x128)
            outVal -= CdfElementProb(cdf, (int)Av1PartitionType.Horz4);
        outVal = CdfProbTop - outVal; // AOM_ICDF
        return new ushort[] { (ushort)outVal, 0, 0 };
    }

    /// <summary>
    /// Mirrors libaom <c>cdf_element_prob</c>: returns the probability mass for
    /// the given element index, computed as (cumprob[element-1] - cumprob[element])
    /// where cumprob is derived from the inverse-CDF storage.
    /// libaom storage: cdf[i] = AOM_ICDF(cumprob[i]) = CDF_PROB_TOP - cumprob[i].
    /// So prob(element) = cdf[element] - cdf[element-1] (or CDF_PROB_TOP for element 0).
    /// Wait, let's recompute: prob(elem) = cumprob(elem) - cumprob(elem-1).
    ///   = (CDF_PROB_TOP - cdf[elem]) - (CDF_PROB_TOP - cdf[elem-1])
    ///   = cdf[elem-1] - cdf[elem].
    /// libaom impl: `(element > 0 ? cdf[element - 1] : CDF_PROB_TOP) - cdf[element]`.
    /// </summary>
    private static int CdfElementProb(ushort[] cdf, int element)
    {
        const int CdfProbTop = 1 << 15;
        int prev = element > 0 ? cdf[element - 1] : CdfProbTop;
        return prev - cdf[element];
    }

    /// <summary>
    /// Mirror of libaom <c>subsize_lookup[partition][bsize]</c> for the
    /// partition shapes the v1 encoder uses (NONE + SPLIT only).
    /// </summary>
    private static int Av1KeyframeWalkerSubsize(int bsize, Av1PartitionType partition)
    {
        // NONE: subsize = bsize. SPLIT: subsize = bsize down one level.
        if (partition == Av1PartitionType.None) return bsize;
        if (partition == Av1PartitionType.Split)
        {
            return bsize switch
            {
                Block64x64 => Block32x32,
                Block32x32 => Block16x16,
                Block16x16 => Av1PartitionContext.Block8x8,
                Av1PartitionContext.Block8x8 => 0, // BLOCK_4X4
                _ => throw new InvalidOperationException($"Unsupported SPLIT bsize {bsize}"),
            };
        }
        throw new NotSupportedException($"v1 encoder doesn't compute subsize for {partition}");
    }

    // ------------------------------------------------------------------
    // Leaf block encode: mode info + per-plane coefs + reconstruction
    // ------------------------------------------------------------------

    private static void EncodeLeafBlock(
        TileEncodeState st,
        int miRow, int miCol, int bsize,
        ReadOnlySpan<byte> ySrc, int ySrcStride,
        ReadOnlySpan<byte> uSrc, int uvSrcStride,
        ReadOnlySpan<byte> vSrc,
        int width, int height,
        int baseQIndex)
    {
        if (bsize != Block16x16 && bsize != Av1PartitionContext.Block8x8)
        {
            throw new NotSupportedException(
                $"v1 encoder only emits BLOCK_16X16 and BLOCK_8X8 leaves, got bsize={bsize}");
        }
        bool is16 = bsize == Block16x16;
        int lumaPx = is16 ? 16 : 8;
        int chromaPx = lumaPx / 2;
        var lumaTx = is16 ? Av1TxSize.Tx16x16 : Av1TxSize.Tx8x8;
        var chromaTx = is16 ? Av1TxSize.Tx8x8 : Av1TxSize.Tx4x4;

        // ---- Mode info bits (mirror Av1ModeInfoReader / write_mb_modes_kf) ----

        // Segmentation disabled -&gt; no segment_id.

        // Skip flag: write 0 (we have residual).
        int leftMiIdx = miRow & (Av1PartitionContext.MaxMibSize - 1);
        int aboveSkipBit = miCol < st.AboveSkip.Length && st.AboveSkip[miCol] ? 1 : 0;
        int leftSkipBit = st.LeftSkip[leftMiIdx] ? 1 : 0;
        int skipCtx = aboveSkipBit + leftSkipBit;
        var skipCdf = Av1DefaultBlockCdfs.DefaultSkipTxfmCdf[skipCtx];
        st.Re.EncodeCdfQ15(0, skipCdf, 2);

        // CDEF skipped (SH.EnableCdef=false).
        // delta_q skipped (delta_q_present=false).

        // Y mode: DC_PRED (sym 0), CDF row by above + left intra context.
        Av1IntraMode aboveYMode = miCol < st.AboveYMode.Length ? st.AboveYMode[miCol] : Av1IntraMode.Dc;
        Av1IntraMode leftYMode = st.LeftYMode[leftMiIdx];
        int aboveCtx = Av1IntraModeContext.Table[(int)aboveYMode];
        int leftCtx = Av1IntraModeContext.Table[(int)leftYMode];
        var yModeCdf = Av1DefaultIntraModeCdfs.DefaultKfYModeCdf[aboveCtx][leftCtx];
        st.Re.EncodeCdfQ15((int)Av1IntraMode.Dc, yModeCdf, Av1ModeInfoReader.IntraModes);

        // Y mode is DC -&gt; not directional -&gt; no angle delta.

        // UV mode: DC_PRED. CFL is allowed for 16x16 and for 8x8 (blocks up to 32x32): cfl_allowed=1.
        // libaom: uv_mode_cdf[cfl_allowed][y_mode], INTRA_MODES syms when cfl, else INTRA_MODES-1.
        // Wait - the formula is UV_INTRA_MODES - !cfl_allowed = 14 - 0 = 14 syms when cfl_allowed = 1.
        int cflAllowed = 1;
        var uvCdf = Av1DefaultIntraModeCdfs.DefaultUvModeCdf[cflAllowed][(int)Av1IntraMode.Dc];
        st.Re.EncodeCdfQ15((int)Av1IntraMode.Dc, uvCdf, Av1ModeInfoReader.UvIntraModes - (1 - cflAllowed));

        // UV mode is DC -&gt; not directional -&gt; no angle delta.

        // Filter intra: SH.EnableFilterIntra=false -&gt; no emission.

        // ---- Per-plane: predict, transform, quantize, emit coefs, reconstruct ----
        EncodePlane(st, plane: 0, miRow, miCol,
            ySrc, ySrcStride, st.SrcLumaW, st.SrcLumaH, st.ReconY, st.LumaW, st.LumaH,
            blockWidthPx: lumaPx, blockHeightPx: lumaPx,
            txSize: lumaTx,
            qDc: Av1DequantTables.DcQuantQtx(baseQIndex, 0, 8),
            qAc: Av1DequantTables.AcQuantQtx(baseQIndex, 0, 8),
            baseQIndex: baseQIndex);
        EncodePlane(st, plane: 1, miRow, miCol,
            uSrc, uvSrcStride, st.SrcChromaW, st.SrcChromaH, st.ReconU, st.ChromaW, st.ChromaH,
            blockWidthPx: chromaPx, blockHeightPx: chromaPx,
            txSize: chromaTx,
            qDc: Av1DequantTables.DcQuantQtx(baseQIndex, 0, 8),
            qAc: Av1DequantTables.AcQuantQtx(baseQIndex, 0, 8),
            baseQIndex: baseQIndex);
        EncodePlane(st, plane: 2, miRow, miCol,
            vSrc, uvSrcStride, st.SrcChromaW, st.SrcChromaH, st.ReconV, st.ChromaW, st.ChromaH,
            blockWidthPx: chromaPx, blockHeightPx: chromaPx,
            txSize: chromaTx,
            qDc: Av1DequantTables.DcQuantQtx(baseQIndex, 0, 8),
            qAc: Av1DequantTables.AcQuantQtx(baseQIndex, 0, 8),
            baseQIndex: baseQIndex);

        // ---- Update mode-info contexts so subsequent blocks see DC + skip=0 ----
        int miW = is16 ? 4 : 2; // BLOCK_16X16 = 4 mi wide, BLOCK_8X8 = 2
        int miH = miW;
        for (int i = 0; i < miW && (miCol + i) < st.AboveYMode.Length; i++)
        {
            st.AboveYMode[miCol + i] = Av1IntraMode.Dc;
            st.AboveSkip[miCol + i] = false;
        }
        for (int i = 0; i < miH; i++)
        {
            int r = (leftMiIdx + i) & (Av1PartitionContext.MaxMibSize - 1);
            st.LeftYMode[r] = Av1IntraMode.Dc;
            st.LeftSkip[r] = false;
        }
    }

    /// <summary>
    /// Predict, forward transform, quantize, emit coefficients, and reconstruct
    /// one plane of one BLOCK_16X16 leaf. Writes the decoded pixels back into
    /// the recon plane so neighboring blocks see the same reconstruction the
    /// decoder will see.
    /// </summary>
    private static void EncodePlane(
        TileEncodeState st, int plane,
        int miRow, int miCol,
        ReadOnlySpan<byte> src, int srcStride, int srcW, int srcH,
        byte[] reconBuf, int planeW, int planeH,
        int blockWidthPx, int blockHeightPx,
        Av1TxSize txSize, short qDc, short qAc,
        int baseQIndex)
    {
        // Block origin in pixels for this plane.
        int xPx, yPx;
        if (plane == 0)
        {
            xPx = miCol * 4;
            yPx = miRow * 4;
        }
        else
        {
            // Chroma is subsampled 2x2 for 4:2:0.
            xPx = (miCol * 4) >> 1;
            yPx = (miRow * 4) >> 1;
        }

        int txW = Av1TxSizeInfo.TxWide[(int)txSize];
        int txH = Av1TxSizeInfo.TxHigh[(int)txSize];
        // For BLOCK_16X16 + plane Y / UV in v1, blockWidthPx == txW and
        // blockHeightPx == txH so txCols = txRows = 1.
        int txCols = blockWidthPx / txW;
        int txRows = blockHeightPx / txH;

        var leftMiIdx = miRow & (Av1PartitionContext.MaxMibSize - 1);
        int planeType = plane == 0 ? 0 : 1;

        for (int ty = 0; ty < txRows; ty++)
        {
            for (int tc = 0; tc < txCols; tc++)
            {
                int xb = xPx + tc * txW;
                int yb = yPx + ty * txH;

                // Build edge buffers from reconstructed pixels.
                // Neighbours exist only inside the tile: nothing above its first row or left of its first column,
                // and the above-right / below-left extensions stop at its right / bottom edge.
                int pxPerMi = plane == 0 ? 4 : 2;
                bool haveAbove = yb > st.TileMiRowStart * pxPerMi;
                bool haveLeft = xb > st.TileMiColStart * pxPerMi;
                var edge = Av1IntraEdge.Build(reconBuf, planeW, st.TileMiColEnd * pxPerMi, st.TileMiRowEnd * pxPerMi,
                    xb, yb, txW, txH, haveAbove, haveLeft);

                // Predict (DC).
                var predict = new byte[txW * txH];
                Av1IntraPredictDispatch.Predict(Av1IntraMode.Dc, edge, predict, txW, txW, txH);

                // Compute residual = source - predict.
                var residual = new short[txW * txH];
                // Past the picture (the coded size is rounded up to 8) the last source column / row repeats.
                for (int r = 0; r < txH; r++)
                {
                    int sRow = Math.Min(yb + r, srcH - 1) * srcStride;
                    int pOff = r * txW;
                    for (int c = 0; c < txW; c++)
                    {
                        residual[r * txW + c] = (short)(src[sRow + Math.Min(xb + c, srcW - 1)] - predict[pOff + c]);
                    }
                }

                // Forward transform.
                var coefsRaster = new int[txW * txH];
                Av1Forward2dTransform.Apply(txSize, Av1TxType.DctDct, residual, coefsRaster);

                // Quantize: DC at coefs[0] (raster (0,0) is also scan[0] for any 2D scan),
                // AC at others.
                Av1ForwardQuantizer.QuantizeBlock(coefsRaster, qDc, qAc);

                // Compute entropy contexts for THIS tx block.
                // Av1EntropyContext is keyed by LUMA mi indices for every plane. A chroma 4x4 unit covers
                // two luma mi, so a chroma transform spans twice its own 4x4 count in luma mi (TX_8X8 of a
                // 16x16 block: 4; TX_4X4 of an 8x8 block: 2) and every one of them is written. (Until the
                // 8x8 edge leaves existed only the first half was written and read, which is the same context
                // as long as all blocks are 16x16; an 8x8 block under the right half of a 16x16 reads the
                // second half.)
                int lumaMiPerUnit = plane == 0 ? 1 : 2;
                int txWMi = Math.Max(1, txW >> 2) * lumaMiPerUnit;
                int txHMi = Math.Max(1, txH >> 2) * lumaMiPerUnit;
                int miRowTx = miRow + ((ty * txH) >> 2) * lumaMiPerUnit;
                int miColTx = miCol + ((tc * txW) >> 2) * lumaMiPerUnit;
                // Every leaf has planeBsize == txsize_to_bsize[tx_size] (16x16: Y TX_16X16 / UV TX_8X8;
                // 8x8: Y TX_8X8 / UV TX_4X4), so planeBsizeIsTxsize=true and
                // planeBsizeLargerThanTxBsize=false.
                int txbSkipCtx = st.EntropyCtx.GetTxbSkipContext(plane, miRowTx, miColTx, txWMi, txHMi,
                    planeBsizeIsTxsize: true, planeBsizeLargerThanTxBsize: false);
                int dcSignCtx = st.EntropyCtx.GetDcSignContext(plane, miRowTx, miColTx, txWMi, txHMi);

                // Emit coefficient stream. Pass baseQIndex so the encoder uses
                // the same q-context CDF buckets as the libaom decoder.
                var encResult = Av1CoefEncoder.WriteCoeffsTxb(
                    st.Re, txSize, plane, Av1IntraMode.Dc,
                    reducedTxSet: true,
                    txbSkipCtx, dcSignCtx,
                    coefsRaster, baseQIndex, Av1TxType.DctDct);

                // Update entropy context for subsequent blocks.
                st.EntropyCtx.Update(plane, miRowTx, miColTx, txWMi, txHMi, encResult.CulLevel);

                // Reconstruct: dequant + inverse transform + add to predictor.
                var dq = new int[txW * txH];
                int shift = Av1TxbCommon.GetTxScale(txSize);
                int maxValue = (1 << (7 + 8)) - 1;
                int minValue = -(1 << (7 + 8));
                for (int i = 0; i < txW * txH; i++)
                {
                    int level = coefsRaster[i];
                    if (level == 0) { dq[i] = 0; continue; }
                    int absLevel = level < 0 ? -level : level;
                    int sign = level < 0 ? 1 : 0;
                    short q = (i == 0) ? qDc : qAc;
                    long dqCoeff = ((long)absLevel * q) & 0xFFFFFF;
                    int dqInt = (int)dqCoeff;
                    dqInt = dqInt >> shift;
                    if (sign != 0) dqInt = -dqInt;
                    if (dqInt > maxValue) dqInt = maxValue;
                    if (dqInt < minValue) dqInt = minValue;
                    dq[i] = dqInt;
                }
                var reconResid = new int[txW * txH];
                if (encResult.Eob > 0)
                {
                    Av1Inverse2dTransform.Apply(txSize, Av1TxType.DctDct, dq, reconResid);
                }

                // Add residual to predictor + clip + write back to recon plane.
                for (int r = 0; r < txH; r++)
                {
                    int dstOff = (yb + r) * planeW + xb;
                    int pOff = r * txW;
                    for (int c = 0; c < txW; c++)
                    {
                        int v = predict[pOff + c] + reconResid[pOff + c];
                        if (v < 0) v = 0;
                        else if (v > 255) v = 255;
                        reconBuf[dstOff + c] = (byte)v;
                    }
                }
            }
        }
    }
}
