// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 keyframe encoder. v1 minimal-but-correct emit path that produces a
// decodable AV1 keyframe bitstream for our 16x16-aligned test frames.
//
// v1 simplifications (mirrors Vp9KeyframeEncoder shape):
//   - Profile 0 (8-bit 4:2:0)
//   - Width and height multiples of 16
//   - Every BLOCK_16X16 uses Y intra mode = DC_PRED, transform = TX_16X16 + DCT_DCT
//   - Every chroma block uses UV intra mode = DC_PRED, transform = TX_8X8 + DCT_DCT
//   - tx_mode = LARGEST (no per-block tx_size signaling)
//   - Single tile (Log2NumTiles = 0)
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
    /// <param name="uSrc">U plane bytes (length &gt;= uvSrcStride * height/2).</param>
    /// <param name="uvSrcStride">UV plane row stride in bytes (&gt;= width/2).</param>
    /// <param name="vSrc">V plane bytes (length &gt;= uvSrcStride * height/2).</param>
    /// <param name="width">Frame width in pixels (multiple of 16 for v1).</param>
    /// <param name="height">Frame height in pixels (multiple of 16 for v1).</param>
    /// <param name="baseQIndex">Base quantizer index 1..255 (lower = higher quality).</param>
    public static byte[] EncodeKeyFrame(
        ReadOnlySpan<byte> ySrc, int ySrcStride,
        ReadOnlySpan<byte> uSrc, int uvSrcStride,
        ReadOnlySpan<byte> vSrc,
        int width, int height,
        int baseQIndex = 32)
    {
        if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width));
        if (height <= 0) throw new ArgumentOutOfRangeException(nameof(height));
        if ((width & 15) != 0 || (height & 15) != 0)
            throw new ArgumentException("Width and height must be multiples of 16 (v1)");
        if (baseQIndex < 1 || baseQIndex > 255)
            throw new ArgumentOutOfRangeException(nameof(baseQIndex),
                "baseQIndex must be in [1, 255]; lossless not supported in v1.");

        // ---- 1. Temporal Delimiter OBU ----
        byte[] tdObu = Av1ObuWriter.EmitObu(
            Av1ObuType.TemporalDelimiter, ReadOnlySpan<byte>.Empty, hasSizeField: true);

        // ---- 2. Sequence Header OBU ----
        var shConfig = Av1KeyframeFraming.BuildSequenceHeaderConfig(width, height);
        byte[] shPayload = Av1SequenceHeaderWriter.EmitPayload(shConfig);
        byte[] shObu = Av1ObuWriter.EmitObu(
            Av1ObuType.SequenceHeader, shPayload, hasSizeField: true);
        // Reify the SH so downstream writers consume a parsed view without re-parsing.
        var sh = Av1KeyframeFraming.SynthesizeSequenceHeader(shConfig);

        // ---- 3. Build the Frame OBU payload (uncompressed_header + tile_group) ----

        // Build the tile entropy stream first; we need to know its byte length
        // to size the frame OBU and (in multi-tile streams) emit per-tile size
        // prefixes. For a single tile no per-tile size is emitted.
        byte[] tileBytes = EncodeSingleTile(
            ySrc, ySrcStride, uSrc, uvSrcStride, vSrc,
            width, height, baseQIndex);

        byte[] uncompressedHeader = Av1KeyframeFraming.BuildUncompressedHeader(sh, width, height, baseQIndex);

        var frameOuter = new byte[uncompressedHeader.Length + tileBytes.Length];
        Buffer.BlockCopy(uncompressedHeader, 0, frameOuter, 0, uncompressedHeader.Length);
        Buffer.BlockCopy(tileBytes, 0, frameOuter, uncompressedHeader.Length, tileBytes.Length);

        byte[] frameObu = Av1ObuWriter.EmitObu(
            Av1ObuType.Frame, frameOuter, hasSizeField: true);

        // ---- 4. Concatenate all three OBUs ----
        var output = new byte[tdObu.Length + shObu.Length + frameObu.Length];
        Buffer.BlockCopy(tdObu, 0, output, 0, tdObu.Length);
        Buffer.BlockCopy(shObu, 0, output, tdObu.Length, shObu.Length);
        Buffer.BlockCopy(frameObu, 0, output, tdObu.Length + shObu.Length, frameObu.Length);
        return output;
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

        // ---- TD OBU ----
        byte[] tdObu = Av1ObuWriter.EmitObu(
            Av1ObuType.TemporalDelimiter, ReadOnlySpan<byte>.Empty, hasSizeField: true);

        // ---- SH OBU ----
        var shConfig = Av1KeyframeFraming.BuildSequenceHeaderConfig(width, height);
        byte[] shPayload = Av1SequenceHeaderWriter.EmitPayload(shConfig);
        byte[] shObu = Av1ObuWriter.EmitObu(
            Av1ObuType.SequenceHeader, shPayload, hasSizeField: true);
        var sh = Av1KeyframeFraming.SynthesizeSequenceHeader(shConfig);

        // ---- Frame OBU = uncompressed header + (GPU-produced) tile bytes ----
        byte[] uncompressedHeader = Av1KeyframeFraming.BuildUncompressedHeader(sh, width, height, baseQIndex);
        var frameOuter = new byte[uncompressedHeader.Length + tileBytes.Length];
        Buffer.BlockCopy(uncompressedHeader, 0, frameOuter, 0, uncompressedHeader.Length);
        Buffer.BlockCopy(tileBytes, 0, frameOuter, uncompressedHeader.Length, tileBytes.Length);
        byte[] frameObu = Av1ObuWriter.EmitObu(
            Av1ObuType.Frame, frameOuter, hasSizeField: true);

        var output = new byte[tdObu.Length + shObu.Length + frameObu.Length];
        Buffer.BlockCopy(tdObu, 0, output, 0, tdObu.Length);
        Buffer.BlockCopy(shObu, 0, output, tdObu.Length, shObu.Length);
        Buffer.BlockCopy(frameObu, 0, output, tdObu.Length + shObu.Length, frameObu.Length);
        return output;
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
        public int LumaW;
        public int LumaH;
        public int ChromaW;
        public int ChromaH;
    }

    /// <summary>
    /// Encode the single keyframe tile and return the raw range-coder
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
        int frameMiCols = ((width + 7) >> 3) << 1;     // mi rows in 4-px units
        int frameMiRows = ((height + 7) >> 3) << 1;
        int sbSize = 64;
        int sbMi = sbSize >> 2; // 16 mi units per SB

        var state = new TileEncodeState
        {
            Re = new Av1RangeEncoder(),
            Pctx = new Av1PartitionContext(frameMiCols),
            EntropyCtx = new Av1EntropyContext(frameMiCols),
            AboveYMode = new Av1IntraMode[frameMiCols],
            AboveSkip = new bool[frameMiCols],
            LeftYMode = new Av1IntraMode[Av1PartitionContext.MaxMibSize],
            LeftSkip = new bool[Av1PartitionContext.MaxMibSize],
            ReconY = new byte[width * height],
            ReconU = new byte[(width / 2) * (height / 2)],
            ReconV = new byte[(width / 2) * (height / 2)],
            LumaW = width,
            LumaH = height,
            ChromaW = width / 2,
            ChromaH = height / 2,
        };
        Array.Fill(state.AboveYMode, Av1IntraMode.Dc);
        Array.Fill(state.LeftYMode, Av1IntraMode.Dc);

        for (int sbRow = 0; sbRow * sbMi < frameMiRows; sbRow++)
        {
            state.Pctx.ResetLeft();
            state.EntropyCtx.ResetLeft();
            Array.Fill(state.LeftYMode, Av1IntraMode.Dc);
            Array.Clear(state.LeftSkip);
            for (int sbCol = 0; sbCol * sbMi < frameMiCols; sbCol++)
            {
                int miRow = sbRow * sbMi;
                int miCol = sbCol * sbMi;
                EncodePartition(
                    state, miRow, miCol, Block64x64,
                    ySrc, ySrcStride, uSrc, uvSrcStride, vSrc,
                    width, height, frameMiRows, frameMiCols, baseQIndex);
            }
        }

        return state.Re.Done();
    }

    /// <summary>
    /// Recursively descend the partition tree, emitting partition symbols
    /// where required and per-leaf block coefficients at BLOCK_16X16.
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

        // For our v1 encoder we want every leaf to be BLOCK_16X16.
        // BLOCK_16X16 has bsl = 2 (4 mi). Larger sizes recurse to SPLIT;
        // BLOCK_16X16 issues PARTITION_NONE.
        Av1PartitionType partition;
        if (bsize >= Av1PartitionContext.Block8x8)
        {
            bool hasRows = (miRow + hbs) < frameMiRows;
            bool hasCols = (miCol + hbs) < frameMiCols;

            if (bsize == Block16x16)
            {
                partition = Av1PartitionType.None;
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
        if (bsize != Block16x16)
        {
            throw new NotSupportedException(
                $"v1 encoder only emits BLOCK_16X16 leaves, got bsize={bsize}");
        }

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

        // UV mode: DC_PRED. CFL is allowed for 16x16 (cfl_allowed=1).
        // libaom: uv_mode_cdf[cfl_allowed][y_mode], INTRA_MODES syms when cfl, else INTRA_MODES-1.
        // Wait - the formula is UV_INTRA_MODES - !cfl_allowed = 14 - 0 = 14 syms when cfl_allowed = 1.
        int cflAllowed = 1;
        var uvCdf = Av1DefaultIntraModeCdfs.DefaultUvModeCdf[cflAllowed][(int)Av1IntraMode.Dc];
        st.Re.EncodeCdfQ15((int)Av1IntraMode.Dc, uvCdf, Av1ModeInfoReader.UvIntraModes - (1 - cflAllowed));

        // UV mode is DC -&gt; not directional -&gt; no angle delta.

        // Filter intra: SH.EnableFilterIntra=false -&gt; no emission.

        // ---- Per-plane: predict, transform, quantize, emit coefs, reconstruct ----
        EncodePlane(st, plane: 0, miRow, miCol,
            ySrc, ySrcStride, st.ReconY, st.LumaW, st.LumaH,
            blockWidthPx: 16, blockHeightPx: 16,
            txSize: Av1TxSize.Tx16x16,
            qDc: Av1DequantTables.DcQuantQtx(baseQIndex, 0, 8),
            qAc: Av1DequantTables.AcQuantQtx(baseQIndex, 0, 8),
            baseQIndex: baseQIndex);
        EncodePlane(st, plane: 1, miRow, miCol,
            uSrc, uvSrcStride, st.ReconU, st.ChromaW, st.ChromaH,
            blockWidthPx: 8, blockHeightPx: 8,
            txSize: Av1TxSize.Tx8x8,
            qDc: Av1DequantTables.DcQuantQtx(baseQIndex, 0, 8),
            qAc: Av1DequantTables.AcQuantQtx(baseQIndex, 0, 8),
            baseQIndex: baseQIndex);
        EncodePlane(st, plane: 2, miRow, miCol,
            vSrc, uvSrcStride, st.ReconV, st.ChromaW, st.ChromaH,
            blockWidthPx: 8, blockHeightPx: 8,
            txSize: Av1TxSize.Tx8x8,
            qDc: Av1DequantTables.DcQuantQtx(baseQIndex, 0, 8),
            qAc: Av1DequantTables.AcQuantQtx(baseQIndex, 0, 8),
            baseQIndex: baseQIndex);

        // ---- Update mode-info contexts so subsequent blocks see DC + skip=0 ----
        int miW = 4; // BLOCK_16X16 = 4 mi wide
        int miH = 4;
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
        ReadOnlySpan<byte> src, int srcStride,
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
                bool haveAbove = yb > 0;
                bool haveLeft = xb > 0;
                var edge = Av1IntraEdge.Build(reconBuf, planeW, planeW, planeH, xb, yb, txW, txH);

                // Predict (DC).
                var predict = new byte[txW * txH];
                Av1IntraPredictDispatch.Predict(Av1IntraMode.Dc, edge, predict, txW, txW, txH);

                // Compute residual = source - predict.
                var residual = new short[txW * txH];
                for (int r = 0; r < txH; r++)
                {
                    int sOff = (yb + r) * srcStride + xb;
                    int pOff = r * txW;
                    for (int c = 0; c < txW; c++)
                    {
                        residual[r * txW + c] = (short)(src[sOff + c] - predict[pOff + c]);
                    }
                }

                // Forward transform.
                var coefsRaster = new int[txW * txH];
                Av1Forward2dTransform.Apply(txSize, Av1TxType.DctDct, residual, coefsRaster);

                // Quantize: DC at coefs[0] (raster (0,0) is also scan[0] for any 2D scan),
                // AC at others.
                Av1ForwardQuantizer.QuantizeBlock(coefsRaster, qDc, qAc);

                // Compute entropy contexts for THIS tx block.
                int txWMi = Math.Max(1, txW >> 2);
                int txHMi = Math.Max(1, txH >> 2);
                int miRowTx = miRow + ((ty * txH) >> 2) - ((plane == 0) ? 0 : 0);
                int miColTx = miCol + ((tc * txW) >> 2) - ((plane == 0) ? 0 : 0);
                // For chroma we still use luma-mi indexing per Av1EntropyContext convention.
                if (plane != 0)
                {
                    // Av1EntropyContext is keyed by luma mi indices; chroma keys
                    // are the same indices for the chroma-ref block. Use the
                    // luma block origin for both above + left lookups.
                    miRowTx = miRow;
                    miColTx = miCol;
                }
                // v1 encoder: BLOCK_16X16+TX_16X16 (Y) and BLOCK_8X8+TX_8X8 (UV).
                // Both have planeBsize == txsize_to_bsize[tx_size], so
                // planeBsizeIsTxsize=true and planeBsizeLargerThanTxBsize=false.
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
