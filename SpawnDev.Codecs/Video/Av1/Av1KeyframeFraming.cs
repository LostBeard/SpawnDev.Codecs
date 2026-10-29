// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 keyframe framing: the Temporal Delimiter + Sequence Header OBUs and the uncompressed frame header for a
// (width, height, q) keyframe. Shared by the GPU encoder (Av1KeyframeEncoderGpu uploads BuildFramingBlob's bytes) and
// the CPU reference encoder in SpawnDev.Codecs.References. Same settings as the encoder's v1 bitstream: profile 0,
// 8-bit 4:2:0, single tile, 64x64 superblocks, loop filter / CDEF / LR / segmentation off.
//
using SpawnDev.Codecs.EntropyCoders;

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>AV1 keyframe framing bytes (TD + SH OBUs, uncompressed header) for the v1 keyframe bitstream.</summary>
internal static class Av1KeyframeFraming
{
    /// <summary>
    /// The fixed part of a keyframe temporal unit for a <paramref name="width"/> x <paramref name="height"/> picture:
    /// the Temporal Delimiter OBU and Sequence Header OBU (<paramref name="aLen"/> bytes) followed by the
    /// uncompressed frame header (<paramref name="bLen"/> bytes). Small, depends only on (width, height, q), and is
    /// uploaded to the accelerator where <see cref="Av1ObuAssembleKernel"/> adds the data-dependent Frame OBU size
    /// and the GPU-produced tile bytes.
    /// </summary>
    internal static byte[] BuildFramingBlob(int width, int height, int baseQIndex, out int aLen, out int bLen)
    {
        if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width));
        if (height <= 0) throw new ArgumentOutOfRangeException(nameof(height));
        byte[] tdObu = Av1ObuWriter.EmitObu(
            Av1ObuType.TemporalDelimiter, ReadOnlySpan<byte>.Empty, hasSizeField: true);
        var shConfig = BuildSequenceHeaderConfig(width, height);
        byte[] shPayload = Av1SequenceHeaderWriter.EmitPayload(shConfig);
        byte[] shObu = Av1ObuWriter.EmitObu(
            Av1ObuType.SequenceHeader, shPayload, hasSizeField: true);
        var sh = SynthesizeSequenceHeader(shConfig);
        byte[] uncompressedHeader = BuildUncompressedHeader(sh, width, height, baseQIndex);

        aLen = tdObu.Length + shObu.Length;
        bLen = uncompressedHeader.Length;
        var blob = new byte[aLen + bLen];
        Buffer.BlockCopy(tdObu, 0, blob, 0, tdObu.Length);
        Buffer.BlockCopy(shObu, 0, blob, tdObu.Length, shObu.Length);
        Buffer.BlockCopy(uncompressedHeader, 0, blob, aLen, bLen);
        return blob;
    }

    // ------------------------------------------------------------------
    // Sequence header
    // ------------------------------------------------------------------

    internal static Av1SequenceHeaderConfig BuildSequenceHeaderConfig(int width, int height)
    {
        return new Av1SequenceHeaderConfig
        {
            SeqProfile = 0,
            SeqLevelIdx0 = 0,
            MaxFrameWidth = width,
            MaxFrameHeight = height,
            BitDepth = 8,
            Monochrome = false,
            SubsamplingX = 1,
            SubsamplingY = 1,
            ColorRangeFull = false,
            Use128x128Superblock = false,
            EnableFilterIntra = false,
            EnableIntraEdgeFilter = false,
            EnableInterintraCompound = false,
            EnableMaskedCompound = false,
            EnableWarpedMotion = false,
            EnableDualFilter = false,
            EnableOrderHint = false,
            SeqChooseScreenContentTools = false,
            SeqForceScreenContentTools = 0,
            EnableSuperres = false,
            EnableCdef = false,
            EnableRestoration = false,
            ColorDescriptionPresent = false,
            ChromaSamplePosition = 0,
            SeparateUvDeltas = false,
            FilmGrainParamsPresent = false,
        };
    }

    /// <summary>
    /// Hand-construct an <see cref="Av1SequenceHeader"/> that matches the values
    /// our SH writer emits. The downstream FH writer + tile encoder only read
    /// a small subset of these fields.
    /// </summary>
    internal static Av1SequenceHeader SynthesizeSequenceHeader(Av1SequenceHeaderConfig cfg)
    {
        return new Av1SequenceHeader
        {
            SeqProfile = cfg.SeqProfile,
            SeqLevelIdx0 = cfg.SeqLevelIdx0,
            MaxFrameWidth = cfg.MaxFrameWidth,
            MaxFrameHeight = cfg.MaxFrameHeight,
            BitDepth = cfg.BitDepth,
            Monochrome = cfg.Monochrome,
            SubsamplingX = cfg.SubsamplingX,
            SubsamplingY = cfg.SubsamplingY,
            ColorRangeFull = cfg.ColorRangeFull,
            Use128x128Superblock = cfg.Use128x128Superblock,
            EnableFilterIntra = cfg.EnableFilterIntra,
            EnableIntraEdgeFilter = cfg.EnableIntraEdgeFilter,
            EnableInterintraCompound = cfg.EnableInterintraCompound,
            EnableMaskedCompound = cfg.EnableMaskedCompound,
            EnableWarpedMotion = cfg.EnableWarpedMotion,
            EnableDualFilter = cfg.EnableDualFilter,
            EnableOrderHint = cfg.EnableOrderHint,
            EnableJntComp = cfg.EnableJntComp,
            EnableRefFrameMvs = cfg.EnableRefFrameMvs,
            OrderHintBitsMinus1 = cfg.OrderHintBitsMinus1,
            SeqChooseScreenContentTools = cfg.SeqChooseScreenContentTools,
            SeqForceScreenContentTools = cfg.SeqChooseScreenContentTools ? 2 : cfg.SeqForceScreenContentTools,
            SeqChooseIntegerMv = cfg.SeqChooseIntegerMv,
            SeqForceIntegerMv = cfg.SeqChooseIntegerMv ? 2 : cfg.SeqForceIntegerMv,
            EnableSuperres = cfg.EnableSuperres,
            EnableCdef = cfg.EnableCdef,
            EnableRestoration = cfg.EnableRestoration,
            ColorDescriptionPresent = cfg.ColorDescriptionPresent,
            ColorPrimaries = cfg.ColorPrimaries,
            TransferCharacteristics = cfg.TransferCharacteristics,
            MatrixCoefficients = cfg.MatrixCoefficients,
            ChromaSamplePosition = cfg.ChromaSamplePosition,
            SeparateUvDeltas = cfg.SeparateUvDeltas,
            FilmGrainParamsPresent = cfg.FilmGrainParamsPresent,
            StillPicture = false,
            ReducedStillPictureHeader = false,
            FrameIdNumbersPresent = false,
            FrameIdLengthMinus7 = 0,
        };
    }

    // ------------------------------------------------------------------
    // Uncompressed header (mirrors Av1CompleteFrameHeaderParser bit-for-bit)
    // ------------------------------------------------------------------

    internal static byte[] BuildUncompressedHeader(
        Av1SequenceHeader sh, int width, int height, int baseQIndex)
    {
        var bw = new Av1BitWriter();

        // ---- Prefix ----
        bw.WriteFlag(false); // show_existing_frame
        bw.WriteBits((int)Av1FrameType.KeyFrame, 2);
        bw.WriteFlag(true);  // show_frame
        // error_resilient_mode is implicit (true) for visible KeyFrame; no bit emitted.
        // disable_cdf_update = 1: tells the decoder NOT to adaptively update
        // CDFs after each symbol read. Our encoder uses the static default
        // CDFs (no adaptation). For the bitstream to round-trip with libaom /
        // libdav1d / ffmpeg, we MUST signal disable_cdf_update so the decoder
        // also keeps the CDFs frozen at the defaults. Without this flag, even
        // a 2-block frame produces a stream the decoder rejects (it expects
        // adapted CDFs that we never wrote).
        bw.WriteFlag(true); // disable_cdf_update
        // SH chose force=0 (no SELECT), so allow_screen_content_tools NOT emitted.
        // SH SCC=0 means force_integer_mv not emitted.
        // FrameIdNumbersPresent=false: no current_frame_id.
        // KeyFrame is not SwitchFrame, SH not reduced -&gt; emit frame_size_override.
        bw.WriteFlag(false); // frame_size_override -&gt; SH defaults define frame size.
        // SH.EnableOrderHint=false: no order_hint.
        // KeyFrame visible: refresh_frame_flags is implicit 0xFF, no bit.
        // frame_size_override=false: no frame_size emission.

        // ---- Post-prefix ----
        // SH.EnableSuperres=false: no superres bits.
        bw.WriteFlag(false); // render_and_frame_size_different
        // allowSccTools=0 -&gt; no allow_intrabc emission.
        // disable_cdf_update=true -&gt; mightBwdAdapt=false -&gt; NO refresh_frame_context bit.
        // (libaom encoder gates this on `!disable_cdf_update`.)

        // ---- Tile info (Av1CompleteFrameHeaderParser.ReadTileInfo) ----
        EmitTileInfo(bw, sh, width, height);

        // ---- Quantization ----
        bw.WriteBits(baseQIndex & 0xFF, 8);
        bw.WriteFlag(false); // y_dc_delta_present
        // numPlanes &gt; 1 -&gt; emit u_dc, u_ac (and v_dc, v_ac if separate_uv_deltas + diff_uv).
        // SH.SeparateUvDeltas=false -&gt; no diff_uv bit; just emit u_dc, u_ac.
        bw.WriteFlag(false); // u_dc_delta_present
        bw.WriteFlag(false); // u_ac_delta_present
        bw.WriteFlag(false); // using_qmatrix

        // ---- Segmentation ----
        bw.WriteFlag(false); // seg.enabled

        // ---- delta_q / delta_lf (gated on baseQIndex &gt; 0) ----
        // baseQIndex &gt; 0 -&gt; emit delta_q_present.
        bw.WriteFlag(false); // delta_q_present_flag

        // ---- Loop filter (skipped when allow_intrabc OR coded_lossless;
        //      neither applies here) ----
        bw.WriteBits(0, 6); // filter_level[0] = 0 (LF disabled)
        bw.WriteBits(0, 6); // filter_level[1] = 0
        // numPlanes > 1 + filter_level[0] == 0 + filter_level[1] == 0
        // -&gt; chroma filter_level NOT emitted.
        bw.WriteBits(0, 3); // sharpness_level
        bw.WriteFlag(false); // mode_ref_delta_enabled

        // ---- CDEF skipped: SH.EnableCdef=false ----

        // ---- LR skipped: SH.EnableRestoration=false ----

        // ---- tx_mode: emit 0 (Largest), since !coded_lossless ----
        bw.WriteFlag(false); // tx_mode == TX_MODE_SELECT? No -&gt; LARGEST

        // ---- reference_mode skipped: intra-only frame ----

        // ---- skip_mode skipped: skip_mode_allowed = false on intra frames ----

        // ---- warped_motion skipped: frame_might_allow_warped_motion = false on intra ----

        // ---- reduced_tx_set ----
        bw.WriteFlag(true); // reduced_tx_set_used = 1

        // ---- film_grain skipped: SH.FilmGrainParamsPresent=false ----

        // Per AV1 spec sec 5.3.4 + 5.3.5, the Frame OBU lays out as:
        //   frame_header_obu (uncompressed_header)
        //   byte_alignment()
        //   tile_group_obu (which itself has tile_start_and_end_present_flag
        //                   inferred 0 for single-tile, then byte_alignment(),
        //                   then the entropy-coded tile bytes).
        //
        // byte_alignment() is zero-pad to byte boundary; it does NOT emit a
        // trailing 1 bit (that's WriteTrailingBits, used by SH and standalone
        // FrameHeader OBUs but NOT by combined Frame OBU).
        bw.ByteAlign();
        return bw.ToArray();
    }

    internal static void EmitTileInfo(Av1BitWriter bw, Av1SequenceHeader sh, int width, int height)
    {
        // Compute the tile geometry the parser uses.
        // Mirrors libaom av1_get_tile_limits (tile_common.c):
        //   max_width_sb     = MAX_TILE_WIDTH >> sb_size_log2  (= 64 for 64-px SB)
        //   max_tile_area_sb = MAX_TILE_AREA  >> (2 * sb_size_log2)  (= 2304 for 64-px SB)
        //   min_log2_cols = tile_log2(max_width_sb, sb_cols)
        //   min_log2      = tile_log2(max_tile_area_sb, sb_cols * sb_rows)
        // MAX_TILE_WIDTH = 4096 px, MAX_TILE_AREA = 4096 * 2304 px (per AV1 spec).
        int sbSize = sh.Use128x128Superblock ? 128 : 64;
        int sbSizeLog2 = sh.Use128x128Superblock ? 7 : 6;
        int miSizeLog2 = 2;
        int mibSizeLog2 = sbSizeLog2 - miSizeLog2;
        // mi_cols = ceil(width_px / 4), mi_rows = ceil(height_px / 4).
        // Width/height are multiples of 16 (v1 invariant) so no rounding needed,
        // but the spec uses ceiling-div so we follow the formula. Note: AV1 mi
        // units are 4-px, NOT 8-px; a previous version incorrectly used >>3 which
        // gave 8-px column count and produced widthSb half what it should be.
        int miCols = (width + 3) >> 2;
        int miRows = (height + 3) >> 2;
        int widthSb = (miCols + (1 << mibSizeLog2) - 1) >> mibSizeLog2;
        int heightSb = (miRows + (1 << mibSizeLog2) - 1) >> mibSizeLog2;

        int maxWidthSb = 4096 >> sbSizeLog2;            // = 64 for 64-px SB
        int maxTileAreaSb = (4096 * 2304) >> (2 * sbSizeLog2); // = 2304 for 64-px SB
        int maxLog2TileCols = TileLog2(1, Math.Min(widthSb, 64));
        int maxLog2TileRows = TileLog2(1, Math.Min(heightSb, 64));
        int minLog2TileCols = Math.Max(0, TileLog2(maxWidthSb, widthSb));
        int minLog2Tiles = Math.Max(minLog2TileCols, TileLog2(maxTileAreaSb, widthSb * heightSb));
        int minLog2TileRows = Math.Max(0, minLog2Tiles - minLog2TileCols);

        // Single tile: log2_cols = minLog2TileCols, log2_rows = max(0, minLog2Tiles - log2_cols).
        // For a 16x16 frame, widthSb=heightSb=1, all the maxes are 0, so no
        // increment bits are emitted for either axis.
        bw.WriteFlag(true); // uniform_tile_spacing_flag

        // log2_cols increments: emit (log2_cols - minLog2TileCols) ones; if &lt; max, emit a 0.
        int log2Cols = minLog2TileCols;
        // We choose to stay at min (= 0 for our small frames).
        if (log2Cols < maxLog2TileCols)
        {
            bw.WriteBits(0, 1); // first 0 bit terminates the increment run
        }

        // log2_rows increments
        int log2RowsMin = Math.Max(0, minLog2Tiles - log2Cols);
        int log2Rows = log2RowsMin;
        if (log2Rows < maxLog2TileRows)
        {
            bw.WriteBits(0, 1);
        }

        // tile_cols * tile_rows == 1 -&gt; no contextUpdateTileId / tileSizeBytes.
        if ((1 << log2Cols) * (1 << log2Rows) > 1)
        {
            bw.WriteBits(0, log2Cols + log2Rows); // context_update_tile_id
            bw.WriteBits(3 - 1, 2);                // tile_size_bytes_minus_1 = 3 (== 4 bytes)
        }
    }

    internal static int TileLog2(int blkSize, int target)
    {
        int k = 0;
        while ((blkSize << k) < target) k++;
        return k;
    }
}
