// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 keyframe framing: the Temporal Delimiter + Sequence Header OBUs and the uncompressed frame header for a
// (width, height, q) keyframe. Shared by the GPU encoder (Av1KeyframeEncoderGpu uploads BuildFramingBlob's bytes) and
// the CPU reference encoder in SpawnDev.Codecs.References. Same settings as the encoder's v1 bitstream: profile 0,
// 8-bit 4:2:0, 64x64 superblocks, uniform tile spacing (the grid is an Av1TileLayout), loop filter / CDEF / LR /
// segmentation off.
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
    internal static byte[] BuildFramingBlob(int width, int height, int baseQIndex, out int aLen, out int bLen) =>
        BuildFramingBlob(width, height, baseQIndex, Av1TileLayout.SingleTile(width, height), out aLen, out bLen);

    /// <summary>The framing blob for a frame coded with the tile grid <paramref name="tiles"/>.</summary>
    internal static byte[] BuildFramingBlob(int width, int height, int baseQIndex, Av1TileLayout tiles, out int aLen, out int bLen)
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
        byte[] uncompressedHeader = BuildUncompressedHeader(sh, width, height, baseQIndex, tiles);

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
        Av1SequenceHeader sh, int width, int height, int baseQIndex) =>
        BuildUncompressedHeader(sh, width, height, baseQIndex, Av1TileLayout.SingleTile(width, height));

    /// <summary>The uncompressed frame header for a frame coded with the tile grid <paramref name="tiles"/>.</summary>
    internal static byte[] BuildUncompressedHeader(
        Av1SequenceHeader sh, int width, int height, int baseQIndex, Av1TileLayout tiles)
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
        EmitTileInfo(bw, tiles);

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
        //   tile_group_obu (tile_start_and_end_present_flag: one bit, 0, when the frame has more than one
        //                   tile, absent for a single tile; then byte_alignment(); then, per tile, a
        //                   TileSizeBytes-byte tile_size_minus_1 (not for the last tile) and the
        //                   entropy-coded tile bytes). The tile group is written by the caller.
        //
        // byte_alignment() is zero-pad to byte boundary; it does NOT emit a
        // trailing 1 bit (that's WriteTrailingBits, used by SH and standalone
        // FrameHeader OBUs but NOT by combined Frame OBU).
        bw.ByteAlign();
        return bw.ToArray();
    }

    /// <summary>Bytes of a <c>tile_size_minus_1</c> field (the header codes <c>tile_size_bytes_minus_1 = 3</c>).</summary>
    internal const int TileSizeBytes = 4;

    /// <summary>
    /// AV1 <c>tile_info()</c> with uniform tile spacing for the grid <paramref name="tiles"/>: the increment bits up
    /// to TileColsLog2 / TileRowsLog2 (a run of ones, then a zero unless the maximum is reached), and, when there is
    /// more than one tile column or row, context_update_tile_id (0: CDF updates are disabled anyway) and
    /// tile_size_bytes_minus_1.
    /// </summary>
    internal static void EmitTileInfo(Av1BitWriter bw, Av1TileLayout tiles)
    {
        bw.WriteFlag(true); // uniform_tile_spacing_flag

        for (int k = tiles.MinLog2TileCols; k < tiles.MaxLog2TileCols; k++)
        {
            bool increment = k < tiles.TileColsLog2;
            bw.WriteBits(increment ? 1 : 0, 1);   // increment_tile_cols_log2
            if (!increment) break;
        }
        for (int k = tiles.MinLog2TileRows; k < tiles.MaxLog2TileRows; k++)
        {
            bool increment = k < tiles.TileRowsLog2;
            bw.WriteBits(increment ? 1 : 0, 1);   // increment_tile_rows_log2
            if (!increment) break;
        }

        if (tiles.TileColsLog2 > 0 || tiles.TileRowsLog2 > 0)
        {
            bw.WriteBits(0, tiles.TileColsLog2 + tiles.TileRowsLog2); // context_update_tile_id
            bw.WriteBits(TileSizeBytes - 1, 2);                        // tile_size_bytes_minus_1
        }
    }

    internal static int TileLog2(int blkSize, int target) => Av1TileLayout.TileLog2(blkSize, target);
}
