// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 keyframe framing: the Temporal Delimiter + Sequence Header OBUs and the uncompressed frame header for a
// (width, height, q) keyframe. Shared by the GPU encoder (Av1KeyframeEncoderGpu uploads BuildFramingBlob's bytes) and
// the CPU reference encoder in SpawnDev.Codecs.References. Same settings as the encoder's v1 bitstream: profile 0,
// 8-bit 4:2:0, 64x64 superblocks, uniform tile spacing (the grid is an Av1TileLayout), loop filter / CDEF / LR /
// segmentation off.
//
// The level. The sequence header signals the level and tier the stream conforms to (Av1Level, AV1 spec Annex A.3).
// Picture size, frame rate and tile grid decide which levels are possible at all; among those, the bytes of the coded
// frame decide (frame size limit, bit rate limit), and those are known only after the tiles are coded. So the framing
// holds one sequence header per possible level / tier (BuildLevelCandidates, in order of preference, each with the
// largest Frame OBU it allows) and the frame gets the first one it fits: Av1ObuAssembleKernel picks it on the
// accelerator, the CPU reference with SelectLevelCandidate. The sequence headers differ only in seq_level_idx and
// seq_tier (and by one bit of length: the tier is coded from level 4.0 up).
//
using SpawnDev.Codecs.EntropyCoders;

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>AV1 keyframe framing bytes (TD + SH OBUs, uncompressed header) for the v1 keyframe bitstream.</summary>
internal static class Av1KeyframeFraming
{
    /// <summary>
    /// The fixed part of a keyframe temporal unit for a <paramref name="width"/> x <paramref name="height"/> picture
    /// coded with the tile grid <paramref name="tiles"/> and shown at <paramref name="frameRateNum"/> /
    /// <paramref name="frameRateDen"/> frames per second: for every level the stream can signal, the Temporal
    /// Delimiter and Sequence Header OBUs, and the uncompressed frame header (<paramref name="bLen"/> bytes at
    /// <paramref name="bOffset"/>). Small, depends only on its arguments, and is uploaded to the accelerator where
    /// <see cref="Av1ObuAssembleKernel"/> picks the level the coded frame fits and adds the data-dependent Frame OBU
    /// size and the GPU-produced tile bytes.
    /// </summary>
    /// <remarks>
    /// Blob layout (<see cref="Av1ObuAssembleKernel"/> reads it): the level table, then the candidates' prefixes,
    /// then the uncompressed header at <paramref name="bOffset"/>. The table is little-endian 32-bit integers: the
    /// candidate count, then per candidate, in order of preference, the largest Frame OBU length it allows, the
    /// offset of its prefix (temporal delimiter and sequence header OBUs) in the blob and the prefix's length. The
    /// last candidate (maximum parameters) allows any length.
    /// </remarks>
    internal static byte[] BuildFramingBlob(
        int width, int height, int baseQIndex, Av1TileLayout tiles, int frameRateNum, int frameRateDen, out int bOffset, out int bLen)
    {
        if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width));
        if (height <= 0) throw new ArgumentOutOfRangeException(nameof(height));
        var candidates = BuildLevelCandidates(width, height, tiles, frameRateNum, frameRateDen);
        var sh = SynthesizeSequenceHeader(BuildSequenceHeaderConfig(width, height, candidates[0].Signal));
        byte[] uncompressedHeader = BuildUncompressedHeader(sh, width, height, baseQIndex, tiles);

        int tableLen = LevelTableHeaderBytes + candidates.Count * LevelTableEntryBytes;
        bOffset = tableLen + candidates.Sum(c => c.Prefix.Length);
        bLen = uncompressedHeader.Length;
        var blob = new byte[bOffset + bLen];
        WriteInt32(blob, 0, candidates.Count);
        int at = tableLen;
        for (int c = 0; c < candidates.Count; c++)
        {
            int entry = LevelTableHeaderBytes + c * LevelTableEntryBytes;
            WriteInt32(blob, entry, (int)Math.Min(candidates[c].MaxFrameObuBytes, int.MaxValue));
            WriteInt32(blob, entry + 4, at);
            WriteInt32(blob, entry + 8, candidates[c].Prefix.Length);
            Buffer.BlockCopy(candidates[c].Prefix, 0, blob, at, candidates[c].Prefix.Length);
            at += candidates[c].Prefix.Length;
        }
        Buffer.BlockCopy(uncompressedHeader, 0, blob, bOffset, bLen);
        return blob;
    }

    /// <summary>Bytes of the candidate count at the start of the framing blob.</summary>
    internal const int LevelTableHeaderBytes = 4;
    /// <summary>Bytes of one candidate's entry in the framing blob's level table (limit, prefix offset, prefix length).</summary>
    internal const int LevelTableEntryBytes = 12;

    private static void WriteInt32(byte[] buffer, int at, int value)
    {
        buffer[at] = (byte)value;
        buffer[at + 1] = (byte)(value >> 8);
        buffer[at + 2] = (byte)(value >> 16);
        buffer[at + 3] = (byte)(value >> 24);
    }

    // ------------------------------------------------------------------
    // Level candidates
    // ------------------------------------------------------------------

    /// <summary>
    /// One level / tier a keyframe stream can signal: its temporal unit prefix (Temporal Delimiter OBU, then the
    /// Sequence Header OBU signalling <paramref name="Signal"/>) and the largest Frame OBU, in bytes, with which the
    /// stream still conforms to it.
    /// </summary>
    internal readonly record struct LevelCandidate(Av1LevelSignal Signal, byte[] Prefix, long MaxFrameObuBytes);

    /// <summary>
    /// The levels and tiers a stream of <paramref name="width"/> x <paramref name="height"/> keyframes with the tile
    /// grid <paramref name="tiles"/>, shown at <paramref name="frameRateNum"/> / <paramref name="frameRateDen"/>
    /// frames per second, can signal (<see cref="Av1Level.Signals"/>: in order of preference, the last one maximum
    /// parameters), each with its prefix bytes and its frame size limit.
    /// </summary>
    internal static List<LevelCandidate> BuildLevelCandidates(int width, int height, Av1TileLayout tiles, int frameRateNum, int frameRateDen)
    {
        var stream = new Av1LevelStream
        {
            SeqProfile = 0, Width = width, Height = height, Tiles = tiles, FrameRateNum = frameRateNum, FrameRateDen = frameRateDen,
        };
        byte[] tdObu = Av1ObuWriter.EmitObu(
            Av1ObuType.TemporalDelimiter, ReadOnlySpan<byte>.Empty, hasSizeField: true);
        var list = new List<LevelCandidate>();
        foreach (var signal in Av1Level.Signals(stream))
        {
            byte[] shPayload = Av1SequenceHeaderWriter.EmitPayload(BuildSequenceHeaderConfig(width, height, signal));
            byte[] shObu = Av1ObuWriter.EmitObu(Av1ObuType.SequenceHeader, shPayload, hasSizeField: true);
            var prefix = new byte[tdObu.Length + shObu.Length];
            Buffer.BlockCopy(tdObu, 0, prefix, 0, tdObu.Length);
            Buffer.BlockCopy(shObu, 0, prefix, tdObu.Length, shObu.Length);
            list.Add(new LevelCandidate(signal, prefix, Av1Level.MaxFrameObuBytes(signal, stream, prefix.Length)));
        }
        return list;
    }

    /// <summary>The first of <paramref name="candidates"/> that allows a Frame OBU of <paramref name="frameObuBytes"/> bytes (the last allows any).</summary>
    internal static LevelCandidate SelectLevelCandidate(List<LevelCandidate> candidates, long frameObuBytes)
    {
        foreach (var c in candidates)
            if (c.MaxFrameObuBytes >= frameObuBytes) return c;
        return candidates[^1];
    }

    // ------------------------------------------------------------------
    // Sequence header
    // ------------------------------------------------------------------

    internal static Av1SequenceHeaderConfig BuildSequenceHeaderConfig(int width, int height, Av1LevelSignal level)
    {
        return new Av1SequenceHeaderConfig
        {
            SeqProfile = 0,
            SeqLevelIdx0 = level.SeqLevelIdx,
            SeqTier0 = level.SeqTier,
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
            SeqTier0 = cfg.SeqTier0,
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
