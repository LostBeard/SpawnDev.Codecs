// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 keyframe encoder. Takes a YUV420 source frame and emits a
// complete VP9 keyframe bitstream that matches what
// <see cref="Vp9KeyframeWalker"/> (and ffmpeg / libvpx) decode back
// to the original pixels modulo quantization loss.
//
// v1 simplifications (mirrors Vp8KeyframeEncoder defaults):
//   - Profile 0 (8-bit YUV 4:2:0)
//   - Width and height multiples of 16
//   - Every Block16x16 uses Y intra mode = DC_PRED, transform =
//     Tx16x16 + DctDct
//   - Every chroma block (Block8x8 per 4:2:0 chroma rule) uses
//     UV intra mode = DC_PRED, transform = Tx8x8 + DctDct
//   - tx_mode = Allow32x32 (so chroma tx clamps to Tx8x8 and luma
//     uses Tx16x16, no per-block tx_size signalling)
//   - Single tile (Log2NumTiles = 0, Log2TileRows = 0)
//   - Loop filter disabled (filter_level = 0)
//   - Segmentation disabled
//   - Default coefficient probabilities (no diff_update_prob bits)
//   - Default skip probabilities (no diff_update_prob bits)
//
// More sophisticated mode selection / RD-optimized quantization /
// loop filtering layers on top of this; the produced bitstream is
// already a fully-valid VP9 keyframe.
//
// libvpx reference for the bitstream layout: vp9/encoder/vp9_bitstream.c
// (write_uncompressed_header, write_compressed_header, write_tile, etc).

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 keyframe encoder (DC-prediction-only, single tile, no LF).</summary>
public static class Vp9KeyframeEncoder
{
    /// <summary>
    /// Encode a single VP9 keyframe from YUV420 source.
    /// </summary>
    /// <param name="ySrc">Y plane bytes, length = ySrcStride * height.</param>
    /// <param name="ySrcStride">Y plane row stride in bytes (>= width).</param>
    /// <param name="uSrc">U plane bytes (length >= uvSrcStride * height/2).</param>
    /// <param name="uvSrcStride">UV plane row stride in bytes (>= width/2).</param>
    /// <param name="vSrc">V plane bytes (length >= uvSrcStride * height/2).</param>
    /// <param name="width">Frame width in pixels (multiple of 16 for v1).</param>
    /// <param name="height">Frame height in pixels (multiple of 16 for v1).</param>
    /// <param name="baseQIndex">Base quantizer index 1..255 (lower = higher quality). 0 forces lossless mode which v1 does not support.</param>
    /// <returns>Complete VP9 frame bytes ready to wrap in IVF / WebM.</returns>
    public static byte[] EncodeKeyFrame(
        ReadOnlySpan<byte> ySrc, int ySrcStride,
        ReadOnlySpan<byte> uSrc, int uvSrcStride,
        ReadOnlySpan<byte> vSrc,
        int width, int height,
        int baseQIndex = 30)
    {
        if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width));
        if (height <= 0) throw new ArgumentOutOfRangeException(nameof(height));
        if ((width & 15) != 0 || (height & 15) != 0)
            throw new ArgumentException("Width and height must be multiples of 16 (v1)");
        if (baseQIndex <= 0 || baseQIndex > 255)
            throw new ArgumentOutOfRangeException(nameof(baseQIndex),
                "baseQIndex must be in [1, 255] (lossless v1 not supported).");

        int miCols = width >> 3;   // = (width + 7) >> 3 since width is multiple of 16
        int miRows = height >> 3;

        // ---- 1. Reconstruction buffer (predictor + residual go here) ----
        var subsampling = Vp9SubsamplingPair.Yuv420;
        var recon = new Vp9FrameBuffer(width, height, subsampling);

        // Quantizer pairs. y_dc_delta / uv_dc_delta / uv_ac_delta = 0.
        var yQuant = Vp9Dequantizer.PlaneQuantizer(baseQIndex, 0, 0);
        var uvQuant = Vp9Dequantizer.PlaneQuantizer(baseQIndex, 0, 0);

        // ---- 2. Build the compressed header bool stream ----
        // tx_mode = Allow32x32 (value 3, two raw bits "11"). This frame
        // is NOT TxModeSelect, so no tx_mode_probs follow.
        // Then: per-tx-size coef probability updates - we emit the "no
        // update" bit for each layer (256 layers per tx_size = 1024
        // bits total of "no update").
        // Then: skip_probs - 3 "no update" bits.
        var compressedEnc = new Vp9BoolEncoder();

        // tx_mode: WriteLiteral(2 bits) = 3 (Allow32x32). The decoder
        // reads ReadLiteral(2) which calls Read(0x80) twice MSB-first.
        compressedEnc.WriteLiteral((int)Vp9TxMode.Allow32x32, 2);

        // coef_probs updates - skip every layer.
        EmitNoCoefProbUpdates(compressedEnc, Vp9TxMode.Allow32x32);

        // skip_probs - 3 "no update" diff_update_prob bits.
        for (int k = 0; k < Vp9SkipProbs.SkipContexts; k++)
            EmitNoDiffUpdate(compressedEnc);

        byte[] compressedBytes = compressedEnc.Stop();

        // ---- 3. Build the tile data bool stream ----
        // Single tile -> the entire frame is one bool-coded payload.
        // Walk SBs, emit partition + mode info + coefficients.
        var tileEnc = new Vp9BoolEncoder();
        EncodeTile(
            tileEnc,
            ySrc, ySrcStride, uSrc, vSrc, uvSrcStride,
            width, height, miCols, miRows,
            yQuant, uvQuant,
            recon);
        byte[] tileBytes = tileEnc.Stop();

        // ---- 4. Build the uncompressed header ----
        byte[] uncompressedHeader = BuildUncompressedHeader(
            width, height, baseQIndex, firstPartitionSize: compressedBytes.Length);

        // ---- 5. Concatenate: uncompressed header + compressed header + tile bytes ----
        // Single tile so no per-tile size prefix - the last (and only)
        // tile spans to end-of-frame.
        var output = new byte[uncompressedHeader.Length + compressedBytes.Length + tileBytes.Length];
        Buffer.BlockCopy(uncompressedHeader, 0, output, 0, uncompressedHeader.Length);
        Buffer.BlockCopy(compressedBytes, 0, output,
            uncompressedHeader.Length, compressedBytes.Length);
        Buffer.BlockCopy(tileBytes, 0, output,
            uncompressedHeader.Length + compressedBytes.Length, tileBytes.Length);
        return output;
    }

    // ------------------------------------------------------------------
    // Uncompressed header
    // ------------------------------------------------------------------

    private static byte[] BuildUncompressedHeader(
        int width, int height, int baseQIndex, int firstPartitionSize)
    {
        var bw = new Vp9BitWriter();

        // frame_marker f(2) = 0b10
        bw.WriteBits(0b10u, 2);

        // profile = 0 -> two bits (low, high) = (0, 0)
        bw.WriteBits(0u, 1);
        bw.WriteBits(0u, 1);

        // show_existing_frame = 0
        bw.WriteBits(0u, 1);

        // frame_type = KEY_FRAME = 0
        bw.WriteBits(0u, 1);

        // show_frame = 1
        bw.WriteBits(1u, 1);

        // error_resilient_mode = 0
        bw.WriteBits(0u, 1);

        // sync_code 0x49 0x83 0x42
        bw.WriteBits(Vp9SyncCode.Byte0, 8);
        bw.WriteBits(Vp9SyncCode.Byte1, 8);
        bw.WriteBits(Vp9SyncCode.Byte2, 8);

        // color_config (profile 0): color_space(3) + (cs!=Srgb ? color_range(1))
        bw.WriteBits((uint)Vp9ColorSpace.Bt601 & 0x7u, 3);
        bw.WriteBits(0u, 1); // color_range = 0 (studio range)

        // frame_width_minus_1 f(16), frame_height_minus_1 f(16)
        bw.WriteBits((uint)(width - 1) & 0xFFFFu, 16);
        bw.WriteBits((uint)(height - 1) & 0xFFFFu, 16);

        // render_and_frame_size_different = 0
        bw.WriteBits(0u, 1);

        // refresh_frame_context = 0 (no need to keep contexts)
        bw.WriteBits(0u, 1);
        // frame_parallel_decoding_mode = 0
        bw.WriteBits(0u, 1);
        // frame_context_idx f(2) = 0
        bw.WriteBits(0u, 2);

        // loop_filter_params: filter_level(6) + sharpness_level(3) +
        // mode_ref_delta_enabled(1)
        bw.WriteBits(0u, 6); // filter_level = 0 (LF disabled)
        bw.WriteBits(0u, 3); // sharpness_level = 0
        bw.WriteBits(0u, 1); // mode_ref_delta_enabled = 0

        // quantization_params: base_q_idx(8) +
        // y_dc_delta f(1)=0 + uv_dc_delta f(1)=0 + uv_ac_delta f(1)=0
        bw.WriteBits((uint)(baseQIndex & 0xFF), 8);
        bw.WriteBits(0u, 1); // y_dc_delta_present
        bw.WriteBits(0u, 1); // uv_dc_delta_present
        bw.WriteBits(0u, 1); // uv_ac_delta_present

        // segmentation_params: enabled f(1) = 0
        bw.WriteBits(0u, 1);

        // tile_info: tile_cols_log2 = MIN, tile_rows_log2 = 0.
        // For width <= 4096 (sb_cols <= 64) MIN=0, MAX depends on
        // frame width. We loop emitting "0" increment bits == MIN, so
        // tile_cols_log2 = MIN.
        int miCols = (width + 7) >> 3;
        var (minLog2Cols, maxLog2Cols) = Vp9TileInfoParser.GetTileNBits(miCols);
        // We want tile_cols_log2 = minLog2Cols. Emit (max - min)
        // increment bits, all 0, to keep at min.
        // Actually the loop reads bits while reader.ReadFlag() returns
        // true; emitting 0 stops the loop early.
        // Simplest: emit one "0" bit for the first attempted increment
        // (decoder stops at first 0). But we must emit (max - min) bits
        // only if the decoder reads that many - actually it stops on
        // the first 0 bit OR when it has read max-min bits.
        // To stay at min: emit a single 0 bit (decoder reads one bit,
        // sees 0, breaks).
        // BUT: if maxLog2Cols == minLog2Cols there are no increment bits.
        if (maxLog2Cols > minLog2Cols)
        {
            bw.WriteBits(0u, 1); // first increment bit = 0 -> stays at min
        }
        // tile_rows_log2 = 0: write a single 0 bit.
        bw.WriteBits(0u, 1);

        // first_partition_size f(16): byte length of compressed header.
        bw.WriteBits((uint)(firstPartitionSize & 0xFFFF), 16);

        // Byte-align so the compressed header (which is bool-coded)
        // starts on a byte boundary.
        bw.PadToByte();

        return bw.ToBytes();
    }

    // ------------------------------------------------------------------
    // Compressed header
    // ------------------------------------------------------------------

    /// <summary>
    /// Emit the "no update" bit for every entry of every coef_probs
    /// table the decoder will read for <paramref name="txMode"/>.
    /// libvpx <c>read_coef_probs_common</c> reads a single LITERAL bit
    /// (probability 128) per tx_size as the update gate; 0 means skip
    /// the whole tx_size, 1 means iterate every (plane, ref, band,
    /// ctx, node) and apply diff_update_prob.
    ///
    /// The gate is a LITERAL bit (prob 128), NOT a diff_update_prob
    /// bit (prob 252). Inside the gate, individual entries use
    /// diff_update_prob. We emit gate=0 at every tx_size to skip the
    /// inner loop entirely.
    /// </summary>
    private static void EmitNoCoefProbUpdates(Vp9BoolEncoder enc, Vp9TxMode txMode)
    {
        Vp9TxSize biggest = txMode switch
        {
            Vp9TxMode.Only4x4 => Vp9TxSize.Tx4x4,
            Vp9TxMode.AllowOnly8x8 => Vp9TxSize.Tx8x8,
            Vp9TxMode.AllowOnly16x16 => Vp9TxSize.Tx16x16,
            Vp9TxMode.Allow32x32 => Vp9TxSize.Tx32x32,
            Vp9TxMode.TxModeSelect => Vp9TxSize.Tx32x32,
            _ => throw new ArgumentOutOfRangeException(nameof(txMode)),
        };
        int max = (int)biggest;
        for (int t = 0; t <= max; t++)
        {
            // update_coef_probs gate: emit 0 (no update for this tx_size).
            // Probability 128 to match Vp9BoolDecoder.ReadBit() == Read(128).
            enc.Write(0, 128);
        }
    }

    /// <summary>
    /// Emit a single "no update" diff_update_prob bit. Mirrors
    /// <see cref="Vp9DiffUpdateProb.Read"/>'s first bit (the gate).
    /// </summary>
    private static void EmitNoDiffUpdate(Vp9BoolEncoder enc)
    {
        enc.Write(0, Vp9DiffUpdateProb.UpdateProb);
    }

    // ------------------------------------------------------------------
    // Tile data: SB walk
    // ------------------------------------------------------------------

    private static void EncodeTile(
        Vp9BoolEncoder enc,
        ReadOnlySpan<byte> ySrc, int ySrcStride,
        ReadOnlySpan<byte> uSrc, ReadOnlySpan<byte> vSrc, int uvSrcStride,
        int width, int height,
        int miCols, int miRows,
        Vp9PlaneQuantizer yQuant, Vp9PlaneQuantizer uvQuant,
        Vp9FrameBuffer recon)
    {
        // Span-bearing context can't be a class member (Span is
        // ref-struct). Use a method-local helper that takes pointers
        // via fixed-pointer reads from the span.
        // Workaround: we don't actually need to flow ySrc through
        // partition recursion as-spans; we can pass the whole Vp9
        // frame planes inline in the method since this tile-walk is
        // self-contained.

        int miColsAligned = (miCols + 7) & ~7;
        var aboveYMode = new Vp9IntraMode[miColsAligned * 2];
        Array.Fill(aboveYMode, Vp9IntraMode.DcPred);
        var aboveSkip = new byte[miColsAligned];
        var abovePartCtx = new byte[miColsAligned];
        var aboveTxSize = new byte[miColsAligned];

        // Per-tile-row left contexts: 8 mi tall (= 16 4x4 cells).
        var leftYMode = new Vp9IntraMode[16];
        var leftSkip = new byte[8];
        var leftPartCtx = new byte[8];
        var leftTxSize = new byte[8];

        for (int miRow = 0; miRow < miRows; miRow += 8)
        {
            Array.Fill(leftYMode, Vp9IntraMode.DcPred);
            Array.Clear(leftSkip);
            Array.Clear(leftPartCtx);
            Array.Clear(leftTxSize);

            for (int miCol = 0; miCol < miCols; miCol += 8)
            {
                EncodePartition(
                    enc,
                    ySrc, ySrcStride, uSrc, vSrc, uvSrcStride,
                    miCols, miRows, miColsAligned,
                    yQuant, uvQuant, recon,
                    aboveYMode, aboveSkip, abovePartCtx, aboveTxSize,
                    leftYMode, leftSkip, leftPartCtx, leftTxSize,
                    miRow, miCol, Vp9BlockSize.Block64x64);
            }
        }
    }

    /// <summary>
    /// Recursively walk the partition tree at (miRow, miCol) for
    /// <paramref name="bsize"/>. Mirrors <see cref="Vp9KeyframeWalker"/>'s
    /// DecodePartition exactly so the bit-reads it makes line up with
    /// the bit-writes here.
    /// </summary>
    private static void EncodePartition(
        Vp9BoolEncoder enc,
        ReadOnlySpan<byte> ySrc, int ySrcStride,
        ReadOnlySpan<byte> uSrc, ReadOnlySpan<byte> vSrc, int uvSrcStride,
        int miCols, int miRows, int miColsAligned,
        Vp9PlaneQuantizer yQuant, Vp9PlaneQuantizer uvQuant,
        Vp9FrameBuffer recon,
        Vp9IntraMode[] aboveYMode, byte[] aboveSkip, byte[] abovePartCtx, byte[] aboveTxSize,
        Vp9IntraMode[] leftYMode, byte[] leftSkip, byte[] leftPartCtx, byte[] leftTxSize,
        int miRow, int miCol, Vp9BlockSize bsize)
    {
        if (miRow >= miRows || miCol >= miCols) return;

        int bsl = Vp9BlockSizes.MiWidthLog2[(int)bsize];
        int hbs = (bsl > 0) ? (1 << (bsl - 1)) : 0;
        bool hasRows = (miRow + hbs) < miRows;
        bool hasCols = (miCol + hbs) < miCols;

        // For our v1 scheme, we want to land on Block16x16 with
        // PARTITION_NONE everywhere. That means:
        //   Block64x64 -> SPLIT  (or forced split if frame < 64)
        //   Block32x32 -> SPLIT  (or forced split if frame < 32)
        //   Block16x16 -> NONE
        Vp9PartitionType partition;
        if (bsize >= Vp9BlockSize.Block8x8)
        {
            int leftIdx = (miRow & 7);
            int aboveIdx = miCol;
            int leftBit = (leftPartCtx[leftIdx] >> bsl) & 1;
            int aboveBit = (abovePartCtx[aboveIdx] >> bsl) & 1;
            int splitState = leftBit * 2 + aboveBit;
            int sizeIdx = bsl;
            ReadOnlySpan<byte> probs = Vp9PartitionProbs.KeyframeProbs(sizeIdx, splitState);

            if (hasRows && hasCols)
            {
                // Pick partition based on bsize.
                partition = bsize switch
                {
                    Vp9BlockSize.Block16x16 => Vp9PartitionType.None,
                    Vp9BlockSize.Block32x32 => Vp9PartitionType.Split,
                    Vp9BlockSize.Block64x64 => Vp9PartitionType.Split,
                    _ => throw new InvalidOperationException(
                        $"v1 encoder reached unexpected bsize {bsize}"),
                };
                EncodePartitionDecision(enc, partition, probs);
            }
            else if (!hasRows && hasCols)
            {
                // Forced: write a single bit choosing Split (1) vs Horz (0).
                // We always pick Split.
                enc.Write(1, probs[1]);
                partition = Vp9PartitionType.Split;
            }
            else if (hasRows && !hasCols)
            {
                // Forced: write a single bit choosing Split (1) vs Vert (0).
                enc.Write(1, probs[2]);
                partition = Vp9PartitionType.Split;
            }
            else
            {
                // Both sides off-frame: forced Split, no bit.
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
                EncodeLeafBlock(enc,
                    ySrc, ySrcStride, uSrc, vSrc, uvSrcStride,
                    miCols, miRows,
                    yQuant, uvQuant, recon,
                    aboveYMode, aboveSkip, aboveTxSize,
                    leftYMode, leftSkip, leftTxSize,
                    miRow, miCol, subsize);
                break;
            case Vp9PartitionType.Split:
                if (bsize == Vp9BlockSize.Block8x8)
                {
                    // 8x8+Split is a single leaf block (4 sub-4x4 cells).
                    // v1 encoder never reaches this since min bsize = 16x16.
                    throw new NotSupportedException("v1 encoder doesn't emit Block8x8+Split");
                }
                EncodePartition(enc, ySrc, ySrcStride, uSrc, vSrc, uvSrcStride,
                    miCols, miRows, miColsAligned, yQuant, uvQuant, recon,
                    aboveYMode, aboveSkip, abovePartCtx, aboveTxSize,
                    leftYMode, leftSkip, leftPartCtx, leftTxSize,
                    miRow, miCol, subsize);
                EncodePartition(enc, ySrc, ySrcStride, uSrc, vSrc, uvSrcStride,
                    miCols, miRows, miColsAligned, yQuant, uvQuant, recon,
                    aboveYMode, aboveSkip, abovePartCtx, aboveTxSize,
                    leftYMode, leftSkip, leftPartCtx, leftTxSize,
                    miRow, miCol + hbs, subsize);
                EncodePartition(enc, ySrc, ySrcStride, uSrc, vSrc, uvSrcStride,
                    miCols, miRows, miColsAligned, yQuant, uvQuant, recon,
                    aboveYMode, aboveSkip, abovePartCtx, aboveTxSize,
                    leftYMode, leftSkip, leftPartCtx, leftTxSize,
                    miRow + hbs, miCol, subsize);
                EncodePartition(enc, ySrc, ySrcStride, uSrc, vSrc, uvSrcStride,
                    miCols, miRows, miColsAligned, yQuant, uvQuant, recon,
                    aboveYMode, aboveSkip, abovePartCtx, aboveTxSize,
                    leftYMode, leftSkip, leftPartCtx, leftTxSize,
                    miRow + hbs, miCol + hbs, subsize);
                break;
            case Vp9PartitionType.Horz:
            case Vp9PartitionType.Vert:
                throw new NotSupportedException(
                    "v1 encoder doesn't emit Horz/Vert partitions");
        }

        // libvpx update_partition_context: only writes when the
        // partition didn't recurse (NONE) or the bsize is exactly 8x8.
        // For our recursion we hit NONE at Block16x16 (subsize=Block16x16).
        if (bsize >= Vp9BlockSize.Block8x8 &&
            (bsize == Vp9BlockSize.Block8x8 || partition != Vp9PartitionType.Split))
        {
            UpdatePartitionContext(abovePartCtx, leftPartCtx, miColsAligned,
                miRow, miCol, bsize, subsize);
        }
    }

    private static void EncodePartitionDecision(
        Vp9BoolEncoder enc, Vp9PartitionType partition, ReadOnlySpan<byte> probs)
    {
        // Walk the partition tree forward. Tree:
        //   i=0 ROOT     -> -None,  2
        //   i=2 H_OR_V_OR_S -> -Horz, 4
        //   i=4 V_OR_S   -> -Vert,  -Split
        // Decoder reads probs[i >> 1] at each level.
        // For NONE: bit 0 with probs[0]
        // For HORZ: bit 1 with probs[0], bit 0 with probs[1]
        // For VERT: bit 1, bit 1, bit 0 (probs[0..2])
        // For SPLIT: bit 1, bit 1, bit 1
        switch (partition)
        {
            case Vp9PartitionType.None:
                enc.Write(0, probs[0]);
                break;
            case Vp9PartitionType.Horz:
                enc.Write(1, probs[0]);
                enc.Write(0, probs[1]);
                break;
            case Vp9PartitionType.Vert:
                enc.Write(1, probs[0]);
                enc.Write(1, probs[1]);
                enc.Write(0, probs[2]);
                break;
            case Vp9PartitionType.Split:
                enc.Write(1, probs[0]);
                enc.Write(1, probs[1]);
                enc.Write(1, probs[2]);
                break;
        }
    }

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

    private static void UpdatePartitionContext(
        byte[] abovePartCtx, byte[] leftPartCtx, int miColsAligned,
        int miRow, int miCol, Vp9BlockSize bsize, Vp9BlockSize subsize)
    {
        int bsl = Vp9BlockSizes.MiWidthLog2[(int)bsize];
        int bs = 1 << bsl;
        int subIdx = (int)subsize < (int)Vp9BlockSize.Invalid ? (int)subsize : (int)bsize;
        var (above, left) = PartitionContextLookup[subIdx];
        for (int i = 0; i < bs; i++)
        {
            int c = miCol + i;
            int r = (miRow + i) & 7;
            if (c < miColsAligned)
                abovePartCtx[c] = above;
            if (r < leftPartCtx.Length)
                leftPartCtx[r] = left;
        }
    }

    // ------------------------------------------------------------------
    // Leaf block encode: mode info + coefficients + recon writeback
    // ------------------------------------------------------------------

    private static void EncodeLeafBlock(
        Vp9BoolEncoder enc,
        ReadOnlySpan<byte> ySrc, int ySrcStride,
        ReadOnlySpan<byte> uSrc, ReadOnlySpan<byte> vSrc, int uvSrcStride,
        int miCols, int miRows,
        Vp9PlaneQuantizer yQuant, Vp9PlaneQuantizer uvQuant,
        Vp9FrameBuffer recon,
        Vp9IntraMode[] aboveYMode, byte[] aboveSkip, byte[] aboveTxSize,
        Vp9IntraMode[] leftYMode, byte[] leftSkip, byte[] leftTxSize,
        int miRow, int miCol, Vp9BlockSize bsize)
    {
        if (miRow >= miRows || miCol >= miCols) return;

        // v1 expects bsize = Block16x16.
        if (bsize != Vp9BlockSize.Block16x16)
            throw new NotSupportedException(
                $"v1 encoder only emits Block16x16 leaves, got {bsize}");

        // ---- Mode-info bits ----
        // 1. segment_id: segmentation disabled, no read/write.
        const int segmentId = 0;
        _ = segmentId;

        // 2. skip flag: pick 0 (we have residual). skip_context = above + left.
        int leftIdxMi = miRow & 7;
        int leftSkipBit = leftSkip[leftIdxMi];
        int aboveSkipBit = miCol < aboveSkip.Length ? aboveSkip[miCol] : 0;
        int skipContext = aboveSkipBit + leftSkipBit;
        const int skipFlag = 0;
        // skip_probs[skipContext] from defaults (no compressed updates).
        enc.Write(skipFlag, Vp9SkipProbs.DefaultProbs[skipContext]);

        // 3. tx_size: tx_mode = Allow32x32 -> NOT Select, no write.
        // The decoder forces tx_size = MIN(maxTxForBlock, biggest) =
        // MIN(Tx16x16, Tx32x32) = Tx16x16.
        var txSize = Vp9TxSize.Tx16x16;

        // 4. y_mode: pick DC_PRED.
        int b4Col = miCol * 2;
        int leftB4Idx = (miRow & 7) * 2;
        var aboveYCell = b4Col < aboveYMode.Length ? aboveYMode[b4Col] : Vp9IntraMode.DcPred;
        var leftYCell = leftYMode[leftB4Idx];
        ReadOnlySpan<byte> yProbs = Vp9IntraModeProbs.KeyframeYProbs(aboveYCell, leftYCell);
        EncodeIntraMode(enc, Vp9IntraMode.DcPred, yProbs);

        // 5. uv_mode: pick DC_PRED.
        ReadOnlySpan<byte> uvProbs = Vp9IntraModeProbs.KeyframeUvProbs(Vp9IntraMode.DcPred);
        EncodeIntraMode(enc, Vp9IntraMode.DcPred, uvProbs);

        // ---- Update mode-info contexts so the next block's reads
        //      mirror what the walker computes. ----
        int b4Wide = Vp9BlockSizes.B4x4Width(bsize);
        int b4High = Vp9BlockSizes.B4x4Height(bsize);
        for (int i = 0; i < b4Wide; i++)
        {
            int c = b4Col + i;
            if (c < aboveYMode.Length) aboveYMode[c] = Vp9IntraMode.DcPred;
        }
        for (int i = 0; i < b4High; i++)
        {
            int r = (leftB4Idx + i) & 15;
            leftYMode[r] = Vp9IntraMode.DcPred;
        }
        int miWide = Vp9BlockSizes.MiWidth(bsize);
        int miHigh = Vp9BlockSizes.MiHeight(bsize);
        for (int i = 0; i < miWide; i++)
        {
            int c = miCol + i;
            if (c < aboveSkip.Length) aboveSkip[c] = (byte)skipFlag;
            if (c < aboveTxSize.Length) aboveTxSize[c] = (byte)txSize;
        }
        for (int i = 0; i < miHigh; i++)
        {
            int r = (leftIdxMi + i) & 7;
            leftSkip[r] = (byte)skipFlag;
            leftTxSize[r] = (byte)txSize;
        }

        // ---- Encode pixels: predict, transform, quantize, emit coefs,
        //      then reconstruct (predict + dequant + iDCT) into recon. ----
        EncodePlanePixels(enc, plane: 0,
            ySrc, ySrcStride, recon, yQuant, miRow, miCol, bsize, txSize,
            yMode: Vp9IntraMode.DcPred);
        EncodePlanePixels(enc, plane: 1,
            uSrc, uvSrcStride, recon, uvQuant, miRow, miCol, bsize, txSize,
            yMode: Vp9IntraMode.DcPred);
        EncodePlanePixels(enc, plane: 2,
            vSrc, uvSrcStride, recon, uvQuant, miRow, miCol, bsize, txSize,
            yMode: Vp9IntraMode.DcPred);
    }

    private static void EncodeIntraMode(
        Vp9BoolEncoder enc, Vp9IntraMode mode, ReadOnlySpan<byte> probs)
    {
        // Walk Vp9IntraModeTree forward, emitting one bit per internal
        // node visited. Tree[i + bit] <= 0 means "leaf with mode -value";
        // > 0 means "next internal node index".
        sbyte[] tree = Vp9IntraModeTree.Tree;
        sbyte target = (sbyte)(-(int)mode);
        int i = 0;
        while (true)
        {
            int leftIndex = i;
            int bit = SubtreeContains(tree, leftIndex, target) ? 0 : 1;
            enc.Write(bit, probs[i >> 1]);
            sbyte next = tree[i + bit];
            if (next <= 0)
            {
                if (next != target)
                    throw new InvalidOperationException(
                        $"EncodeIntraMode landed on {-next} but expected {(int)mode}");
                return;
            }
            i = next;
        }
    }

    private static bool SubtreeContains(sbyte[] tree, int rootIndex, sbyte target)
    {
        sbyte v = tree[rootIndex];
        if (v <= 0) return v == target;
        return SubtreeContains(tree, v, target) || SubtreeContains(tree, v + 1, target);
    }

    // ------------------------------------------------------------------
    // Per-plane pixel encode
    // ------------------------------------------------------------------

    private static void EncodePlanePixels(
        Vp9BoolEncoder enc, int plane,
        ReadOnlySpan<byte> src, int srcStride,
        Vp9FrameBuffer recon, Vp9PlaneQuantizer planeQuant,
        int miRow, int miCol, Vp9BlockSize bsize, Vp9TxSize lumaTxSize,
        Vp9IntraMode yMode)
    {
        bool isUv = plane != 0;
        int ssX = isUv ? recon.Subsampling.SubsamplingX : 0;
        int ssY = isUv ? recon.Subsampling.SubsamplingY : 0;

        Vp9BlockSize planeBsize = isUv
            ? Vp9ChromaBlockSize.ForLumaBlock(bsize)
            : bsize;
        Vp9TxSize txSize = isUv
            ? Vp9ChromaBlockSize.GetChromaTxSize(lumaTxSize, bsize)
            : lumaTxSize;
        Vp9IntraMode mode = yMode;  // both Y and UV use DC_PRED in v1

        int blockWidthPx = Vp9BlockSizes.Width(planeBsize);
        int blockHeightPx = Vp9BlockSizes.Height(planeBsize);
        int txN = Vp9IntraBlockDecode.TxSizeToN(txSize);
        int txCols = Math.Max(1, blockWidthPx / txN);
        int txRows = Math.Max(1, blockHeightPx / txN);

        byte[] planeBuf = plane switch
        {
            0 => recon.Y,
            1 => recon.U,
            _ => recon.V,
        };
        int planeStride = plane == 0 ? recon.LumaWidth : recon.ChromaWidth;
        int planeHeight = plane == 0 ? recon.LumaHeight : recon.ChromaHeight;

        int blockX0 = (miCol << 3) >> ssX;
        int blockY0 = (miRow << 3) >> ssY;

        var planeType = isUv
            ? Vp9BlockCoefDecoder.PlaneType.Uv
            : Vp9BlockCoefDecoder.PlaneType.Y;
        var coefProbs = Vp9CoefProbs.DefaultCoefProbsFor(txSize);

        Span<byte> aboveBuf = stackalloc byte[64];
        Span<byte> leftBuf = stackalloc byte[32];
        Span<byte> dstLocal = stackalloc byte[32 * 32];
        Span<short> residual = stackalloc short[1024];
        Span<int> coefsInt = stackalloc int[1024];
        Span<short> coefsShort = stackalloc short[1024];

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

                var txType = (txSize == Vp9TxSize.Tx32x32 || isUv)
                    ? Vp9TxType.DctDct
                    : Vp9IntraTxType.ForMode(mode);
                var scanType = Vp9ScanTables.ScanTypeForTxType(txType);

                // Build above row (2*txN samples).
                if (hasAbove)
                {
                    int aboveCount = Math.Min(2 * txN, planeStride - xPx);
                    int aboveRowOff = (yPx - 1) * planeStride + xPx;
                    for (int i = 0; i < aboveCount; i++)
                        aboveBuf[i] = planeBuf[aboveRowOff + i];
                    for (int i = aboveCount; i < 2 * txN; i++)
                        aboveBuf[i] = (aboveCount > 0) ? aboveBuf[aboveCount - 1]
                            : Vp9IntraEdgeFill.AboveFill;
                }
                else
                {
                    for (int i = 0; i < 2 * txN; i++) aboveBuf[i] = Vp9IntraEdgeFill.AboveFill;
                }

                // Build left column (txN samples).
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

                // Predict.
                dstLocal[..(txN * txN)].Clear();
                Vp9IntraPredictor.Predict(
                    mode, topLeft, aboveBuf[..(2 * txN)], leftBuf[..txN],
                    dstLocal[..(txN * txN)], txN, txN,
                    haveAbove: hasAbove, haveLeft: hasLeft);

                // Compute residual = source - prediction.
                int srcRowOff0 = yPx * srcStride + xPx;
                for (int r = 0; r < txN; r++)
                {
                    int srcOff = srcRowOff0 + r * srcStride;
                    int predOff = r * txN;
                    for (int c = 0; c < txN; c++)
                        residual[r * txN + c] = (short)(src[srcOff + c] - dstLocal[predOff + c]);
                }

                // Forward transform into int[] (libvpx forward DCT
                // produces int range).
                Vp9ForwardTransform.Apply(txSize, txType, residual[..(txN * txN)],
                    rowStrideShorts: txN, coefsInt[..(txN * txN)]);

                // Quantize in scan-position order. The forward DCT
                // produces raster-laid coefs; the scan determines which
                // raster slot maps to scan slot 0 (DC). We assume DC
                // quantizer applies at raster[0] (which is also scan[0])
                // since every scan in libvpx puts raster 0 at scan 0.
                Vp9ForwardQuantizer.QuantizeBlock(
                    coefsInt[..(txN * txN)], planeQuant.Dc, planeQuant.Ac);

                // Cast to short for the encoder + dequantizer.
                int slots = txN * txN;
                for (int i = 0; i < slots; i++)
                {
                    int v = coefsInt[i];
                    if (v > short.MaxValue) v = short.MaxValue;
                    if (v < short.MinValue) v = short.MinValue;
                    coefsShort[i] = (short)v;
                }

                // Emit coefficient bool bits.
                var coefBlockArray = coefsShort[..slots].ToArray();
                Vp9BlockCoefEncoder.EncodeBlockCoefficients(
                    (prob, bit) => enc.Write(bit, prob),
                    txSize, scanType, planeType,
                    Vp9BlockCoefDecoder.RefType.Intra,
                    coefBlockArray,
                    isHighBitDepth: false,
                    coefProbs: coefProbs);

                // ---- Reconstruct so subsequent blocks see the same
                //      context the decoder will compute. ----
                Vp9Dequantizer.DequantizeInPlace(coefsShort[..slots], planeQuant);
                Vp9InverseTransform.Apply(
                    txType, txSize, coefsShort[..slots],
                    dstLocal[..slots], stride: txN);

                // Copy reconstructed block into the recon plane.
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
