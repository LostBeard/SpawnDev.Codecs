// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 keyframe per-block mode info reader. Mirrors the keyframe path
// of libvpx vp9/decoder/vp9_decodemv.c read_intra_frame_mode_info().
//
// libvpx order for a keyframe leaf block:
//   1. read_intra_segment_id     (only if seg.update_map is set;
//                                 for BBB seg is OFF so seg=0)
//   2. read_skip                 (skip_probs[skip_context])
//   3. read_tx_size              (only TxModeSelect + bsize >= 8x8)
//   4. read Y intra modes        (sub-8x8: one per 4x4 sub-block via
//                                 KfYProbs(above, left); 8x8+: one
//                                 mode for the whole block driven by
//                                 the above/left 4x4 corner modes)
//   5. read uv_mode              (KfUvProbs(yMode))
//
// This file packages each step as a static helper - the walker
// composes them.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// VP9 keyframe per-block mode-info reader. Decomposes the libvpx
/// <c>read_intra_frame_mode_info</c> flow into composable pieces so
/// the walker can drive each step explicitly.
/// </summary>
public static class Vp9KeyframeModeInfo
{
    /// <summary>
    /// libvpx <c>read_intra_segment_id</c>. For keyframes:
    ///   - if !segmentation.Enabled OR !UpdateMap -> segment_id = 0.
    ///   - else read a tree-coded segment id 0..7 via SegmentTree
    ///     using the per-frame tree probabilities.
    /// Keyframes never use temporal segmentation update.
    /// </summary>
    public static int ReadIntraSegmentId(
        Vp9SegmentationParams segmentation,
        Vp9BoolDecoder reader)
    {
        ArgumentNullException.ThrowIfNull(segmentation);
        ArgumentNullException.ThrowIfNull(reader);
        if (!segmentation.Enabled || !segmentation.UpdateMap) return 0;
        return Vp9SegmentTree.Decode(p => reader.Read(p), segmentation.TreeProbsArray);
    }

    /// <summary>
    /// libvpx <c>read_skip</c>. Returns 0 (residual present) or 1 (no
    /// residual) using skip_probs[skip_context]. If segmentation has
    /// SKIP active for this segment, returns 1 unconditionally without
    /// reading bits.
    /// </summary>
    public static int ReadSkip(
        Vp9SegmentationParams segmentation,
        int segmentId,
        Vp9SkipProbs skipProbs,
        int skipContext,
        Vp9BoolDecoder reader)
    {
        ArgumentNullException.ThrowIfNull(segmentation);
        ArgumentNullException.ThrowIfNull(skipProbs);
        ArgumentNullException.ThrowIfNull(reader);
        if (Vp9SegmentationLookup.IsFeatureActive(segmentation, segmentId, Vp9SegFeature.Skip))
            return 1;
        if ((uint)skipContext >= 3)
            throw new ArgumentOutOfRangeException(nameof(skipContext), skipContext, "skip_context must be 0..2");
        return reader.Read(skipProbs.Probs[skipContext]);
    }

    /// <summary>
    /// libvpx <c>read_tx_size</c>. Returns the per-block tx_size given
    /// the frame-level tx_mode and block size. For TxModeSelect frames
    /// reads from the bitstream guided by per-context tx_size probs;
    /// for forced modes returns min(maxTxForBlock, biggestTxForMode).
    /// </summary>
    public static Vp9TxSize ReadTxSize(
        Vp9TxMode txMode,
        Vp9BlockSize blockSize,
        int txSizeContext,
        Vp9TxModeProbs txModeProbs,
        Vp9BoolDecoder reader)
    {
        ArgumentNullException.ThrowIfNull(txModeProbs);
        ArgumentNullException.ThrowIfNull(reader);
        var maxTx = Vp9MaxTxSize.ForBlockSize(blockSize);
        if (txMode != Vp9TxMode.TxModeSelect || maxTx <= Vp9TxSize.Tx4x4)
        {
            // Forced mode (or sub-8x8 block): no read, just clamp.
            return Vp9TxSizeDecoder.ReadTxSize(txMode, maxTx, reader: null, default);
        }

        // SELECT mode: pull the right probability row by max_tx_size.
        Span<byte> probs = stackalloc byte[3];
        switch (maxTx)
        {
            case Vp9TxSize.Tx8x8:
                probs[0] = txModeProbs.P8x8[txSizeContext, 0];
                return Vp9TxSizeDecoder.ReadSelectedTxSize(maxTx, reader, probs[..1]);
            case Vp9TxSize.Tx16x16:
                probs[0] = txModeProbs.P16x16[txSizeContext, 0];
                probs[1] = txModeProbs.P16x16[txSizeContext, 1];
                return Vp9TxSizeDecoder.ReadSelectedTxSize(maxTx, reader, probs[..2]);
            case Vp9TxSize.Tx32x32:
                probs[0] = txModeProbs.P32x32[txSizeContext, 0];
                probs[1] = txModeProbs.P32x32[txSizeContext, 1];
                probs[2] = txModeProbs.P32x32[txSizeContext, 2];
                return Vp9TxSizeDecoder.ReadSelectedTxSize(maxTx, reader, probs[..3]);
            default:
                throw new InvalidOperationException($"unexpected max_tx_size {maxTx}");
        }
    }
}
