// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP8 macroblock mode info decoder (key-frame path). Mirrors libvpx
// vp8/decoder/decodemv.c read_kf_mb_mode_info plus the segment-ID +
// mb_skip_coeff prelude from vp8_decode_mode_mvs.
//
// For a key frame each macroblock (16x16 luma + 2x 8x8 chroma) carries:
//   1. segment_id          (only if segmentation_enabled && update_map)
//   2. mb_skip_coeff       (only if mb_no_skip_coeff_enabled)
//   3. y_mode              (Vp8YMode 0..4 - DC/V/H/TM/B)
//   4. if y_mode == B_PRED: 16 sub-block modes (Vp8IntraMode4x4 each,
//                            decoded with the kf_bmode_prob context table
//                            indexed by [above_block_mode][left_block_mode])
//   5. uv_mode             (Vp8UvMode 0..3 - DC/V/H/TM)
//
// Inter-frame additions (ref_frame, mv_ref, mv_components) layer in the
// next slice.
//
// kf_bmode_prob is the 10 x 10 x 9-byte context table; this slice does
// not yet include it (it's a 900-byte normative table). Until that lands,
// callers can pass the default DefaultBModeProb + a callback that
// returns it for any (above, left) pair, OR a dummy zero context if they
// only need 16x16 modes.

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>VP8 per-macroblock mode info (decoded key-frame).</summary>
public sealed record Vp8KeyFrameMbModeInfo
{
    /// <summary>Segment ID 0..3 (default 0 if segmentation map not updated).</summary>
    public required int SegmentId { get; init; }
    /// <summary>True if the macroblock has no coefficients (skip flag).</summary>
    public required bool SkipCoeff { get; init; }
    /// <summary>16x16 Y mode (or B_PRED to indicate per-block sub-modes).</summary>
    public required Vp8YMode YMode { get; init; }
    /// <summary>
    /// 16 sub-block modes when <see cref="YMode"/> == <see cref="Vp8YMode.BPred"/>;
    /// null otherwise. Indexed in 4x4 raster order within the MB.
    /// </summary>
    public required Vp8IntraMode4x4[]? SubBlockModes { get; init; }
    /// <summary>8x8 chroma UV mode.</summary>
    public required Vp8UvMode UvMode { get; init; }
}

/// <summary>
/// VP8 keyframe MB mode info decoder. Composes the bool decoder + mode
/// trees + default probabilities into per-macroblock mode info.
/// </summary>
public static class Vp8MbModeInfoDecoder
{
    /// <summary>
    /// Decode one macroblock's mode info from the keyframe bitstream.
    /// </summary>
    /// <param name="reader">Bool decoder positioned at the MB.</param>
    /// <param name="frameHeader">Decoded frame header (carries seg / skip enables).</param>
    /// <param name="bModeProbResolver">
    /// Callback that supplies the 9-entry probability vector for each
    /// 4x4 sub-block when y_mode == B_PRED. Receives:
    ///   subIndex - 4x4 sub-block index in raster order (0..15)
    ///   alreadyDecoded - the sub-block modes already decoded for THIS MB
    ///                    (entries [0..subIndex-1] are valid; the rest are default values)
    /// The callback should resolve the (above, left) sub-mode context per
    /// libvpx <c>above_block_mode</c> / <c>left_block_mode</c> rules using
    /// the MB-above and MB-left state the walker tracks externally PLUS
    /// the in-progress entries for sub-blocks
    /// already processed in this MB. It MUST then return the 9-entry
    /// probability vector at <c>vp8_kf_bmode_prob[above][left]</c>.
    /// Pass null to use the default (uses <see cref="Vp8ModeTrees.DefaultBModeProb"/>
    /// for every block - functionally correct only when all neighbors are BDcPred).
    /// </param>
    public static Vp8KeyFrameMbModeInfo DecodeKeyFrameMb(
        Vp8BoolDecoder reader,
        Vp8FrameHeader frameHeader,
        Func<int, Vp8IntraMode4x4[], byte[]>? bModeProbResolver = null)
    {
        ArgumentNullException.ThrowIfNull(reader);
        ArgumentNullException.ThrowIfNull(frameHeader);
        bModeProbResolver ??= DefaultBModeProbResolver;

        // 1. Segment ID (if segmentation_enabled && update_map)
        int segmentId = 0;
        if (frameHeader.Segmentation.Enabled && frameHeader.Segmentation.UpdateMap)
        {
            segmentId = DecodeSegmentId(reader, frameHeader.Segmentation.SegmentTreeProbs);
        }

        // 2. mb_skip_coeff
        bool skipCoeff = false;
        if (frameHeader.MbNoSkipCoeffEnabled)
        {
            skipCoeff = reader.DecodeBool(frameHeader.ProbSkipFalse) != 0;
        }

        // 3. Y mode
        var yMode = (Vp8YMode)Vp8ModeTrees.DecodeTree(
            reader, Vp8ModeTrees.KfYModeTree, Vp8ModeTrees.DefaultKfYModeProb);

        // 4. Sub-block modes if Y mode == B_PRED. We pass the
        // partially-filled subBlockModes buffer to the resolver so it can
        // compute (above, left) using already-decoded neighbors WITHIN the
        // current MB plus the walker-tracked above/left MB state.
        Vp8IntraMode4x4[]? subBlockModes = null;
        if (yMode == Vp8YMode.BPred)
        {
            subBlockModes = new Vp8IntraMode4x4[16];
            for (int i = 0; i < 16; i++) subBlockModes[i] = Vp8IntraMode4x4.BDcPred; // default for "not yet decoded"
            for (int i = 0; i < 16; i++)
            {
                var probs = bModeProbResolver(i, subBlockModes);
                subBlockModes[i] = (Vp8IntraMode4x4)Vp8ModeTrees.DecodeTree(
                    reader, Vp8ModeTrees.BModeTree, probs);
            }
        }

        // 5. UV mode
        var uvMode = (Vp8UvMode)Vp8ModeTrees.DecodeTree(
            reader, Vp8ModeTrees.UvModeTree, Vp8ModeTrees.DefaultKfUvModeProb);

        return new Vp8KeyFrameMbModeInfo
        {
            SegmentId = segmentId,
            SkipCoeff = skipCoeff,
            YMode = yMode,
            SubBlockModes = subBlockModes,
            UvMode = uvMode,
        };
    }

    /// <summary>
    /// Decode a 0..3 segment ID from a 3-leaf segment tree.
    /// libvpx reads this as: bit -> bit -> 0/1; the 4-segment tree shape
    /// is fixed.
    /// </summary>
    private static int DecodeSegmentId(Vp8BoolDecoder reader, byte[] segmentTreeProbs)
    {
        // libvpx vp8_kf_ymode_tree-style: tree = { 2, 4, -0, -1, -2, -3 }
        // i.e., bit0 -> branch0; bit0=0 -> read bit1 to choose seg 0/1;
        //       bit0=1 -> read bit2 to choose seg 2/3.
        if (reader.DecodeBool(segmentTreeProbs[0]) == 0)
            return reader.DecodeBool(segmentTreeProbs[1]);
        else
            return 2 + reader.DecodeBool(segmentTreeProbs[2]);
    }

    /// <summary>
    /// Default bmode probability resolver - returns the static
    /// <see cref="Vp8ModeTrees.DefaultBModeProb"/> regardless of position
    /// or already-decoded neighbors. Only correct when neighbors happen to
    /// be the trivial all-BDcPred case; the walker overrides this with a
    /// real <see cref="Vp8KfBmodeProbs.GetProbs"/> lookup using the proper
    /// (above, left) sub-mode context.
    /// </summary>
    public static byte[] DefaultBModeProbResolver(int subIndex, Vp8IntraMode4x4[] alreadyDecoded)
        => Vp8ModeTrees.DefaultBModeProb;
}
