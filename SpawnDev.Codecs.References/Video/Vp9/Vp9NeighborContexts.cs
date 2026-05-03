// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 neighbor-derived context helpers. Pure functions over the
// above/left neighbor flags - caller queries the mode-info grid for
// the relevant edge state and passes it in as bool / int args.
//
// This file packages the simplest of libvpx's context-derivation
// helpers from vp9/common/vp9_pred_common.h:
//
//   get_skip_context        : above_skip + left_skip in [0, 2]
//   get_intra_inter_context : 0..3 from above/left availability
//                             plus their intra-vs-inter flags
//
// Block-edge neighbor state is encoded as `bool? aboveIntra` /
// `bool? leftIntra`:
//   null       = neighbor unavailable (image edge or block-zero
//                during decode)
//   true       = neighbor is intra-coded
//   false      = neighbor is inter-coded
//
// More involved helpers (reference-frame context predictors) need
// richer neighbor state - kept for downstream slices.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 neighbor-derived context helpers.</summary>
public static class Vp9NeighborContexts
{
    /// <summary>
    /// libvpx <c>get_skip_context</c>. Returns 0..2 = (above_skip + left_skip).
    /// Missing neighbors contribute 0. Used as the index into the
    /// <see cref="Vp9SkipProbs"/> array for the per-block skip flag.
    /// </summary>
    /// <param name="aboveSkip">
    /// Skip flag of the above neighbor, or null if unavailable.
    /// </param>
    /// <param name="leftSkip">
    /// Skip flag of the left neighbor, or null if unavailable.
    /// </param>
    public static int GetSkipContext(bool? aboveSkip, bool? leftSkip)
    {
        int above = aboveSkip == true ? 1 : 0;
        int left = leftSkip == true ? 1 : 0;
        return above + left;
    }

    /// <summary>
    /// libvpx <c>get_intra_inter_context</c>. Returns 0..3 indexing the
    /// <see cref="Vp9IntraInterProbs"/> 4-context probability table.
    /// </summary>
    /// <param name="aboveIntra">
    /// True if the above neighbor is intra-coded; false if inter-coded;
    /// null if no above neighbor.
    /// </param>
    /// <param name="leftIntra">
    /// True if the left neighbor is intra-coded; false if inter-coded;
    /// null if no left neighbor.
    /// </param>
    /// <returns>
    /// Both edges present:
    /// <list type="bullet">
    /// <item><description>0 = both inter</description></item>
    /// <item><description>1 = exactly one of (above, left) is intra</description></item>
    /// <item><description>3 = both intra</description></item>
    /// </list>
    /// One edge present:
    /// <list type="bullet">
    /// <item><description>0 = the present edge is inter</description></item>
    /// <item><description>2 = the present edge is intra</description></item>
    /// </list>
    /// Neither edge present: 0.
    /// </returns>
    public static int GetIntraInterContext(bool? aboveIntra, bool? leftIntra)
    {
        bool hasAbove = aboveIntra.HasValue;
        bool hasLeft = leftIntra.HasValue;

        if (hasAbove && hasLeft)
        {
            bool ai = aboveIntra!.Value;
            bool li = leftIntra!.Value;
            if (li && ai) return 3;
            return (li || ai) ? 1 : 0;
        }

        if (hasAbove || hasLeft)
        {
            bool edgeIntra = hasAbove ? aboveIntra!.Value : leftIntra!.Value;
            return edgeIntra ? 2 : 0;
        }

        return 0;
    }

    /// <summary>
    /// libvpx <c>get_tx_size_context</c>. Returns 0 or 1 indexing the
    /// <see cref="Vp9TxModeProbs.TxSizeContexts"/> dimension of the
    /// per-context tx_size probability tables.
    /// </summary>
    /// <param name="blockSize">Current block's <see cref="Vp9BlockSize"/>.</param>
    /// <param name="above">
    /// Above neighbor's (tx_size, skip) tuple, or null if no above
    /// neighbor. Skipped neighbors do not contribute their tx_size.
    /// </param>
    /// <param name="left">
    /// Left neighbor's (tx_size, skip) tuple, or null if no left
    /// neighbor. Skipped neighbors do not contribute their tx_size.
    /// </param>
    /// <remarks>
    /// libvpx logic: when a neighbor is missing entirely (image edge),
    /// inherit from the other neighbor. When a neighbor is present
    /// but skip-coded, treat its tx_size as max_tx_size for the
    /// CURRENT block. Context = 1 if the (above + left) tx_size sum
    /// exceeds max_tx_size, else 0.
    /// </remarks>
    public static int GetTxSizeContext(
        Vp9BlockSize blockSize,
        (Vp9TxSize TxSize, bool Skip)? above,
        (Vp9TxSize TxSize, bool Skip)? left)
    {
        var maxTxSize = Vp9MaxTxSize.ForBlockSize(blockSize);
        int maxIdx = (int)maxTxSize;

        bool hasAbove = above.HasValue;
        bool hasLeft = left.HasValue;

        int aboveCtx = hasAbove && !above!.Value.Skip ? (int)above.Value.TxSize : maxIdx;
        int leftCtx = hasLeft && !left!.Value.Skip ? (int)left.Value.TxSize : maxIdx;
        if (!hasAbove) aboveCtx = leftCtx;
        if (!hasLeft) leftCtx = aboveCtx;
        return (aboveCtx + leftCtx) > maxIdx ? 1 : 0;
    }

    /// <summary>
    /// libvpx <c>PARTITION_PLOFFSET</c> = 4: stride between bsl groups
    /// in the 16-context partition probability tables.
    /// </summary>
    public const int PartitionPlaneOffset = 4;

    /// <summary>
    /// libvpx <c>partition_plane_context</c>. Returns 0..15 indexing
    /// either <see cref="Vp9PartitionProbs.KfPartitionProbs"/> or
    /// <see cref="Vp9PartitionProbs.DefaultPartitionProbs"/>.
    ///
    /// Layout: <c>(left_split * 2 + above_split) + bsl * 4</c>, where
    /// <c>bsl = mi_width_log2_lookup[blockSize]</c> picks the bsl
    /// group (0 = 8x8 -> 4x4, 1 = 16x16 -> 8x8, 2 = 32x32 -> 16x16,
    /// 3 = 64x64 -> 32x32) and the (left_split, above_split) bits
    /// pick the 4 contexts within that group.
    /// </summary>
    /// <param name="aboveSegCtx">
    /// Above-row segment-context byte at column <c>mi_col</c>.
    /// libvpx stores 1 bit per bsl level; only the bit at
    /// <c>bsl = mi_width_log2_lookup[blockSize]</c> matters here.
    /// </param>
    /// <param name="leftSegCtx">
    /// Left-column segment-context byte at <c>mi_row &amp; MI_MASK</c>.
    /// </param>
    /// <param name="blockSize">
    /// Block size whose partition decision is being decoded. Must be
    /// >= Block8x8 (libvpx never decodes a partition for sub-8x8).
    /// </param>
    public static int GetPartitionPlaneContext(
        byte aboveSegCtx,
        byte leftSegCtx,
        Vp9BlockSize blockSize)
    {
        int idx = (int)blockSize;
        if ((uint)idx >= (uint)Vp9BlockSizes.Count)
            throw new ArgumentOutOfRangeException(nameof(blockSize), blockSize,
                "VP9 block size index out of range.");
        int bsl = Vp9BlockSizes.MiWidthLog2[idx];
        if (bsl < 0 || bsl > 3)
            throw new ArgumentOutOfRangeException(nameof(blockSize), blockSize,
                "Partition decoding only applies to Block8x8..Block64x64.");

        int above = (aboveSegCtx >> bsl) & 1;
        int left = (leftSegCtx >> bsl) & 1;
        return (left * 2 + above) + bsl * PartitionPlaneOffset;
    }

    /// <summary>
    /// libvpx <c>SWITCHABLE_FILTERS</c> = 3. Used both as the count of
    /// per-block selectable filters (EightTap / EightTapSmooth /
    /// EightTapSharp) and as the sentinel "missing or non-inter
    /// neighbor" value when computing switchable-interp context.
    /// </summary>
    public const int SwitchableFiltersCount = 3;

    /// <summary>
    /// libvpx <c>vp9_get_pred_context_switchable_interp</c>. Returns
    /// 0..3 indexing the <see cref="Vp9SwitchableInterpProbs"/> 4-context
    /// dimension. Logic: a neighbor contributes its actual filter only
    /// if it is present AND inter-coded; otherwise it contributes the
    /// sentinel <see cref="SwitchableFiltersCount"/>.
    /// </summary>
    /// <remarks>
    /// Result rules (after substituting non-inter / missing neighbors
    /// with the sentinel):
    /// <list type="bullet">
    /// <item><description>If both sides agree, return that filter (or
    /// sentinel if both sides agree on sentinel).</description></item>
    /// <item><description>If exactly one side is the sentinel, return
    /// the other side's filter.</description></item>
    /// <item><description>If both sides are valid but differ, return
    /// sentinel.</description></item>
    /// </list>
    /// libvpx clamps incoming filter values against
    /// <see cref="Vp9InterpFilter.Bilinear"/> by treating it as a fixed
    /// filter; switchable contexts only ever see values in [0,
    /// SwitchableFiltersCount]. Callers from
    /// non-switchable-frame paths must pass the sentinel directly via
    /// <c>isInter = false</c>.
    /// </remarks>
    public static int GetSwitchableInterpContext(
        (bool IsInter, Vp9InterpFilter Filter)? above,
        (bool IsInter, Vp9InterpFilter Filter)? left)
    {
        int aboveType = above.HasValue && above.Value.IsInter
            ? (int)above.Value.Filter
            : SwitchableFiltersCount;
        int leftType = left.HasValue && left.Value.IsInter
            ? (int)left.Value.Filter
            : SwitchableFiltersCount;

        if (leftType == aboveType) return leftType;
        if (leftType == SwitchableFiltersCount) return aboveType;
        if (aboveType == SwitchableFiltersCount) return leftType;
        return SwitchableFiltersCount;
    }

    /// <summary>
    /// libvpx <c>vp9_get_pred_context_seg_id</c>. Returns 0..2 =
    /// (above_seg_id_predicted + left_seg_id_predicted), where
    /// missing neighbors contribute 0. Used to index the segmentation
    /// temporal-update probability array
    /// (<see cref="Vp9SegmentationParams.PredProbs"/>).
    /// </summary>
    /// <param name="aboveSegIdPredicted">
    /// seg_id_predicted flag of the above neighbor, or null if no
    /// above neighbor (top-of-frame edge).
    /// </param>
    /// <param name="leftSegIdPredicted">
    /// seg_id_predicted flag of the left neighbor, or null if no
    /// left neighbor (start-of-tile-row edge).
    /// </param>
    public static int GetSegIdContext(bool? aboveSegIdPredicted, bool? leftSegIdPredicted)
    {
        int above = aboveSegIdPredicted == true ? 1 : 0;
        int left = leftSegIdPredicted == true ? 1 : 0;
        return above + left;
    }
}
