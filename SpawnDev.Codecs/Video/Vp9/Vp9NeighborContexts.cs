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
// More involved helpers (partition_plane_context, get_tx_size_context,
// reference-frame context predictors) need richer neighbor state -
// kept for downstream slices.

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
}
