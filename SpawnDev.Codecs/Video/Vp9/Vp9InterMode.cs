// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 inter mode tree topology + decoder. Inter-coded blocks select
// one of 4 motion-vector modes from this tree at decode time.
//
// libvpx reference: vp9/common/vp9_entropymode.c (vp9_inter_mode_tree)
// and vp9_blockd.h (INTER_OFFSET macro). VP9 spec sec 6.4.4 mode info.
//
// Inter mode encoding uses a "mode offset" (0..3) rather than the
// absolute PREDICTION_MODE values (10..13 in vp9_enums.h). The
// offset = mode - NEARESTMV, so:
//   NEARESTMV (10) - 10 = 0
//   NEARMV    (11) - 10 = 1
//   ZEROMV    (12) - 10 = 2
//   NEWMV     (13) - 10 = 3
//
// The probability tables, tree leaves, and counters all index by the
// offset (libvpx convention), not the absolute mode. This file
// follows the same convention - Vp9InterMode is the OFFSET.
//
// Tree structure (3 internal nodes, 4 leaves; libvpx layout):
//   i=0  ROOT       -> -ZeroMv,  2 = NEAREST_OR_FAR_OR_NEW
//   i=2  NEAREST_NM -> -NearestMv (offset 0 -> non-negative leaf), 4
//   i=4  NEAR_OR_NEW -> -NearMv, -NewMv

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// VP9 inter mode "offset" alphabet (libvpx INTER_OFFSET, 0..3).
/// The absolute PREDICTION_MODE values are 10..13; the offset is
/// what every probability table and tree leaf encodes.
/// </summary>
public enum Vp9InterMode : byte
{
    /// <summary>0 - Nearest motion-vector candidate.</summary>
    NearestMv = 0,
    /// <summary>1 - Near motion-vector candidate.</summary>
    NearMv = 1,
    /// <summary>2 - Zero motion vector.</summary>
    ZeroMv = 2,
    /// <summary>3 - New motion vector explicitly transmitted.</summary>
    NewMv = 3,
}

/// <summary>VP9 inter mode tree topology and decoder.</summary>
public static class Vp9InterModeTree
{
    /// <summary>Number of inter modes (libvpx INTER_MODES = 4).</summary>
    public const int InterModes = 4;

    /// <summary>
    /// libvpx vp9_inter_mode_tree, 6 entries (3 internal nodes x 2
    /// branches). Negative-or-zero values are leaf inter modes;
    /// positive values are byte indices of the next node.
    ///
    /// Note: ZeroMv = 2 (positive when negated to -2 in tree) but
    /// NearestMv = 0 negates to 0 which is a non-negative leaf - the
    /// walker treats <c>next &lt;= 0</c> as a leaf so offset 0 lands
    /// correctly. Same convention as slice 153's intra mode tree
    /// where DcPred = 0.
    /// </summary>
    public static readonly sbyte[] Tree = new sbyte[]
    {
        -(sbyte)Vp9InterMode.ZeroMv,    2,                              // 0 = ROOT
        -(sbyte)Vp9InterMode.NearestMv, 4,                              // 1 = NEAREST_NM
        -(sbyte)Vp9InterMode.NearMv,    -(sbyte)Vp9InterMode.NewMv,     // 2 = NEAR_OR_NEW
    };

    /// <summary>
    /// Walk the inter mode tree using a 3-entry probability vector.
    /// Same shape as <see cref="Vp9CoefTrees.DecodeConToken"/>,
    /// <see cref="Vp9PartitionTree.Decode"/>, and
    /// <see cref="Vp9IntraModeTree.Decode"/>. The probability vector
    /// is indexed by tree pair index; libvpx exposes
    /// inter_mode_probs[ctx][3] arrays that match.
    /// </summary>
    public static Vp9InterMode Decode(Func<byte, int> readBit, ReadOnlySpan<byte> probs)
    {
        ArgumentNullException.ThrowIfNull(readBit);
        if (probs.Length < 3)
            throw new ArgumentException("probs must hold 3 entries for the inter mode tree", nameof(probs));

        int i = 0;
        while (true)
        {
            int bit = readBit(probs[i >> 1]);
            sbyte next = Tree[i + bit];
            if (next <= 0)
                return (Vp9InterMode)(-next);
            i = next;
        }
    }
}
