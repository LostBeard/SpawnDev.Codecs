// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 intra prediction mode alphabet + tree topology used by the
// entropy decoder to read the y_mode and uv_mode fields of an
// intra-frame block.
//
// libvpx reference: vp9/common/vp9_enums.h (DC_PRED..TM_PRED = 0..9)
// and vp9_entropymode.c (vp9_intra_mode_tree, 18 entries, 9 internal
// nodes / 10 leaves).
//
// Mode reference (libvpx vp9_enums.h comments):
//   DC_PRED       0  Average of above and left pixels
//   V_PRED        1  Vertical
//   H_PRED        2  Horizontal
//   D45_PRED      3  Directional 45 deg
//   D135_PRED     4  Directional 135 deg
//   D117_PRED     5  Directional 117 deg
//   D153_PRED     6  Directional 153 deg
//   D207_PRED     7  Directional 207 deg
//   D63_PRED      8  Directional 63 deg
//   TM_PRED       9  True-motion
//
// Tree structure (libvpx vp9_intra_mode_tree, byte-indexed, 18
// entries, 9 internal nodes / 10 leaves):
//   i=0  DC_NODE   -> -DC_PRED,  2 = TM_NODE
//   i=2  TM_NODE   -> -TM_PRED,  4 = V_NODE
//   i=4  V_NODE    -> -V_PRED,   6 = COM_NODE
//   i=6  COM_NODE  ->  8 = H_NODE,         12 = D45_NODE
//   i=8  H_NODE    -> -H_PRED,   10 = D135_NODE
//   i=10 D135_NODE -> -D135_PRED, -D117_PRED
//   i=12 D45_NODE  -> -D45_PRED, 14 = D63_NODE
//   i=14 D63_NODE  -> -D63_PRED, 16 = D153_NODE
//   i=16 D153_NODE -> -D153_PRED, -D207_PRED

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 intra prediction modes (libvpx PREDICTION_MODE 0..9).</summary>
public enum Vp9IntraMode : byte
{
    /// <summary>0 - Average of above and left pixels.</summary>
    DcPred = 0,
    /// <summary>1 - Vertical (copy top row).</summary>
    VPred = 1,
    /// <summary>2 - Horizontal (copy left column).</summary>
    HPred = 2,
    /// <summary>3 - Directional 45 deg.</summary>
    D45Pred = 3,
    /// <summary>4 - Directional 135 deg.</summary>
    D135Pred = 4,
    /// <summary>5 - Directional 117 deg.</summary>
    D117Pred = 5,
    /// <summary>6 - Directional 153 deg.</summary>
    D153Pred = 6,
    /// <summary>7 - Directional 207 deg.</summary>
    D207Pred = 7,
    /// <summary>8 - Directional 63 deg.</summary>
    D63Pred = 8,
    /// <summary>9 - True-motion.</summary>
    TmPred = 9,
}

/// <summary>VP9 intra mode tree topology and decoder.</summary>
public static class Vp9IntraModeTree
{
    /// <summary>
    /// Number of intra prediction modes (libvpx INTRA_MODES = 10).
    /// </summary>
    public const int IntraModes = 10;

    /// <summary>
    /// libvpx vp9_intra_mode_tree, 18 entries (9 internal nodes x 2
    /// branches). Negative values are leaf modes; non-negative values
    /// are byte indices of the next node within this same array.
    ///
    /// DcPred = 0 case: the tree value at slot 0 is `-DcPred` = 0.
    /// The walker treats the value as &quot;&lt;= 0 means leaf&quot; so 0
    /// is a valid leaf for mode value 0.
    /// </summary>
    public static readonly sbyte[] Tree = new sbyte[]
    {
        -(sbyte)Vp9IntraMode.DcPred,    2,                            // 0 = DC_NODE
        -(sbyte)Vp9IntraMode.TmPred,    4,                            // 1 = TM_NODE
        -(sbyte)Vp9IntraMode.VPred,     6,                            // 2 = V_NODE
        8,                              12,                           // 3 = COM_NODE
        -(sbyte)Vp9IntraMode.HPred,     10,                           // 4 = H_NODE
        -(sbyte)Vp9IntraMode.D135Pred,  -(sbyte)Vp9IntraMode.D117Pred,// 5 = D135_NODE
        -(sbyte)Vp9IntraMode.D45Pred,   14,                           // 6 = D45_NODE
        -(sbyte)Vp9IntraMode.D63Pred,   16,                           // 7 = D63_NODE
        -(sbyte)Vp9IntraMode.D153Pred,  -(sbyte)Vp9IntraMode.D207Pred,// 8 = D153_NODE
    };

    /// <summary>
    /// Walk the intra mode tree using the supplied 9-entry probability
    /// vector. Same shape as <see cref="Vp9CoefTrees.DecodeConToken"/>
    /// and <see cref="Vp9PartitionTree.Decode"/>. The probability
    /// vector is indexed by tree pair index; libvpx exposes
    /// y_mode_prob and uv_mode_prob arrays that match.
    /// </summary>
    public static Vp9IntraMode Decode(Func<byte, int> readBit, ReadOnlySpan<byte> probs)
    {
        ArgumentNullException.ThrowIfNull(readBit);
        if (probs.Length < 9)
            throw new ArgumentException("probs must hold 9 entries for the intra mode tree", nameof(probs));

        int i = 0;
        while (true)
        {
            int bit = readBit(probs[i >> 1]);
            sbyte next = Tree[i + bit];
            if (next <= 0)
                return (Vp9IntraMode)(-next);
            i = next;
        }
    }
}
