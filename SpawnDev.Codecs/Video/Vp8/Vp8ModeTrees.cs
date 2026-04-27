// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP8 intra-mode decision trees + default mode probabilities.
// Bit-exact copy of libvpx vp8/common/entropymode.c trees +
// vp8_entropymodedata.h prob tables.
//
// VP8 uses signed-int trees:
//   - positive value = next node index (byte slot in the array)
//   - negative value = leaf token (the prediction-mode value)
//
// Tree walk: at node `i`, read a bit at probs[i >> 1]; advance to
// tree[i + bit]. Stop when the result is non-positive; the leaf token
// is the negation of that value.
//
// Trees:
//   vp8_kf_ymode_tree (5 leaves: B_PRED, DC, V, H, TM) - keyframe Y mode
//   vp8_uv_mode_tree  (4 leaves: DC, V, H, TM)        - chroma UV mode
//   vp8_bmode_tree    (10 leaves: 10 4x4 modes)        - per-block 4x4 mode
//
// Default probability tables (4/3/9 entries respectively) used as
// initial probs for the tree walks.

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>VP8 16x16 luma intra mode (RFC 6386 sec 11.2).</summary>
public enum Vp8YMode : byte
{
    /// <summary>DC predictor.</summary>
    DcPred = 0,
    /// <summary>Vertical predictor.</summary>
    VPred = 1,
    /// <summary>Horizontal predictor.</summary>
    HPred = 2,
    /// <summary>TrueMotion predictor.</summary>
    TmPred = 3,
    /// <summary>16x16 split into 4x4 sub-blocks, each with its own intra mode.</summary>
    BPred = 4,
}

/// <summary>VP8 8x8 chroma intra mode. Same alphabet as 16x16 luma but no B_PRED.</summary>
public enum Vp8UvMode : byte
{
    /// <summary>DC predictor.</summary>
    DcPred = 0,
    /// <summary>Vertical predictor.</summary>
    VPred = 1,
    /// <summary>Horizontal predictor.</summary>
    HPred = 2,
    /// <summary>TrueMotion predictor.</summary>
    TmPred = 3,
}

/// <summary>VP8 mode trees + default probabilities (RFC 6386 sec 11).</summary>
public static class Vp8ModeTrees
{
    /// <summary>
    /// Keyframe Y mode tree. 8 entries, leaves at negative values; matches
    /// libvpx <c>vp8_kf_ymode_tree</c>. Note keyframe order has B_PRED first,
    /// distinct from inter-frame <c>vp8_ymode_tree</c>.
    /// </summary>
    public static readonly sbyte[] KfYModeTree = new sbyte[]
    {
        -(sbyte)Vp8YMode.BPred, 2,
        4, 6,
        -(sbyte)Vp8YMode.DcPred, -(sbyte)Vp8YMode.VPred,
        -(sbyte)Vp8YMode.HPred, -(sbyte)Vp8YMode.TmPred,
    };

    /// <summary>UV mode tree. Same shape for keyframes and inter frames.</summary>
    public static readonly sbyte[] UvModeTree = new sbyte[]
    {
        -(sbyte)Vp8UvMode.DcPred, 2,
        -(sbyte)Vp8UvMode.VPred, 4,
        -(sbyte)Vp8UvMode.HPred, -(sbyte)Vp8UvMode.TmPred,
    };

    /// <summary>
    /// 4x4 intra-mode tree. 18 entries, 10 leaves; matches libvpx
    /// <c>vp8_bmode_tree</c>. Used for sub-block mode decode under
    /// keyframe Y_PRED = B_PRED.
    /// </summary>
    public static readonly sbyte[] BModeTree = new sbyte[]
    {
        -(sbyte)Vp8IntraMode4x4.BDcPred, 2,
        -(sbyte)Vp8IntraMode4x4.BTmPred, 4,
        -(sbyte)Vp8IntraMode4x4.BVePred, 6,
        8, 12,
        -(sbyte)Vp8IntraMode4x4.BHePred, 10,
        -(sbyte)Vp8IntraMode4x4.BRdPred, -(sbyte)Vp8IntraMode4x4.BVrPred,
        -(sbyte)Vp8IntraMode4x4.BLdPred, 14,
        -(sbyte)Vp8IntraMode4x4.BVlPred, 16,
        -(sbyte)Vp8IntraMode4x4.BHdPred, -(sbyte)Vp8IntraMode4x4.BHuPred,
    };

    /// <summary>Default keyframe Y mode probabilities (libvpx vp8_kf_ymode_prob).</summary>
    public static readonly byte[] DefaultKfYModeProb = new byte[] { 145, 156, 163, 128 };

    /// <summary>Default keyframe UV mode probabilities (libvpx vp8_kf_uv_mode_prob).</summary>
    public static readonly byte[] DefaultKfUvModeProb = new byte[] { 142, 114, 183 };

    /// <summary>Default 4x4 intra-mode probabilities (libvpx vp8_bmode_prob).</summary>
    public static readonly byte[] DefaultBModeProb = new byte[] { 120, 90, 79, 133, 87, 85, 80, 111, 151 };

    /// <summary>
    /// Walk a VP8 mode tree using the supplied probabilities. Mirrors
    /// libvpx <c>treed_read</c>: starts at index 0, reads a bit at
    /// <paramref name="probs"/>[i &gt;&gt; 1], advances to tree[i + bit],
    /// stops when the result is non-positive (leaf), returns -result.
    /// </summary>
    public static int DecodeTree(Vp8BoolDecoder reader, sbyte[] tree, ReadOnlySpan<byte> probs)
    {
        ArgumentNullException.ThrowIfNull(reader);
        ArgumentNullException.ThrowIfNull(tree);
        int i = 0;
        while (true)
        {
            int bit = reader.DecodeBool(probs[i >> 1]);
            sbyte next = tree[i + bit];
            if (next <= 0) return -next;
            i = next;
        }
    }
}
