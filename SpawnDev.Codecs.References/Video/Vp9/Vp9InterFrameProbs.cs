// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Small VP9 default probability tables for inter-frame block parsing.
// Bit-exact with libvpx vp9/common/vp9_entropymode.c.
//
// Three tables in this slice:
//   - DefaultInterModeProbs (7 x 3 = 21 bytes)
//     Per-context probability slice for the inter mode tree (slice
//     158's Vp9InterModeTree). 7 contexts cover the (above, left)
//     mode-info combinations; 3 probs per context drive the 4-leaf
//     tree decision. libvpx default_inter_mode_probs.
//
//   - DefaultSkipProbs (3 bytes)
//     Per-context probability that the current block has no decoded
//     coefficients (the "skip" bit at the start of mode-info parsing).
//     3 contexts based on the skip-bit values of the above + left
//     blocks. libvpx default_skip_probs = { 192, 128, 64 }.
//
//   - DefaultIntraInterProb (4 bytes)
//     Per-context probability that the current block is intra-coded
//     (vs. inter-coded). 4 contexts. libvpx default_intra_inter_p =
//     { 9, 102, 187, 225 }.
//
// All three are "default" tables - the compressed frame header may
// update individual entries during a frame, but a fresh decode at
// keyframe boundaries starts from these values.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// Small VP9 inter-frame default probability tables.
/// </summary>
public static class Vp9InterFrameProbs
{
    /// <summary>Number of inter mode contexts (libvpx INTER_MODE_CONTEXTS = 7).</summary>
    public const int InterModeContexts = 7;

    /// <summary>Number of skip contexts (libvpx SKIP_CONTEXTS = 3).</summary>
    public const int SkipContexts = 3;

    /// <summary>Number of intra/inter contexts (libvpx INTRA_INTER_CONTEXTS = 4).</summary>
    public const int IntraInterContexts = 4;

    /// <summary>
    /// Default probabilities for the inter mode tree, 7 contexts x 3
    /// probs (libvpx default_inter_mode_probs). Index by context to
    /// get a 3-byte slice for <see cref="Vp9InterModeTree.Decode"/>.
    /// </summary>
    public static readonly byte[] DefaultInterModeProbs = new byte[]
    {
        2, 173, 34,    // 0 = both zero mv
        7, 145, 85,    // 1 = one zero mv + one a predicted mv
        7, 166, 63,    // 2 = two predicted mvs
        7, 94,  66,    // 3 = one predicted/zero and one new mv
        8, 64,  46,    // 4 = two new mvs
        17, 81, 31,    // 5 = one intra neighbour + x
        25, 29, 30,    // 6 = two intra neighbours
    };

    /// <summary>
    /// Default skip-bit probabilities, 3 contexts (libvpx
    /// default_skip_probs).
    /// </summary>
    public static readonly byte[] DefaultSkipProbs = new byte[] { 192, 128, 64 };

    /// <summary>
    /// Default intra-vs-inter probabilities, 4 contexts (libvpx
    /// default_intra_inter_p).
    /// </summary>
    public static readonly byte[] DefaultIntraInterProb = new byte[] { 9, 102, 187, 225 };

    /// <summary>
    /// Get the 3-byte inter mode probability slice for the given
    /// context. Pass directly to <see cref="Vp9InterModeTree.Decode"/>.
    /// </summary>
    public static ReadOnlySpan<byte> InterModeProbs(int context)
    {
        if ((uint)context >= InterModeContexts)
            throw new ArgumentOutOfRangeException(nameof(context));
        return DefaultInterModeProbs.AsSpan(context * 3, 3);
    }
}
