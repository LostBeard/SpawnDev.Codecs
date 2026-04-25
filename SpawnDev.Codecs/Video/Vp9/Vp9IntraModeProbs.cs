// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 default intra mode probability tables. Bit-exact with libvpx
// vp9/common/vp9_entropymode.c.
//
// Three arrays in this slice:
//   - KfUvModeProbs    [10 INTRA_MODES][9 INTRA_MODES - 1]  = 90 bytes
//     Keyframe UV plane: probabilities indexed by the just-decoded Y
//     plane intra mode. libvpx vp9_kf_uv_mode_prob.
//   - DefaultIfYProbs  [4 BLOCK_SIZE_GROUPS][9 INTRA_MODES - 1] = 36 bytes
//     Inter-frame Y plane initial state: probabilities indexed by the
//     block size group. libvpx default_if_y_probs.
//   - DefaultIfUvProbs [10 INTRA_MODES][9 INTRA_MODES - 1] = 90 bytes
//     Inter-frame UV plane initial state: same shape as KfUvModeProbs
//     but different content. libvpx default_if_uv_probs.
//
// Each 9-byte slice feeds <see cref="Vp9IntraModeTree.Decode"/> for
// one (context, plane) pair. The compressed-header machinery may
// adjust the inter-frame defaults during a frame; the keyframe
// table is fixed normative data that never updates.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 default intra mode probability tables.</summary>
public static class Vp9IntraModeProbs
{
    /// <summary>Number of intra mode tree probabilities (libvpx INTRA_MODES - 1).</summary>
    public const int ProbsPerMode = 9;

    /// <summary>Number of block-size groups for inter-frame Y intra (libvpx BLOCK_SIZE_GROUPS).</summary>
    public const int BlockSizeGroups = 4;

    /// <summary>
    /// Keyframe UV plane intra mode probabilities (libvpx
    /// vp9_kf_uv_mode_prob). Flat 10 x 9 = 90 bytes; index by the
    /// decoded Y plane intra mode.
    /// </summary>
    public static readonly byte[] KfUvModeProbs = new byte[]
    {
        144, 11, 54, 157, 195, 130, 46, 58, 108,   // y = DcPred
        118, 15, 123, 148, 131, 101, 44, 93, 131,  // y = VPred
        113, 12, 23, 188, 226, 142, 26, 32, 125,   // y = HPred
        120, 11, 50, 123, 163, 135, 64, 77, 103,   // y = D45Pred
        113, 9, 36, 155, 111, 157, 32, 44, 161,    // y = D135Pred
        116, 9, 55, 176, 76, 96, 37, 61, 149,      // y = D117Pred
        115, 9, 28, 141, 161, 167, 21, 25, 193,    // y = D153Pred
        120, 12, 32, 145, 195, 142, 32, 38, 86,    // y = D207Pred
        116, 12, 64, 120, 140, 125, 49, 115, 121,  // y = D63Pred
        102, 19, 66, 162, 182, 122, 35, 59, 128,   // y = TmPred
    };

    /// <summary>
    /// Inter-frame Y plane intra mode probabilities by block size
    /// group (libvpx default_if_y_probs). Flat 4 x 9 = 36 bytes.
    /// Block size groups: 0 = block_size &lt; 8x8, 1 = &lt; 16x16,
    /// 2 = &lt; 32x32, 3 = &gt;= 32x32.
    /// </summary>
    public static readonly byte[] DefaultIfYProbs = new byte[]
    {
        65, 32, 18, 144, 162, 194, 41, 51, 98,    // block_size < 8x8
        132, 68, 18, 165, 217, 196, 45, 40, 78,   // block_size < 16x16
        173, 80, 19, 176, 240, 193, 64, 35, 46,   // block_size < 32x32
        221, 135, 38, 194, 248, 121, 96, 85, 29,  // block_size >= 32x32
    };

    /// <summary>
    /// Inter-frame UV plane intra mode probabilities (libvpx
    /// default_if_uv_probs). Flat 10 x 9 = 90 bytes; index by the
    /// decoded Y plane intra mode. Initial state - the compressed
    /// frame header may update entries during a frame.
    /// </summary>
    public static readonly byte[] DefaultIfUvProbs = new byte[]
    {
        120, 7, 76, 176, 208, 126, 28, 54, 103,    // y = DcPred
        48, 12, 154, 155, 139, 90, 34, 117, 119,   // y = VPred
        67, 6, 25, 204, 243, 158, 13, 21, 96,      // y = HPred
        97, 5, 44, 131, 176, 139, 48, 68, 97,      // y = D45Pred
        83, 5, 42, 156, 111, 152, 26, 49, 152,     // y = D135Pred
        80, 5, 58, 178, 74, 83, 33, 62, 145,       // y = D117Pred
        86, 5, 32, 154, 192, 168, 14, 22, 163,     // y = D153Pred
        85, 5, 32, 156, 216, 148, 19, 29, 73,      // y = D207Pred
        77, 7, 64, 116, 132, 122, 37, 126, 120,    // y = D63Pred
        101, 21, 107, 181, 192, 103, 19, 67, 125,  // y = TmPred
    };

    /// <summary>
    /// Get the 9-byte probability slice for keyframe UV decoding given
    /// the just-decoded Y plane intra mode. Pass directly to
    /// <see cref="Vp9IntraModeTree.Decode"/>.
    /// </summary>
    public static ReadOnlySpan<byte> KeyframeUvProbs(Vp9IntraMode yMode)
    {
        int idx = (int)yMode;
        if ((uint)idx >= Vp9IntraModeTree.IntraModes)
            throw new ArgumentOutOfRangeException(nameof(yMode));
        return KfUvModeProbs.AsSpan(idx * ProbsPerMode, ProbsPerMode);
    }

    /// <summary>
    /// Get the 9-byte initial-state probability slice for inter-frame
    /// Y plane intra mode decoding given the block size group.
    /// </summary>
    public static ReadOnlySpan<byte> InterFrameYProbs(int blockSizeGroup)
    {
        if ((uint)blockSizeGroup >= BlockSizeGroups)
            throw new ArgumentOutOfRangeException(nameof(blockSizeGroup));
        return DefaultIfYProbs.AsSpan(blockSizeGroup * ProbsPerMode, ProbsPerMode);
    }

    /// <summary>
    /// Get the 9-byte initial-state probability slice for inter-frame
    /// UV plane intra mode decoding given the just-decoded Y plane
    /// intra mode.
    /// </summary>
    public static ReadOnlySpan<byte> InterFrameUvProbs(Vp9IntraMode yMode)
    {
        int idx = (int)yMode;
        if ((uint)idx >= Vp9IntraModeTree.IntraModes)
            throw new ArgumentOutOfRangeException(nameof(yMode));
        return DefaultIfUvProbs.AsSpan(idx * ProbsPerMode, ProbsPerMode);
    }
}
