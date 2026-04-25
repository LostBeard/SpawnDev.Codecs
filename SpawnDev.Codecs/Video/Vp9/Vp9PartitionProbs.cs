// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 default partition probability tables. Bit-exact with libvpx
// vp9/common/vp9_entropymode.c (vp9_kf_partition_probs and
// default_partition_probs).
//
// Both arrays are shaped [16 contexts][3 probs]. The 16 contexts
// decompose into 4 transform-size pairs (8x8 -> 4x4, 16x16 -> 8x8,
// 32x32 -> 16x16, 64x64 -> 32x32) times 4 above/left split states
// (a/l both unsplit, a split + l unsplit, a unsplit + l split,
// both split). The 3 probabilities feed slice 151's partition tree:
//   probs[0] = None vs (Horz/Vert/Split)
//   probs[1] = Horz vs (Vert/Split)
//   probs[2] = Vert vs Split
//
// Storage: flat byte[48] in row-major order. The flat index for
// (sizeIdx, splitState, probNode) is:
//   ((sizeIdx * 4 + splitState) * 3 + probNode)
// where sizeIdx = 0..3 covers 8x8/16x16/32x32/64x64 and splitState
// = 0..3 covers the four above/left combinations.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 default partition probability tables.</summary>
public static class Vp9PartitionProbs
{
    /// <summary>Number of partition contexts (libvpx PARTITION_CONTEXTS).</summary>
    public const int PartitionContexts = 16;

    /// <summary>Probabilities per context (libvpx PARTITION_TYPES - 1).</summary>
    public const int ProbsPerContext = 3;

    /// <summary>
    /// Keyframe partition probabilities (libvpx vp9_kf_partition_probs).
    /// 16 contexts x 3 probabilities = 48 bytes.
    /// </summary>
    public static readonly byte[] KfPartitionProbs = new byte[]
    {
        // 8x8 -> 4x4
        158, 97, 94,    // a/l both not split
        93,  24, 99,    // a split, l not split
        85,  119, 44,   // l split, a not split
        62,  59, 67,    // a/l both split

        // 16x16 -> 8x8
        149, 53, 53,
        94,  20, 48,
        83,  53, 24,
        52,  18, 18,

        // 32x32 -> 16x16
        150, 40, 39,
        78,  12, 26,
        67,  33, 11,
        24,  7,  5,

        // 64x64 -> 32x32
        174, 35, 49,
        68,  11, 27,
        57,  15, 9,
        12,  3,  3,
    };

    /// <summary>
    /// Inter-frame default partition probabilities (libvpx
    /// default_partition_probs). 16 contexts x 3 probabilities = 48 bytes.
    /// </summary>
    public static readonly byte[] DefaultPartitionProbs = new byte[]
    {
        // 8x8 -> 4x4
        199, 122, 141,
        147, 63,  159,
        148, 133, 118,
        121, 104, 114,

        // 16x16 -> 8x8
        174, 73, 87,
        92,  41, 83,
        82,  99, 50,
        53,  39, 39,

        // 32x32 -> 16x16
        177, 58, 59,
        68,  26, 63,
        52,  79, 25,
        17,  14, 12,

        // 64x64 -> 32x32
        222, 34, 30,
        72,  16, 44,
        58,  32, 12,
        10,  7,  6,
    };

    /// <summary>
    /// Compute the flat index into either probability array for the
    /// given (sizeIdx, splitState, probNode) tuple.
    /// </summary>
    /// <param name="sizeIdx">
    /// 0 = 8x8 -&gt; 4x4, 1 = 16x16 -&gt; 8x8, 2 = 32x32 -&gt; 16x16,
    /// 3 = 64x64 -&gt; 32x32.
    /// </param>
    /// <param name="splitState">
    /// 0 = a/l both unsplit, 1 = a split + l unsplit,
    /// 2 = a unsplit + l split, 3 = both split.
    /// </param>
    /// <param name="probNode">0..2 (the 3 partition tree nodes).</param>
    public static int Index(int sizeIdx, int splitState, int probNode)
    {
        if ((uint)sizeIdx >= 4) throw new ArgumentOutOfRangeException(nameof(sizeIdx));
        if ((uint)splitState >= 4) throw new ArgumentOutOfRangeException(nameof(splitState));
        if ((uint)probNode >= 3) throw new ArgumentOutOfRangeException(nameof(probNode));
        return (sizeIdx * 4 + splitState) * 3 + probNode;
    }

    /// <summary>
    /// Get the 3-element probability slice for a given (sizeIdx,
    /// splitState) tuple from the keyframe array. Pass the result
    /// directly to <see cref="Vp9PartitionTree.Decode"/>.
    /// </summary>
    public static ReadOnlySpan<byte> KeyframeProbs(int sizeIdx, int splitState)
    {
        int start = Index(sizeIdx, splitState, 0);
        return KfPartitionProbs.AsSpan(start, 3);
    }

    /// <summary>
    /// Get the 3-element probability slice for a given (sizeIdx,
    /// splitState) tuple from the default (inter-frame) array.
    /// </summary>
    public static ReadOnlySpan<byte> DefaultProbs(int sizeIdx, int splitState)
    {
        int start = Index(sizeIdx, splitState, 0);
        return DefaultPartitionProbs.AsSpan(start, 3);
    }
}
