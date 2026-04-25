// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 segment ID tree. 8 leaves (segment ids 0..7), 7 internal
// nodes. Balanced binary tree structure used to decode the
// per-block segment id when segmentation is enabled and
// update_map is set.
//
// libvpx reference: vp9/common/vp9_entropymode.c vp9_segment_tree
// (14 entries) and vp9/decoder/vp9_decodemv.c read_segment_id.
//
// Tree shape:
//   i=0   : 2,  4
//   i=2   : 6,  8
//   i=4   : 10, 12
//   i=6   : -0, -1   (leaves: segment 0, segment 1)
//   i=8   : -2, -3
//   i=10  : -4, -5
//   i=12  : -6, -7

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 segment ID tree topology and decoder.</summary>
public static class Vp9SegmentTree
{
    /// <summary>libvpx <c>MAX_SEGMENTS</c>.</summary>
    public const int MaxSegments = Vp9SegmentationParams.MaxSegments;

    /// <summary>libvpx <c>SEG_TREE_PROBS</c>.</summary>
    public const int TreeProbs = Vp9SegmentationParams.TreeProbs;

    /// <summary>
    /// libvpx <c>vp9_segment_tree</c>, 14 entries (7 internal nodes
    /// x 2 branches). Negative values are leaf segment ids; non-
    /// negative values are byte indices of the next node within this
    /// same array.
    /// </summary>
    public static readonly sbyte[] Tree = new sbyte[]
    {
        2, 4,           // i=0  ROOT
        6, 8,           // i=2
        10, 12,         // i=4
        -0, -1,         // i=6  leaves seg 0/1
        -2, -3,         // i=8  leaves seg 2/3
        -4, -5,         // i=10 leaves seg 4/5
        -6, -7,         // i=12 leaves seg 6/7
    };

    /// <summary>
    /// Walk the segment tree using a 7-entry probability vector
    /// (libvpx <see cref="Vp9SegmentationParams.TreeProbsArray"/>).
    /// Returns 0..7 segment id.
    /// </summary>
    public static int Decode(Func<byte, int> readBit, ReadOnlySpan<byte> probs)
    {
        ArgumentNullException.ThrowIfNull(readBit);
        if (probs.Length < TreeProbs)
            throw new ArgumentException(
                $"probs must hold {TreeProbs} entries for the segment tree", nameof(probs));

        int i = 0;
        while (true)
        {
            int bit = readBit(probs[i >> 1]);
            sbyte next = Tree[i + bit];
            if (next <= 0)
                return -next;
            i = next;
        }
    }
}
