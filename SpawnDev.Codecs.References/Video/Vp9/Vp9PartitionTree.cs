// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 partition tree topology + decoder. The partition tree controls
// how each superblock is recursively split during decode: at each
// level, one of four partition decisions is made, and SPLIT recurses
// into 4 sub-blocks.
//
// libvpx reference: vp9/common/vp9_entropymode.c vp9_partition_tree
// (6 entries) and vp9_entropy.h PARTITION_TYPES / PARTITION_NONE
// enum constants. VP9 spec sec 6.4.3 "Partition syntax".
//
// Tree layout (3 internal nodes, 4 leaves):
//   i=0  ROOT  ->  -None, 2
//   i=2  H_OR_V ->  -Horz, 4
//   i=4  V_OR_S ->  -Vert, -Split

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 partition decision at a given level (libvpx PARTITION_TYPES).</summary>
public enum Vp9PartitionType : byte
{
    /// <summary>No split: this block is decoded as one transform block.</summary>
    None = 0,
    /// <summary>Horizontal split: one wide block on top, one wide block on bottom.</summary>
    Horz = 1,
    /// <summary>Vertical split: one tall block on left, one tall block on right.</summary>
    Vert = 2,
    /// <summary>Split into 4 quarter-sized sub-blocks; recurse.</summary>
    Split = 3,
}

/// <summary>VP9 partition tree topology and decoder.</summary>
public static class Vp9PartitionTree
{
    /// <summary>
    /// libvpx vp9_partition_tree, 6 entries laid out as 3 internal
    /// nodes of (left, right) byte-index branches. Negative values
    /// are leaf partition types; non-negative values are byte indices
    /// of the next node within this same array.
    ///
    /// Tree structure:
    ///   i=0 ROOT     -> -None,  2 = H_OR_V_OR_S
    ///   i=2 H_OR_V_OR_S -> -Horz, 4 = V_OR_S
    ///   i=4 V_OR_S   -> -Vert,  -Split
    /// </summary>
    public static readonly sbyte[] Tree = new sbyte[]
    {
        -(sbyte)Vp9PartitionType.None,  2,
        -(sbyte)Vp9PartitionType.Horz,  4,
        -(sbyte)Vp9PartitionType.Vert, -(sbyte)Vp9PartitionType.Split,
    };

    /// <summary>
    /// Walk the partition tree given a 3-entry probability vector,
    /// using <paramref name="readBit"/> to read each decision. Same
    /// shape as <see cref="Vp9CoefTrees.DecodeConToken"/>.
    /// </summary>
    /// <param name="readBit">
    /// Caller-supplied bit reader (typically a closure over a
    /// <see cref="Vp9BoolDecoder"/>: <c>b =&gt; reader.Read(b)</c>).
    /// </param>
    /// <param name="probs">
    /// 3-entry probability vector indexed by tree pair index:
    /// probs[0] = ROOT (None vs not), probs[1] = H_OR_V_OR_S (Horz
    /// vs not), probs[2] = V_OR_S (Vert vs Split).
    /// </param>
    public static Vp9PartitionType Decode(Func<byte, int> readBit, ReadOnlySpan<byte> probs)
    {
        ArgumentNullException.ThrowIfNull(readBit);
        if (probs.Length < 3)
            throw new ArgumentException("probs must hold 3 entries for the partition tree", nameof(probs));

        int i = 0;
        while (true)
        {
            int bit = readBit(probs[i >> 1]);
            sbyte next = Tree[i + bit];
            if (next <= 0)
                return (Vp9PartitionType)(-next);
            i = next;
        }
    }
}
