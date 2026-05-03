// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 motion vector joint-type tree. The 4-way joint tells the
// decoder whether (h, v) MV components are both zero, only-h
// nonzero, only-v nonzero, or both nonzero. The tree shape:
//
//   ROOT  : -Zero,  2
//   i=2   : -Hnzvz, 4
//   i=4   : -Hzvnz, -Hnzvnz
//
// libvpx reference: vp9/common/vp9_entropymv.c vp9_mv_joint_tree.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 motion vector joint type (libvpx MV_JOINT_TYPE).</summary>
public enum Vp9MvJointType : byte
{
    /// <summary>(h, v) = (zero, zero) - no movement.</summary>
    Zero = 0,
    /// <summary>(h, v) = (nonzero, zero) - horizontal-only movement.</summary>
    Hnzvz = 1,
    /// <summary>(h, v) = (zero, nonzero) - vertical-only movement.</summary>
    Hzvnz = 2,
    /// <summary>(h, v) = (nonzero, nonzero) - diagonal movement.</summary>
    Hnzvnz = 3,
}

/// <summary>VP9 motion vector joint type tree topology and decoder.</summary>
public static class Vp9MvJointTree
{
    /// <summary>libvpx <c>MV_JOINTS</c>.</summary>
    public const int Joints = 4;

    /// <summary>
    /// libvpx <c>vp9_mv_joint_tree</c>, 6 entries laid out as 3
    /// internal nodes of (left, right) byte-index branches. Negative
    /// values are leaf joint types; non-negative values are byte
    /// indices of the next node within this same array.
    /// </summary>
    public static readonly sbyte[] Tree = new sbyte[]
    {
        -(sbyte)Vp9MvJointType.Zero,    2,
        -(sbyte)Vp9MvJointType.Hnzvz,   4,
        -(sbyte)Vp9MvJointType.Hzvnz,  -(sbyte)Vp9MvJointType.Hnzvnz,
    };

    /// <summary>
    /// Walk the joint-type tree given a 3-entry probability vector,
    /// using <paramref name="readBit"/> to read each decision. Same
    /// shape as <see cref="Vp9PartitionTree.Decode"/>.
    /// </summary>
    /// <param name="readBit">
    /// Caller-supplied bit reader (typically a closure over a
    /// <see cref="Vp9BoolDecoder"/>: <c>p =&gt; reader.Read(p)</c>).
    /// </param>
    /// <param name="probs">
    /// 3-entry probability vector (libvpx <c>nmv_context.joints</c>:
    /// <see cref="Vp9MvProbs.Joints"/>).
    /// </param>
    public static Vp9MvJointType Decode(Func<byte, int> readBit, ReadOnlySpan<byte> probs)
    {
        ArgumentNullException.ThrowIfNull(readBit);
        if (probs.Length < 3)
            throw new ArgumentException("probs must hold 3 entries for the MV joint tree", nameof(probs));

        int i = 0;
        while (true)
        {
            int bit = readBit(probs[i >> 1]);
            sbyte next = Tree[i + bit];
            if (next <= 0)
                return (Vp9MvJointType)(-next);
            i = next;
        }
    }
}
