// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 subsize_lookup: given a parent block size and a partition
// decision, return the child block size. Used by the partition
// tree traversal in block decode (sec 6.4.3).
//
// libvpx reference: vp9/common/vp9_common_data.c subsize_lookup
// 4x13 table indexed by [PARTITION_TYPE][BLOCK_SIZE].
//
// Many (parent, partition) combinations are illegal in the VP9
// bitstream - the bitstream encodes partition decisions in a
// constrained way that prevents the decoder from ever needing
// these entries. They are represented by Vp9BlockSize.Invalid
// here, matching libvpx BLOCK_INVALID. Reaching one indicates
// a corrupted bitstream.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// (block size, partition type) -> child block size lookup.
/// </summary>
public static class Vp9SubsizeLookup
{
    /// <summary>
    /// 4x13 table indexed by <c>[PARTITION_TYPE][BLOCK_SIZE]</c>.
    /// Returns <see cref="Vp9BlockSize.Invalid"/> for combinations
    /// that the bitstream is forbidden from emitting.
    /// </summary>
    public static readonly Vp9BlockSize[,] Lookup = new Vp9BlockSize[4, Vp9BlockSizes.Count]
    {
        // PARTITION_NONE: same size as parent.
        {
            Vp9BlockSize.Block4x4,   Vp9BlockSize.Block4x8,   Vp9BlockSize.Block8x4,
            Vp9BlockSize.Block8x8,   Vp9BlockSize.Block8x16,  Vp9BlockSize.Block16x8,
            Vp9BlockSize.Block16x16, Vp9BlockSize.Block16x32, Vp9BlockSize.Block32x16,
            Vp9BlockSize.Block32x32, Vp9BlockSize.Block32x64, Vp9BlockSize.Block64x32,
            Vp9BlockSize.Block64x64,
        },
        // PARTITION_HORZ: only legal for square parents 8x8 and up.
        {
            Vp9BlockSize.Invalid,    Vp9BlockSize.Invalid,    Vp9BlockSize.Invalid,
            Vp9BlockSize.Block8x4,   Vp9BlockSize.Invalid,    Vp9BlockSize.Invalid,
            Vp9BlockSize.Block16x8,  Vp9BlockSize.Invalid,    Vp9BlockSize.Invalid,
            Vp9BlockSize.Block32x16, Vp9BlockSize.Invalid,    Vp9BlockSize.Invalid,
            Vp9BlockSize.Block64x32,
        },
        // PARTITION_VERT: only legal for square parents 8x8 and up.
        {
            Vp9BlockSize.Invalid,    Vp9BlockSize.Invalid,    Vp9BlockSize.Invalid,
            Vp9BlockSize.Block4x8,   Vp9BlockSize.Invalid,    Vp9BlockSize.Invalid,
            Vp9BlockSize.Block8x16,  Vp9BlockSize.Invalid,    Vp9BlockSize.Invalid,
            Vp9BlockSize.Block16x32, Vp9BlockSize.Invalid,    Vp9BlockSize.Invalid,
            Vp9BlockSize.Block32x64,
        },
        // PARTITION_SPLIT: only legal for square parents 8x8 and up.
        {
            Vp9BlockSize.Invalid,    Vp9BlockSize.Invalid,    Vp9BlockSize.Invalid,
            Vp9BlockSize.Block4x4,   Vp9BlockSize.Invalid,    Vp9BlockSize.Invalid,
            Vp9BlockSize.Block8x8,   Vp9BlockSize.Invalid,    Vp9BlockSize.Invalid,
            Vp9BlockSize.Block16x16, Vp9BlockSize.Invalid,    Vp9BlockSize.Invalid,
            Vp9BlockSize.Block32x32,
        },
    };

    /// <summary>
    /// Look up the child block size for a (parent, partition) pair.
    /// Returns <see cref="Vp9BlockSize.Invalid"/> if the combination
    /// is forbidden by the VP9 bitstream rules.
    /// </summary>
    public static Vp9BlockSize Subsize(Vp9BlockSize parent, Vp9PartitionType partition)
    {
        int p = (int)partition;
        int b = (int)parent;
        if ((uint)p >= 4u)
            throw new ArgumentOutOfRangeException(nameof(partition), partition,
                "VP9 partition type must be one of None/Horz/Vert/Split.");
        if ((uint)b >= (uint)Vp9BlockSizes.Count)
            throw new ArgumentOutOfRangeException(nameof(parent), parent,
                "VP9 block size index out of range.");
        return Lookup[p, b];
    }
}
