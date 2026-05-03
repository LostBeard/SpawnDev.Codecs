// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 size_group_lookup: maps a block size to a 4-way size group
// index used as context for the y_mode and uv_mode probability
// tables. Block sizes are bucketed by area (log2 area / 2).
//
// libvpx reference: vp9/common/vp9_blockd.h size_group_lookup
// (13 entries) and BLOCK_SIZE_GROUPS = 4.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// Block-size to 4-way size group mapping (libvpx
/// <c>size_group_lookup</c>). Used to index intra mode probability
/// tables (Y / UV) by block area bucket.
/// </summary>
public static class Vp9SizeGroupLookup
{
    /// <summary>libvpx <c>BLOCK_SIZE_GROUPS</c>.</summary>
    public const int Groups = 4;

    /// <summary>
    /// 13-entry lookup. Group 0 covers 4x4..8x4, group 1 covers
    /// 8x8..16x8, group 2 covers 16x16..32x16, group 3 covers
    /// 32x32..64x64.
    /// </summary>
    public static readonly byte[] Lookup = new byte[Vp9BlockSizes.Count]
    {
        0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3,
    };

    /// <summary>Look up the 0..3 size group index for a block size.</summary>
    public static int ForBlockSize(Vp9BlockSize size)
    {
        int idx = (int)size;
        if ((uint)idx >= (uint)Vp9BlockSizes.Count)
            throw new ArgumentOutOfRangeException(nameof(size), size,
                "VP9 block size index out of range.");
        return Lookup[idx];
    }
}
