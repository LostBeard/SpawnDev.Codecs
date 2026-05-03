// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 MV reference candidate scan offsets. The candidate generator
// (libvpx vp9_find_mv_refs_idx) scans 8 specific neighbor positions
// per block size, looking for usable MVs to seed the
// NewMV / NearestMV / NearMV reference list.
//
// libvpx reference: vp9/common/vp9_mvref_common.h mv_ref_blocks.
//
// Table shape: BLOCK_SIZES (13) x MVREF_NEIGHBOURS (8) of (row, col)
// offsets in mi units relative to the current block's top-left mi
// position. Negative row = above; negative col = left.
//
// Storage uses sbyte since all values fit in [-3, 6]. Flat row-major
// layout: position (bs, n) = Lookup[bs * 16 + n*2 .. n*2+1].

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 MV reference candidate scan offsets.</summary>
public static class Vp9MvRefBlockOffsets
{
    /// <summary>libvpx <c>MVREF_NEIGHBOURS</c>.</summary>
    public const int Neighbours = 8;

    /// <summary>
    /// libvpx <c>mv_ref_blocks</c>. Indexed [block_size, neighbor]
    /// where each neighbor is a (row, col) offset in mi units.
    /// </summary>
    public static readonly sbyte[,,] Lookup = new sbyte[Vp9BlockSizes.Count, Neighbours, 2]
    {
        // Block4x4
        { { -1,  0 }, {  0, -1 }, { -1, -1 }, { -2,  0 },
          {  0, -2 }, { -2, -1 }, { -1, -2 }, { -2, -2 } },
        // Block4x8
        { { -1,  0 }, {  0, -1 }, { -1, -1 }, { -2,  0 },
          {  0, -2 }, { -2, -1 }, { -1, -2 }, { -2, -2 } },
        // Block8x4
        { { -1,  0 }, {  0, -1 }, { -1, -1 }, { -2,  0 },
          {  0, -2 }, { -2, -1 }, { -1, -2 }, { -2, -2 } },
        // Block8x8
        { { -1,  0 }, {  0, -1 }, { -1, -1 }, { -2,  0 },
          {  0, -2 }, { -2, -1 }, { -1, -2 }, { -2, -2 } },
        // Block8x16
        { {  0, -1 }, { -1,  0 }, {  1, -1 }, { -1, -1 },
          {  0, -2 }, { -2,  0 }, { -2, -1 }, { -1, -2 } },
        // Block16x8
        { { -1,  0 }, {  0, -1 }, { -1,  1 }, { -1, -1 },
          { -2,  0 }, {  0, -2 }, { -1, -2 }, { -2, -1 } },
        // Block16x16
        { { -1,  0 }, {  0, -1 }, { -1,  1 }, {  1, -1 },
          { -1, -1 }, { -3,  0 }, {  0, -3 }, { -3, -3 } },
        // Block16x32
        { {  0, -1 }, { -1,  0 }, {  2, -1 }, { -1, -1 },
          { -1,  1 }, {  0, -3 }, { -3,  0 }, { -3, -3 } },
        // Block32x16
        { { -1,  0 }, {  0, -1 }, { -1,  2 }, { -1, -1 },
          {  1, -1 }, { -3,  0 }, {  0, -3 }, { -3, -3 } },
        // Block32x32
        { { -1,  1 }, {  1, -1 }, { -1,  2 }, {  2, -1 },
          { -1, -1 }, { -3,  0 }, {  0, -3 }, { -3, -3 } },
        // Block32x64
        { {  0, -1 }, { -1,  0 }, {  4, -1 }, { -1,  2 },
          { -1, -1 }, {  0, -3 }, { -3,  0 }, {  2, -1 } },
        // Block64x32
        { { -1,  0 }, {  0, -1 }, { -1,  4 }, {  2, -1 },
          { -1, -1 }, { -3,  0 }, {  0, -3 }, { -1,  2 } },
        // Block64x64
        { { -1,  3 }, {  3, -1 }, { -1,  4 }, {  4, -1 },
          { -1, -1 }, { -1,  0 }, {  0, -1 }, { -1,  6 } },
    };

    /// <summary>
    /// Get the (row, col) offset for the n-th neighbor of a block of
    /// size <paramref name="blockSize"/>. Both indices bounds-checked.
    /// </summary>
    public static (sbyte Row, sbyte Col) GetOffset(Vp9BlockSize blockSize, int neighbor)
    {
        int b = (int)blockSize;
        if ((uint)b >= (uint)Vp9BlockSizes.Count)
            throw new ArgumentOutOfRangeException(nameof(blockSize), blockSize,
                "block size index out of range.");
        if ((uint)neighbor >= (uint)Neighbours)
            throw new ArgumentOutOfRangeException(nameof(neighbor), neighbor,
                "neighbor must be in [0, 8).");
        return (Lookup[b, neighbor, 0], Lookup[b, neighbor, 1]);
    }
}
