// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 block size enum + dimension lookup tables. Mirror of libvpx
// vp9/common/vp9_enums.h BLOCK_SIZE plus the various dimension
// lookups in vp9/common/vp9_blockd.h:
//
//   num_pels_log2_lookup
//   num_8x8_blocks_wide_lookup / num_8x8_blocks_high_lookup
//   num_4x4_blocks_wide_lookup / num_4x4_blocks_high_lookup
//   mi_width_log2_lookup / mi_height_log2_lookup (8x8 mi units)
//   b_width_log2_lookup / b_height_log2_lookup (4x4 b units)
//
// VP9 has 13 block sizes covering 4x4 -> 64x64 with non-square
// variants. Each dimension is a power of 2 (4, 8, 16, 32, 64).
//
// Internal arithmetic uses two units interchangeably:
//   "mi" = 8x8 mode info units (libvpx MI_BLOCK_SIZE = 8 px)
//   "b"  = 4x4 transform-block units
// A 64x64 SB is 8x8 mi or 16x16 b.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 block size (libvpx BLOCK_SIZE).</summary>
public enum Vp9BlockSize : byte
{
    /// <summary>4x4 pixels.</summary>
    Block4x4 = 0,
    /// <summary>4x8 pixels.</summary>
    Block4x8 = 1,
    /// <summary>8x4 pixels.</summary>
    Block8x4 = 2,
    /// <summary>8x8 pixels.</summary>
    Block8x8 = 3,
    /// <summary>8x16 pixels.</summary>
    Block8x16 = 4,
    /// <summary>16x8 pixels.</summary>
    Block16x8 = 5,
    /// <summary>16x16 pixels.</summary>
    Block16x16 = 6,
    /// <summary>16x32 pixels.</summary>
    Block16x32 = 7,
    /// <summary>32x16 pixels.</summary>
    Block32x16 = 8,
    /// <summary>32x32 pixels.</summary>
    Block32x32 = 9,
    /// <summary>32x64 pixels.</summary>
    Block32x64 = 10,
    /// <summary>64x32 pixels.</summary>
    Block64x32 = 11,
    /// <summary>64x64 pixels (super-block).</summary>
    Block64x64 = 12,
}

/// <summary>VP9 block-size dimension lookups.</summary>
public static class Vp9BlockSizes
{
    /// <summary>libvpx <c>BLOCK_SIZES</c>.</summary>
    public const int Count = 13;

    /// <summary>Largest block size (64x64 super-block).</summary>
    public const Vp9BlockSize Largest = Vp9BlockSize.Block64x64;

    /// <summary>Block width in pixels, indexed by Vp9BlockSize.</summary>
    public static readonly int[] WidthPx = new int[Count]
    {
        4,  4,  8,  8,  8, 16, 16, 16, 32, 32, 32, 64, 64,
    };

    /// <summary>Block height in pixels, indexed by Vp9BlockSize.</summary>
    public static readonly int[] HeightPx = new int[Count]
    {
        4,  8,  4,  8, 16,  8, 16, 32, 16, 32, 64, 32, 64,
    };

    /// <summary>libvpx <c>num_8x8_blocks_wide_lookup</c>.</summary>
    public static readonly int[] Num8x8Wide = new int[Count]
    {
        1, 1, 1, 1, 1, 2, 2, 2, 4, 4, 4, 8, 8,
    };

    /// <summary>libvpx <c>num_8x8_blocks_high_lookup</c>.</summary>
    public static readonly int[] Num8x8High = new int[Count]
    {
        1, 1, 1, 1, 2, 1, 2, 4, 2, 4, 8, 4, 8,
    };

    /// <summary>libvpx <c>num_4x4_blocks_wide_lookup</c>.</summary>
    public static readonly int[] Num4x4Wide = new int[Count]
    {
        1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16,
    };

    /// <summary>libvpx <c>num_4x4_blocks_high_lookup</c>.</summary>
    public static readonly int[] Num4x4High = new int[Count]
    {
        1, 2, 1, 2, 4, 2, 4, 8, 4, 8, 16, 8, 16,
    };

    /// <summary>
    /// libvpx <c>mi_width_log2_lookup</c>: log2 of the block width in
    /// 8x8 mode info units.
    /// </summary>
    public static readonly int[] MiWidthLog2 = new int[Count]
    {
        0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3,
    };

    /// <summary>libvpx <c>mi_height_log2_lookup</c>.</summary>
    public static readonly int[] MiHeightLog2 = new int[Count]
    {
        0, 0, 0, 0, 1, 0, 1, 2, 1, 2, 3, 2, 3,
    };

    /// <summary>libvpx <c>num_pels_log2_lookup</c>: log2(num_pels).</summary>
    public static readonly int[] NumPelsLog2 = new int[Count]
    {
        4, 5, 5, 6, 7, 7, 8, 9, 9, 10, 11, 11, 12,
    };

    /// <summary>Block width in pixels.</summary>
    public static int Width(Vp9BlockSize size) => WidthPx[(int)size];

    /// <summary>Block height in pixels.</summary>
    public static int Height(Vp9BlockSize size) => HeightPx[(int)size];

    /// <summary>Block width in 8x8 mode info units.</summary>
    public static int MiWidth(Vp9BlockSize size) => Num8x8Wide[(int)size];

    /// <summary>Block height in 8x8 mode info units.</summary>
    public static int MiHeight(Vp9BlockSize size) => Num8x8High[(int)size];

    /// <summary>Block width in 4x4 transform-block units.</summary>
    public static int B4x4Width(Vp9BlockSize size) => Num4x4Wide[(int)size];

    /// <summary>Block height in 4x4 transform-block units.</summary>
    public static int B4x4Height(Vp9BlockSize size) => Num4x4High[(int)size];

    /// <summary>True if the block is square (width == height).</summary>
    public static bool IsSquare(Vp9BlockSize size) =>
        WidthPx[(int)size] == HeightPx[(int)size];
}
