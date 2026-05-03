// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 super-block / mi / pixel dimension math. Mirror of libvpx
// vp9/common/vp9_common.h ALIGN_POWER_OF_TWO + the MI_SIZE /
// MI_BLOCK_SIZE constants in vp9_blockd.h.
//
// VP9 layout primer:
//   - Pixel: smallest addressable unit.
//   - 4x4 b unit: 1 transform-block side at the smallest tx size.
//   - 8x8 mi unit: mode-info grid cell. 1 mi = 8 pixels per side.
//   - 64x64 super-block (SB): the largest partition tree root.
//     1 SB = 8 mi = 64 pixels per side.
//
// Tile widths / row scan boundaries align to SB.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 super-block / mi / pixel dimension math helpers.</summary>
public static class Vp9SuperblockMath
{
    /// <summary>libvpx <c>MI_SIZE_LOG2</c> = 3 (8 pixels per mi unit).</summary>
    public const int MiSizeLog2 = 3;

    /// <summary>libvpx <c>MI_SIZE</c> = 8.</summary>
    public const int MiSize = 1 << MiSizeLog2;

    /// <summary>libvpx <c>MI_BLOCK_SIZE_LOG2</c> = 3 (8 mi units per SB side).</summary>
    public const int MiPerSbLog2 = 3;

    /// <summary>libvpx <c>MI_BLOCK_SIZE</c> = 8.</summary>
    public const int MiPerSb = 1 << MiPerSbLog2;

    /// <summary>SB size in pixels: 64.</summary>
    public const int SbSizeLog2 = MiSizeLog2 + MiPerSbLog2;

    /// <summary>SB side in pixels: 64.</summary>
    public const int SbSize = 1 << SbSizeLog2;

    /// <summary>Convert pixels to mi units (truncated divide by 8).</summary>
    public static int PixelToMi(int px) => px >> MiSizeLog2;

    /// <summary>Convert mi units to pixels (multiply by 8).</summary>
    public static int MiToPixel(int mi) => mi << MiSizeLog2;

    /// <summary>Convert mi units to SB units (truncated divide by 8).</summary>
    public static int MiToSb(int mi) => mi >> MiPerSbLog2;

    /// <summary>
    /// Round <paramref name="px"/> up to the next SB-aligned pixel
    /// count. libvpx <c>ALIGN_POWER_OF_TWO(px, SB_LOG2)</c>.
    /// </summary>
    public static int AlignToSbPixels(int px) =>
        (px + SbSize - 1) & ~(SbSize - 1);

    /// <summary>
    /// Round <paramref name="mi"/> up to the next SB-aligned mi count.
    /// libvpx <c>mi_cols_aligned_to_sb</c>.
    /// </summary>
    public static int AlignToSbMi(int mi) =>
        (mi + MiPerSb - 1) & ~(MiPerSb - 1);

    /// <summary>
    /// Number of SBs needed to cover <paramref name="px"/> pixels,
    /// rounding up.
    /// </summary>
    public static int SbsForPixels(int px) =>
        (px + SbSize - 1) >> SbSizeLog2;

    /// <summary>
    /// Number of SBs needed to cover <paramref name="mi"/> mi units,
    /// rounding up.
    /// </summary>
    public static int SbsForMi(int mi) =>
        (mi + MiPerSb - 1) >> MiPerSbLog2;
}
