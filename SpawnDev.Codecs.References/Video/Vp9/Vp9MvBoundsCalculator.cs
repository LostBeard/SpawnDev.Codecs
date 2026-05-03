// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 MV bounds calculator. Computes the per-block MV clamping
// envelope so the reference window stays within the (padded)
// reference frame. Mirror of libvpx vp9/common/vp9_mvref_common.h
// clamp_mv2 + the LEFT_TOP_MARGIN / RIGHT_BOTTOM_MARGIN constants
// from vp9_blockd.h.
//
// Block edges in 1/8-pel units:
//   mb_to_top_edge    = -(mi_row * MI_SIZE) << 3
//   mb_to_bottom_edge = (mi_rows - bh - mi_row) * MI_SIZE << 3
//   mb_to_left_edge   = -(mi_col * MI_SIZE) << 3
//   mb_to_right_edge  = (mi_cols - bw - mi_col) * MI_SIZE << 3
//
// MI_SIZE = 8 px, and the &lt;&lt; 3 shifts pixels to 1/8-pel.
//
// Then clamp_mv2 expands the envelope by LEFT_TOP_MARGIN /
// RIGHT_BOTTOM_MARGIN (= (192 - 4) &lt;&lt; 3 = 1504) which is the
// frame border padding minus the interpolation extension.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 MV bounds (in 1/8-pel units, matching Vp9Mv storage).</summary>
public readonly record struct Vp9MvBounds(int MinRow, int MaxRow, int MinCol, int MaxCol);

/// <summary>VP9 MV bounds calculator.</summary>
public static class Vp9MvBoundsCalculator
{
    /// <summary>libvpx <c>MI_SIZE</c> = 8 pixels per mi unit.</summary>
    public const int MiSize = 8;

    /// <summary>libvpx <c>VP9_ENC_BORDER_IN_PIXELS</c>.</summary>
    public const int EncBorderInPixels = 192;

    /// <summary>libvpx <c>VP9_INTERP_EXTEND</c>.</summary>
    public const int InterpExtend = 4;

    /// <summary>
    /// libvpx <c>LEFT_TOP_MARGIN</c> = (192 - 4) &lt;&lt; 3 = 1504 (in 1/8-pel
    /// units). Distance the MV may extend beyond the frame's left /
    /// top edge.
    /// </summary>
    public const int LeftTopMargin = (EncBorderInPixels - InterpExtend) << 3;

    /// <summary>
    /// libvpx <c>RIGHT_BOTTOM_MARGIN</c> = (192 - 4) &lt;&lt; 3 = 1504. Distance
    /// the MV may extend beyond the right / bottom edge.
    /// </summary>
    public const int RightBottomMargin = (EncBorderInPixels - InterpExtend) << 3;

    /// <summary>
    /// Compute MV clamp bounds for a block of size <paramref name="blockSize"/>
    /// at frame mi position (<paramref name="miRow"/>,
    /// <paramref name="miCol"/>) within a frame of mi dimensions
    /// (<paramref name="frameMiRows"/>, <paramref name="frameMiCols"/>).
    ///
    /// Mirror of libvpx <c>set_mb_offsets</c> + <c>clamp_mv2</c>.
    /// </summary>
    public static Vp9MvBounds Compute(
        int miRow, int miCol,
        Vp9BlockSize blockSize,
        int frameMiRows, int frameMiCols)
    {
        if (miRow < 0)
            throw new ArgumentOutOfRangeException(nameof(miRow), miRow, "miRow must be >= 0.");
        if (miCol < 0)
            throw new ArgumentOutOfRangeException(nameof(miCol), miCol, "miCol must be >= 0.");
        if (frameMiRows <= 0)
            throw new ArgumentOutOfRangeException(nameof(frameMiRows), frameMiRows, "frameMiRows must be > 0.");
        if (frameMiCols <= 0)
            throw new ArgumentOutOfRangeException(nameof(frameMiCols), frameMiCols, "frameMiCols must be > 0.");

        int bw = Vp9BlockSizes.MiWidth(blockSize);
        int bh = Vp9BlockSizes.MiHeight(blockSize);

        int mbToTopEdge = -((miRow * MiSize) << 3);
        int mbToBottomEdge = (frameMiRows - bh - miRow) * MiSize << 3;
        int mbToLeftEdge = -((miCol * MiSize) << 3);
        int mbToRightEdge = (frameMiCols - bw - miCol) * MiSize << 3;

        return new Vp9MvBounds(
            MinRow: mbToTopEdge - LeftTopMargin,
            MaxRow: mbToBottomEdge + RightBottomMargin,
            MinCol: mbToLeftEdge - LeftTopMargin,
            MaxCol: mbToRightEdge + RightBottomMargin);
    }
}
