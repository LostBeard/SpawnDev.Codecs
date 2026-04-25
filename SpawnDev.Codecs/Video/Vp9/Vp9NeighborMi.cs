// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 neighbor mi-position helper. The MV reference candidate scan
// applies offsets from <see cref="Vp9MvRefBlockOffsets"/> to the
// current block's (mi_row, mi_col) to look up neighbor blocks. Some
// of those offsets will land outside the frame at top/left edges
// (rows -2 or -3, cols -3) or past the right/bottom edges - those
// neighbors must be skipped, not crash.
//
// libvpx reference: the bounds checks scattered throughout
// vp9/common/vp9_mvref_common.c that test whether
// xd->mb_to_*_edge permits the candidate position.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 neighbor mi-position bounds-check helper.</summary>
public static class Vp9NeighborMi
{
    /// <summary>
    /// Compute the neighbor mi position given a current block's
    /// (mi_row, mi_col), an offset, and the frame's mi bounds.
    /// Returns the (row, col) tuple if the offset lands inside the
    /// frame, or null if it goes out of bounds (above row 0, left of
    /// col 0, or past the frame's mi rows / cols).
    /// </summary>
    public static (int Row, int Col)? GetNeighbor(
        int currentMiRow, int currentMiCol,
        int rowOffset, int colOffset,
        int frameMiRows, int frameMiCols)
    {
        if (frameMiRows <= 0)
            throw new ArgumentOutOfRangeException(nameof(frameMiRows), frameMiRows, "frameMiRows must be > 0.");
        if (frameMiCols <= 0)
            throw new ArgumentOutOfRangeException(nameof(frameMiCols), frameMiCols, "frameMiCols must be > 0.");

        int r = currentMiRow + rowOffset;
        int c = currentMiCol + colOffset;
        if (r < 0 || r >= frameMiRows) return null;
        if (c < 0 || c >= frameMiCols) return null;
        return (r, c);
    }

    /// <summary>
    /// True when the offset lands inside the frame's mi bounds.
    /// Convenience wrapper over <see cref="GetNeighbor"/>.
    /// </summary>
    public static bool IsInBounds(
        int currentMiRow, int currentMiCol,
        int rowOffset, int colOffset,
        int frameMiRows, int frameMiCols)
    {
        return GetNeighbor(currentMiRow, currentMiCol, rowOffset, colOffset,
            frameMiRows, frameMiCols).HasValue;
    }
}
