// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 MV reference candidate scanner. Walks the 8 neighbor
// positions per block size from <see cref="Vp9MvRefBlockOffsets"/>,
// looks up each position via a caller-provided mode-info accessor,
// and populates <see cref="Vp9MvRefCandidatesByRef"/> with each
// neighbor's MV(s) bucketed by reference frame.
//
// libvpx reference: the per-neighbor inner loop of
// vp9/common/vp9_mvref_common.c vp9_find_mv_refs_idx. The full
// libvpx routine also handles temporal MV reference, mode-context
// counting, and additional dedup logic - those are downstream
// slices.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 MV reference candidate scanner.</summary>
public static class Vp9MvRefScanner
{
    /// <summary>
    /// Scan the 8 neighbor positions for the given block and populate
    /// <paramref name="result"/> with MVs from each non-intra,
    /// in-bounds neighbor. The result is cleared at start.
    /// </summary>
    /// <param name="curMiRow">Current block top-left mi row.</param>
    /// <param name="curMiCol">Current block top-left mi col.</param>
    /// <param name="blockSize">Current block size.</param>
    /// <param name="frameMiRows">Frame mi row count (for bounds check).</param>
    /// <param name="frameMiCols">Frame mi col count (for bounds check).</param>
    /// <param name="miAt">
    /// Mode-info lookup callback. Returns the
    /// <see cref="Vp9MbModeInfo"/> at (mi_row, mi_col) or null if no
    /// mode info has been written yet (above the frame's "valid"
    /// region during the early SB rows).
    /// </param>
    /// <param name="result">Output candidates store, cleared at start.</param>
    public static void ScanCandidates(
        int curMiRow, int curMiCol,
        Vp9BlockSize blockSize,
        int frameMiRows, int frameMiCols,
        Func<int, int, Vp9MbModeInfo?> miAt,
        Vp9MvRefCandidatesByRef result)
    {
        ArgumentNullException.ThrowIfNull(miAt);
        ArgumentNullException.ThrowIfNull(result);
        result.Clear();

        for (int n = 0; n < Vp9MvRefBlockOffsets.Neighbours; n++)
        {
            var (rowOffset, colOffset) = Vp9MvRefBlockOffsets.GetOffset(blockSize, n);
            var pos = Vp9NeighborMi.GetNeighbor(
                curMiRow, curMiCol, rowOffset, colOffset,
                frameMiRows, frameMiCols);
            if (pos is null) continue;

            var neighbor = miAt(pos.Value.Row, pos.Value.Col);
            if (neighbor is null) continue;
            if (neighbor.IsIntra) continue;

            // Primary reference contributes its MV.
            if (Vp9MvReferenceFrames.IsInter(neighbor.PrimaryRefFrame))
                result.ForRef(neighbor.PrimaryRefFrame).TryAdd(neighbor.PrimaryMv);

            // Compound reference (when present) contributes its MV.
            if (neighbor.IsCompound && neighbor.CompoundRefFrame.HasValue)
                result.ForRef(neighbor.CompoundRefFrame.Value).TryAdd(neighbor.CompoundMv);
        }
    }
}
