// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP8 per-macroblock entropy contexts. Tracks the "had any non-zero
// coefficient" flag for each 4x4 sub-block of the previously-decoded
// MBs above and to the left of the current MB. Used by
// Vp8CoefBlockDecoder.Decode to seed the initial prev_coef_context (0,
// 1, or 2) for each block.
//
// Layout per MB (libvpx ENTROPY_CONTEXT_PLANES = 9 chars):
//   slots 0..3 = Y4 above contexts (one per 4x4 column of the MB's Y plane)
//   slot 4..5  = U above contexts (one per 4x4 column of the MB's U plane)
//   slot 6..7  = V above contexts (one per 4x4 column of the MB's V plane)
//   slot 8     = Y2 above context (single MB-level slot for the Y2 block)
//
// The "above" array is per-MB-COLUMN (one entry for each MB column in
// the frame), and the "left" array is per-MB-ROW (one entry kept for
// the current row). Both are reset at frame boundaries.
//
// Reference: libvpx vp8/common/blockd.h (ENTROPY_CONTEXT_PLANES) +
// vp8/decoder/decodeframe.c decode_mb_rows (the
// `xd->above_context = pc->above_context;
//  memset(xd->left_context, 0, sizeof(ENTROPY_CONTEXT_PLANES));`
//  pattern).

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>VP8 per-MB entropy contexts (ENTROPY_CONTEXT_PLANES).</summary>
public sealed class Vp8EntropyContexts
{
    /// <summary>Number of byte slots per macroblock entropy-context plane (libvpx).</summary>
    public const int PlanesPerMb = 9;

    /// <summary>Above context: <see cref="PlanesPerMb"/> bytes per MB column.</summary>
    public byte[] Above { get; }
    /// <summary>Left context: <see cref="PlanesPerMb"/> bytes for the current row.</summary>
    public byte[] Left { get; }

    /// <summary>Number of MB columns in the frame.</summary>
    public int MbCols { get; }

    /// <summary>Allocate above/left contexts for a frame with <paramref name="mbCols"/> MB columns.</summary>
    public Vp8EntropyContexts(int mbCols)
    {
        if (mbCols <= 0) throw new ArgumentOutOfRangeException(nameof(mbCols));
        MbCols = mbCols;
        Above = new byte[mbCols * PlanesPerMb];
        Left = new byte[PlanesPerMb];
    }

    /// <summary>Clear all contexts to zero (start of frame).</summary>
    public void ClearAll()
    {
        Array.Clear(Above);
        Array.Clear(Left);
    }

    /// <summary>Clear the left context (start of MB row).</summary>
    public void ClearLeft()
    {
        Array.Clear(Left);
    }

    /// <summary>Get a span over the above context for MB column <paramref name="mbCol"/>.</summary>
    public Span<byte> GetAbove(int mbCol)
    {
        if ((uint)mbCol >= (uint)MbCols)
            throw new ArgumentOutOfRangeException(nameof(mbCol));
        return Above.AsSpan(mbCol * PlanesPerMb, PlanesPerMb);
    }

    /// <summary>Index helpers for the per-plane slots within a MB context.</summary>
    public static class Plane
    {
        /// <summary>Y4 above starts at slot 0 (4 columns).</summary>
        public const int YBase = 0;
        /// <summary>U starts at slot 4 (2 columns).</summary>
        public const int UBase = 4;
        /// <summary>V starts at slot 6 (2 columns).</summary>
        public const int VBase = 6;
        /// <summary>Y2 single slot at index 8.</summary>
        public const int Y2Slot = 8;
    }
}
