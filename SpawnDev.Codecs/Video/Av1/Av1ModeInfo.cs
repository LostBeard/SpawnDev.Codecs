// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 per-block mode info + neighbor tracking. Mirrors a subset of libaom
// MB_MODE_INFO + the above/left "context buffer" pattern used by
// MACROBLOCKD.
//
// Spec reference: AV1 Bitstream and Decoding Process Specification
//   sec 5.11.5  Decode block syntax (read_intra_frame_mode_info)
//   sec 6.10.2  Mode info semantics
//
// We track per-mi (4-px) cells for the lifetime of a tile, since many
// neighbor lookups (intra mode CDF context, skip CDF context, partition
// context, txfm partition context) read above/left at the mi grid.

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>
/// AV1 per-block decoded info. Slim libaom MB_MODE_INFO subset focused on
/// what the keyframe block decode pipeline needs.
/// </summary>
public sealed class Av1ModeInfo
{
    /// <summary>Block size enum value (libaom BLOCK_SIZE).</summary>
    public int BlockSize;
    /// <summary>Luma intra prediction mode.</summary>
    public Av1IntraMode YMode;
    /// <summary>Chroma intra prediction mode (UV_INTRA_MODE; same enum as Y plus UV_CFL_PRED at index 13).</summary>
    public byte UvMode;
    /// <summary>Y angle delta (signed -3..3) when YMode is directional.</summary>
    public sbyte YAngleDelta;
    /// <summary>UV angle delta when UvMode is directional.</summary>
    public sbyte UvAngleDelta;
    /// <summary>True when this block has no transform coefficients.</summary>
    public bool SkipTxfm;
    /// <summary>Selected transform size (libaom TX_SIZE).</summary>
    public Av1TxSize TxSize;
    /// <summary>Segment id (always 0 when segmentation disabled).</summary>
    public byte SegmentId;
    /// <summary>filter_intra mode index (0..4) when use_filter_intra true.</summary>
    public byte FilterIntraMode;
    /// <summary>True when filter_intra is enabled for this block.</summary>
    public bool UseFilterIntra;
    /// <summary>True when this block uses CFL_PRED for chroma (uv_mode == UV_CFL_PRED).</summary>
    public bool UseCfl;
    /// <summary>CFL packed alpha index (libaom <c>cfl_alpha_idx</c>): bits 0..3 = V mag, bits 4..7 = U mag.</summary>
    public byte CflAlphaIdx;
    /// <summary>CFL joint sign (libaom <c>cfl_alpha_signs</c>): 0..7 (CFL_JOINT_SIGNS).</summary>
    public sbyte CflAlphaSigns;
}

/// <summary>
/// AV1 mode-info grid + neighbor query helpers. Tracks the latest
/// <see cref="Av1ModeInfo"/> written into each (mi_row, mi_col) cell so
/// the block decoder can answer above/left queries with constant-time
/// indexed reads.
/// </summary>
public sealed class Av1ModeInfoGrid
{
    private readonly Av1ModeInfo?[] _cells;
    private readonly int _miCols;

    /// <summary>Construct a mode info grid sized to (miRows x miCols).</summary>
    public Av1ModeInfoGrid(int miRows, int miCols)
    {
        if (miRows < 0) throw new ArgumentOutOfRangeException(nameof(miRows));
        if (miCols < 0) throw new ArgumentOutOfRangeException(nameof(miCols));
        _miCols = miCols;
        _cells = new Av1ModeInfo?[Math.Max(1, miRows * miCols)];
    }

    /// <summary>Total number of mi columns in the grid.</summary>
    public int MiCols => _miCols;

    /// <summary>Total number of mi rows.</summary>
    public int MiRows => _cells.Length / Math.Max(1, _miCols);

    /// <summary>
    /// Get the mode info above (mi_row - 1, mi_col), or null if out of frame.
    /// Mirrors libaom <c>xd->above_mbmi</c>.
    /// </summary>
    public Av1ModeInfo? Above(int miRow, int miCol)
    {
        if (miRow <= 0 || miCol < 0 || miCol >= _miCols) return null;
        return _cells[(miRow - 1) * _miCols + miCol];
    }

    /// <summary>
    /// Get the mode info left of (mi_row, mi_col - 1), or null if out of frame.
    /// Mirrors libaom <c>xd->left_mbmi</c>.
    /// </summary>
    public Av1ModeInfo? Left(int miRow, int miCol)
    {
        if (miRow < 0 || miRow >= MiRows || miCol <= 0) return null;
        return _cells[miRow * _miCols + (miCol - 1)];
    }

    /// <summary>
    /// Write <paramref name="mi"/> into every mi cell covered by the block at
    /// (miRow, miCol) with size <paramref name="bsize"/>.
    /// </summary>
    public void Write(int miRow, int miCol, int bsize, Av1ModeInfo mi)
    {
        ArgumentNullException.ThrowIfNull(mi);
        if (bsize < 0 || bsize >= Av1PartitionContext.MiSizeWide.Length)
            throw new ArgumentOutOfRangeException(nameof(bsize));
        int bw = Av1PartitionContext.MiSizeWide[bsize];
        int bh = Av1PartitionContext.MiSizeHigh[bsize];
        int rEnd = Math.Min(MiRows, miRow + bh);
        int cEnd = Math.Min(_miCols, miCol + bw);
        for (int r = miRow; r < rEnd; r++)
        {
            int rowBase = r * _miCols;
            for (int c = miCol; c < cEnd; c++)
            {
                _cells[rowBase + c] = mi;
            }
        }
    }
}
