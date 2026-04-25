// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 per-block mode info. Mirror of libvpx vp9/common/vp9_blockd.h
// MB_MODE_INFO (the "macroblock mode info" struct, despite the name
// VP9 doesn't have macroblocks - it has the SuperBlock partition
// tree and the leaves carry MB_MODE_INFO).
//
// Stored at every decoded leaf of the partition tree; the per-frame
// mode info grid maps (mi_row, mi_col) tuples to these records.
// Used by:
//   - intra prediction (BlockSize + YMode + UvMode + TxSize)
//   - inter prediction (BlockSize + RefFrames + Mvs + InterpFilter
//     + Skip + TxSize)
//   - loop filter (Skip + TxSize + RefFrames + InterMode)
//   - context predictors (above/left neighbor lookups)

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// VP9 per-block decoded mode info. Mirror of libvpx
/// <c>MB_MODE_INFO</c>.
/// </summary>
/// <remarks>
/// IsIntra-discriminated: when <see cref="RefFrames"/>[0] is
/// <see cref="Vp9MvReferenceFrame.Intra"/>, the block is intra-coded
/// and YMode / UvMode are meaningful (InterMode / Mvs / InterpFilter
/// have no meaning). Otherwise the block is inter-coded and the
/// inter fields apply.
/// </remarks>
public sealed record Vp9MbModeInfo
{
    /// <summary>libvpx <c>sb_type</c>: block size at this leaf.</summary>
    public required Vp9BlockSize BlockSize { get; init; }

    /// <summary>
    /// libvpx <c>ref_frame[2]</c>: ref_frame[0] is the primary
    /// reference (Intra for intra-coded blocks); ref_frame[1] is the
    /// second reference for compound prediction (Vp9MvReferenceFrame
    /// values 1..3 only) or null for single-ref blocks.
    /// </summary>
    public required Vp9MvReferenceFrame PrimaryRefFrame { get; init; }

    /// <summary>Second compound reference frame, or null for single-ref / intra blocks.</summary>
    public Vp9MvReferenceFrame? CompoundRefFrame { get; init; }

    /// <summary>True when this block is intra-coded.</summary>
    public bool IsIntra => PrimaryRefFrame == Vp9MvReferenceFrame.Intra;

    /// <summary>True when this block uses compound (two-reference) prediction.</summary>
    public bool IsCompound => CompoundRefFrame.HasValue;

    /// <summary>libvpx <c>mode</c> for intra blocks (Y plane).</summary>
    public Vp9IntraMode? YMode { get; init; }

    /// <summary>libvpx <c>uv_mode</c> for intra blocks (UV plane).</summary>
    public Vp9IntraMode? UvMode { get; init; }

    /// <summary>libvpx <c>mode</c> for inter blocks.</summary>
    public Vp9InterMode? InterMode { get; init; }

    /// <summary>libvpx <c>tx_size</c>.</summary>
    public required Vp9TxSize TxSize { get; init; }

    /// <summary>libvpx <c>skip</c>: skip transform coefficient coding.</summary>
    public required bool Skip { get; init; }

    /// <summary>libvpx <c>segment_id</c>: 0..7.</summary>
    public required int SegmentId { get; init; }

    /// <summary>libvpx <c>seg_id_predicted</c>: temporal-update predictor flag.</summary>
    public bool SegmentIdPredicted { get; init; }

    /// <summary>
    /// libvpx <c>mv[0]</c>: motion vector for the primary reference.
    /// <see cref="Vp9Mv.Zero"/> for intra blocks (unused).
    /// </summary>
    public Vp9Mv PrimaryMv { get; init; } = Vp9Mv.Zero;

    /// <summary>
    /// libvpx <c>mv[1]</c>: motion vector for the second reference
    /// (compound prediction). <see cref="Vp9Mv.Zero"/> for single-ref /
    /// intra blocks.
    /// </summary>
    public Vp9Mv CompoundMv { get; init; } = Vp9Mv.Zero;

    /// <summary>libvpx <c>interp_filter</c>: per-block filter selection.</summary>
    public Vp9InterpFilter InterpFilter { get; init; } = Vp9InterpFilter.EightTap;
}
