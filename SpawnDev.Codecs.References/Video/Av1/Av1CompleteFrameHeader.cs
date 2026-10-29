// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 complete uncompressed frame header. Carries the union of every
// keyframe-relevant structure from the AV1 spec sec 5.9 (uncompressed
// header) past the prefix subset surfaced by Av1FrameHeader:
//
//   - Tile info (sec 5.9.15)
//   - Quantization params (sec 5.9.12)
//   - Segmentation (sec 5.9.14)
//   - Delta quantizer / loop filter signaling (sec 5.9.16 / 5.9.17)
//   - Loop filter params (sec 5.9.11)
//   - CDEF params (sec 5.9.19)
//   - Loop restoration params (sec 5.9.20)
//   - Tx mode (sec 5.9.21)
//   - Frame reference mode (sec 5.9.23)
//   - Skip mode params (sec 5.9.22)
//   - Reduced tx set used (sec 5.9.25)
//   - Global motion params (sec 5.9.24, intra frames default to identity)
//   - Film grain params (sec 5.9.30)
//
// Inter-only sections (frame_refs, interpolation_filter, etc.) are
// represented as defaults / null on intra-only frames.

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>AV1 tile info (spec sec 5.9.15).</summary>
public sealed record Av1TileInfo
{
    /// <summary>True for uniform tile spacing (libaom uniform_tile_spacing_flag).</summary>
    public required bool UniformSpacing { get; init; }
    /// <summary>log2(tile cols).</summary>
    public required int Log2TileCols { get; init; }
    /// <summary>log2(tile rows).</summary>
    public required int Log2TileRows { get; init; }
    /// <summary>Number of tile columns.</summary>
    public required int TileCols { get; init; }
    /// <summary>Number of tile rows.</summary>
    public required int TileRows { get; init; }
    /// <summary>tile_size_bytes from frame header (1..4); only present when tiles>1.</summary>
    public int TileSizeBytes { get; init; }
    /// <summary>context_update_tile_id (cdf update tile); 0 for single-tile.</summary>
    public int ContextUpdateTileId { get; init; }
    /// <summary>Per-tile column start positions in superblock units (uniform spacing case populates these too).</summary>
    public int[] ColStartSb { get; init; } = Array.Empty<int>();
    /// <summary>Per-tile row start positions in superblock units.</summary>
    public int[] RowStartSb { get; init; } = Array.Empty<int>();
}

/// <summary>AV1 quantization params (spec sec 5.9.12).</summary>
public sealed record Av1QuantParams
{
    /// <summary>base_q_idx (0..255).</summary>
    public required int BaseQindex { get; init; }
    /// <summary>y_dc_delta_q (-63..63).</summary>
    public int YDcDeltaQ { get; init; }
    /// <summary>u_dc_delta_q.</summary>
    public int UDcDeltaQ { get; init; }
    /// <summary>u_ac_delta_q.</summary>
    public int UAcDeltaQ { get; init; }
    /// <summary>v_dc_delta_q.</summary>
    public int VDcDeltaQ { get; init; }
    /// <summary>v_ac_delta_q.</summary>
    public int VAcDeltaQ { get; init; }
    /// <summary>using_qmatrix flag.</summary>
    public bool UsingQmatrix { get; init; }
    /// <summary>qmatrix_level for Y.</summary>
    public int QmatrixLevelY { get; init; }
    /// <summary>qmatrix_level for U.</summary>
    public int QmatrixLevelU { get; init; }
    /// <summary>qmatrix_level for V.</summary>
    public int QmatrixLevelV { get; init; }
}

/// <summary>AV1 segmentation params (spec sec 5.9.14).</summary>
public sealed record Av1SegmentationParams
{
    /// <summary>seg_enabled.</summary>
    public required bool Enabled { get; init; }
    /// <summary>update_map.</summary>
    public bool UpdateMap { get; init; }
    /// <summary>temporal_update.</summary>
    public bool TemporalUpdate { get; init; }
    /// <summary>update_data.</summary>
    public bool UpdateData { get; init; }
    /// <summary>Per-segment feature_enabled[8][8] flags (8 segments, 8 features).</summary>
    public bool[,] FeatureEnabled { get; init; } = new bool[8, 8];
    /// <summary>Per-segment feature_data[8][8] values (8 segments, 8 features).</summary>
    public int[,] FeatureData { get; init; } = new int[8, 8];
}

/// <summary>AV1 loop filter params (spec sec 5.9.11).</summary>
public sealed record Av1LoopFilterParams
{
    /// <summary>filter_level[0] (Y vertical).</summary>
    public required int FilterLevel0 { get; init; }
    /// <summary>filter_level[1] (Y horizontal).</summary>
    public required int FilterLevel1 { get; init; }
    /// <summary>filter_level_u (chroma U).</summary>
    public int FilterLevelU { get; init; }
    /// <summary>filter_level_v (chroma V).</summary>
    public int FilterLevelV { get; init; }
    /// <summary>sharpness_level (0..7).</summary>
    public int SharpnessLevel { get; init; }
    /// <summary>mode_ref_delta_enabled.</summary>
    public bool ModeRefDeltaEnabled { get; init; }
    /// <summary>mode_ref_delta_update.</summary>
    public bool ModeRefDeltaUpdate { get; init; }
    /// <summary>ref_deltas[8] (signed, -64..63).</summary>
    public int[] RefDeltas { get; init; } = new int[8] { 1, 0, 0, 0, -1, 0, -1, -1 };
    /// <summary>mode_deltas[2] (signed).</summary>
    public int[] ModeDeltas { get; init; } = new int[2];
}

/// <summary>AV1 CDEF params (spec sec 5.9.19).</summary>
public sealed record Av1CdefParams
{
    /// <summary>cdef_damping (3..6).</summary>
    public required int Damping { get; init; }
    /// <summary>cdef_bits (0..3).</summary>
    public required int Bits { get; init; }
    /// <summary>cdef_y_pri_strength[8] / cdef_y_sec_strength[8] packed: cdef_strengths[i].</summary>
    public int[] YStrengths { get; init; } = new int[8];
    /// <summary>cdef_uv_strengths[i] (0 if monochrome).</summary>
    public int[] UvStrengths { get; init; } = new int[8];
}

/// <summary>AV1 loop restoration params (spec sec 5.9.20).</summary>
public sealed record Av1LrParams
{
    /// <summary>frame_restoration_type per plane (NONE / WIENER / SGRPROJ / SWITCHABLE).</summary>
    public Av1RestorationType[] PerPlane { get; init; } = new Av1RestorationType[3];
    /// <summary>Per-plane restoration unit size (in pixels).</summary>
    public int[] UnitSize { get; init; } = new int[3];
}

/// <summary>AV1 frame_restoration_type values (spec sec 6.10.15).</summary>
public enum Av1RestorationType : byte
{
    /// <summary>No restoration filter.</summary>
    None = 0,
    /// <summary>Wiener filter.</summary>
    Wiener = 1,
    /// <summary>Self-guided projection filter.</summary>
    SgrProj = 2,
    /// <summary>Per-block switchable.</summary>
    Switchable = 3,
}

/// <summary>AV1 tx_mode values (spec sec 6.8.21).</summary>
public enum Av1TxMode : byte
{
    /// <summary>Only 4x4 transforms (lossless).</summary>
    Only4x4 = 0,
    /// <summary>Largest tx size for the block.</summary>
    Largest = 1,
    /// <summary>Per-block tx size selection.</summary>
    Select = 2,
}

/// <summary>AV1 reference_mode values (spec sec 6.8.23).</summary>
public enum Av1ReferenceMode : byte
{
    /// <summary>Single ref only.</summary>
    SingleReference = 0,
    /// <summary>Compound ref only.</summary>
    CompoundReference = 1,
    /// <summary>Per-block selection.</summary>
    ReferenceModeSelect = 2,
}

/// <summary>AV1 film grain params (spec sec 5.9.30).</summary>
public sealed record Av1FilmGrainParams
{
    /// <summary>apply_grain flag.</summary>
    public required bool ApplyGrain { get; init; }
    /// <summary>random_seed (16-bit).</summary>
    public int RandomSeed { get; init; }
    /// <summary>update_parameters.</summary>
    public bool UpdateParameters { get; init; }
}

/// <summary>
/// AV1 complete uncompressed frame header for a keyframe / intra-only frame.
/// </summary>
public sealed record Av1CompleteFrameHeader
{
    /// <summary>The prefix-fields header.</summary>
    public required Av1FrameHeader Prefix { get; init; }
    /// <summary>Tile info structure.</summary>
    public required Av1TileInfo TileInfo { get; init; }
    /// <summary>Quantization params.</summary>
    public required Av1QuantParams Quant { get; init; }
    /// <summary>Segmentation params.</summary>
    public required Av1SegmentationParams Segmentation { get; init; }
    /// <summary>Loop filter params.</summary>
    public required Av1LoopFilterParams LoopFilter { get; init; }
    /// <summary>CDEF params (only valid when SH.EnableCdef and not coded_lossless and not allow_intrabc).</summary>
    public Av1CdefParams? Cdef { get; init; }
    /// <summary>Loop restoration params (only valid when SH.EnableRestoration and not all_lossless).</summary>
    public Av1LrParams? Lr { get; init; }
    /// <summary>delta_q_present_flag.</summary>
    public bool DeltaQPresent { get; init; }
    /// <summary>delta_q_res (1 / 2 / 4 / 8) - only valid when DeltaQPresent.</summary>
    public int DeltaQRes { get; init; } = 1;
    /// <summary>delta_lf_present_flag.</summary>
    public bool DeltaLfPresent { get; init; }
    /// <summary>delta_lf_res.</summary>
    public int DeltaLfRes { get; init; } = 1;
    /// <summary>delta_lf_multi.</summary>
    public bool DeltaLfMulti { get; init; }
    /// <summary>tx_mode (ONLY_4X4 / LARGEST / SELECT).</summary>
    public required Av1TxMode TxMode { get; init; }
    /// <summary>frame_reference_mode.</summary>
    public required Av1ReferenceMode ReferenceMode { get; init; }
    /// <summary>skip_mode_present (intra-only frames: always false).</summary>
    public bool SkipModePresent { get; init; }
    /// <summary>reduced_tx_set_used.</summary>
    public required bool ReducedTxSetUsed { get; init; }
    /// <summary>Film grain params (only valid when SH.FilmGrainParamsPresent).</summary>
    public Av1FilmGrainParams? FilmGrain { get; init; }
    /// <summary>Total bytes consumed by the uncompressed header (round up to byte).</summary>
    public required int HeaderSizeBytes { get; init; }
    /// <summary>True when the segmentation + quant produce coded_lossless (skip CDEF/LF).</summary>
    public required bool CodedLossless { get; init; }
    /// <summary>True when frame is fully lossless.</summary>
    public required bool AllLossless { get; init; }
}
