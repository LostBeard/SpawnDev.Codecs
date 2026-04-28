// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 read_intra_frame_mode_info port. Bit-exact translation of libaom
// av1/decoder/decodemv.c read_intra_frame_mode_info() for keyframes /
// intra-only frames.
//
// Upstream Copyright (c) 2016, Alliance for Open Media. All rights reserved.
// Upstream license: BSD 2-Clause + AV1 Patent License 1.0. See NOTICE.md.
// Upstream source: aomedia.googlesource.com/aom
//   av1/decoder/decodemv.c read_intra_frame_mode_info (line 773)
//   av1/common/av1_common_int.h get_y_mode_cdf (line 1433)
//   av1/common/blockd.c av1_above_block_mode / av1_left_block_mode
//   av1/common/common_data.h intra_mode_context (line 411)
//   av1/common/reconintra.h av1_is_directional_mode / av1_use_angle_delta /
//                            av1_filter_intra_allowed
//   av1/common/pred_common.h av1_get_skip_txfm_context (line 175)
//
// Spec reference: AV1 Bitstream and Decoding Process Specification
//   sec 5.11.5  Decode block syntax
//   sec 6.10.4  Mode info semantics
//
// What is wired up here:
//   - read_intra_segment_id      (no-op when segmentation disabled - BBB case)
//   - read_skip_txfm             (uses Av1DefaultBlockCdfs.DefaultSkipTxfmCdf)
//   - read_cdef                  (reads cdef_bits raw bits per CDEF unit; BBB has cdef_bits=0)
//   - read_delta_q_params        (reads delta_q_cdf-driven abs + sign once per superblock origin)
//   - read_intra_mode (Y)        (uses Av1DefaultIntraModeCdfs.DefaultKfYModeCdf
//                                 with above+left intra-mode contexts)
//   - read_angle_delta (Y)       (uses Av1DefaultIntraModeCdfs.DefaultAngleDeltaCdf
//                                 when YMode is directional and bsize >= 8x8)
//   - read_intra_mode_uv         (uses Av1DefaultIntraModeCdfs.DefaultUvModeCdf)
//   - read_angle_delta (UV)
//   - read_filter_intra_mode_info (uses DefaultFilterIntraCdfs / DefaultFilterIntraModeCdf)
//
// What is NOT wired up here (out of scope for mode info read):
//   - intrabc (allow_intrabc is false on natural-content streams like BBB)
//   - palette mode (only enabled with screen content tools)
//   - CFL alpha (only when uv_mode == UV_CFL_PRED)
//
// The reader writes the decoded block info into the supplied
// <see cref="Av1ModeInfoGrid"/> so subsequent neighbor lookups (skip CDF
// context, intra mode CDF context) see the correct above/left state.

using SpawnDev.Codecs.EntropyCoders;

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>
/// AV1 keyframe mode-info reader. Stateless; takes the entropy decoder + the
/// mode info grid and decodes the per-block mode information bits.
/// </summary>
public static class Av1ModeInfoReader
{
    /// <summary>libaom <c>UV_INTRA_MODES</c>: 14 (13 luma modes + UV_CFL_PRED).</summary>
    public const int UvIntraModes = 14;

    /// <summary>libaom <c>INTRA_MODES</c>: 13.</summary>
    public const int IntraModes = 13;

    /// <summary>libaom <c>MAX_ANGLE_DELTA</c>: 3 (deltas in [-3, +3]).</summary>
    public const int MaxAngleDelta = 3;

    /// <summary>libaom <c>FILTER_INTRA_MODES</c>: 5.</summary>
    public const int FilterIntraModes = 5;

    /// <summary>
    /// libaom <c>intra_mode_context[INTRA_MODES]</c> from
    /// av1/common/common_data.h line 411. Maps a PREDICTION_MODE to a 0..4
    /// "context bucket" used by <c>get_y_mode_cdf</c> to pick the
    /// <see cref="Av1DefaultIntraModeCdfs.DefaultKfYModeCdf"/> row.
    /// </summary>
    public static readonly int[] IntraModeContext = new int[]
    {
        0, 1, 2, 3, 4, 4, 4, 4, 3, 0, 1, 2, 0,
    };

    /// <summary>
    /// Decode the per-block mode info for one keyframe block at (miRow, miCol)
    /// of size <paramref name="bsize"/>. Mirrors libaom
    /// <c>read_intra_frame_mode_info</c>.
    /// </summary>
    /// <param name="rd">Per-tile entropy decoder (state is mutated as bits are consumed).</param>
    /// <param name="sh">Sequence header (drives feature gating: filter_intra, monochrome, subsampling).</param>
    /// <param name="header">Complete frame header (drives delta_q signaling, allow_intrabc, etc).</param>
    /// <param name="grid">Per-tile mode info grid (used for above/left neighbor queries).</param>
    /// <param name="superblockState">
    /// Per-superblock decode state (cdef_transmitted flags + current base qindex).
    /// Caller resets at superblock origin and keeps it across all blocks within
    /// the superblock so CDEF + delta_q reads happen exactly once per superblock.
    /// </param>
    /// <param name="miRow">Block row in 4-px mi units within the frame.</param>
    /// <param name="miCol">Block column in 4-px mi units within the frame.</param>
    /// <param name="bsize">libaom BLOCK_SIZE enum value.</param>
    public static Av1ModeInfo Read(
        Av1RangeDecoder rd,
        Av1SequenceHeader sh,
        Av1CompleteFrameHeader header,
        Av1ModeInfoGrid grid,
        Av1SuperblockState superblockState,
        int miRow, int miCol, int bsize)
    {
        ArgumentNullException.ThrowIfNull(rd);
        ArgumentNullException.ThrowIfNull(sh);
        ArgumentNullException.ThrowIfNull(header);
        ArgumentNullException.ThrowIfNull(grid);
        ArgumentNullException.ThrowIfNull(superblockState);

        var mi = new Av1ModeInfo
        {
            BlockSize = bsize,
        };

        // 1. read_intra_segment_id - skip when segmentation disabled (BBB case).
        if (header.Segmentation.Enabled)
        {
            throw new NotImplementedException(
                "AV1 segmentation read path not implemented. " +
                "BBB stream has segmentation disabled, so this branch is unreachable.");
        }
        mi.SegmentId = 0;

        // 2. read_skip_txfm
        // libaom av1/common/pred_common.h:av1_get_skip_txfm_context:
        //   ctx = (above ? above->skip_txfm : 0) + (left ? left->skip_txfm : 0)
        var above = grid.Above(miRow, miCol);
        var left = grid.Left(miRow, miCol);
        int skipCtx = (above is not null ? (above.SkipTxfm ? 1 : 0) : 0)
                    + (left is not null ? (left.SkipTxfm ? 1 : 0) : 0);
        var skipCdf = Av1DefaultBlockCdfs.DefaultSkipTxfmCdf[skipCtx];
        mi.SkipTxfm = rd.DecodeCdfQ15(skipCdf, 2) == 1;

        // 3. read_cdef - reads cdef_bits raw bits from the FIRST non-skip block in
        //    each CDEF unit (64x64 region inside the superblock).
        //    BBB has cdef_bits=0 so this reads zero bits, but the cdef_transmitted
        //    bookkeeping still applies for streams with CDEF strength signaling.
        ReadCdef(rd, sh, header, superblockState, miRow, miCol, mi.SkipTxfm);

        // 4. read_delta_q_params + read_delta_lf - per-superblock-origin, conditional
        //    on (bsize != sb_size || skip_txfm == 0) AND we're at the superblock origin.
        ReadDeltaQ(rd, sh, header, superblockState, miRow, miCol, bsize, mi.SkipTxfm);

        // 5. allow_intrabc - off for natural-content streams.
        if (header.Prefix.AllowIntraBc)
        {
            throw new NotImplementedException(
                "AV1 intrabc read path not implemented. " +
                "BBB stream has allow_intrabc=false; this branch is unreachable for it.");
        }

        // 5. read_intra_mode (Y) - uses kf_y CDF with above+left intra mode context.
        Av1IntraMode aboveYMode = above is not null ? above.YMode : Av1IntraMode.Dc;
        Av1IntraMode leftYMode = left is not null ? left.YMode : Av1IntraMode.Dc;
        int aboveCtx = IntraModeContext[(int)aboveYMode];
        int leftCtx = IntraModeContext[(int)leftYMode];
        var yModeCdf = Av1DefaultIntraModeCdfs.DefaultKfYModeCdf[aboveCtx][leftCtx];
        mi.YMode = (Av1IntraMode)rd.DecodeCdfQ15(yModeCdf, IntraModes);

        // 6. read_angle_delta (Y) - directional modes only, bsize >= 8x8.
        if (UseAngleDelta(bsize) && IsDirectionalMode(mi.YMode))
        {
            // libaom angle_delta_cdf is indexed by mode - V_PRED.
            var angleCdf = Av1DefaultIntraModeCdfs.DefaultAngleDeltaCdf[(int)mi.YMode - (int)Av1IntraMode.Vertical];
            int sym = rd.DecodeCdfQ15(angleCdf, 2 * MaxAngleDelta + 1);
            mi.YAngleDelta = (sbyte)(sym - MaxAngleDelta);
        }

        // 7. read_intra_mode_uv - skip on monochrome / sub-chroma blocks.
        // For 4:2:0 + sufficient block size (>=8x8), every block has a chroma ref.
        bool isChromaRef = !sh.Monochrome && IsChromaReference(miRow, miCol, bsize, sh.SubsamplingX, sh.SubsamplingY);
        if (isChromaRef)
        {
            // libaom: uv_mode_cdf[cfl_allowed][y_mode], 14 syms with CFL else 13.
            // is_cfl_allowed: bsize <= 32x32 AND not lossless (we treat as enabled
            // when subsampling is 4:2:0 + bsize <=32x32).
            int cflAllowed = IsCflAllowed(bsize, sh.SubsamplingX, sh.SubsamplingY) ? 1 : 0;
            int nsyms = UvIntraModes - (cflAllowed == 0 ? 1 : 0);
            var uvCdf = Av1DefaultIntraModeCdfs.DefaultUvModeCdf[cflAllowed][(int)mi.YMode];
            mi.UvMode = (byte)rd.DecodeCdfQ15(uvCdf, nsyms);

            // CFL alphas (UV_CFL_PRED == 13). Reads joint sign + per-channel
            // magnitudes per libaom read_cfl_alphas. The decoded values are
            // stored on mi for the chroma reconstruction pipeline.
            if (mi.UvMode == 13)
            {
                ReadCflAlphas(rd, mi);
                // Keep mi.UvMode == 13 (CFL) so the walker dispatches CFL
                // prediction. The walker maps CFL -> DC predictor + alpha*AC.
            }

            // libaom maps uv_mode -> intra_mode for the angle delta lookup; for
            // non-CFL modes uv_mode IS the intra mode directly.
            var uvIntraMode = (Av1IntraMode)mi.UvMode;
            if (UseAngleDelta(bsize) && IsDirectionalMode(uvIntraMode))
            {
                var angleCdf = Av1DefaultIntraModeCdfs.DefaultAngleDeltaCdf[(int)uvIntraMode - (int)Av1IntraMode.Vertical];
                int sym = rd.DecodeCdfQ15(angleCdf, 2 * MaxAngleDelta + 1);
                mi.UvAngleDelta = (sbyte)(sym - MaxAngleDelta);
            }
        }
        else
        {
            mi.UvMode = (byte)Av1IntraMode.Dc;
        }

        // 8. palette - skip when allow_screen_content_tools is off (BBB case).
        if (header.Prefix.AllowScreenContentTools != 0)
        {
            throw new NotImplementedException(
                "AV1 palette mode info read path not implemented. " +
                "BBB stream has allow_screen_content_tools=0; unreachable for it.");
        }

        // 9. read_filter_intra_mode_info
        // Only allowed when SH.EnableFilterIntra && mode==DC_PRED && palette_size==0
        // && bsize <= 32x32. BBB has EnableFilterIntra=false, so this is a no-op.
        if (sh.EnableFilterIntra && mi.YMode == Av1IntraMode.Dc && IsFilterIntraAllowedBsize(bsize))
        {
            // filter_intra_cdfs is indexed by bsize.
            var filterCdf = Av1DefaultIntraModeCdfs.DefaultFilterIntraCdfs[bsize];
            bool useFilterIntra = rd.DecodeCdfQ15(filterCdf, 2) == 1;
            mi.UseFilterIntra = useFilterIntra;
            if (useFilterIntra)
            {
                mi.FilterIntraMode = (byte)rd.DecodeCdfQ15(
                    Av1DefaultIntraModeCdfs.DefaultFilterIntraModeCdf,
                    FilterIntraModes);
            }
        }

        // tx_size is read OUTSIDE read_intra_frame_mode_info (in a separate
        // loop in libaom decode_partition); we leave it at the largest tx
        // size for the block here. The actual value is set by the caller
        // (Av1KeyframeWalker) before invoking the coefficient decoder.
        mi.TxSize = Av1TxSizeInfo.MaxTxSizeRect[bsize];

        // Write into the grid so future Above/Left queries see this block.
        grid.Write(miRow, miCol, bsize, mi);
        return mi;
    }

    /// <summary>
    /// libaom <c>read_cfl_alphas</c>: read the joint sign symbol from
    /// cfl_sign_cdf, then per-channel 4-bit magnitude indices from
    /// cfl_alpha_cdf. Stores the result on <paramref name="mi"/>.
    ///
    /// Magnitudes:
    ///   CFL_SIGN_U(js) = ((js + 1) * 11) &gt;&gt; 5   -- 0/1/2 = ZERO/NEG/POS
    ///   CFL_SIGN_V(js) = (js + 1) - 3 * CFL_SIGN_U(js)
    ///   CFL_CONTEXT_U(js) = js + 1 - 3
    ///   CFL_CONTEXT_V(js) = CFL_SIGN_V(js) * 3 + CFL_SIGN_U(js) - 3
    /// </summary>
    private static void ReadCflAlphas(Av1RangeDecoder rd, Av1ModeInfo mi)
    {
        int jointSign = rd.DecodeCdfQ15(Av1DefaultIntraModeCdfs.DefaultCflSignCdf, 8);
        int signU = ((jointSign + 1) * 11) >> 5;
        int signV = (jointSign + 1) - 3 * signU;
        byte idx = 0;
        if (signU != 0)
        {
            int ctxU = jointSign + 1 - 3;
            var cdfU = Av1DefaultIntraModeCdfs.DefaultCflAlphaCdf[ctxU];
            int mag = rd.DecodeCdfQ15(cdfU, 16);
            idx = (byte)(mag << 4);
        }
        if (signV != 0)
        {
            int ctxV = signV * 3 + signU - 3;
            var cdfV = Av1DefaultIntraModeCdfs.DefaultCflAlphaCdf[ctxV];
            int mag = rd.DecodeCdfQ15(cdfV, 16);
            idx |= (byte)mag;
        }
        mi.UseCfl = true;
        mi.CflAlphaSigns = (sbyte)jointSign;
        mi.CflAlphaIdx = idx;
    }

    /// <summary>libaom <c>av1_is_directional_mode</c>: V_PRED..D67_PRED.</summary>
    public static bool IsDirectionalMode(Av1IntraMode mode)
    {
        // Per libaom enum order matches Av1IntraMode here:
        //   V=1, H=2, D45=3, D135=4, D113=5, D157=6, D203=7, D67=8
        return mode >= Av1IntraMode.Vertical && mode <= Av1IntraMode.D67;
    }

    /// <summary>libaom <c>av1_use_angle_delta</c>: bsize >= BLOCK_8X8.</summary>
    public static bool UseAngleDelta(int bsize)
    {
        return bsize >= Av1PartitionContext.Block8x8;
    }

    /// <summary>libaom <c>av1_filter_intra_allowed_bsize</c>: bw and bh both at most 32.</summary>
    public static bool IsFilterIntraAllowedBsize(int bsize)
    {
        if (bsize < 0 || bsize >= Av1PartitionContext.MiSizeWide.Length) return false;
        // mi_size_wide[bsize] * 4 <= 32 == mi_size_wide[bsize] <= 8
        return Av1PartitionContext.MiSizeWide[bsize] <= 8
            && Av1PartitionContext.MiSizeHigh[bsize] <= 8;
    }

    /// <summary>libaom <c>is_chroma_reference</c>: chroma plane has a sample for this block.</summary>
    public static bool IsChromaReference(int miRow, int miCol, int bsize, int subX, int subY)
    {
        if (bsize < 0 || bsize >= Av1PartitionContext.MiSizeWide.Length) return false;
        int bw = Av1PartitionContext.MiSizeWide[bsize];
        int bh = Av1PartitionContext.MiSizeHigh[bsize];
        // Per libaom: when bsize is < 8x8 in either chroma dim, we need to be at
        // an even mi position to "own" the chroma sample for this block.
        bool refX = bw >= 2 || (miCol & 1) == 0 || subX == 0;
        bool refY = bh >= 2 || (miRow & 1) == 0 || subY == 0;
        return refX && refY;
    }

    /// <summary>libaom <c>is_cfl_allowed</c>: 4:2:0 + bsize at most 32x32 + lossless not yet supported.</summary>
    public static bool IsCflAllowed(int bsize, int subX, int subY)
    {
        if (bsize < 0 || bsize >= Av1PartitionContext.MiSizeWide.Length) return false;
        if (subX == 0 && subY == 0) return false;  // 4:4:4 disables CFL
        // mi_size <= 8 (==32 px) AND chroma block size <=32 px.
        return Av1PartitionContext.MiSizeWide[bsize] <= 8
            && Av1PartitionContext.MiSizeHigh[bsize] <= 8;
    }

    /// <summary>
    /// libaom <c>read_cdef</c> port. Reads cdef_bits raw bits at the first
    /// non-skip block of each CDEF unit (64x64) within the superblock.
    /// </summary>
    private static void ReadCdef(
        Av1RangeDecoder rd,
        Av1SequenceHeader sh,
        Av1CompleteFrameHeader header,
        Av1SuperblockState sbState,
        int miRow, int miCol, bool skipTxfm)
    {
        if (header.CodedLossless) return;
        if (header.Prefix.AllowIntraBc) return;
        if (header.Cdef is null) return;

        // CDEF unit is 64x64 = 16 mi units (MI_SIZE_LOG2 = 2).
        int sbMask = (sh.Use128x128Superblock ? 32 : 16) - 1;
        int miRowInSb = miRow & sbMask;
        int miColInSb = miCol & sbMask;
        if (miRowInSb == 0 && miColInSb == 0)
        {
            for (int i = 0; i < 4; i++) sbState.CdefTransmitted[i] = false;
        }

        // CDEF unit index within the superblock.
        const int cdefSize = 16;  // 64 px / 4 px-per-mi
        int cdefRowInSb = ((miRow & cdefSize) != 0) ? 1 : 0;
        int cdefColInSb = ((miCol & cdefSize) != 0) ? 1 : 0;
        int idx = sh.Use128x128Superblock ? cdefColInSb + 2 * cdefRowInSb : 0;

        if (!sbState.CdefTransmitted[idx] && !skipTxfm)
        {
            int cdefBits = header.Cdef.Bits;
            if (cdefBits > 0)
            {
                rd.DecodeBits(cdefBits);
            }
            sbState.CdefTransmitted[idx] = true;
        }
    }

    /// <summary>
    /// libaom <c>read_delta_q_params</c> port. Reads delta_qindex (and optionally
    /// delta_lf*) once per superblock, gated by the per-block-skip-flag rule.
    /// </summary>
    private static void ReadDeltaQ(
        Av1RangeDecoder rd,
        Av1SequenceHeader sh,
        Av1CompleteFrameHeader header,
        Av1SuperblockState sbState,
        int miRow, int miCol, int bsize, bool skipTxfm)
    {
        if (!header.DeltaQPresent) return;

        // libaom: the read happens only at the superblock origin (b_col==0,
        // b_row==0 in superblock-mi space), AND only when bsize != sb_size
        // OR skip_txfm == 0.
        int sbMask = (sh.Use128x128Superblock ? 32 : 16) - 1;
        bool atSbOrigin = ((miCol & sbMask) == 0) && ((miRow & sbMask) == 0);
        if (!atSbOrigin) return;

        int sbBsize = sh.Use128x128Superblock ? 15 : 12;  // BLOCK_128X128 / BLOCK_64X64
        bool readFlag = (bsize != sbBsize) || !skipTxfm;
        if (!readFlag) return;

        int deltaQ = ReadDeltaQindex(rd) * header.DeltaQRes;
        sbState.CurrentBaseQindex = Math.Clamp(
            sbState.CurrentBaseQindex + deltaQ, 1, Av1DequantTables.MaxQ);

        if (header.DeltaLfPresent)
        {
            // libaom: per-LF-id delta when delta_lf_multi, else single delta.
            if (header.DeltaLfMulti)
            {
                int frameLfCount = sh.Monochrome ? 2 : 4;
                for (int lfId = 0; lfId < frameLfCount; lfId++)
                {
                    int delta = ReadDeltaLflevel(rd, Av1DefaultDeltaCdfs.DefaultDeltaLfMultiCdf[lfId])
                              * header.DeltaLfRes;
                    sbState.DeltaLf[lfId] = Math.Clamp(
                        sbState.DeltaLf[lfId] + delta, -63, 63);
                }
            }
            else
            {
                int delta = ReadDeltaLflevel(rd, Av1DefaultDeltaCdfs.DefaultDeltaLfCdf)
                          * header.DeltaLfRes;
                sbState.DeltaLfFromBase = Math.Clamp(
                    sbState.DeltaLfFromBase + delta, -63, 63);
            }
        }
    }

    /// <summary>
    /// libaom <c>read_delta_qindex</c> port. Returns the signed delta_qindex
    /// (NOT scaled by delta_q_res; caller multiplies).
    /// </summary>
    private static int ReadDeltaQindex(Av1RangeDecoder rd)
    {
        int abs = rd.DecodeCdfQ15(
            Av1DefaultDeltaCdfs.DefaultDeltaQCdf,
            Av1DefaultDeltaCdfs.DeltaQProbs + 1);
        if (abs >= Av1DefaultDeltaCdfs.DeltaQSmall)
        {
            int remBits = (int)rd.DecodeBits(3) + 1;
            int thr = (1 << remBits) + 1;
            abs = (int)rd.DecodeBits(remBits) + thr;
        }
        int sign = abs != 0 ? (int)rd.DecodeBits(1) : 1;
        return sign != 0 ? -abs : abs;
    }

    /// <summary>libaom <c>read_delta_lflevel</c> port.</summary>
    private static int ReadDeltaLflevel(Av1RangeDecoder rd, ushort[] cdf)
    {
        int abs = rd.DecodeCdfQ15(cdf, Av1DefaultDeltaCdfs.DeltaLfProbs + 1);
        if (abs >= Av1DefaultDeltaCdfs.DeltaLfSmall)
        {
            int remBits = (int)rd.DecodeBits(3) + 1;
            int thr = (1 << remBits) + 1;
            abs = (int)rd.DecodeBits(remBits) + thr;
        }
        int sign = abs != 0 ? (int)rd.DecodeBits(1) : 1;
        return sign != 0 ? -abs : abs;
    }
}

/// <summary>
/// AV1 per-superblock decode state. Tracks the CDEF strength bookkeeping and
/// the running delta_q / delta_lf levels accumulated across superblocks.
/// Caller resets the CdefTransmitted bits at superblock origin.
/// </summary>
public sealed class Av1SuperblockState
{
    /// <summary>cdef_transmitted[4] - per-CDEF-unit "have we read the strength" flags.</summary>
    public bool[] CdefTransmitted { get; } = new bool[4];

    /// <summary>Running base qindex (initialized from CompleteFrameHeader.Quant.BaseQindex).</summary>
    public int CurrentBaseQindex;

    /// <summary>delta_lf accumulator, single (when DeltaLfMulti is false).</summary>
    public int DeltaLfFromBase;

    /// <summary>delta_lf[4] per loop-filter id (when DeltaLfMulti).</summary>
    public int[] DeltaLf { get; } = new int[4];
}
