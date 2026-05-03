// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Full AV1 uncompressed frame header parser for keyframe / intra_only
// frames. Mirrors libaom <c>read_uncompressed_header</c> in
// av1/decoder/decodeframe.c.
//
// Upstream Copyright (c) 2016, Alliance for Open Media.
// Upstream license: BSD 2-Clause + AV1 Patent License 1.0. See NOTICE.md.
//
// Pipes the bit reader through the optional sections in spec order:
//   reduced_still / OP / frame_size / refresh_flags  (already in prefix)
//   render_size / superres / tile_info / quant / segmentation /
//   delta_q+lf / loop_filter / cdef / lr / tx_mode / reference_mode /
//   skip_mode / warped_motion / reduced_tx_set / film_grain.
//
// The parser is deliberately KEYFRAME-FOCUSED: many inter-frame paths
// throw NotImplementedException so silent decode failures don't ship
// against bitstreams we can't validate.

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>Full AV1 frame header parser.</summary>
public static class Av1CompleteFrameHeaderParser
{
    /// <summary>
    /// Parse the complete uncompressed header from the start of an OBU
    /// Frame / FrameHeader payload. Returns the parsed structure plus
    /// the byte offset to the start of the tile data (post-byte-align).
    /// </summary>
    public static Av1CompleteFrameHeader Parse(ReadOnlySpan<byte> payload, Av1SequenceHeader sh)
    {
        ArgumentNullException.ThrowIfNull(sh);

        // Re-parse the prefix fields. Keeps the parser self-contained; the
        // small cost is paying the prefix bit cursor twice on the same OBU.
        var prefix = Av1FrameHeaderParser.Parse(payload, sh);

        // Build a fresh bit reader and replay the prefix bits to advance
        // the cursor to the post-prefix position.
        var br = new Av1BitReader(payload);
        AdvancePrefix(ref br, sh, prefix);

        if (prefix.ShowExistingFrame)
        {
            // No additional structures past show_existing_frame index.
            return MakeMinimal(prefix, finalBitPos: br.Position);
        }

        bool isIntra = prefix.FrameIsIntra;
        if (!isIntra)
        {
            // Inter-frame path requires reference frame setup, frame_refs,
            // interpolation filter, switchable_motion_mode etc - well outside
            // the keyframe-decode scope of this parser.
            throw new NotImplementedException(
                "AV1 complete frame header parser only supports key / intra-only frames; " +
                "inter-frame parsing is downstream work.");
        }

        // For keyframes / intra-only with frame_size_override=0, the prefix
        // walked frame_size already (via SH defaults). Now we still need:
        //   1. setup_superres - 1 bit gated on SH.EnableSuperres
        //   2. setup_render_size - 1 bit (+ 32 if true)
        //   3. allow_intrabc - 1 bit gated on SCC + !superres_scaled
        //   4. refresh_frame_context (might_bwd_adapt) - 1 bit
        bool useSuperres = false;
        if (sh.EnableSuperres)
        {
            useSuperres = br.ReadFlag();
            if (useSuperres) br.ReadBits(3); // SUPERRES_SCALE_BITS = 3
        }
        bool renderDifferent = br.ReadFlag();
        if (renderDifferent) { br.ReadBits(16); br.ReadBits(16); }

        bool allowIntraBc = false;
        if (prefix.AllowScreenContentTools != 0 && !useSuperres)
        {
            allowIntraBc = br.ReadFlag();
        }

        // Replace the prefix with one carrying allow_intrabc set.
        prefix = prefix with { AllowIntraBc = allowIntraBc };

        // refresh_frame_context (might_bwd_adapt = !disableCdfUpdate)
        bool mightBwdAdapt = !prefix.DisableCdfUpdate;
        if (mightBwdAdapt) br.ReadBits(1);

        // Tile info
        var tileInfo = ReadTileInfo(ref br, sh, prefix);

        // Quantization
        bool monochrome = sh.Monochrome;
        int numPlanes = monochrome ? 1 : 3;
        var quant = ReadQuantization(ref br, sh, numPlanes);

        // Segmentation - keyframes ALWAYS have primary_ref_frame == NONE,
        // which means update_map=update_data=1 (no temporal_update),
        // libaom skips reading the update flags then.
        var seg = ReadSegmentation(ref br, prefix, segmentationPrimaryRefIsNone: true);

        // Loop filter / CDEF / LR / delta_q / delta_lf are gated on
        // allow_intrabc / coded_lossless. Compute coded_lossless first.
        bool isLosslessSegment0 = (quant.BaseQindex == 0
            && quant.YDcDeltaQ == 0
            && quant.UDcDeltaQ == 0 && quant.UAcDeltaQ == 0
            && quant.VDcDeltaQ == 0 && quant.VAcDeltaQ == 0);
        bool codedLossless = isLosslessSegment0 && !seg.Enabled;
        bool allLossless = codedLossless && !sh.EnableSuperres;

        // delta_q_present + delta_lf_present
        bool deltaQPresent = false;
        int deltaQRes = 1;
        bool deltaLfPresent = false;
        int deltaLfRes = 1;
        bool deltaLfMulti = false;
        if (quant.BaseQindex > 0)
        {
            deltaQPresent = br.ReadFlag();
            if (deltaQPresent)
            {
                deltaQRes = 1 << (int)br.ReadBits(2);
                if (!prefix.AllowIntraBc)
                {
                    deltaLfPresent = br.ReadFlag();
                    if (deltaLfPresent)
                    {
                        deltaLfRes = 1 << (int)br.ReadBits(2);
                        deltaLfMulti = br.ReadFlag();
                    }
                }
            }
        }

        // Loop filter (skipped when allow_intrabc OR coded_lossless)
        Av1LoopFilterParams lf;
        if (prefix.AllowIntraBc || codedLossless)
        {
            lf = new Av1LoopFilterParams { FilterLevel0 = 0, FilterLevel1 = 0 };
        }
        else
        {
            lf = ReadLoopFilter(ref br, numPlanes);
        }

        // CDEF (skipped when allow_intrabc OR coded_lossless OR !SH.EnableCdef)
        Av1CdefParams? cdef = null;
        if (!codedLossless && sh.EnableCdef && !prefix.AllowIntraBc)
        {
            cdef = ReadCdef(ref br, numPlanes);
        }

        // Loop restoration (skipped when all_lossless OR !SH.EnableRestoration OR allow_intrabc)
        Av1LrParams? lr = null;
        if (!allLossless && sh.EnableRestoration && !prefix.AllowIntraBc)
        {
            lr = ReadLrParams(ref br, sh, numPlanes);
        }

        // tx_mode
        var txMode = codedLossless ? Av1TxMode.Only4x4
            : (br.ReadFlag() ? Av1TxMode.Select : Av1TxMode.Largest);

        // reference_mode (intra-only frames: forced to SINGLE_REFERENCE)
        var refMode = Av1ReferenceMode.SingleReference;

        // skip_mode (intra-only: skip_mode_allowed = false -> not read)
        bool skipModePresent = false;

        // allow_warped_motion (skipped on intra: frame_might_allow_warped_motion = false)
        // Then reduced_tx_set
        bool reducedTxSetUsed = br.ReadFlag();

        // global_motion (skipped on intra)
        // film_grain
        Av1FilmGrainParams? filmGrain = null;
        if (sh.FilmGrainParamsPresent && (prefix.ShowFrame || prefix.ShowableFrame))
        {
            filmGrain = ReadFilmGrain(ref br);
        }

        // Trailing bits / byte-align - libaom decodeframe wraps this by
        // advancing the bit cursor to the next byte; that's where the
        // tile data begins.
        int finalBitPos = (br.Position + 7) & ~7;

        return new Av1CompleteFrameHeader
        {
            Prefix = prefix,
            TileInfo = tileInfo,
            Quant = quant,
            Segmentation = seg,
            LoopFilter = lf,
            Cdef = cdef,
            Lr = lr,
            DeltaQPresent = deltaQPresent,
            DeltaQRes = deltaQRes,
            DeltaLfPresent = deltaLfPresent,
            DeltaLfRes = deltaLfRes,
            DeltaLfMulti = deltaLfMulti,
            TxMode = txMode,
            ReferenceMode = refMode,
            SkipModePresent = skipModePresent,
            ReducedTxSetUsed = reducedTxSetUsed,
            FilmGrain = filmGrain,
            HeaderSizeBytes = finalBitPos / 8,
            CodedLossless = codedLossless,
            AllLossless = allLossless,
        };
    }

    private static Av1CompleteFrameHeader MakeMinimal(Av1FrameHeader prefix, int finalBitPos)
    {
        return new Av1CompleteFrameHeader
        {
            Prefix = prefix,
            TileInfo = new Av1TileInfo { UniformSpacing = true, Log2TileCols = 0, Log2TileRows = 0, TileCols = 1, TileRows = 1 },
            Quant = new Av1QuantParams { BaseQindex = 0 },
            Segmentation = new Av1SegmentationParams { Enabled = false },
            LoopFilter = new Av1LoopFilterParams { FilterLevel0 = 0, FilterLevel1 = 0 },
            TxMode = Av1TxMode.Largest,
            ReferenceMode = Av1ReferenceMode.SingleReference,
            ReducedTxSetUsed = false,
            HeaderSizeBytes = (finalBitPos + 7) / 8,
            CodedLossless = false,
            AllLossless = false,
        };
    }

    /// <summary>
    /// Replay the bit cursor advancement that <see cref="Av1FrameHeaderParser"/>
    /// performs, so the post-prefix reader is positioned correctly. This
    /// keeps the parsers in lock-step without duplicating all the bit
    /// branches.
    /// </summary>
    private static void AdvancePrefix(ref Av1BitReader br, Av1SequenceHeader sh, Av1FrameHeader prefix)
    {
        if (sh.ReducedStillPictureHeader)
        {
            return;
        }

        bool showExisting = br.ReadFlag();
        if (showExisting)
        {
            br.ReadBits(3);
            return;
        }

        br.ReadBits(2); // frame_type
        bool showFrame = br.ReadFlag();
        bool showableFrame;
        if (!showFrame) showableFrame = br.ReadFlag();
        else showableFrame = prefix.FrameType != Av1FrameType.KeyFrame;

        bool errorResilient;
        if (prefix.FrameType == Av1FrameType.SwitchFrame
            || (prefix.FrameType == Av1FrameType.KeyFrame && showFrame))
            errorResilient = true;
        else
            errorResilient = br.ReadFlag();

        br.ReadFlag(); // disable_cdf_update

        // allow_screen_content_tools
        if (sh.SeqForceScreenContentTools == 2) br.ReadBits(1);
        int allowSccTools = prefix.AllowScreenContentTools;
        if (allowSccTools != 0)
        {
            if (sh.SeqForceIntegerMv == 2) br.ReadBits(1);
        }

        if (sh.FrameIdNumbersPresent)
        {
            br.ReadBits(sh.FrameIdLengthMinus7 + 7);
        }

        bool frameSizeOverride;
        if (prefix.FrameType == Av1FrameType.SwitchFrame) frameSizeOverride = true;
        else if (sh.ReducedStillPictureHeader) frameSizeOverride = false;
        else frameSizeOverride = br.ReadFlag();

        bool isIntra = prefix.FrameIsIntra;
        if (sh.EnableOrderHint)
        {
            // Mirror Av1FrameHeaderParser bug fix: order_hint is read for
            // every non-reduced-still frame when EnableOrderHint is set.
            br.ReadBits(sh.OrderHintBitsMinus1 + 1);
        }

        // refresh_frame_flags
        if (!((prefix.FrameType == Av1FrameType.KeyFrame && showFrame)
              || prefix.FrameType == Av1FrameType.SwitchFrame))
        {
            br.ReadBits(8);
        }

        if (frameSizeOverride)
        {
            br.ReadBits(16); // width_minus_1
            br.ReadBits(16); // height_minus_1
        }
        // allow_intrabc is NOT in the prefix - parsed by Av1CompleteFrameHeaderParser
        // after setup_superres + setup_render_size.
    }

    private static Av1TileInfo ReadTileInfo(ref Av1BitReader br, Av1SequenceHeader sh, Av1FrameHeader prefix)
    {
        // Compute mi grid size. AV1 superblock size = 64x64 (Use128=false) or 128x128.
        // AV1 mi units are 4 luma samples; mi_cols = ceil(width_px / 4).
        int sbSize = sh.Use128x128Superblock ? 128 : 64;
        int sbSizeLog2 = sh.Use128x128Superblock ? 7 : 6;
        int miSizeLog2 = 2; // 4x4 mi units
        int mibSizeLog2 = sbSizeLog2 - miSizeLog2;
        int miCols = (prefix.FrameWidth + 3) >> 2;
        int miRows = (prefix.FrameHeight + 3) >> 2;
        // CEIL to mibSize
        int widthSb = (miCols + (1 << mibSizeLog2) - 1) >> mibSizeLog2;
        int heightSb = (miRows + (1 << mibSizeLog2) - 1) >> mibSizeLog2;

        // Mirrors libaom av1_get_tile_limits (tile_common.c):
        //   max_width_sb     = MAX_TILE_WIDTH >> sb_size_log2
        //   max_tile_area_sb = MAX_TILE_AREA  >> (2 * sb_size_log2)
        // MAX_TILE_WIDTH = 4096 px, MAX_TILE_AREA = 4096 * 2304 px.
        int maxWidthSb = 4096 >> sbSizeLog2;            // = 64 for 64-px SB
        int maxTileAreaSb = (4096 * 2304) >> (2 * sbSizeLog2); // = 2304 for 64-px SB
        int maxLog2TileCols = TileLog2(1, Math.Min(widthSb, 64));
        int maxLog2TileRows = TileLog2(1, Math.Min(heightSb, 64));
        int minLog2TileCols = Math.Max(0, TileLog2(maxWidthSb, widthSb));
        int minLog2Tiles = Math.Max(minLog2TileCols, TileLog2(maxTileAreaSb, widthSb * heightSb));
        int minLog2TileRows = Math.Max(0, minLog2Tiles - minLog2TileCols);

        bool uniform = br.ReadFlag();
        int log2Cols, log2Rows, tileCols, tileRows;
        var colStartSb = new int[64 + 1];
        var rowStartSb = new int[64 + 1];

        if (uniform)
        {
            log2Cols = minLog2TileCols;
            while (log2Cols < maxLog2TileCols)
            {
                if (!br.ReadFlag()) break;
                log2Cols++;
            }
            int log2RowsLocal = Math.Max(0, minLog2Tiles - log2Cols);
            log2Rows = log2RowsLocal;
            while (log2Rows < maxLog2TileRows)
            {
                if (!br.ReadFlag()) break;
                log2Rows++;
            }
            tileCols = 1 << log2Cols;
            tileRows = 1 << log2Rows;
            // Compute uniform start positions
            int stride = (widthSb + tileCols - 1) / tileCols;
            int startSb = 0;
            for (int i = 0; i < tileCols; i++) { colStartSb[i] = startSb; startSb += stride; }
            colStartSb[tileCols] = widthSb;
            stride = (heightSb + tileRows - 1) / tileRows;
            startSb = 0;
            for (int i = 0; i < tileRows; i++) { rowStartSb[i] = startSb; startSb += stride; }
            rowStartSb[tileRows] = heightSb;
        }
        else
        {
            // Variable tile spacing - rare path. Read per-tile size_minus_1 in
            // SBs via uniform variable-length integer (rb_read_uniform).
            int wsb = widthSb;
            int i = 0;
            int startSb = 0;
            while (wsb > 0 && i < 64)
            {
                int sizeSb = 1 + RbReadUniform(ref br, Math.Min(wsb, 64));
                colStartSb[i] = startSb;
                startSb += sizeSb;
                wsb -= sizeSb;
                i++;
            }
            tileCols = i;
            colStartSb[i] = startSb + wsb;
            log2Cols = TileLog2(1, tileCols);

            int hsb = heightSb;
            int j = 0;
            startSb = 0;
            while (hsb > 0 && j < 64)
            {
                int sizeSb = 1 + RbReadUniform(ref br, Math.Min(hsb, 64));
                rowStartSb[j] = startSb;
                startSb += sizeSb;
                hsb -= sizeSb;
                j++;
            }
            tileRows = j;
            rowStartSb[j] = startSb + hsb;
            log2Rows = TileLog2(1, tileRows);
        }

        int contextUpdateTileId = 0;
        int tileSizeBytes = 1;
        if (tileCols * tileRows > 1)
        {
            contextUpdateTileId = (int)br.ReadBits(log2Cols + log2Rows);
            tileSizeBytes = (int)br.ReadBits(2) + 1;
        }

        // Trim arrays to actual count + 1 sentinel
        var trimmedCols = new int[tileCols + 1];
        Array.Copy(colStartSb, trimmedCols, tileCols + 1);
        var trimmedRows = new int[tileRows + 1];
        Array.Copy(rowStartSb, trimmedRows, tileRows + 1);

        return new Av1TileInfo
        {
            UniformSpacing = uniform,
            Log2TileCols = log2Cols,
            Log2TileRows = log2Rows,
            TileCols = tileCols,
            TileRows = tileRows,
            TileSizeBytes = tileSizeBytes,
            ContextUpdateTileId = contextUpdateTileId,
            ColStartSb = trimmedCols,
            RowStartSb = trimmedRows,
        };
    }

    /// <summary>libaom <c>tile_log2(blkSize, target)</c>: ceil(log2(target/blkSize)).</summary>
    private static int TileLog2(int blkSize, int target)
    {
        int k = 0;
        while ((blkSize << k) < target) k++;
        return k;
    }

    /// <summary>libaom <c>rb_read_uniform(n)</c>: read uniformly-distributed integer in [0, n).</summary>
    private static int RbReadUniform(ref Av1BitReader br, int n)
    {
        if (n <= 1) return 0; // degenerate case: only one possible value
        int l = GetUnsignedBits(n);
        int m = (1 << l) - n;
        int v = (int)br.ReadBits(l - 1);
        if (v < m) return v;
        return (v << 1) - m + (int)br.ReadBits(1);
    }

    private static int GetUnsignedBits(int n)
    {
        int b = 0;
        while ((1 << b) < n) b++;
        return b;
    }

    private static int ReadDeltaQ(ref Av1BitReader br) =>
        br.ReadFlag() ? ReadInvSignedLiteral(ref br, 6) : 0;

    private static int ReadInvSignedLiteral(ref Av1BitReader br, int bits)
    {
        int v = (int)br.ReadBits(bits);
        bool sign = br.ReadFlag();
        return sign ? -v : v;
    }

    private static Av1QuantParams ReadQuantization(ref Av1BitReader br, Av1SequenceHeader sh, int numPlanes)
    {
        int baseQindex = (int)br.ReadBits(8);
        int yDcDelta = ReadDeltaQ(ref br);
        int uDcDelta = 0, uAcDelta = 0, vDcDelta = 0, vAcDelta = 0;
        if (numPlanes > 1)
        {
            int diffUv = sh.SeparateUvDeltas ? (int)br.ReadBits(1) : 0;
            uDcDelta = ReadDeltaQ(ref br);
            uAcDelta = ReadDeltaQ(ref br);
            if (diffUv != 0)
            {
                vDcDelta = ReadDeltaQ(ref br);
                vAcDelta = ReadDeltaQ(ref br);
            }
            else
            {
                vDcDelta = uDcDelta;
                vAcDelta = uAcDelta;
            }
        }
        bool usingQmatrix = br.ReadFlag();
        int qmY = 0, qmU = 0, qmV = 0;
        if (usingQmatrix)
        {
            qmY = (int)br.ReadBits(4);
            qmU = (int)br.ReadBits(4);
            qmV = sh.SeparateUvDeltas ? (int)br.ReadBits(4) : qmU;
        }
        return new Av1QuantParams
        {
            BaseQindex = baseQindex,
            YDcDeltaQ = yDcDelta,
            UDcDeltaQ = uDcDelta,
            UAcDeltaQ = uAcDelta,
            VDcDeltaQ = vDcDelta,
            VAcDeltaQ = vAcDelta,
            UsingQmatrix = usingQmatrix,
            QmatrixLevelY = qmY,
            QmatrixLevelU = qmU,
            QmatrixLevelV = qmV,
        };
    }

    private static Av1SegmentationParams ReadSegmentation(
        ref Av1BitReader br, Av1FrameHeader prefix, bool segmentationPrimaryRefIsNone)
    {
        bool enabled = br.ReadFlag();
        if (!enabled)
        {
            return new Av1SegmentationParams { Enabled = false };
        }
        bool updateMap, temporalUpdate, updateData;
        if (segmentationPrimaryRefIsNone)
        {
            updateMap = true;
            temporalUpdate = false;
            updateData = true;
        }
        else
        {
            updateMap = br.ReadFlag();
            temporalUpdate = updateMap ? br.ReadFlag() : false;
            updateData = br.ReadFlag();
        }

        var featureEnabled = new bool[8, 8];
        var featureData = new int[8, 8];
        if (updateData)
        {
            // Per AV1 spec 5.9.14: 8 segments, 8 features (SEG_LVL_MAX).
            // SEG_FEATURE_DATA_MAX[8] = {255, 63, 63, 63, 63, 7, 0, 0}; signed[6]=true,5,4,3,2,1
            int[] dataMaxs = { 255, 63, 63, 63, 63, 7, 0, 0 };
            bool[] isSigned = { true, true, true, true, true, false, false, false };
            for (int i = 0; i < 8; i++)
            {
                for (int j = 0; j < 8; j++)
                {
                    bool fe = br.ReadFlag();
                    if (fe)
                    {
                        int dataMax = dataMaxs[j];
                        int dataMin = -dataMax;
                        int ubits = GetUnsignedBits(dataMax + (isSigned[j] ? 0 : 1));
                        int data = isSigned[j]
                            ? ReadInvSignedLiteral(ref br, ubits)
                            : (int)br.ReadBits(ubits);
                        if (data < dataMin) data = dataMin;
                        if (data > dataMax) data = dataMax;
                        featureEnabled[i, j] = true;
                        featureData[i, j] = data;
                    }
                }
            }
        }
        return new Av1SegmentationParams
        {
            Enabled = enabled,
            UpdateMap = updateMap,
            TemporalUpdate = temporalUpdate,
            UpdateData = updateData,
            FeatureEnabled = featureEnabled,
            FeatureData = featureData,
        };
    }

    private static Av1LoopFilterParams ReadLoopFilter(ref Av1BitReader br, int numPlanes)
    {
        int level0 = (int)br.ReadBits(6);
        int level1 = (int)br.ReadBits(6);
        int levelU = 0, levelV = 0;
        if (numPlanes > 1 && (level0 != 0 || level1 != 0))
        {
            levelU = (int)br.ReadBits(6);
            levelV = (int)br.ReadBits(6);
        }
        int sharpness = (int)br.ReadBits(3);
        bool modeRefDeltaEnabled = br.ReadFlag();
        bool modeRefDeltaUpdate = false;
        var refDeltas = new int[8] { 1, 0, 0, 0, -1, 0, -1, -1 };
        var modeDeltas = new int[2];
        if (modeRefDeltaEnabled)
        {
            modeRefDeltaUpdate = br.ReadFlag();
            if (modeRefDeltaUpdate)
            {
                for (int i = 0; i < 8; i++)
                {
                    if (br.ReadFlag()) refDeltas[i] = ReadInvSignedLiteral(ref br, 6);
                }
                for (int i = 0; i < 2; i++)
                {
                    if (br.ReadFlag()) modeDeltas[i] = ReadInvSignedLiteral(ref br, 6);
                }
            }
        }
        return new Av1LoopFilterParams
        {
            FilterLevel0 = level0,
            FilterLevel1 = level1,
            FilterLevelU = levelU,
            FilterLevelV = levelV,
            SharpnessLevel = sharpness,
            ModeRefDeltaEnabled = modeRefDeltaEnabled,
            ModeRefDeltaUpdate = modeRefDeltaUpdate,
            RefDeltas = refDeltas,
            ModeDeltas = modeDeltas,
        };
    }

    private static Av1CdefParams ReadCdef(ref Av1BitReader br, int numPlanes)
    {
        int damping = (int)br.ReadBits(2) + 3;
        int bits = (int)br.ReadBits(2);
        int n = 1 << bits;
        var ys = new int[8];
        var uvs = new int[8];
        // CDEF_STRENGTH_BITS = 6
        for (int i = 0; i < n; i++)
        {
            ys[i] = (int)br.ReadBits(6);
            if (numPlanes > 1) uvs[i] = (int)br.ReadBits(6);
        }
        return new Av1CdefParams
        {
            Damping = damping,
            Bits = bits,
            YStrengths = ys,
            UvStrengths = uvs,
        };
    }

    private static Av1LrParams ReadLrParams(ref Av1BitReader br, Av1SequenceHeader sh, int numPlanes)
    {
        var perPlane = new Av1RestorationType[3];
        var unitSize = new int[3];
        bool allNone = true;
        bool chromaNone = true;
        for (int p = 0; p < numPlanes; p++)
        {
            // 2 bits (or 1 + optional 1) per plane:
            //   bit0=1 -> bit1 chooses SGRPROJ vs WIENER
            //   bit0=0 -> bit1 chooses SWITCHABLE vs NONE
            Av1RestorationType t;
            if (br.ReadFlag())
                t = br.ReadFlag() ? Av1RestorationType.SgrProj : Av1RestorationType.Wiener;
            else
                t = br.ReadFlag() ? Av1RestorationType.Switchable : Av1RestorationType.None;
            perPlane[p] = t;
            if (t != Av1RestorationType.None)
            {
                allNone = false;
                if (p > 0) chromaNone = false;
            }
        }
        if (!allNone)
        {
            int sbSize = sh.Use128x128Superblock ? 128 : 64;
            // luma: bits to choose between 64/128/256
            unitSize[0] = sbSize;
            if (sbSize == 64)
            {
                if (br.ReadFlag()) unitSize[0] <<= 1;
            }
            if (unitSize[0] > 64)
            {
                if (br.ReadFlag()) unitSize[0] <<= 1;
            }
            // chroma derived
            if (numPlanes > 1)
            {
                int s = Math.Min(sh.SubsamplingX, sh.SubsamplingY);
                if (s != 0 && !chromaNone)
                {
                    unitSize[1] = unitSize[0] >> (br.ReadFlag() ? s : 0);
                }
                else
                {
                    unitSize[1] = unitSize[0];
                }
                unitSize[2] = unitSize[1];
            }
        }
        else
        {
            int max = 256;
            for (int p = 0; p < numPlanes; p++) unitSize[p] = max;
        }
        return new Av1LrParams { PerPlane = perPlane, UnitSize = unitSize };
    }

    private static Av1FilmGrainParams ReadFilmGrain(ref Av1BitReader br)
    {
        bool apply = br.ReadFlag();
        if (!apply) return new Av1FilmGrainParams { ApplyGrain = false };
        int seed = (int)br.ReadBits(16);
        // Intra frames force update_parameters = 1 implicitly per spec sec 5.9.30,
        // libaom only reads update_parameters for INTER_FRAME. For keyframes we
        // pin it to true, and skip the inherit-via-ref-idx branch.
        bool updateParams = true;
        // Past this point AV1 film grain has many fields (scaling points,
        // ar coeffs, ...) - downstream work; we surface only the headline
        // flags so the bit cursor doesn't end up off-by-many. Keyframe AV1
        // streams in BBB have FilmGrainParamsPresent=false so this is dormant.
        return new Av1FilmGrainParams
        {
            ApplyGrain = apply,
            RandomSeed = seed,
            UpdateParameters = updateParams,
        };
    }
}
