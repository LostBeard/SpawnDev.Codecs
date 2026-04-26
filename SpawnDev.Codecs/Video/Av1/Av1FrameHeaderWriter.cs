// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 Frame Header writer - inverse of Av1FrameHeaderParser for the
// prefix fields (show_existing_frame / frame_type / show_frame /
// showable_frame / error_resilient_mode).
//
// Matches the parser's scope: emits the leading bits a consumer can
// use right away. The full uncompressed_header continues with
// frame_size, loop filter, quant, segmentation, tile info, CDEF, loop
// restoration, etc., all of which depend on active SequenceHeader and
// reference-frame state - those are downstream encoder work, not the
// foundation layer.

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>Caller-facing config for emitting the AV1 frame header prefix.</summary>
public sealed record Av1FrameHeaderConfig
{
    /// <summary>True for show_existing_frame OBUs (replays a reference slot).</summary>
    public bool ShowExistingFrame { get; init; }

    /// <summary>Reference map index. Only used when ShowExistingFrame=true.</summary>
    public int FrameToShowMapIdx { get; init; }

    /// <summary>Frame type when ShowExistingFrame=false.</summary>
    public Av1FrameType FrameType { get; init; } = Av1FrameType.KeyFrame;

    /// <summary>True if this frame is displayed.</summary>
    public bool ShowFrame { get; init; } = true;

    /// <summary>
    /// True if this frame is later showable via show_existing_frame.
    /// Only signaled when ShowFrame=false; otherwise computed by the
    /// parser as <c>FrameType != KeyFrame</c>.
    /// </summary>
    public bool ShowableFrame { get; init; }

    /// <summary>error_resilient_mode flag.</summary>
    public bool ErrorResilientMode { get; init; }

    /// <summary>disable_cdf_update flag.</summary>
    public bool DisableCdfUpdate { get; init; }

    /// <summary>
    /// allow_screen_content_tools value (0 or 1). Only emitted explicitly
    /// when SH.SeqForceScreenContentTools is SELECT (2); otherwise inherited
    /// from SH and the writer skips emission to match what the parser reads.
    /// </summary>
    public int AllowScreenContentTools { get; init; }

    /// <summary>
    /// force_integer_mv value (0 or 1). Only emitted when
    /// AllowScreenContentTools != 0 and SH.SeqForceIntegerMv is SELECT (2).
    /// For intra frames the parser forces this to 1 regardless.
    /// </summary>
    public int ForceIntegerMv { get; init; }

    /// <summary>current_frame_id (only emitted when SH.FrameIdNumbersPresent).</summary>
    public int CurrentFrameId { get; init; }

    /// <summary>
    /// frame_size_override_flag. Implicit for SwitchFrame (true) and when
    /// ReducedStillPictureHeader (false); otherwise emitted.
    /// </summary>
    public bool FrameSizeOverride { get; init; }

    /// <summary>
    /// frame_width when FrameSizeOverride=true. Falls back to SH.MaxFrameWidth
    /// when override is false (writer doesn't emit override-size bits).
    /// Per current parser scope, written/read as 16-bit (max - 1).
    /// </summary>
    public int FrameWidth { get; init; }

    /// <summary>frame_height when FrameSizeOverride=true. See <see cref="FrameWidth"/>.</summary>
    public int FrameHeight { get; init; }

    /// <summary>
    /// allow_intrabc flag (only for intra_only/key frames with screen
    /// content tools enabled). See Av1FrameHeader.AllowIntraBc.
    /// </summary>
    public bool AllowIntraBc { get; init; }

    /// <summary>
    /// Build a writer config from a parsed <see cref="Av1FrameHeader"/>.
    /// Round-trip helper: parse(FH) -&gt; FromHeader -&gt; EmitPayload should
    /// produce byte-equivalent output for any frame header whose fields
    /// our parser surfaces.
    /// </summary>
    public static Av1FrameHeaderConfig FromHeader(Av1FrameHeader fh)
    {
        ArgumentNullException.ThrowIfNull(fh);
        return new Av1FrameHeaderConfig
        {
            ShowExistingFrame = fh.ShowExistingFrame,
            FrameToShowMapIdx = fh.FrameToShowMapIdx,
            FrameType = fh.FrameType,
            ShowFrame = fh.ShowFrame,
            ShowableFrame = fh.ShowableFrame,
            ErrorResilientMode = fh.ErrorResilientMode,
            DisableCdfUpdate = fh.DisableCdfUpdate,
            AllowScreenContentTools = fh.AllowScreenContentTools,
            ForceIntegerMv = fh.ForceIntegerMv,
            CurrentFrameId = fh.CurrentFrameId,
            FrameSizeOverride = fh.FrameSizeOverride,
            OrderHint = fh.OrderHint,
            RefreshFrameFlags = fh.RefreshFrameFlags,
            FrameWidth = fh.FrameWidth,
            FrameHeight = fh.FrameHeight,
            AllowIntraBc = fh.AllowIntraBc,
        };
    }

    /// <summary>
    /// order_hint value. Only emitted when SH.EnableOrderHint=true AND
    /// the frame is non-intra AND not error_resilient_mode.
    /// </summary>
    public int OrderHint { get; init; }

    /// <summary>
    /// refresh_frame_flags (8-bit mask). Implicit 0xFF for visible
    /// KeyFrame and SwitchFrame; otherwise emitted as f(8).
    /// </summary>
    public int RefreshFrameFlags { get; init; }
}

/// <summary>AV1 frame header prefix writer.</summary>
public static class Av1FrameHeaderWriter
{
    /// <summary>
    /// Emit the AV1 uncompressed_header prefix bits matching
    /// <see cref="Av1FrameHeaderParser.Parse"/>'s read scope. Returns
    /// the bit stream as a byte payload (not byte-aligned - the trailing
    /// portion of the AV1 frame header continues with more fields the
    /// caller is responsible for emitting downstream).
    /// </summary>
    public static byte[] EmitPayload(Av1FrameHeaderConfig cfg, Av1SequenceHeader sh)
    {
        ArgumentNullException.ThrowIfNull(cfg);
        ArgumentNullException.ThrowIfNull(sh);

        var bw = new Av1BitWriter();

        if (sh.ReducedStillPictureHeader)
        {
            // The parser returns a fixed key/show/intra header with no bits
            // read. Match by emitting nothing and trailing-bits aligning.
            bw.WriteTrailingBits();
            return bw.ToArray();
        }

        bw.WriteFlag(cfg.ShowExistingFrame);
        if (cfg.ShowExistingFrame)
        {
            if ((uint)cfg.FrameToShowMapIdx > 7)
                throw new ArgumentOutOfRangeException(nameof(cfg.FrameToShowMapIdx),
                    "frame_to_show_map_idx is f(3): 0..7.");
            bw.WriteBits(cfg.FrameToShowMapIdx, 3);
            bw.WriteTrailingBits();
            return bw.ToArray();
        }

        bw.WriteBits((int)cfg.FrameType, 2);
        bw.WriteFlag(cfg.ShowFrame);

        if (!cfg.ShowFrame)
        {
            bw.WriteFlag(cfg.ShowableFrame);
        }

        // error_resilient_mode is implicit in two cases - matches parser:
        bool errImplicit = cfg.FrameType == Av1FrameType.SwitchFrame
            || (cfg.FrameType == Av1FrameType.KeyFrame && cfg.ShowFrame);
        if (errImplicit)
        {
            if (!cfg.ErrorResilientMode)
                throw new ArgumentException(
                    "error_resilient_mode is implicit (true) for SwitchFrame and visible KeyFrame.",
                    nameof(cfg));
        }
        else
        {
            bw.WriteFlag(cfg.ErrorResilientMode);
        }

        bw.WriteFlag(cfg.DisableCdfUpdate);

        if (sh.SeqForceScreenContentTools == 2 /* SELECT */)
        {
            if ((uint)cfg.AllowScreenContentTools > 1)
                throw new ArgumentOutOfRangeException(nameof(cfg.AllowScreenContentTools),
                    "allow_screen_content_tools is f(1): 0 or 1 when SH chose SELECT.");
            bw.WriteBits(cfg.AllowScreenContentTools, 1);
        }
        // else: inherited from SH, no emission.

        bool isIntra = cfg.FrameType == Av1FrameType.KeyFrame
            || cfg.FrameType == Av1FrameType.IntraOnlyFrame;
        if (cfg.AllowScreenContentTools != 0 && !isIntra
            && sh.SeqForceIntegerMv == 2 /* SELECT */)
        {
            if ((uint)cfg.ForceIntegerMv > 1)
                throw new ArgumentOutOfRangeException(nameof(cfg.ForceIntegerMv),
                    "force_integer_mv is f(1): 0 or 1 when SH chose SELECT and frame is non-intra.");
            bw.WriteBits(cfg.ForceIntegerMv, 1);
        }
        // else: implicit per AV1 spec - no emission.

        if (sh.FrameIdNumbersPresent)
        {
            int idLen = sh.FrameIdLengthMinus7 + 7;
            if ((uint)cfg.CurrentFrameId >= (1u << idLen))
                throw new ArgumentOutOfRangeException(nameof(cfg.CurrentFrameId),
                    $"current_frame_id must fit in {idLen} bits.");
            bw.WriteBits(cfg.CurrentFrameId, idLen);
        }

        if (cfg.FrameType == Av1FrameType.SwitchFrame)
        {
            if (!cfg.FrameSizeOverride)
                throw new ArgumentException(
                    "frame_size_override_flag is implicitly true for SwitchFrame.",
                    nameof(cfg));
        }
        else if (sh.ReducedStillPictureHeader)
        {
            if (cfg.FrameSizeOverride)
                throw new ArgumentException(
                    "frame_size_override_flag is implicitly false for reduced_still_picture_header.",
                    nameof(cfg));
        }
        else
        {
            bw.WriteFlag(cfg.FrameSizeOverride);
        }

        // When override is set, emit frame_width + frame_height. Parser
        // reads 16 bits each (simplified scope); we mirror that here.
        if (cfg.FrameSizeOverride)
        {
            int w = cfg.FrameWidth > 0 ? cfg.FrameWidth : sh.MaxFrameWidth;
            int h = cfg.FrameHeight > 0 ? cfg.FrameHeight : sh.MaxFrameHeight;
            if ((uint)(w - 1) > 0xFFFF) throw new ArgumentOutOfRangeException(nameof(cfg.FrameWidth));
            if ((uint)(h - 1) > 0xFFFF) throw new ArgumentOutOfRangeException(nameof(cfg.FrameHeight));
            bw.WriteBits(w - 1, 16);
            bw.WriteBits(h - 1, 16);
        }

        if (sh.EnableOrderHint && !cfg.ErrorResilientMode && !isIntra)
        {
            int orderHintBits = sh.OrderHintBitsMinus1 + 1;
            if ((uint)cfg.OrderHint >= (1u << orderHintBits))
                throw new ArgumentOutOfRangeException(nameof(cfg.OrderHint),
                    $"order_hint must fit in {orderHintBits} bits.");
            bw.WriteBits(cfg.OrderHint, orderHintBits);
        }

        // refresh_frame_flags
        bool refreshImplicit = (cfg.FrameType == Av1FrameType.KeyFrame && cfg.ShowFrame)
            || cfg.FrameType == Av1FrameType.SwitchFrame;
        if (refreshImplicit)
        {
            if (cfg.RefreshFrameFlags != 0xFF && cfg.RefreshFrameFlags != 0)
                throw new ArgumentException(
                    "refresh_frame_flags is implicit 0xFF for visible KeyFrame and SwitchFrame.",
                    nameof(cfg));
        }
        else
        {
            if ((uint)cfg.RefreshFrameFlags > 0xFF)
                throw new ArgumentOutOfRangeException(nameof(cfg.RefreshFrameFlags),
                    "refresh_frame_flags is f(8): 0..255.");
            bw.WriteBits(cfg.RefreshFrameFlags, 8);
        }

        // allow_intrabc only for intra frames with SCC tools active.
        if (isIntra && cfg.AllowScreenContentTools != 0)
        {
            bw.WriteFlag(cfg.AllowIntraBc);
        }

        bw.WriteTrailingBits();
        return bw.ToArray();
    }
}
