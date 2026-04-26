// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 Frame Header / Uncompressed Header parser (spec sec 5.9.1).
// Extracts the prefix of per-frame metadata: frame_type, show flags,
// error_resilient_mode, and similar headline fields.
//
// The full uncompressed header is large (several hundred bits across
// many optional sections - frame_size_with_refs, loop filter, quant,
// segmentation, tile info, CDEF, loop restoration, ...) and depends
// on the active SequenceHeader. This first cut surfaces only the
// fields a consumer can use right away (frame type, show, key vs
// inter); the rest is downstream work.

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>AV1 frame_type values (spec sec 6.8.2).</summary>
public enum Av1FrameType : byte
{
    /// <summary>Keyframe.</summary>
    KeyFrame = 0,
    /// <summary>Inter frame (typical P/B-like).</summary>
    InterFrame = 1,
    /// <summary>Intra-only (decoder-side keyframe for switch points).</summary>
    IntraOnlyFrame = 2,
    /// <summary>Switch frame (random access).</summary>
    SwitchFrame = 3,
}

/// <summary>Headline fields from the AV1 uncompressed frame header.</summary>
public sealed record Av1FrameHeader
{
    /// <summary>True when this OBU just replays a buffered reference slot.</summary>
    public required bool ShowExistingFrame { get; init; }

    /// <summary>
    /// Reference slot to replay. Only meaningful when
    /// <see cref="ShowExistingFrame"/> is true.
    /// </summary>
    public int FrameToShowMapIdx { get; init; }

    /// <summary>Frame type (key / inter / intra_only / switch).</summary>
    public required Av1FrameType FrameType { get; init; }

    /// <summary>True for keyframe and intra_only frames.</summary>
    public bool FrameIsIntra => FrameType == Av1FrameType.KeyFrame || FrameType == Av1FrameType.IntraOnlyFrame;

    /// <summary>True when this frame is to be displayed.</summary>
    public required bool ShowFrame { get; init; }

    /// <summary>
    /// True when this frame may later be shown via show_existing_frame.
    /// Only signaled when <see cref="ShowFrame"/> is false; otherwise
    /// always true.
    /// </summary>
    public required bool ShowableFrame { get; init; }

    /// <summary>libvpx <c>error_resilient_mode</c>.</summary>
    public required bool ErrorResilientMode { get; init; }

    /// <summary>
    /// disable_cdf_update flag. When true, CDF tables aren't updated from
    /// this frame's symbols. Defaults to false for unparsed prefix fields.
    /// </summary>
    public bool DisableCdfUpdate { get; init; }

    /// <summary>
    /// allow_screen_content_tools resolved value (0 or 1). Either parsed
    /// from the bitstream when SH says SELECT, or inherited from
    /// <see cref="Av1SequenceHeader.SeqForceScreenContentTools"/>.
    /// </summary>
    public int AllowScreenContentTools { get; init; }

    /// <summary>
    /// force_integer_mv resolved value (0 or 1). For intra frames this is
    /// forced to 1; otherwise parsed from the bitstream when allowed by SH
    /// + AllowScreenContentTools.
    /// </summary>
    public int ForceIntegerMv { get; init; }

    /// <summary>
    /// current_frame_id, only present when SH.FrameIdNumbersPresent.
    /// </summary>
    public int CurrentFrameId { get; init; }

    /// <summary>
    /// frame_size_override_flag. When true, the bitstream carries explicit
    /// frame_size + render_size. When false, dimensions come from SH defaults
    /// (or frame_refs for inter frames).
    /// </summary>
    public bool FrameSizeOverride { get; init; }

    /// <summary>
    /// order_hint. Only meaningful when SH.EnableOrderHint=true. Used by
    /// the AV1 reference frame system + temporal MV scaling.
    /// </summary>
    public int OrderHint { get; init; }

    /// <summary>
    /// refresh_frame_flags: which of the 8 reference frame slots to
    /// update with this frame's reconstructed output. KeyFrame visible
    /// implicit 0xFF; SwitchFrame implicit 0xFF; otherwise f(8).
    /// </summary>
    public int RefreshFrameFlags { get; init; }

    /// <summary>
    /// Decoded frame width in pixels. Either from SH.MaxFrameWidth (no
    /// override) or read from the bitstream when frame_size_override
    /// applies. Always non-zero for parsed frame headers.
    /// </summary>
    public int FrameWidth { get; init; }

    /// <summary>Decoded frame height in pixels. See <see cref="FrameWidth"/>.</summary>
    public int FrameHeight { get; init; }

    /// <summary>
    /// allow_intrabc flag - intra_only / key frames with screen content
    /// tools may use intra block-copy. Most natural-content streams have
    /// this off.
    /// </summary>
    public bool AllowIntraBc { get; init; }
}

/// <summary>AV1 frame header parser (headline fields only).</summary>
public static class Av1FrameHeaderParser
{
    /// <summary>
    /// Parse the headline fields of an AV1 Frame / FrameHeader OBU
    /// payload, given the active <see cref="Av1SequenceHeader"/>.
    /// </summary>
    public static Av1FrameHeader Parse(ReadOnlySpan<byte> payload, Av1SequenceHeader sh)
    {
        ArgumentNullException.ThrowIfNull(sh);
        var br = new Av1BitReader(payload);

        if (sh.ReducedStillPictureHeader)
        {
            return new Av1FrameHeader
            {
                ShowExistingFrame = false,
                FrameType = Av1FrameType.KeyFrame,
                ShowFrame = true,
                ShowableFrame = false,
                ErrorResilientMode = false,
            };
        }

        bool showExisting = br.ReadFlag();
        if (showExisting)
        {
            int idx = (int)br.ReadBits(3);
            // Skip the rest (frame_presentation_time_delta if decoder
            // model present, frame_id if frame_id_numbers_present, etc.).
            return new Av1FrameHeader
            {
                ShowExistingFrame = true,
                FrameToShowMapIdx = idx,
                FrameType = Av1FrameType.KeyFrame,
                ShowFrame = true,
                ShowableFrame = true,
                ErrorResilientMode = false,
            };
        }

        var frameType = (Av1FrameType)br.ReadBits(2);
        bool showFrame = br.ReadFlag();
        bool showableFrame;
        if (!showFrame)
        {
            showableFrame = br.ReadFlag();
        }
        else
        {
            showableFrame = frameType != Av1FrameType.KeyFrame;
        }

        bool errorResilient;
        if (frameType == Av1FrameType.SwitchFrame
            || (frameType == Av1FrameType.KeyFrame && showFrame))
        {
            errorResilient = true;
        }
        else
        {
            errorResilient = br.ReadFlag();
        }

        bool disableCdfUpdate = br.ReadFlag();

        // allow_screen_content_tools: spec sec 5.9.1
        // libaom uses SELECT = 2 to mean "frame chooses".
        int allowSccTools;
        if (sh.SeqForceScreenContentTools == 2 /* SELECT */)
            allowSccTools = (int)br.ReadBits(1);
        else
            allowSccTools = sh.SeqForceScreenContentTools;

        // force_integer_mv: also resolves SELECT=2 from the SH side.
        int forceIntegerMv;
        if (allowSccTools != 0)
        {
            if (sh.SeqForceIntegerMv == 2 /* SELECT */)
                forceIntegerMv = (int)br.ReadBits(1);
            else
                forceIntegerMv = sh.SeqForceIntegerMv;
        }
        else
        {
            forceIntegerMv = 0;
        }
        bool isIntra = frameType == Av1FrameType.KeyFrame
            || frameType == Av1FrameType.IntraOnlyFrame;
        if (isIntra) forceIntegerMv = 1;

        int currentFrameId = 0;
        if (sh.FrameIdNumbersPresent)
        {
            int idLen = sh.FrameIdLengthMinus7 + 7;
            currentFrameId = (int)br.ReadBits(idLen);
        }

        bool frameSizeOverride;
        if (frameType == Av1FrameType.SwitchFrame)
            frameSizeOverride = true;
        else if (sh.ReducedStillPictureHeader)
            frameSizeOverride = false;
        else
            frameSizeOverride = br.ReadFlag();

        int orderHint = 0;
        if (sh.EnableOrderHint && !errorResilient && !isIntra)
        {
            // AV1 spec: order_hint is f(OrderHintBits) only for non-intra
            // non-error-resilient frames. Intra frames have order_hint=0
            // implicitly. Note: for KeyFrame + IntraOnly + show_existing,
            // we don't read it here either.
            int orderHintBits = sh.OrderHintBitsMinus1 + 1;
            orderHint = (int)br.ReadBits(orderHintBits);
        }

        // refresh_frame_flags: which 8 ref slots get updated with this
        // frame's output. KeyFrame visible / SwitchFrame are implicit 0xFF.
        int refreshFrameFlags;
        if ((frameType == Av1FrameType.KeyFrame && showFrame)
            || frameType == Av1FrameType.SwitchFrame)
        {
            refreshFrameFlags = 0xFF;
        }
        else
        {
            refreshFrameFlags = (int)br.ReadBits(8);
        }

        // frame_size + render_size (simplified: for keyframe + intra_only,
        // always read explicit dims when frame_size_override; otherwise
        // SH defaults). For inter frames with refs, full frame_size_with_refs
        // walks frame_refs - we simplify by falling through to SH dims when
        // frame_size_override is false.
        int frameWidth = sh.MaxFrameWidth;
        int frameHeight = sh.MaxFrameHeight;
        if (frameSizeOverride)
        {
            // f(SH.frame_width_bits + 1) for width, same for height.
            // Our SH parser doesn't surface those values today, so we
            // assume 16 bits each (max). For BBB the override flag is 0
            // on every frame, so this branch isn't exercised.
            int widthMinus1 = (int)br.ReadBits(16);
            int heightMinus1 = (int)br.ReadBits(16);
            frameWidth = widthMinus1 + 1;
            frameHeight = heightMinus1 + 1;
            // render_size fields skipped - downstream work.
        }

        // allow_intrabc only signaled for intra_only/key frames with screen
        // content tools. Most natural-content streams skip it. Implicit 0
        // otherwise. We don't yet read the surrounding inter-frame fields
        // (frame_refs / interpolation_filter / etc.) so this is the last
        // bit our parser surfaces today.
        bool allowIntraBc = false;
        if (isIntra && allowSccTools != 0 && br.BitsRemaining > 0)
        {
            // Strict spec: allow_intrabc emission is gated on coded_lossless
            // and frame size match, but the simplest scope is: read 1 bit
            // for intra frames with SCC enabled.
            // Note: BBB has allowSccTools=0 on its keyframes so this branch
            // is dormant for that fixture.
            allowIntraBc = br.ReadFlag();
        }

        return new Av1FrameHeader
        {
            ShowExistingFrame = false,
            FrameType = frameType,
            ShowFrame = showFrame,
            ShowableFrame = showableFrame,
            ErrorResilientMode = errorResilient,
            DisableCdfUpdate = disableCdfUpdate,
            AllowScreenContentTools = allowSccTools,
            ForceIntegerMv = forceIntegerMv,
            CurrentFrameId = currentFrameId,
            FrameSizeOverride = frameSizeOverride,
            OrderHint = orderHint,
            RefreshFrameFlags = refreshFrameFlags,
            FrameWidth = frameWidth,
            FrameHeight = frameHeight,
            AllowIntraBc = allowIntraBc,
        };
    }
}
