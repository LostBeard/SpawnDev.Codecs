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

        return new Av1FrameHeader
        {
            ShowExistingFrame = false,
            FrameType = frameType,
            ShowFrame = showFrame,
            ShowableFrame = showableFrame,
            ErrorResilientMode = errorResilient,
        };
    }
}
