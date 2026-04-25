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

        bw.WriteTrailingBits();
        return bw.ToArray();
    }
}
