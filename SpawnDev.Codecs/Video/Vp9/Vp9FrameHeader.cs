// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 uncompressed frame header parse per the VP9 Bitstream Specification
// section 6.2 / 7.2.
// https://storage.googleapis.com/downloads.webmproject.org/docs/vp9/vp9-bitstream-specification-v0.6-20160331-draft.pdf
//
// This slice parses enough of the uncompressed header to populate:
//   - frame_marker validity
//   - profile (0-3)
//   - show_existing_frame + frame_to_show_map_idx
//   - frame_type (KEY_FRAME / NON_KEY_FRAME)
//   - show_frame
//   - error_resilient_mode
//   - keyframe / intra-only specifics: bit_depth, color_space, color_range,
//     subsampling_x / subsampling_y, frame width+height, render width+height
//
// We intentionally stop at the point where per-segment / per-tile data begins
// - those are decoder-state items and will land with the compressed-header
// slice (116) that precedes entropy-decoded data.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 frame type code (spec §6.2 "frame_type").</summary>
public enum Vp9FrameType
{
    /// <summary>Keyframe (refreshes all reference frames).</summary>
    Key = 0,
    /// <summary>Inter-predicted frame or intra-only frame.</summary>
    NonKey = 1,
}

/// <summary>VP9 color-space code (spec §6.2 "color_space"). Matches libvpx VP9_CS_*.</summary>
public enum Vp9ColorSpace
{
    /// <summary>Unknown (signals the decoder to infer).</summary>
    Unknown = 0,
    /// <summary>ITU-R BT.601.</summary>
    Bt601 = 1,
    /// <summary>ITU-R BT.709.</summary>
    Bt709 = 2,
    /// <summary>SMPTE-170.</summary>
    Smpte170 = 3,
    /// <summary>SMPTE-240.</summary>
    Smpte240 = 4,
    /// <summary>ITU-R BT.2020.</summary>
    Bt2020 = 5,
    /// <summary>Reserved.</summary>
    Reserved = 6,
    /// <summary>sRGB (profile 1 or 3 only).</summary>
    Srgb = 7,
}

/// <summary>Parsed VP9 uncompressed frame header.</summary>
public sealed record Vp9FrameHeader
{
    /// <summary>VP9 profile 0..3 (4:2:0 8-bit, 4:2:0 high-bit-depth, 4:4:4 8-bit, 4:4:4 high-bit-depth).</summary>
    public required int Profile { get; init; }

    /// <summary>
    /// True when the frame is just a re-display of an existing reference
    /// frame. When set, <see cref="FrameToShowMapIdx"/> indicates which
    /// reference slot to show; no other fields below are populated.
    /// </summary>
    public required bool ShowExistingFrame { get; init; }

    /// <summary>Reference-frame slot (0-7) to re-display. Only valid when <see cref="ShowExistingFrame"/>.</summary>
    public int FrameToShowMapIdx { get; init; }

    /// <summary>KEY_FRAME vs NON_KEY_FRAME.</summary>
    public Vp9FrameType FrameType { get; init; }

    /// <summary>Whether this frame is to be displayed (false for altref and intra-only hidden).</summary>
    public bool ShowFrame { get; init; }

    /// <summary>Error-resilient-mode flag.</summary>
    public bool ErrorResilientMode { get; init; }

    /// <summary>True when this is an intra-only frame (non-key, not shown, altref-resolved).</summary>
    public bool IntraOnly { get; init; }

    /// <summary>Luma bit depth: 8, 10, or 12. Only populated for keyframes / intra-only / profile>0.</summary>
    public int BitDepth { get; init; } = 8;

    /// <summary>Color-space enumeration (only populated when the header provided it).</summary>
    public Vp9ColorSpace ColorSpace { get; init; } = Vp9ColorSpace.Unknown;

    /// <summary>True for full range (0..255 at 8-bit), false for studio range.</summary>
    public bool ColorRangeFull { get; init; }

    /// <summary>True when chroma-X is subsampled (4:2:0 or 4:2:2).</summary>
    public bool SubsamplingX { get; init; }

    /// <summary>True when chroma-Y is subsampled (4:2:0).</summary>
    public bool SubsamplingY { get; init; }

    /// <summary>Luma width in pixels. Valid for keyframe / intra-only.</summary>
    public int FrameWidth { get; init; }

    /// <summary>Luma height in pixels. Valid for keyframe / intra-only.</summary>
    public int FrameHeight { get; init; }

    /// <summary>Render-target width, or 0 when absent.</summary>
    public int RenderWidth { get; init; }

    /// <summary>Render-target height, or 0 when absent.</summary>
    public int RenderHeight { get; init; }
}

/// <summary>Stateless parser for the VP9 uncompressed frame header.</summary>
public static class Vp9FrameHeaderParser
{
    private const int FrameMarkerExpected = 0b10;
    private const byte SyncByte0 = 0x49;
    private const byte SyncByte1 = 0x83;
    private const byte SyncByte2 = 0x42;

    /// <summary>Parse the first bits of <paramref name="frame"/> as the uncompressed header.</summary>
    public static Vp9FrameHeader Parse(ReadOnlySpan<byte> frame)
    {
        if (frame.Length < 1) throw new InvalidDataException("VP9 frame is empty.");
        var r = new Vp9BitReader(frame);

        // frame_marker: f(2) must be 0b10.
        uint marker = r.ReadBits(2);
        if (marker != FrameMarkerExpected)
            throw new InvalidDataException($"VP9 frame_marker 0b{Convert.ToString(marker, 2).PadLeft(2, '0')} invalid, expected 0b10.");

        // profile_low_bit: f(1), profile_high_bit: f(1). profile = (high<<1) | low.
        int profileLow = (int)r.ReadBits(1);
        int profileHigh = (int)r.ReadBits(1);
        int profile = (profileHigh << 1) | profileLow;
        if (profile == 3)
        {
            // reserved_zero_bit: f(1) must be 0.
            uint reserved = r.ReadBits(1);
            if (reserved != 0)
                throw new InvalidDataException("VP9 profile=3 reserved_zero_bit must be 0.");
        }

        // show_existing_frame: f(1)
        bool showExisting = r.ReadFlag();
        if (showExisting)
        {
            int mapIdx = (int)r.ReadBits(3);
            return new Vp9FrameHeader
            {
                Profile = profile,
                ShowExistingFrame = true,
                FrameToShowMapIdx = mapIdx,
            };
        }

        // frame_type, show_frame, error_resilient_mode.
        var frameType = (Vp9FrameType)r.ReadBits(1);
        bool showFrame = r.ReadFlag();
        bool errorResilient = r.ReadFlag();

        int bitDepth = 8;
        var colorSpace = Vp9ColorSpace.Unknown;
        bool colorRangeFull = false;
        bool subX = true, subY = true; // default 4:2:0
        int frameWidth = 0, frameHeight = 0;
        int renderWidth = 0, renderHeight = 0;
        bool intraOnly = false;

        if (frameType == Vp9FrameType.Key)
        {
            ReadFrameSyncCode(ref r);
            (bitDepth, colorSpace, colorRangeFull, subX, subY) = ReadColorConfig(ref r, profile);
            (frameWidth, frameHeight, renderWidth, renderHeight) = ReadFrameAndRenderSize(ref r);
        }
        else
        {
            // NON_KEY_FRAME: intra_only = show_frame==0 ? f(1) : 0
            intraOnly = !showFrame && r.ReadFlag();
            if (!errorResilient) r.ReadBits(2); // reset_frame_context (not used in this slice)
            if (intraOnly)
            {
                ReadFrameSyncCode(ref r);
                if (profile > 0)
                {
                    (bitDepth, colorSpace, colorRangeFull, subX, subY) = ReadColorConfig(ref r, profile);
                }
                else
                {
                    // profile 0 intra-only defaults per spec.
                    bitDepth = 8;
                    colorSpace = Vp9ColorSpace.Bt601;
                    subX = subY = true;
                }
                r.ReadBits(8); // refresh_frame_flags
                (frameWidth, frameHeight, renderWidth, renderHeight) = ReadFrameAndRenderSize(ref r);
            }
            // Inter frame: skipping remainder (ref indices, interp filter, loop_filter,
            // etc.) - those come in slice 116.
        }

        return new Vp9FrameHeader
        {
            Profile = profile,
            ShowExistingFrame = false,
            FrameType = frameType,
            ShowFrame = showFrame,
            ErrorResilientMode = errorResilient,
            IntraOnly = intraOnly,
            BitDepth = bitDepth,
            ColorSpace = colorSpace,
            ColorRangeFull = colorRangeFull,
            SubsamplingX = subX,
            SubsamplingY = subY,
            FrameWidth = frameWidth,
            FrameHeight = frameHeight,
            RenderWidth = renderWidth,
            RenderHeight = renderHeight,
        };
    }

    private static void ReadFrameSyncCode(ref Vp9BitReader r)
    {
        // Three f(8) bytes: 0x49 0x83 0x42. Verifies we're on a valid frame boundary.
        if (r.ReadBits(8) != SyncByte0 || r.ReadBits(8) != SyncByte1 || r.ReadBits(8) != SyncByte2)
            throw new InvalidDataException("VP9 frame_sync_code mismatch (expected 49 83 42).");
    }

    private static (int bitDepth, Vp9ColorSpace cs, bool rangeFull, bool subX, bool subY)
        ReadColorConfig(ref Vp9BitReader r, int profile)
    {
        int bitDepth = 8;
        if (profile >= 2)
        {
            bool tenOrTwelve = r.ReadFlag();
            bitDepth = tenOrTwelve ? 12 : 10;
        }
        var cs = (Vp9ColorSpace)r.ReadBits(3);
        bool rangeFull = false;
        bool subX, subY;
        if (cs != Vp9ColorSpace.Srgb)
        {
            rangeFull = r.ReadFlag();
            if (profile == 1 || profile == 3)
            {
                subX = r.ReadFlag();
                subY = r.ReadFlag();
                uint reserved = r.ReadBits(1);
                if (reserved != 0)
                    throw new InvalidDataException("VP9 color_config reserved bit must be 0.");
            }
            else
            {
                // Profile 0/2: 4:2:0 only.
                subX = subY = true;
            }
        }
        else
        {
            // sRGB implies 4:4:4 + full range; only valid in profile 1 or 3.
            if (profile != 1 && profile != 3)
                throw new InvalidDataException("VP9 sRGB color space requires profile 1 or 3.");
            rangeFull = true;
            subX = subY = false;
            uint reserved = r.ReadBits(1);
            if (reserved != 0)
                throw new InvalidDataException("VP9 sRGB color_config reserved bit must be 0.");
        }
        return (bitDepth, cs, rangeFull, subX, subY);
    }

    private static (int frameWidth, int frameHeight, int renderWidth, int renderHeight)
        ReadFrameAndRenderSize(ref Vp9BitReader r)
    {
        int fw = (int)r.ReadBits(16) + 1;
        int fh = (int)r.ReadBits(16) + 1;
        int rw = 0, rh = 0;
        if (r.ReadFlag()) // render_and_frame_size_different
        {
            rw = (int)r.ReadBits(16) + 1;
            rh = (int)r.ReadBits(16) + 1;
        }
        return (fw, fh, rw, rh);
    }
}
