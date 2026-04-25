// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 frame_size_with_refs parser. For inter frames, the
// uncompressed header can either reuse one of the 3 reference
// frames' sizes or carry an explicit width+height.
//
// Bitstream layout (libvpx setup_frame_size_with_refs):
//   For each of REFS_PER_FRAME=3 references in order (LAST,
//     GOLDEN, ALTREF):
//     found_ref f(1)
//     if found_ref: take width/height from that ref's y_crop_width/height
//                   and stop reading the per-ref loop early.
//   if no ref was selected:
//     frame_width_minus_1 f(16); width = value + 1
//     frame_height_minus_1 f(16); height = value + 1
//   render_and_frame_size_different f(1)
//   if set:
//     render_width_minus_1 f(16); rwidth = value + 1
//     render_height_minus_1 f(16); rheight = value + 1
//
// Same render-size sub-field as setup_frame_size for keyframes.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>Parsed frame size for an inter frame.</summary>
public sealed record Vp9FrameSizeWithRefs
{
    /// <summary>Frame width in pixels.</summary>
    public required int FrameWidth { get; init; }

    /// <summary>Frame height in pixels.</summary>
    public required int FrameHeight { get; init; }

    /// <summary>
    /// Reference slot whose size was reused (0=LAST, 1=GOLDEN,
    /// 2=ALTREF), or -1 if explicit width/height were read.
    /// </summary>
    public required int RefFoundIdx { get; init; }

    /// <summary>Render width in pixels (0 if not signaled).</summary>
    public required int RenderWidth { get; init; }

    /// <summary>Render height in pixels (0 if not signaled).</summary>
    public required int RenderHeight { get; init; }

    /// <summary>True when render_and_frame_size_different was set.</summary>
    public required bool RenderSizeOverride { get; init; }
}

/// <summary>Parser for the VP9 inter-frame frame_size_with_refs section.</summary>
public static class Vp9FrameSizeWithRefsParser
{
    /// <summary>libvpx <c>REFS_PER_FRAME</c>.</summary>
    public const int RefsPerFrame = 3;

    /// <summary>
    /// Parse frame_size_with_refs from <paramref name="reader"/>.
    /// </summary>
    /// <param name="reader">Uncompressed-header bit reader.</param>
    /// <param name="refFrameSizes">
    /// Array of 3 <c>(width, height)</c> tuples for the LAST / GOLDEN /
    /// ALTREF references. Caller resolves these from the reference
    /// frame buffer pool using the indices parsed by
    /// <see cref="Vp9ReferenceFrameInfoParser"/>.
    /// </param>
    internal static Vp9FrameSizeWithRefs Parse(
        ref Vp9BitReader reader,
        ReadOnlySpan<(int Width, int Height)> refFrameSizes)
    {
        if (refFrameSizes.Length < RefsPerFrame)
            throw new ArgumentException(
                $"refFrameSizes must hold at least {RefsPerFrame} entries",
                nameof(refFrameSizes));

        int foundIdx = -1;
        int frameWidth = 0;
        int frameHeight = 0;
        for (int i = 0; i < RefsPerFrame; i++)
        {
            // libvpx breaks out of the loop on the first set bit, so
            // subsequent ref bits are NOT read.
            if (reader.ReadFlag())
            {
                foundIdx = i;
                frameWidth = refFrameSizes[i].Width;
                frameHeight = refFrameSizes[i].Height;
                break;
            }
        }

        if (foundIdx < 0)
        {
            frameWidth = (int)reader.ReadBits(16) + 1;
            frameHeight = (int)reader.ReadBits(16) + 1;
        }

        bool renderOverride = reader.ReadFlag();
        int renderWidth = 0;
        int renderHeight = 0;
        if (renderOverride)
        {
            renderWidth = (int)reader.ReadBits(16) + 1;
            renderHeight = (int)reader.ReadBits(16) + 1;
        }

        return new Vp9FrameSizeWithRefs
        {
            FrameWidth = frameWidth,
            FrameHeight = frameHeight,
            RefFoundIdx = foundIdx,
            RenderWidth = renderWidth,
            RenderHeight = renderHeight,
            RenderSizeOverride = renderOverride,
        };
    }

    /// <summary>Convenience overload for unit tests.</summary>
    public static Vp9FrameSizeWithRefs Parse(
        ReadOnlySpan<byte> data,
        ReadOnlySpan<(int Width, int Height)> refFrameSizes)
    {
        var r = new Vp9BitReader(data);
        return Parse(ref r, refFrameSizes);
    }
}
