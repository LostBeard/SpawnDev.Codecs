// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 reconstructed frame buffer. Plain CPU-side YUV420 (or other
// subsampling) buffer that the keyframe walker writes pixel blocks
// into and the frame sink reads at the end of decode.
//
// libvpx reference: vp9/common/vp9_blockd.h struct macroblockd_plane
// pre/dst buffers that the per-block decoder writes through.
//
// Layout: each plane is row-major width*height bytes. The walker
// asks for the offset of (mi_row, mi_col) at a given subsampling
// and the writable Span<byte> that covers the block, plus the
// stride.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// VP9 reconstructed frame buffer (Y + U + V planes). Allocated per
/// frame; the keyframe walker writes per-block pixels into it; the
/// frame sink consumes it at end-of-frame.
/// </summary>
public sealed class Vp9FrameBuffer
{
    /// <summary>Luma plane width in pixels.</summary>
    public int LumaWidth { get; }

    /// <summary>Luma plane height in pixels.</summary>
    public int LumaHeight { get; }

    /// <summary>Chroma plane width in pixels.</summary>
    public int ChromaWidth { get; }

    /// <summary>Chroma plane height in pixels.</summary>
    public int ChromaHeight { get; }

    /// <summary>Subsampling pair used to size the chroma planes.</summary>
    public Vp9SubsamplingPair Subsampling { get; }

    /// <summary>Luma plane (row-major, length = LumaWidth * LumaHeight).</summary>
    public byte[] Y { get; }

    /// <summary>Chroma-U plane (row-major, length = ChromaWidth * ChromaHeight).</summary>
    public byte[] U { get; }

    /// <summary>Chroma-V plane.</summary>
    public byte[] V { get; }

    /// <summary>Initialize a fresh frame buffer at the given dimensions.</summary>
    public Vp9FrameBuffer(int lumaWidth, int lumaHeight, Vp9SubsamplingPair subsampling)
    {
        if (lumaWidth <= 0) throw new ArgumentOutOfRangeException(nameof(lumaWidth));
        if (lumaHeight <= 0) throw new ArgumentOutOfRangeException(nameof(lumaHeight));
        LumaWidth = lumaWidth;
        LumaHeight = lumaHeight;
        Subsampling = subsampling;
        ChromaWidth = subsampling.ChromaWidth(lumaWidth);
        ChromaHeight = subsampling.ChromaHeight(lumaHeight);
        Y = new byte[LumaWidth * LumaHeight];
        U = new byte[ChromaWidth * ChromaHeight];
        V = new byte[ChromaWidth * ChromaHeight];
    }
}
