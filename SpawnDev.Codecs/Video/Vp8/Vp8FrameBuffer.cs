// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP8 reconstructed-frame buffer. Holds the Y / U / V planes for one
// decoded frame in 4:2:0 layout. The macroblock walker fills this
// in-place; the renderer reads from it.
//
// Layout:
//   Y plane: stride * height bytes, stride >= roundUp(width, 16)
//   U plane: uvStride * (height/2) bytes, uvStride >= roundUp(width/2, 8)
//   V plane: same as U
//
// The stride is rounded up so MB-edge writes don't have to worry about
// fractional MBs at the right/bottom edges. The renderer uses the
// nominal width/height to know which pixels are real.

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>VP8 reconstructed-frame YUV420 buffer.</summary>
public sealed class Vp8FrameBuffer
{
    /// <summary>Logical frame width (the nominal width from the frame tag).</summary>
    public int Width { get; }

    /// <summary>Logical frame height.</summary>
    public int Height { get; }

    /// <summary>Y plane stride in bytes (>= rounded-up Width to 16-MB boundary).</summary>
    public int YStride { get; }

    /// <summary>U/V plane stride in bytes.</summary>
    public int UvStride { get; }

    /// <summary>Y plane backing buffer.</summary>
    public byte[] YPlane { get; }
    /// <summary>U plane backing buffer.</summary>
    public byte[] UPlane { get; }
    /// <summary>V plane backing buffer.</summary>
    public byte[] VPlane { get; }

    /// <summary>
    /// Allocate a frame buffer for <paramref name="width"/> x <paramref name="height"/>
    /// with strides rounded up to the macroblock boundary (16 for Y, 8 for UV).
    /// </summary>
    public Vp8FrameBuffer(int width, int height)
    {
        if (width <= 0 || width > 0x3FFF) throw new ArgumentOutOfRangeException(nameof(width));
        if (height <= 0 || height > 0x3FFF) throw new ArgumentOutOfRangeException(nameof(height));

        Width = width;
        Height = height;

        // Round Y stride up to MB (16) and UV stride up to MB chroma (8).
        YStride = (width + 15) & ~15;
        UvStride = ((width / 2) + 7) & ~7;
        // For odd widths, add 1 to the chroma plane allocation.
        int uvWidthAlloc = (width + 1) / 2;
        if (UvStride < uvWidthAlloc) UvStride = (uvWidthAlloc + 7) & ~7;

        int yHeight = (height + 15) & ~15;
        int uvHeight = (height + 1) / 2;
        uvHeight = (uvHeight + 7) & ~7;

        YPlane = new byte[YStride * yHeight];
        UPlane = new byte[UvStride * uvHeight];
        VPlane = new byte[UvStride * uvHeight];
    }

    /// <summary>Macroblock columns in the Y plane (ceil(width / 16)).</summary>
    public int MbCols => (Width + 15) / 16;

    /// <summary>Macroblock rows in the Y plane (ceil(height / 16)).</summary>
    public int MbRows => (Height + 15) / 16;

    /// <summary>Get a span over the Y plane for a 16x16 MB at column <paramref name="mbCol"/>, row <paramref name="mbRow"/>.</summary>
    public Span<byte> GetYMb(int mbCol, int mbRow)
    {
        int offset = mbRow * 16 * YStride + mbCol * 16;
        return YPlane.AsSpan(offset);
    }

    /// <summary>Get a span over the U plane for an 8x8 chroma MB at the supplied MB position.</summary>
    public Span<byte> GetUMb(int mbCol, int mbRow)
    {
        int offset = mbRow * 8 * UvStride + mbCol * 8;
        return UPlane.AsSpan(offset);
    }

    /// <summary>Get a span over the V plane for an 8x8 chroma MB at the supplied MB position.</summary>
    public Span<byte> GetVMb(int mbCol, int mbRow)
    {
        int offset = mbRow * 8 * UvStride + mbCol * 8;
        return VPlane.AsSpan(offset);
    }
}
