// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt). See NOTICE.md for upstream attributions.
//
// VP9 frame buffers on the accelerator (libvpx BufferPool frame_bufs, indices from Vp9HeaderReader): one byte buffer
// per frame buffer holding Y, U, V. Planes are padded to whole 64x64 superblocks, so a prediction or reconstruction of
// a block that crosses the right / bottom frame edge stays inside the allocation (libvpx writes those pixels into its
// frame borders; nothing reads them back as picture data). Sizes: the frame's crop size (libvpx y_crop_width), the
// 8-aligned decoded size (y_width, what intra prediction treats as the frame edge) and the padded allocation.

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs.Video.Vp9.Decoder;

/// <summary>Geometry of one frame buffer's planes.</summary>
internal readonly record struct Vp9PlaneLayout(int Width, int Height, int SsX, int SsY, int YStride, int YRows, int UvStride, int UvRows)
{
    public long UOffset => (long)YStride * YRows;
    public long VOffset => UOffset + (long)UvStride * UvRows;
    public long Bytes => VOffset + (long)UvStride * UvRows;
    /// <summary>libvpx y_width / y_height (8-aligned).</summary>
    public int AlignedWidth => (Width + 7) & ~7;
    public int AlignedHeight => (Height + 7) & ~7;
    /// <summary>libvpx uv_crop_width / height (rounded up).</summary>
    public int UvCropWidth => (Width + SsX) >> SsX;
    public int UvCropHeight => (Height + SsY) >> SsY;
    /// <summary>libvpx uv_width / uv_height (from the aligned size).</summary>
    public int UvAlignedWidth => AlignedWidth >> SsX;
    public int UvAlignedHeight => AlignedHeight >> SsY;

    public static Vp9PlaneLayout For(int width, int height, int ssX, int ssY)
    {
        int w64 = (width + 63) & ~63, h64 = (height + 63) & ~63;
        return new Vp9PlaneLayout(width, height, ssX, ssY, w64, h64, w64 >> ssX, h64 >> ssY);
    }
}

/// <summary>The decoder's frame buffers on the accelerator.</summary>
internal sealed class Vp9FrameBuffersGpu : IDisposable
{
    private readonly Accelerator _acc;
    private readonly MemoryBuffer1D<byte, Stride1D.Dense>?[] _buffers = new MemoryBuffer1D<byte, Stride1D.Dense>?[Vp9HeaderReader.FrameBuffers];
    private readonly Vp9PlaneLayout[] _layouts = new Vp9PlaneLayout[Vp9HeaderReader.FrameBuffers];

    public Vp9FrameBuffersGpu(Accelerator accelerator) => _acc = accelerator ?? throw new ArgumentNullException(nameof(accelerator));

    /// <summary>Make frame buffer <paramref name="index"/> hold a frame of the given size (reallocates only when it grows).</summary>
    public void Ensure(int index, int width, int height, int ssX, int ssY)
    {
        var layout = Vp9PlaneLayout.For(width, height, ssX, ssY);
        var buf = _buffers[index];
        if (buf == null || buf.Length < layout.Bytes)
        {
            buf?.Dispose();
            _buffers[index] = _acc.Allocate1D<byte>(layout.Bytes);
        }
        _layouts[index] = layout;
    }

    public ArrayView<byte> View(int index) => _buffers[index]!.View;
    public Vp9PlaneLayout Layout(int index) => _layouts[index];

    public void Dispose()
    {
        foreach (var b in _buffers) b?.Dispose();
    }
}

/// <summary>Output packing: the cropped planes of a frame buffer into one tight I420-style buffer (Y, then U, then V).</summary>
internal static class Vp9OutputKernels
{
    /// <summary>One thread per output byte.</summary>
    public static void PackPlanesKernel(Index1D i, ArrayView<byte> src, ArrayView<byte> dst, int w, int h, int uvW, int uvH,
        int yStride, int uvStride, long uOffset, long vOffset)
    {
        long ySize = (long)w * h, uvSize = (long)uvW * uvH;
        long at = i;
        if (at < ySize)
        {
            dst[at] = src[(at / w) * yStride + at % w];
            return;
        }
        at -= ySize;
        long plane = at < uvSize ? uOffset : vOffset;
        if (at >= uvSize) at -= uvSize;
        dst[i] = src[plane + (at / uvW) * uvStride + at % uvW];
    }
}
