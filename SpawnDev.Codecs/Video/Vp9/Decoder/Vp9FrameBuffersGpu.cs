// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt). See NOTICE.md for upstream attributions.
//
// VP9 frame buffers on the accelerator (libvpx BufferPool frame_bufs, indices from Vp9HeaderReader), all in ONE byte
// buffer so a kernel reaches any reference by offset (a block picks LAST / GOLDEN / ALTREF at run time). Each frame
// buffer owns a fixed-size region holding Y, U, V. Planes are padded to whole 64x64 superblocks, so a prediction or
// reconstruction of a block that crosses the right / bottom frame edge stays inside the region (libvpx writes those
// pixels into its frame borders; nothing reads them back as picture data). When a larger frame arrives the pool grows
// and every region is copied over with its layout (references of another size stay valid: scaled prediction).

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs.Video.Vp9.Decoder;

/// <summary>Geometry of one frame buffer's planes (offsets relative to the frame buffer's region).</summary>
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

    public long PlaneOffset(int plane) => plane == 0 ? 0 : plane == 1 ? UOffset : VOffset;
    public int Stride(int plane) => plane == 0 ? YStride : UvStride;
    public int CropWidth(int plane) => plane == 0 ? Width : UvCropWidth;
    public int CropHeight(int plane) => plane == 0 ? Height : UvCropHeight;

    public static Vp9PlaneLayout For(int width, int height, int ssX, int ssY)
    {
        int w64 = (width + 63) & ~63, h64 = (height + 63) & ~63;
        return new Vp9PlaneLayout(width, height, ssX, ssY, w64, h64, w64 >> ssX, h64 >> ssY);
    }
}

/// <summary>The decoder's frame buffers on the accelerator, one region per libvpx frame buffer index.</summary>
internal sealed class Vp9FrameBuffersGpu : IDisposable
{
    private readonly Accelerator _acc;
    private MemoryBuffer1D<byte, Stride1D.Dense>? _pool;
    private long _slotBytes;
    private readonly Vp9PlaneLayout[] _layouts = new Vp9PlaneLayout[Vp9HeaderReader.FrameBuffers];

    public Vp9FrameBuffersGpu(Accelerator accelerator) => _acc = accelerator ?? throw new ArgumentNullException(nameof(accelerator));

    /// <summary>Make frame buffer <paramref name="index"/> hold a frame of the given size (the pool grows when needed).</summary>
    public void Ensure(int index, int width, int height, int ssX, int ssY)
    {
        var layout = Vp9PlaneLayout.For(width, height, ssX, ssY);
        if (_pool == null || layout.Bytes > _slotBytes)
        {
            long slot = Math.Max(layout.Bytes, _slotBytes);
            // The kernels address the pool with 32-bit offsets (64-bit math is emulated, and slow to compile, on WebGPU):
            // every frame buffer must end below 2 GiB.
            if (slot * Vp9HeaderReader.FrameBuffers > int.MaxValue)
                throw new NotSupportedException($"VP9 frames of {width}x{height} need {slot * Vp9HeaderReader.FrameBuffers:N0} " +
                    $"bytes of frame buffers; the decoder addresses at most {int.MaxValue:N0}.");
            var pool = _acc.Allocate1D<byte>(slot * Vp9HeaderReader.FrameBuffers);
            if (_pool != null)
                for (int i = 0; i < Vp9HeaderReader.FrameBuffers; i++)
                    if (_layouts[i].Bytes > 0)
                        _pool.View.SubView(i * _slotBytes, _layouts[i].Bytes).CopyTo(pool.View.SubView(i * slot, _layouts[i].Bytes));
            _pool?.Dispose();
            _pool = pool;
            _slotBytes = slot;
        }
        _layouts[index] = layout;
    }

    /// <summary>All frame buffers.</summary>
    public ArrayView<byte> Pool => _pool!.View;
    /// <summary>Where frame buffer <paramref name="index"/> starts in <see cref="Pool"/>.</summary>
    public long Offset(int index) => index * _slotBytes;
    public Vp9PlaneLayout Layout(int index) => _layouts[index];

    public void Dispose() => _pool?.Dispose();
}

/// <summary>Output packing: the cropped planes of a frame buffer into one tight I420-style buffer (Y, then U, then V).</summary>
internal static class Vp9OutputKernels
{
    /// <summary>One thread per output byte; <paramref name="src"/> offsets are absolute in the pool.</summary>
    public static void PackPlanesKernel(Index1D i, ArrayView<byte> src, ArrayView<byte> dst, int w, int h, int uvW, int uvH,
        int yStride, int uvStride, long yOffset, long uOffset, long vOffset)
    {
        long ySize = (long)w * h, uvSize = (long)uvW * uvH;
        long at = i;
        if (at < ySize)
        {
            dst[at] = src[yOffset + (at / w) * yStride + at % w];
            return;
        }
        at -= ySize;
        long plane = at < uvSize ? uOffset : vOffset;
        if (at >= uvSize) at -= uvSize;
        dst[i] = src[plane + (at / uvW) * uvStride + at % uvW];
    }
}
