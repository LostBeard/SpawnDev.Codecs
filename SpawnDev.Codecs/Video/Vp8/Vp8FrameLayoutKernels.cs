// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP8 frame buffer <-> per-block-packed layout glue. The transform
// kernels in this codebase operate on per-block-packed buffers (16 or
// 64 contiguous bytes per block, MB-major) but the encoder source and
// decoder output live in YUV plane buffers with row strides. These
// utilities bridge the two layouts entirely on the GPU.
//
// Two kernels:
//   - GatherY16x16 / GatherUv8x8: frame plane bytes -> per-block-packed
//     bytes. One thread per block. Each block reads 16 (Y) or 8 (UV)
//     rows from the frame and writes them packed.
//   - ScatterY16x16 / ScatterUv8x8: inverse - per-block-packed bytes
//     -> frame plane.
//
// The Y MB plane uses 16x16 macroblocks; UV uses 8x8 (4:2:0
// subsampling). Y4 sub-blocks are not stored as separate per-block
// buffers in these utilities - the encoder handles that split itself
// after the gather since the per-block-packed Y buffer is already
// laid out for it.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>
/// VP8 frame-buffer <-> per-block-packed layout kernels. Holds the
/// pre-compiled gather/scatter kernels for Y (16x16 MB) and UV (8x8
/// MB) planes. One thread per block.
/// </summary>
public sealed class Vp8FrameLayoutKernels : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, int, int, int> _gatherY16Kernel;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, int, int, int> _scatterY16Kernel;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, int, int, int> _gatherUv8Kernel;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, int, int, int> _scatterUv8Kernel;

    /// <summary>Compile kernels onto <paramref name="accelerator"/>.</summary>
    public Vp8FrameLayoutKernels(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _gatherY16Kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<byte>, int, int, int>(GatherY16Kernel);
        _scatterY16Kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<byte>, int, int, int>(ScatterY16Kernel);
        _gatherUv8Kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<byte>, int, int, int>(GatherUv8Kernel);
        _scatterUv8Kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<byte>, int, int, int>(ScatterUv8Kernel);
    }

    /// <summary>
    /// Gather Y plane bytes into per-MB packed 16x16 blocks. Output
    /// layout: block-major, mbCount * 256 bytes total.
    /// </summary>
    public void GatherY16(ArrayView<byte> yPlane, ArrayView<byte> y16Packed,
        int mbCols, int mbRows, int yStride)
    {
        int mbCount = mbCols * mbRows;
        if (mbCount == 0) return;
        if (y16Packed.Length < mbCount * 256L)
            throw new ArgumentException("y16Packed must hold mbCount*256 bytes.", nameof(y16Packed));
        if (yPlane.Length < (long)yStride * mbRows * 16)
            throw new ArgumentException("yPlane too short for given stride/dimensions.", nameof(yPlane));
        _gatherY16Kernel(mbCount, yPlane, y16Packed, mbCols, mbRows, yStride);
    }

    /// <summary>Scatter per-MB packed 16x16 blocks into the Y plane.</summary>
    public void ScatterY16(ArrayView<byte> y16Packed, ArrayView<byte> yPlane,
        int mbCols, int mbRows, int yStride)
    {
        int mbCount = mbCols * mbRows;
        if (mbCount == 0) return;
        if (y16Packed.Length < mbCount * 256L)
            throw new ArgumentException("y16Packed must hold mbCount*256 bytes.", nameof(y16Packed));
        if (yPlane.Length < (long)yStride * mbRows * 16)
            throw new ArgumentException("yPlane too short for given stride/dimensions.", nameof(yPlane));
        _scatterY16Kernel(mbCount, y16Packed, yPlane, mbCols, mbRows, yStride);
    }

    /// <summary>Gather UV plane bytes into per-MB packed 8x8 blocks.</summary>
    public void GatherUv8(ArrayView<byte> uvPlane, ArrayView<byte> uv8Packed,
        int mbCols, int mbRows, int uvStride)
    {
        int mbCount = mbCols * mbRows;
        if (mbCount == 0) return;
        if (uv8Packed.Length < mbCount * 64L)
            throw new ArgumentException("uv8Packed must hold mbCount*64 bytes.", nameof(uv8Packed));
        if (uvPlane.Length < (long)uvStride * mbRows * 8)
            throw new ArgumentException("uvPlane too short for given stride/dimensions.", nameof(uvPlane));
        _gatherUv8Kernel(mbCount, uvPlane, uv8Packed, mbCols, mbRows, uvStride);
    }

    /// <summary>Scatter per-MB packed 8x8 blocks into the UV plane.</summary>
    public void ScatterUv8(ArrayView<byte> uv8Packed, ArrayView<byte> uvPlane,
        int mbCols, int mbRows, int uvStride)
    {
        int mbCount = mbCols * mbRows;
        if (mbCount == 0) return;
        if (uv8Packed.Length < mbCount * 64L)
            throw new ArgumentException("uv8Packed must hold mbCount*64 bytes.", nameof(uv8Packed));
        if (uvPlane.Length < (long)uvStride * mbRows * 8)
            throw new ArgumentException("uvPlane too short for given stride/dimensions.", nameof(uvPlane));
        _scatterUv8Kernel(mbCount, uv8Packed, uvPlane, mbCols, mbRows, uvStride);
    }

    private static void GatherY16Kernel(
        Index1D mbIdx,
        ArrayView<byte> yPlane,
        ArrayView<byte> y16Packed,
        int mbCols, int mbRows, int yStride)
    {
        int idx = mbIdx;
        if (idx >= mbCols * mbRows) return;
        int mbRow = idx / mbCols;
        int mbCol = idx % mbCols;
        long pBase = (long)idx * 256;
        long fBase = (long)(mbRow * 16) * yStride + (long)(mbCol * 16);
        for (int r = 0; r < 16; r++)
        {
            long fRow = fBase + (long)r * yStride;
            long pRow = pBase + (long)r * 16;
            for (int c = 0; c < 16; c++) y16Packed[pRow + c] = yPlane[fRow + c];
        }
    }

    private static void ScatterY16Kernel(
        Index1D mbIdx,
        ArrayView<byte> y16Packed,
        ArrayView<byte> yPlane,
        int mbCols, int mbRows, int yStride)
    {
        int idx = mbIdx;
        if (idx >= mbCols * mbRows) return;
        int mbRow = idx / mbCols;
        int mbCol = idx % mbCols;
        long pBase = (long)idx * 256;
        long fBase = (long)(mbRow * 16) * yStride + (long)(mbCol * 16);
        for (int r = 0; r < 16; r++)
        {
            long fRow = fBase + (long)r * yStride;
            long pRow = pBase + (long)r * 16;
            for (int c = 0; c < 16; c++) yPlane[fRow + c] = y16Packed[pRow + c];
        }
    }

    private static void GatherUv8Kernel(
        Index1D mbIdx,
        ArrayView<byte> uvPlane,
        ArrayView<byte> uv8Packed,
        int mbCols, int mbRows, int uvStride)
    {
        int idx = mbIdx;
        if (idx >= mbCols * mbRows) return;
        int mbRow = idx / mbCols;
        int mbCol = idx % mbCols;
        long pBase = (long)idx * 64;
        long fBase = (long)(mbRow * 8) * uvStride + (long)(mbCol * 8);
        for (int r = 0; r < 8; r++)
        {
            long fRow = fBase + (long)r * uvStride;
            long pRow = pBase + (long)r * 8;
            for (int c = 0; c < 8; c++) uv8Packed[pRow + c] = uvPlane[fRow + c];
        }
    }

    private static void ScatterUv8Kernel(
        Index1D mbIdx,
        ArrayView<byte> uv8Packed,
        ArrayView<byte> uvPlane,
        int mbCols, int mbRows, int uvStride)
    {
        int idx = mbIdx;
        if (idx >= mbCols * mbRows) return;
        int mbRow = idx / mbCols;
        int mbCol = idx % mbCols;
        long pBase = (long)idx * 64;
        long fBase = (long)(mbRow * 8) * uvStride + (long)(mbCol * 8);
        for (int r = 0; r < 8; r++)
        {
            long fRow = fBase + (long)r * uvStride;
            long pRow = pBase + (long)r * 8;
            for (int c = 0; c < 8; c++) uvPlane[fRow + c] = uv8Packed[pRow + c];
        }
    }

    /// <summary>Release kernel resources.</summary>
    public void Dispose() { /* auto-grouped */ }
}
