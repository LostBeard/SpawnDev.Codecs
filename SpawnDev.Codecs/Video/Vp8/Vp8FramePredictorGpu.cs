// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP8 frame-level GPU predictor builder. Reads recon-frame neighbour
// pixels, computes per-MB intra predictors (16x16 luma + 8x8 UV) for
// every MB, writes them into per-block-packed predictor buffers.
//
// One thread per MB for the neighbour-gather step; then the existing
// Vp8IntraPredict16x16Kernel + Vp8IntraPredict8x8UvKernel batched
// dispatches do the actual prediction math.
//
// Caller supplies:
//   - reconY / reconU / reconV: row-strided plane buffers holding
//     previously-reconstructed pixels (for an in-progress encode the
//     completed-MB rows; for a fresh frame, the caller can leave them
//     uninitialized when haveAbove/haveLeft flags are all false).
//   - per-MB modes (4-bit Vp8IntraMode16x16 in low nibble of one byte;
//     same byte format used by the existing IntraPredict kernels).
//   - per-MB haveAbove/haveLeft flags packed into the same byte
//     (bit 4 = haveAbove, bit 5 = haveLeft).
//
// Output:
//   - per-MB packed predictor buffers (Y: 256 bytes/MB, U: 64
//     bytes/MB, V: 64 bytes/MB).

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>
/// VP8 frame-level GPU predictor builder. Reads neighbour pixels from
/// the recon plane, runs intra predict for each MB. Holds the
/// pre-compiled gather + intra-predict kernels.
/// </summary>
public sealed class Vp8FramePredictorGpu : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Vp8IntraPredict16x16Kernel _intraY16Kernel;
    private readonly Vp8IntraPredict8x8UvKernel _intraUv8Kernel;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, int, int, int> _gatherY16NeighboursKernel;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, int, int, int> _gatherUv8NeighboursKernel;

    /// <summary>Compile kernels onto <paramref name="accelerator"/>.</summary>
    public Vp8FramePredictorGpu(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _intraY16Kernel = new Vp8IntraPredict16x16Kernel(accelerator);
        _intraUv8Kernel = new Vp8IntraPredict8x8UvKernel(accelerator);
        _gatherY16NeighboursKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, int, int, int>(GatherY16NeighboursKernel);
        _gatherUv8NeighboursKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, int, int, int>(GatherUv8NeighboursKernel);
    }

    /// <summary>
    /// Run intra prediction for an entire frame.
    /// </summary>
    /// <param name="yRecon">Reconstructed Y plane (row-strided). Read for above/left/top-left samples; not modified.</param>
    /// <param name="uRecon">Reconstructed U plane. Same shape semantics as yRecon, sub-sampled.</param>
    /// <param name="vRecon">Reconstructed V plane.</param>
    /// <param name="modesY">Per-MB Y mode + flags (one byte per MB, mode in low nibble, haveAbove bit 4, haveLeft bit 5).</param>
    /// <param name="modesUv">Per-MB UV mode + flags. UV uses the same Vp8IntraMode16x16 enum.</param>
    /// <param name="yPredOut">Output: per-MB packed Y predictor (256 bytes/MB), block-major.</param>
    /// <param name="uPredOut">Output: per-MB packed U predictor (64 bytes/MB).</param>
    /// <param name="vPredOut">Output: per-MB packed V predictor (64 bytes/MB).</param>
    /// <param name="mbCols">Number of macroblocks per row.</param>
    /// <param name="mbRows">Number of macroblock rows.</param>
    /// <param name="yStride">Y plane row stride in bytes.</param>
    /// <param name="uvStride">UV plane row stride in bytes.</param>
    public void Run(
        ArrayView<byte> yRecon,
        ArrayView<byte> uRecon,
        ArrayView<byte> vRecon,
        ArrayView<byte> modesY,
        ArrayView<byte> modesUv,
        ArrayView<byte> yPredOut,
        ArrayView<byte> uPredOut,
        ArrayView<byte> vPredOut,
        int mbCols, int mbRows,
        int yStride, int uvStride)
    {
        int mbCount = mbCols * mbRows;
        if (mbCount == 0) return;

        // Allocate per-MB neighbour buffers: above[16] + left[16] +
        // topLeft[1] for Y; above[8] + left[8] + topLeft[1] for UV.
        // GPU-resident. We allocate temporaries inside Run (one set per
        // call) since predictor build runs once per frame and the
        // allocations are small relative to the source planes.
        using var yAbove = _accelerator.Allocate1D<byte>(mbCount * 16);
        using var yLeft = _accelerator.Allocate1D<byte>(mbCount * 16);
        using var yTopLeft = _accelerator.Allocate1D<byte>(mbCount);
        using var uAbove = _accelerator.Allocate1D<byte>(mbCount * 8);
        using var uLeft = _accelerator.Allocate1D<byte>(mbCount * 8);
        using var uTopLeft = _accelerator.Allocate1D<byte>(mbCount);
        using var vAbove = _accelerator.Allocate1D<byte>(mbCount * 8);
        using var vLeft = _accelerator.Allocate1D<byte>(mbCount * 8);
        using var vTopLeft = _accelerator.Allocate1D<byte>(mbCount);

        _gatherY16NeighboursKernel(mbCount, yRecon, yAbove.View, yLeft.View, yTopLeft.View,
            mbCols, mbRows, yStride);
        _gatherUv8NeighboursKernel(mbCount, uRecon, uAbove.View, uLeft.View, uTopLeft.View,
            mbCols, mbRows, uvStride);
        _gatherUv8NeighboursKernel(mbCount, vRecon, vAbove.View, vLeft.View, vTopLeft.View,
            mbCols, mbRows, uvStride);

        // Run the existing batched intra predict kernels.
        _intraY16Kernel.Run(yAbove.View, yLeft.View, yTopLeft.View, modesY, yPredOut, mbCount);
        _intraUv8Kernel.Run(uAbove.View, uLeft.View, uTopLeft.View, modesUv, uPredOut, mbCount);
        _intraUv8Kernel.Run(vAbove.View, vLeft.View, vTopLeft.View, modesUv, vPredOut, mbCount);
    }

    /// <summary>
    /// Gather 16-byte above row, 16-byte left column, and 1-byte
    /// top-left sample from the recon Y plane for each MB. One thread
    /// per MB. Out-of-frame neighbours are filled with the VP8
    /// defaults (above = 127, left = 129, top-left = 128) so the
    /// predictor's haveAbove/haveLeft flags correctly elide the
    /// neighbour at the prediction step.
    /// </summary>
    private static void GatherY16NeighboursKernel(
        Index1D mbIdx,
        ArrayView<byte> yRecon,
        ArrayView<byte> yAbove,
        ArrayView<byte> yLeft,
        ArrayView<byte> yTopLeft,
        int mbCols, int mbRows, int yStride)
    {
        int idx = mbIdx;
        if (idx >= mbCols * mbRows) return;
        int mbRow = idx / mbCols;
        int mbCol = idx % mbCols;
        long aBase = (long)idx * 16;
        long lBase = (long)idx * 16;
        // Above row: yRecon[(mbRow*16 - 1) * yStride + mbCol*16 + c] for c=0..15.
        if (mbRow == 0)
        {
            for (int c = 0; c < 16; c++) yAbove[aBase + c] = 127;
            yTopLeft[idx] = 128;
        }
        else
        {
            long fRow = (long)(mbRow * 16 - 1) * yStride + (long)(mbCol * 16);
            for (int c = 0; c < 16; c++) yAbove[aBase + c] = yRecon[fRow + c];
            // Top-left: yRecon[(mbRow*16-1) * yStride + (mbCol*16-1)].
            if (mbCol == 0) yTopLeft[idx] = 129;
            else yTopLeft[idx] = yRecon[fRow - 1];
        }
        // Left column: yRecon[(mbRow*16+r) * yStride + mbCol*16 - 1] for r=0..15.
        if (mbCol == 0)
        {
            for (int r = 0; r < 16; r++) yLeft[lBase + r] = 129;
        }
        else
        {
            long fCol = (long)(mbCol * 16 - 1);
            for (int r = 0; r < 16; r++)
                yLeft[lBase + r] = yRecon[(long)(mbRow * 16 + r) * yStride + fCol];
        }
    }

    /// <summary>
    /// Same as GatherY16NeighboursKernel but for an 8x8 UV plane MB.
    /// </summary>
    private static void GatherUv8NeighboursKernel(
        Index1D mbIdx,
        ArrayView<byte> uvRecon,
        ArrayView<byte> uvAbove,
        ArrayView<byte> uvLeft,
        ArrayView<byte> uvTopLeft,
        int mbCols, int mbRows, int uvStride)
    {
        int idx = mbIdx;
        if (idx >= mbCols * mbRows) return;
        int mbRow = idx / mbCols;
        int mbCol = idx % mbCols;
        long aBase = (long)idx * 8;
        long lBase = (long)idx * 8;
        if (mbRow == 0)
        {
            for (int c = 0; c < 8; c++) uvAbove[aBase + c] = 127;
            uvTopLeft[idx] = 128;
        }
        else
        {
            long fRow = (long)(mbRow * 8 - 1) * uvStride + (long)(mbCol * 8);
            for (int c = 0; c < 8; c++) uvAbove[aBase + c] = uvRecon[fRow + c];
            if (mbCol == 0) uvTopLeft[idx] = 129;
            else uvTopLeft[idx] = uvRecon[fRow - 1];
        }
        if (mbCol == 0)
        {
            for (int r = 0; r < 8; r++) uvLeft[lBase + r] = 129;
        }
        else
        {
            long fCol = (long)(mbCol * 8 - 1);
            for (int r = 0; r < 8; r++)
                uvLeft[lBase + r] = uvRecon[(long)(mbRow * 8 + r) * uvStride + fCol];
        }
    }

    /// <summary>Release kernel resources.</summary>
    public void Dispose()
    {
        _intraY16Kernel.Dispose();
        _intraUv8Kernel.Dispose();
    }
}
