// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP8 GPU-resident frame-level transform driver. Runs the full
// encoder transform stack on the device for an entire frame's worth
// of macroblocks in one go: FDCT (Y4 + UV), forward Walsh (Y2),
// forward quantization (Y2 + Y4 + UV) - all without touching the
// CPU between stages.
//
// Caller supplies the per-block residuals already in GPU memory; this
// class assumes the predictor + subtract step has already happened on
// the GPU side (per the GPU-resident pipeline goal: no CPU<->GPU
// bouncing through the encoder hot path).
//
// Output is the quantized coefs in GPU memory. Callers either (a)
// read them back to feed the CPU entropy coder, or (b) keep them on
// GPU for future GPU-side entropy parallelism.
//
// Why this lives separately from the existing CPU Vp8KeyframeEncoder:
// the CPU encoder is the single-threaded reference. The GPU path is a
// performance-oriented re-implementation that calls into the same
// CPU-side bitstream writer for the entropy stage but does ALL the
// transform / quantization work on the device. Entropy coding remains
// sequential per VP8 spec.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>
/// VP8 frame-level GPU transform pipeline. Holds the pre-compiled
/// kernel set + reusable GPU buffers. Run the forward-direction
/// transforms (FDCT + Walsh + quant) for an entire frame on the
/// device in one call.
/// </summary>
public sealed class Vp8FrameTransformGpu : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Vp8ForwardDct4x4Kernel _fdctKernel;
    private readonly Vp8ForwardWalsh4x4Kernel _walshKernel;
    private readonly Vp8ForwardQuantizerKernel _quantKernel;
    private readonly Action<Index1D, ArrayView<short>, ArrayView<short>, int> _gatherY4DcKernel;
    private readonly Action<Index1D, ArrayView<short>, ArrayView<short>, int> _scatterY2DcKernel;

    /// <summary>Compile kernels onto <paramref name="accelerator"/>.</summary>
    public Vp8FrameTransformGpu(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _fdctKernel = new Vp8ForwardDct4x4Kernel(accelerator);
        _walshKernel = new Vp8ForwardWalsh4x4Kernel(accelerator);
        _quantKernel = new Vp8ForwardQuantizerKernel(accelerator);
        _gatherY4DcKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<short>, ArrayView<short>, int>(GatherY4DcsKernel);
        _scatterY2DcKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<short>, ArrayView<short>, int>(ScatterY2DcsKernel);
    }

    /// <summary>
    /// Run the full forward transform stack on <paramref name="mbCount"/>
    /// macroblocks. Inputs are residuals (block-major); outputs are the
    /// quantized coefs in the same layout.
    /// </summary>
    /// <param name="y4Residual">16 Y4 4x4 residual blocks per MB, 16 shorts each. Layout: MB-major, then block index 0..15, then 16 shorts. Total = mbCount*16*16.</param>
    /// <param name="uResidual">4 U 4x4 residual blocks per MB. mbCount*4*16.</param>
    /// <param name="vResidual">4 V 4x4 residual blocks per MB. mbCount*4*16.</param>
    /// <param name="y4Coefs">Output: quantized Y4 coefs, same layout as y4Residual.</param>
    /// <param name="y2Coefs">Output: quantized Y2 coefs (one 4x4 block per MB). mbCount*16.</param>
    /// <param name="uCoefs">Output: quantized U coefs.</param>
    /// <param name="vCoefs">Output: quantized V coefs.</param>
    /// <param name="y1DcQ">Per-MB Y1 DC quantizer (length = mbCount).</param>
    /// <param name="y1AcQ">Per-MB Y1 AC quantizer.</param>
    /// <param name="y2DcQ">Per-MB Y2 DC quantizer.</param>
    /// <param name="y2AcQ">Per-MB Y2 AC quantizer.</param>
    /// <param name="uvDcQ">Per-MB UV DC quantizer.</param>
    /// <param name="uvAcQ">Per-MB UV AC quantizer.</param>
    /// <param name="mbCount">Number of macroblocks.</param>
    public void Run(
        ArrayView<short> y4Residual,
        ArrayView<short> uResidual,
        ArrayView<short> vResidual,
        ArrayView<short> y4Coefs,
        ArrayView<short> y2Coefs,
        ArrayView<short> uCoefs,
        ArrayView<short> vCoefs,
        ArrayView<short> y1DcQ,
        ArrayView<short> y1AcQ,
        ArrayView<short> y2DcQ,
        ArrayView<short> y2AcQ,
        ArrayView<short> uvDcQ,
        ArrayView<short> uvAcQ,
        int mbCount)
    {
        if (mbCount < 0) throw new ArgumentOutOfRangeException(nameof(mbCount));
        if (mbCount == 0) return;

        int y4BlockCount = mbCount * 16;
        int uvBlockCount = mbCount * 4;

        // 1. FDCT all Y4 + U + V residual blocks. They all live in
        //    different memory regions so we dispatch three times. Each
        //    dispatch writes into the matching coef buffer.
        _fdctKernel.Run(y4Residual, y4Coefs, y4BlockCount);
        _fdctKernel.Run(uResidual, uCoefs, uvBlockCount);
        _fdctKernel.Run(vResidual, vCoefs, uvBlockCount);

        // 2. Gather Y4 DCs into a temporary Y2 buffer (one short per Y4
        //    block, 16 shorts per MB), then run forward Walsh on it.
        //    The Y2 buffer for the kernel is mbCount*16 shorts, same as
        //    the final y2Coefs buffer - we use y2Coefs as scratch.
        _gatherY4DcKernel((Index1D)y4BlockCount, y4Coefs, y2Coefs, mbCount);

        // 3. Forward Walsh on the gathered Y2 block (one Walsh per MB).
        //    We need a temporary because Walsh writes into a different
        //    buffer than it reads. Use uCoefs as scratch is risky; pass
        //    a dedicated buffer instead.
        // SAFE: walshKernel reads input, writes output; both must be
        // distinct. Use a tmp on the device of size mbCount*16.
        using var walshTmp = _accelerator.Allocate1D<short>(mbCount * 16);
        _walshKernel.Run(y2Coefs, walshTmp.View, mbCount);

        // Copy walshTmp -> y2Coefs (in-place semantics for the caller).
        // Use the scatter kernel which also clears Y4[0] (zero out the Y4
        // DC slot since the encoder writes the Y2-derived DC there at
        // dequant time).
        _scatterY2DcKernel((Index1D)y4BlockCount, walshTmp.View, y2Coefs, mbCount);
        // After scatter, y2Coefs holds the post-Walsh values; the kernel
        // also zeroed out coef[0] of every Y4 block (mirrors libvpx
        // encoder which clears Y4 DC after Y2 transform).
        ClearY4DcKernelDispatch(y4Coefs, y4BlockCount);

        // 4. Quantize all coef blocks. Y4 uses y1Dc/y1Ac, Y2 uses
        //    y2Dc/y2Ac, UV uses uvDc/uvAc. Quantizer kernel takes
        //    parallel ArrayViews of length blockCount, so we need to
        //    expand the per-MB quantizer arrays to per-block.
        QuantizeAllPlanes(
            y4Coefs, y2Coefs, uCoefs, vCoefs,
            y1DcQ, y1AcQ, y2DcQ, y2AcQ, uvDcQ, uvAcQ,
            mbCount);
    }

    /// <summary>
    /// Quantize every Y4 + Y2 + U + V block. Expands per-MB quantizer
    /// vectors into per-block ones via dedicated expand kernels.
    /// </summary>
    private void QuantizeAllPlanes(
        ArrayView<short> y4Coefs,
        ArrayView<short> y2Coefs,
        ArrayView<short> uCoefs,
        ArrayView<short> vCoefs,
        ArrayView<short> y1DcQ,
        ArrayView<short> y1AcQ,
        ArrayView<short> y2DcQ,
        ArrayView<short> y2AcQ,
        ArrayView<short> uvDcQ,
        ArrayView<short> uvAcQ,
        int mbCount)
    {
        int y4BlockCount = mbCount * 16;
        int uvBlockCount = mbCount * 4;

        // Expand per-MB quantizers to per-block.
        using var y4Dc = _accelerator.Allocate1D<short>(y4BlockCount);
        using var y4Ac = _accelerator.Allocate1D<short>(y4BlockCount);
        using var uDc = _accelerator.Allocate1D<short>(uvBlockCount);
        using var uAc = _accelerator.Allocate1D<short>(uvBlockCount);
        using var vDc = _accelerator.Allocate1D<short>(uvBlockCount);
        using var vAc = _accelerator.Allocate1D<short>(uvBlockCount);

        ExpandPerMbToPerBlockDispatch(y1DcQ, y4Dc.View, mbCount, blocksPerMb: 16);
        ExpandPerMbToPerBlockDispatch(y1AcQ, y4Ac.View, mbCount, blocksPerMb: 16);
        ExpandPerMbToPerBlockDispatch(uvDcQ, uDc.View, mbCount, blocksPerMb: 4);
        ExpandPerMbToPerBlockDispatch(uvAcQ, uAc.View, mbCount, blocksPerMb: 4);
        ExpandPerMbToPerBlockDispatch(uvDcQ, vDc.View, mbCount, blocksPerMb: 4);
        ExpandPerMbToPerBlockDispatch(uvAcQ, vAc.View, mbCount, blocksPerMb: 4);

        _quantKernel.Run(y4Coefs, y4Dc.View, y4Ac.View, y4BlockCount);
        _quantKernel.Run(y2Coefs, y2DcQ, y2AcQ, mbCount);
        _quantKernel.Run(uCoefs, uDc.View, uAc.View, uvBlockCount);
        _quantKernel.Run(vCoefs, vDc.View, vAc.View, uvBlockCount);
    }

    /// <summary>
    /// Gather coef[0] of every Y4 block in an MB into a 16-short Y2
    /// pre-Walsh block. One thread per Y4 block.
    /// </summary>
    private static void GatherY4DcsKernel(
        Index1D blockIdx,
        ArrayView<short> y4Coefs,
        ArrayView<short> y2PreWalsh,
        int mbCount)
    {
        int idx = blockIdx;
        if (idx >= mbCount * 16) return;
        int mbIdx = idx >> 4;       // / 16
        int slot = idx & 0xF;       // % 16
        // Y4 block layout: y4Coefs[mbIdx * 256 + slot * 16 + 0] = DC.
        long y4Off = (long)mbIdx * 256 + (long)slot * 16;
        long y2Off = (long)mbIdx * 16 + slot;
        y2PreWalsh[y2Off] = y4Coefs[y4Off];
    }

    /// <summary>
    /// After forward Walsh, the post-Walsh Y2 block lives in walshOut.
    /// Copy it into the final y2Coefs buffer. Also we DON'T touch Y4
    /// coef[0] here - that's done by ClearY4DcKernelDispatch.
    /// </summary>
    private static void ScatterY2DcsKernel(
        Index1D blockIdx,
        ArrayView<short> walshOut,
        ArrayView<short> y2Coefs,
        int mbCount)
    {
        int idx = blockIdx;
        if (idx >= mbCount * 16) return;
        // walshOut and y2Coefs are both [mbCount*16] in the same packed
        // layout, so this is just a copy. Keeping the dispatch as a
        // kernel rather than a CopyTo so we avoid any host-side stream
        // sync.
        y2Coefs[idx] = walshOut[idx];
    }

    /// <summary>Zero out coef[0] of every Y4 block (libvpx-style after Y2 transform).</summary>
    private void ClearY4DcKernelDispatch(ArrayView<short> y4Coefs, int y4BlockCount)
    {
        var k = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<short>, int>(ClearY4DcKernel);
        k((Index1D)y4BlockCount, y4Coefs, y4BlockCount);
    }

    private static void ClearY4DcKernel(
        Index1D blockIdx,
        ArrayView<short> y4Coefs,
        int y4BlockCount)
    {
        int idx = blockIdx;
        if (idx >= y4BlockCount) return;
        y4Coefs[(long)idx * 16] = 0;
    }

    /// <summary>
    /// Expand a per-MB short array into a per-block array by repeating
    /// each entry <paramref name="blocksPerMb"/> times. Used to feed the
    /// per-block quantizer kernel from the encoder's per-MB quantizer
    /// state.
    /// </summary>
    private void ExpandPerMbToPerBlockDispatch(
        ArrayView<short> perMb, ArrayView<short> perBlock,
        int mbCount, int blocksPerMb)
    {
        var k = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<short>, ArrayView<short>, int, int>(ExpandPerMbToPerBlockKernel);
        k((Index1D)(mbCount * blocksPerMb), perMb, perBlock, mbCount, blocksPerMb);
    }

    private static void ExpandPerMbToPerBlockKernel(
        Index1D blockIdx,
        ArrayView<short> perMb,
        ArrayView<short> perBlock,
        int mbCount,
        int blocksPerMb)
    {
        int idx = blockIdx;
        if (idx >= mbCount * blocksPerMb) return;
        int mbIdx = idx / blocksPerMb;
        perBlock[idx] = perMb[mbIdx];
    }

    /// <summary>Release kernel resources. Does NOT dispose the accelerator.</summary>
    public void Dispose()
    {
        _fdctKernel.Dispose();
        _walshKernel.Dispose();
        _quantKernel.Dispose();
    }
}
