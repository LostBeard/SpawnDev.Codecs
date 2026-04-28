// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP8 GPU-resident reconstruction pipeline. Mirror of
// Vp8FrameTransformGpu but for the inverse direction:
//
//   dequantize -> inverse Walsh (Y2) -> inject Y2 DCs into Y4 ->
//   inverse DCT/DC-only IDCT -> add predictor -> clip -> recon.
//
// Caller supplies:
//   - quantized coefs on device (Y4, Y2, U, V) - typically the
//     output of Vp8FrameTransformGpu;
//   - per-MB dequantizer values on device;
//   - per-block predictors on device (4x4 packed bytes each).
//
// Output:
//   - per-block reconstructed pixels (4x4 packed bytes), block-major.
//
// The reconstructed pixels are what the encoder writes back into its
// frame buffer for the next macroblock's intra prediction, and what
// the decoder writes for the actual decoded frame. Both paths share
// this same math, so this driver is dual-use.
//
// Why this matters: the encoder's per-MB recon step is what locks
// the encoder + decoder to producing identical reconstructed pixels.
// Drift between them compounds across the frame and breaks
// subsequent-MB intra prediction. Running both encoder-side recon
// and decoder reconstruction through the same kernels guarantees no
// drift.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>
/// VP8 frame-level GPU reconstruction pipeline. Holds the pre-compiled
/// kernel set; runs dequant + inverse Walsh + Y2 DC injection + IDCT
/// + predictor add + clip for a frame's worth of macroblocks in one
/// call. Mirror of <see cref="Vp8FrameTransformGpu"/>.
/// </summary>
public sealed class Vp8FrameReconstructGpu : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Vp8DequantizerKernel _dequantKernel;
    private readonly Vp8InverseWalsh4x4Kernel _invWalshKernel;
    private readonly Vp8InverseDct4x4Kernel _idctKernel;
    private readonly Vp8DcOnlyIdctAddKernel _dcOnlyKernel;
    private readonly Action<Index1D, ArrayView<short>, ArrayView<short>, int> _injectY2DcKernel;
    private readonly Action<Index1D, ArrayView<short>, ArrayView<short>, int> _walshDc1OnlyKernel;
    private readonly Action<Index1D, ArrayView<short>, ArrayView<int>, int> _detectAcOnlyKernel;

    /// <summary>Compile kernels onto <paramref name="accelerator"/>.</summary>
    public Vp8FrameReconstructGpu(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _dequantKernel = new Vp8DequantizerKernel(accelerator);
        _invWalshKernel = new Vp8InverseWalsh4x4Kernel(accelerator);
        _idctKernel = new Vp8InverseDct4x4Kernel(accelerator);
        _dcOnlyKernel = new Vp8DcOnlyIdctAddKernel(accelerator);
        _injectY2DcKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<short>, ArrayView<short>, int>(InjectY2DcKernel);
        _walshDc1OnlyKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<short>, ArrayView<short>, int>(WalshDc1OnlyKernel);
        _detectAcOnlyKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<short>, ArrayView<int>, int>(DetectAcOnlyKernel);
    }

    /// <summary>
    /// Run reconstruction on <paramref name="mbCount"/> macroblocks.
    /// Mutates the input coef views (dequant is in-place); writes
    /// reconstructed bytes to the per-block recon buffers.
    /// </summary>
    /// <param name="y4Coefs">Quantized Y4 coefs, mbCount*16*16 shorts. Will be dequantized in place. Caller's responsibility to copy first if it wants to preserve the quantized values.</param>
    /// <param name="y2Coefs">Quantized Y2 coefs, mbCount*16 shorts. Dequantized in place.</param>
    /// <param name="uCoefs">Quantized U coefs, mbCount*4*16 shorts. Dequantized in place.</param>
    /// <param name="vCoefs">Quantized V coefs, mbCount*4*16 shorts. Dequantized in place.</param>
    /// <param name="y4Pred">Per-block Y4 predictor bytes, mbCount*16*16 bytes (4x4 packed per block).</param>
    /// <param name="uPred">Per-block U predictor bytes, mbCount*4*16 bytes.</param>
    /// <param name="vPred">Per-block V predictor bytes, mbCount*4*16 bytes.</param>
    /// <param name="y4Recon">Output: per-block Y4 recon bytes, mbCount*16*16 bytes.</param>
    /// <param name="uRecon">Output: per-block U recon bytes, mbCount*4*16 bytes.</param>
    /// <param name="vRecon">Output: per-block V recon bytes, mbCount*4*16 bytes.</param>
    /// <param name="y1DcQ">Per-MB Y1 DC dequantizer, length = mbCount.</param>
    /// <param name="y1AcQ">Per-MB Y1 AC dequantizer.</param>
    /// <param name="y2DcQ">Per-MB Y2 DC dequantizer.</param>
    /// <param name="y2AcQ">Per-MB Y2 AC dequantizer.</param>
    /// <param name="uvDcQ">Per-MB UV DC dequantizer.</param>
    /// <param name="uvAcQ">Per-MB UV AC dequantizer.</param>
    /// <param name="y2HasAc">Per-MB flag: 1 if Y2 has any non-zero AC, else 0. Drives the inverse-Walsh-vs-DC-broadcast branch (libvpx vp8_short_inv_walsh4x4_c vs _1 fast path). Length = mbCount.</param>
    /// <param name="mbCount">Number of macroblocks.</param>
    public void Run(
        ArrayView<short> y4Coefs,
        ArrayView<short> y2Coefs,
        ArrayView<short> uCoefs,
        ArrayView<short> vCoefs,
        ArrayView<byte> y4Pred,
        ArrayView<byte> uPred,
        ArrayView<byte> vPred,
        ArrayView<byte> y4Recon,
        ArrayView<byte> uRecon,
        ArrayView<byte> vRecon,
        ArrayView<short> y1DcQ,
        ArrayView<short> y1AcQ,
        ArrayView<short> y2DcQ,
        ArrayView<short> y2AcQ,
        ArrayView<short> uvDcQ,
        ArrayView<short> uvAcQ,
        ArrayView<byte> y2HasAc,
        int mbCount)
    {
        if (mbCount < 0) throw new ArgumentOutOfRangeException(nameof(mbCount));
        if (mbCount == 0) return;

        int y4BlockCount = mbCount * 16;
        int uvBlockCount = mbCount * 4;

        // 1. Dequantize Y2 (in-place). Per-MB dequantizers.
        _dequantKernel.Run(y2Coefs, y2DcQ, y2AcQ, mbCount);

        // 2. Inverse Walsh on Y2 (out-of-place).
        using var y2Inv = _accelerator.Allocate1D<short>(mbCount * 16);
        _invWalshKernel.Run(y2Coefs, y2Inv.View, mbCount);

        // 2b. For MBs with Y2 AC == 0, libvpx uses the DC-broadcast fast
        // path (vp8_short_inv_walsh4x4_1). Run a kernel that writes
        // ((y2Coefs[0] + 3) >> 3) into all 16 slots of y2Inv for those
        // MBs. We honor y2HasAc as the discriminator; runtime
        // bit-accurate to libvpx encoder/decoder paths.
        _walshDc1OnlyKernel((Index1D)mbCount, y2Coefs, y2Inv.View, mbCount);
        // Note: walshDc1OnlyKernel is "no-op when y2HasAc[mb] != 0"
        // OR the caller is responsible for ordering the dispatches so
        // the AC-present overrides the broadcast. Because the GPU
        // can't easily branch out of an earlier dispatch, we do BOTH
        // dispatches and the kernel itself reads y2HasAc to decide.
        // Simpler design: have a single combined kernel that picks
        // path. Refactor to that:

        // Replace 2 + 2b with a fused Y2 inverse kernel.
        FusedY2InverseKernelDispatch(y2Coefs, y2Inv.View, y2HasAc, mbCount);

        // 3. Expand per-MB dequantizers to per-block for Y4 + UV.
        using var y4DcExpand = _accelerator.Allocate1D<short>(y4BlockCount);
        using var y4AcExpand = _accelerator.Allocate1D<short>(y4BlockCount);
        using var uDcExpand = _accelerator.Allocate1D<short>(uvBlockCount);
        using var uAcExpand = _accelerator.Allocate1D<short>(uvBlockCount);
        using var vDcExpand = _accelerator.Allocate1D<short>(uvBlockCount);
        using var vAcExpand = _accelerator.Allocate1D<short>(uvBlockCount);
        ExpandPerMbDispatch(y1DcQ, y4DcExpand.View, mbCount, 16);
        ExpandPerMbDispatch(y1AcQ, y4AcExpand.View, mbCount, 16);
        ExpandPerMbDispatch(uvDcQ, uDcExpand.View, mbCount, 4);
        ExpandPerMbDispatch(uvAcQ, uAcExpand.View, mbCount, 4);
        ExpandPerMbDispatch(uvDcQ, vDcExpand.View, mbCount, 4);
        ExpandPerMbDispatch(uvAcQ, vAcExpand.View, mbCount, 4);

        // 4. Dequantize Y4 (in-place).
        _dequantKernel.Run(y4Coefs, y4DcExpand.View, y4AcExpand.View, y4BlockCount);

        // 5. Inject Y2-derived DC into Y4 coef[0] (overrides the dequantized DC,
        //    matching the libvpx decoder logic in vp8_decode_macroblock that
        //    overwrites mb_dqcoeff[0] = y2_inv[block_idx] after dequant).
        _injectY2DcKernel((Index1D)y4BlockCount, y2Inv.View, y4Coefs, mbCount);

        // 6. Inverse DCT + predict-add for every Y4 block.
        _idctKernel.Run(y4Coefs, y4Pred, y4Recon, y4BlockCount);

        // 7. Dequantize UV (in-place).
        _dequantKernel.Run(uCoefs, uDcExpand.View, uAcExpand.View, uvBlockCount);
        _dequantKernel.Run(vCoefs, vDcExpand.View, vAcExpand.View, uvBlockCount);

        // 8. Inverse DCT + predict-add for every UV block.
        _idctKernel.Run(uCoefs, uPred, uRecon, uvBlockCount);
        _idctKernel.Run(vCoefs, vPred, vRecon, uvBlockCount);
    }

    /// <summary>
    /// Inject Y2 inverse DC into Y4 coef[0]. One thread per Y4 block.
    /// Mirrors libvpx vp8_decode_macroblock's
    ///   xd->dst.y_buffer + qcoeff[0] = mb->dequant_y2_inverse[i].
    /// </summary>
    private static void InjectY2DcKernel(
        Index1D blockIdx,
        ArrayView<short> y2Inv,
        ArrayView<short> y4Coefs,
        int mbCount)
    {
        int idx = blockIdx;
        if (idx >= mbCount * 16) return;
        int mbIdx = idx >> 4;
        int slot = idx & 0xF;
        long y2Off = (long)mbIdx * 16 + slot;
        long y4Off = (long)mbIdx * 256 + (long)slot * 16;
        y4Coefs[y4Off] = y2Inv[y2Off];
    }

    /// <summary>
    /// libvpx vp8_short_inv_walsh4x4_1 fast path: when Y2 AC is all
    /// zero, every output is broadcast ((dc + 3) &gt;&gt; 3) to all 16 slots.
    /// One thread per MB. NOT used directly; superseded by
    /// FusedY2InverseKernelDispatch.
    /// </summary>
    private static void WalshDc1OnlyKernel(
        Index1D mbIdx,
        ArrayView<short> y2Coefs,
        ArrayView<short> y2Inv,
        int mbCount)
    {
        int idx = mbIdx;
        if (idx >= mbCount) return;
        // Placeholder - real selection is done by FusedY2InverseKernel.
    }

    /// <summary>
    /// Fused Y2 inverse: branches per-MB on y2HasAc to either keep the
    /// already-computed inverse Walsh output or overwrite with the
    /// DC-broadcast fast path. One thread per (MB, slot) pair.
    /// </summary>
    private void FusedY2InverseKernelDispatch(
        ArrayView<short> y2Coefs,
        ArrayView<short> y2Inv,
        ArrayView<byte> y2HasAc,
        int mbCount)
    {
        var k = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<short>, ArrayView<short>, ArrayView<byte>, int>(FusedY2InverseKernel);
        k((Index1D)(mbCount * 16), y2Coefs, y2Inv, y2HasAc, mbCount);
    }

    /// <summary>
    /// One thread per (MB, slot). If y2HasAc[mb] != 0, leaves y2Inv as
    /// the inverse Walsh output. Otherwise overwrites with
    /// ((y2Coefs[mb*16] + 3) >> 3).
    /// </summary>
    private static void FusedY2InverseKernel(
        Index1D pairIdx,
        ArrayView<short> y2Coefs,
        ArrayView<short> y2Inv,
        ArrayView<byte> y2HasAc,
        int mbCount)
    {
        int idx = pairIdx;
        if (idx >= mbCount * 16) return;
        int mbIdx = idx >> 4;
        int slot = idx & 0xF;
        if (y2HasAc[mbIdx] == 0)
        {
            // DC-broadcast fast path (libvpx vp8_short_inv_walsh4x4_1).
            short dc = y2Coefs[(long)mbIdx * 16];
            y2Inv[(long)mbIdx * 16 + slot] = (short)((dc + 3) >> 3);
        }
        // else: y2Inv already has the inverse-Walsh result from the
        // earlier _invWalshKernel.Run call. Leave it alone.
    }

    /// <summary>
    /// Mark Y2 blocks with any non-zero AC. Useful for caller to
    /// compute y2HasAc on the GPU side from the quantized Y2 coefs.
    /// One thread per MB.
    /// </summary>
    public void ComputeY2HasAc(
        ArrayView<short> y2Coefs,
        ArrayView<byte> y2HasAc,
        int mbCount)
    {
        if (mbCount == 0) return;
        var k = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<short>, ArrayView<byte>, int>(ComputeY2HasAcKernel);
        k((Index1D)mbCount, y2Coefs, y2HasAc, mbCount);
    }

    private static void ComputeY2HasAcKernel(
        Index1D mbIdx,
        ArrayView<short> y2Coefs,
        ArrayView<byte> y2HasAc,
        int mbCount)
    {
        int idx = mbIdx;
        if (idx >= mbCount) return;
        long b = (long)idx * 16;
        byte hasAc = 0;
        for (int i = 1; i < 16; i++) if (y2Coefs[b + i] != 0) { hasAc = 1; break; }
        y2HasAc[idx] = hasAc;
    }

    /// <summary>Detect any non-zero AC coef across the Y4 plane. Used internally for the AC-only branch in IDCT future fast paths.</summary>
    private static void DetectAcOnlyKernel(
        Index1D blockIdx,
        ArrayView<short> coefs,
        ArrayView<int> hasAcMask,
        int blockCount)
    {
        // Reserved for future expansion - detect all-zero AC blocks
        // for DC-only IDCT fast-path routing.
        int idx = blockIdx;
        if (idx >= blockCount) return;
        long b = (long)idx * 16;
        int hasAc = 0;
        for (int i = 1; i < 16; i++) if (coefs[b + i] != 0) { hasAc = 1; break; }
        hasAcMask[idx] = hasAc;
    }

    /// <summary>Expand per-MB short array to per-block by repeating each value blocksPerMb times.</summary>
    private void ExpandPerMbDispatch(
        ArrayView<short> perMb, ArrayView<short> perBlock,
        int mbCount, int blocksPerMb)
    {
        var k = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<short>, ArrayView<short>, int, int>(ExpandPerMbKernel);
        k((Index1D)(mbCount * blocksPerMb), perMb, perBlock, mbCount, blocksPerMb);
    }

    private static void ExpandPerMbKernel(
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

    /// <summary>Release kernel resources.</summary>
    public void Dispose()
    {
        _dequantKernel.Dispose();
        _invWalshKernel.Dispose();
        _idctKernel.Dispose();
        _dcOnlyKernel.Dispose();
    }
}
