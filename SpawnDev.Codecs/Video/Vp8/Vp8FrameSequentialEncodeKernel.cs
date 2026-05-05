// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP8 wave-front MB encoder kernel. Per-diagonal dispatch: every MB
// on a given anti-diagonal of the MB grid runs in parallel (one thread
// per MB). The host driver dispatches one kernel per diagonal
// 0..mbCols+mbRows-2 in order; within a diagonal the MBs are
// independent because DC_PRED only reads MB(r-1, c) (diagonal d-1) and
// MB(r, c-1) (diagonal d-1).
//
// Replaces the previous extent=1 single-thread implementation. On a
// 32x16 grid the peak parallelism is 16 (the smaller dimension); the
// average diagonal width is ~11, so the speedup over the single-thread
// path is ~10x for the per-MB transform/quant/IDCT/recon stage.
//
// Kernel saves quantized coefs to global buffers (y4Coefs, y2Coefs,
// uCoefs, vCoefs). After this kernel runs, Vp8FrameEntropyKernel
// processes those coefs into the bool-coded bitstream.
//
// v1 simplifications (matches Vp8KeyframeEncoder):
// - All MBs use Y_PRED = DC_PRED, UV_PRED = DC_PRED.
// - No segmentation, no loop filter, single token partition.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>
/// Per-frame slot strides for the batch wave-front sequential encode kernel.
/// Threads in batch dispatch each compute (frameIdx, mbOnDiagonal) from the
/// global thread index and SubView all per-frame buffers by frameIdx*stride.
/// </summary>
public struct Vp8FrameSeqEncodeBatchParams
{
    /// <summary>Y plane bytes per frame (width*height).</summary>
    public int YStride;
    /// <summary>UV plane bytes per frame (uvWidth*uvHeight).</summary>
    public int UvStride;
    /// <summary>Y4 coefs per frame (mbCount*256).</summary>
    public int Y4CoefStride;
    /// <summary>Y2 coefs per frame (mbCount*16).</summary>
    public int Y2CoefStride;
    /// <summary>UV coefs per frame (mbCount*64).</summary>
    public int UvCoefStride;
    /// <summary>Dequant ints per frame (6).</summary>
    public int DequantStride;
    /// <summary>mbCols.</summary>
    public int MbCols;
    /// <summary>mbRows.</summary>
    public int MbRows;
    /// <summary>diagD packed in high 16 bits, rMin in low 16 (matches single-frame layout).</summary>
    public int DiagAndRMin;
    /// <summary>Number of MBs on this diagonal (used to extract frameIdx from idx.X).</summary>
    public int DiagCount;
}

/// <summary>
/// VP8 multi-MB sequential encoder kernel. Single thread per frame;
/// processes all MBs in row-major order with inline math. Saves
/// quantized coefs for the downstream entropy kernel; updates the
/// recon plane in place.
/// </summary>
public sealed class Vp8FrameSequentialEncodeKernel : IDisposable
{
    // libvpx Q16 IDCT constants (vp8_short_idct4x4llm_c).
    private const int CospiSqrt2Minus1 = 20091;
    private const int SinpiSqrt2 = 35468;

    private readonly Accelerator _accelerator;
    private readonly Action<
        Index1D,
        ArrayView<byte>, ArrayView<byte>, ArrayView<byte>,
        ArrayView<byte>, ArrayView<byte>, ArrayView<byte>,
        ArrayView<short>, ArrayView<short>, ArrayView<short>, ArrayView<short>,
        ArrayView<int>,
        int, int, int> _kernel;

    private readonly Action<
        Index1D,
        ArrayView<byte>, ArrayView<byte>, ArrayView<byte>,
        ArrayView<byte>, ArrayView<byte>, ArrayView<byte>,
        ArrayView<short>, ArrayView<short>, ArrayView<short>, ArrayView<short>,
        ArrayView<int>,
        Vp8FrameSeqEncodeBatchParams> _batchKernel;

    /// <summary>
    /// Single-dispatch wave-front kernel. All wave-front diagonals run inside
    /// one grouped kernel using Group.Barrier between diagonals. Eliminates
    /// the per-diagonal launch overhead of the multi-dispatch path.
    /// Constraint: total threads (numFrames * peakDiag) must fit in one block
    /// (1024 max on CUDA).
    /// </summary>
    private readonly Action<
        KernelConfig,
        ArrayView<byte>, ArrayView<byte>, ArrayView<byte>,
        ArrayView<byte>, ArrayView<byte>, ArrayView<byte>,
        ArrayView<short>, ArrayView<short>, ArrayView<short>, ArrayView<short>,
        ArrayView<int>,
        Vp8FrameSeqEncodeBatchParams, int, int> _singleDispatchBatchKernel;

    /// <summary>Compile.</summary>
    public Vp8FrameSequentialEncodeKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<byte>, ArrayView<byte>, ArrayView<byte>,
            ArrayView<byte>, ArrayView<byte>, ArrayView<byte>,
            ArrayView<short>, ArrayView<short>, ArrayView<short>, ArrayView<short>,
            ArrayView<int>,
            int, int, int>(EncodeWavefrontKernel);
        _batchKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<byte>, ArrayView<byte>, ArrayView<byte>,
            ArrayView<byte>, ArrayView<byte>, ArrayView<byte>,
            ArrayView<short>, ArrayView<short>, ArrayView<short>, ArrayView<short>,
            ArrayView<int>,
            Vp8FrameSeqEncodeBatchParams>(EncodeWavefrontBatchKernel);
        _singleDispatchBatchKernel = accelerator.LoadStreamKernel<
            ArrayView<byte>, ArrayView<byte>, ArrayView<byte>,
            ArrayView<byte>, ArrayView<byte>, ArrayView<byte>,
            ArrayView<short>, ArrayView<short>, ArrayView<short>, ArrayView<short>,
            ArrayView<int>,
            Vp8FrameSeqEncodeBatchParams, int, int>(EncodeWavefrontSingleDispatchKernel);
    }

    /// <summary>
    /// Encode all MBs in a frame using wave-front per-diagonal dispatch.
    /// Calls the kernel once per anti-diagonal d=0..mbCols+mbRows-2.
    /// Within each diagonal the MBs are independent (DC_PRED reads only
    /// from diagonal d-1), so a kernel dispatched at extent = diagonal-MB-count
    /// runs every MB on that diagonal in parallel.
    /// </summary>
    public void Run(
        ArrayView<byte> yPlane, ArrayView<byte> uPlane, ArrayView<byte> vPlane,
        ArrayView<byte> yRecon, ArrayView<byte> uRecon, ArrayView<byte> vRecon,
        ArrayView<short> y4Coefs, ArrayView<short> y2Coefs,
        ArrayView<short> uCoefs, ArrayView<short> vCoefs,
        ArrayView<int> dequant,
        int mbCols, int mbRows)
    {
        if (mbCols <= 0 || mbRows <= 0) throw new ArgumentOutOfRangeException();
        if (dequant.Length < 6) throw new ArgumentException("dequant must hold 6 ints.", nameof(dequant));
        int totalDiagonals = mbCols + mbRows - 1;
        for (int d = 0; d < totalDiagonals; d++)
        {
            // Range of valid (r,c) on diagonal d (r + c = d, 0<=r<mbRows, 0<=c<mbCols):
            //   r in [max(0, d - mbCols + 1), min(d, mbRows - 1)]
            int rMin = d - (mbCols - 1); if (rMin < 0) rMin = 0;
            int rMax = d; if (rMax > mbRows - 1) rMax = mbRows - 1;
            int diagCount = rMax - rMin + 1;
            // Pack diagD (high 16 bits) + rMin (low 16 bits). mb dims fit in 16b
            // for any reasonable VP8 frame (max 65535 MBs per axis).
            int diagAndRMin = (d << 16) | (rMin & 0xFFFF);
            _kernel(diagCount,
                yPlane, uPlane, vPlane,
                yRecon, uRecon, vRecon,
                y4Coefs, y2Coefs, uCoefs, vCoefs,
                dequant, mbCols, mbRows, diagAndRMin);
        }
    }

    /// <summary>
    /// Batch wave-front: encode N frames concurrently. Per diagonal d,
    /// dispatch a single kernel of extent = numFrames * diagCount; each
    /// thread is one (frame, MB-on-diagonal) pair. All N frames execute
    /// their diagonal-d work in parallel on independent CUDA cores.
    /// </summary>
    public void RunBatch(
        ArrayView<byte> yPlane, ArrayView<byte> uPlane, ArrayView<byte> vPlane,
        ArrayView<byte> yRecon, ArrayView<byte> uRecon, ArrayView<byte> vRecon,
        ArrayView<short> y4Coefs, ArrayView<short> y2Coefs,
        ArrayView<short> uCoefs, ArrayView<short> vCoefs,
        ArrayView<int> dequant,
        int mbCols, int mbRows, int frameCount,
        int yStride, int uvStride,
        int y4CoefStride, int y2CoefStride, int uvCoefStride,
        int dequantStride)
    {
        if (frameCount <= 0) throw new ArgumentOutOfRangeException(nameof(frameCount));
        int totalDiagonals = mbCols + mbRows - 1;
        for (int d = 0; d < totalDiagonals; d++)
        {
            int rMin = d - (mbCols - 1); if (rMin < 0) rMin = 0;
            int rMax = d; if (rMax > mbRows - 1) rMax = mbRows - 1;
            int diagCount = rMax - rMin + 1;
            int diagAndRMin = (d << 16) | (rMin & 0xFFFF);
            var p = new Vp8FrameSeqEncodeBatchParams
            {
                YStride = yStride,
                UvStride = uvStride,
                Y4CoefStride = y4CoefStride,
                Y2CoefStride = y2CoefStride,
                UvCoefStride = uvCoefStride,
                DequantStride = dequantStride,
                MbCols = mbCols,
                MbRows = mbRows,
                DiagAndRMin = diagAndRMin,
                DiagCount = diagCount,
            };
            _batchKernel(frameCount * diagCount,
                yPlane, uPlane, vPlane,
                yRecon, uRecon, vRecon,
                y4Coefs, y2Coefs, uCoefs, vCoefs,
                dequant, p);
        }
    }

    /// <summary>
    /// Wave-front kernel: each thread encodes one MB on the supplied
    /// diagonal. Thread index t maps to (row=rMin+t, col=d-row). All
    /// threads on the same diagonal are independent.
    /// </summary>
    private static void EncodeWavefrontKernel(
        Index1D idx,
        ArrayView<byte> yPlane, ArrayView<byte> uPlane, ArrayView<byte> vPlane,
        ArrayView<byte> yRecon, ArrayView<byte> uRecon, ArrayView<byte> vRecon,
        ArrayView<short> y4Coefs, ArrayView<short> y2Coefs,
        ArrayView<short> uCoefs, ArrayView<short> vCoefs,
        ArrayView<int> dequant,
        int mbCols, int mbRows, int diagAndRMin)
    {
        int diagD = diagAndRMin >> 16;
        int rMin = diagAndRMin & 0xFFFF;
        int yStride = mbCols * 16;
        int uvStride = mbCols * 8;
        int y1Dc = dequant[0]; int y1Ac = dequant[1];
        int y2Dc = dequant[2]; int y2Ac = dequant[3];
        int uvDc = dequant[4]; int uvAc = dequant[5];

        int mbRow = rMin + idx.X;
        int mbCol = diagD - mbRow;
        if (mbRow < 0 || mbRow >= mbRows || mbCol < 0 || mbCol >= mbCols) return;

        EncodeMacroblock(
            mbRow, mbCol, mbCols, yStride, uvStride,
            yPlane, uPlane, vPlane,
            yRecon, uRecon, vRecon,
            y4Coefs, y2Coefs, uCoefs, vCoefs,
            y1Dc, y1Ac, y2Dc, y2Ac, uvDc, uvAc);
    }

    /// <summary>
    /// Try to use the single-dispatch kernel: gridSize = numFrames blocks,
    /// each block has peakDiag threads. Each block does ONE frame's full
    /// wave-front internally with Group.Barrier between diagonals; blocks
    /// run in parallel on different SMs. CUDA caps block at 1024 threads
    /// AND has register-file pressure - peakDiag rarely exceeds ~64 for
    /// typical resolutions, so register pressure is fine per block.
    /// Returns false only if a hard limit is hit (peakDiag > 1024).
    /// </summary>
    public bool TryRunBatchSingleDispatch(
        ArrayView<byte> yPlane, ArrayView<byte> uPlane, ArrayView<byte> vPlane,
        ArrayView<byte> yRecon, ArrayView<byte> uRecon, ArrayView<byte> vRecon,
        ArrayView<short> y4Coefs, ArrayView<short> y2Coefs,
        ArrayView<short> uCoefs, ArrayView<short> vCoefs,
        ArrayView<int> dequant,
        int mbCols, int mbRows, int frameCount,
        int yStride, int uvStride,
        int y4CoefStride, int y2CoefStride, int uvCoefStride,
        int dequantStride)
    {
        if (frameCount <= 0) throw new ArgumentOutOfRangeException(nameof(frameCount));
        int peakDiag = mbCols < mbRows ? mbCols : mbRows;
        int totalDiagonals = mbCols + mbRows - 1;
        if (peakDiag > 1024) return false;

        var p = new Vp8FrameSeqEncodeBatchParams
        {
            YStride = yStride,
            UvStride = uvStride,
            Y4CoefStride = y4CoefStride,
            Y2CoefStride = y2CoefStride,
            UvCoefStride = uvCoefStride,
            DequantStride = dequantStride,
            MbCols = mbCols,
            MbRows = mbRows,
            DiagAndRMin = 0,
            DiagCount = peakDiag,
        };
        // gridSize = numFrames, blockSize = peakDiag.
        _singleDispatchBatchKernel(
            new KernelConfig(frameCount, peakDiag),
            yPlane, uPlane, vPlane,
            yRecon, uRecon, vRecon,
            y4Coefs, y2Coefs, uCoefs, vCoefs,
            dequant, p, frameCount, totalDiagonals);
        return true;
    }

    /// <summary>
    /// Single-dispatch wave-front kernel. Each block (Grid.IdxX = frame
    /// index) processes one full frame's wave-front; threads within a block
    /// (Group.IdxX = MB position on diagonal) handle parallel MBs per
    /// diagonal with Group.Barrier between diagonals.
    /// </summary>
    private static void EncodeWavefrontSingleDispatchKernel(
        ArrayView<byte> yPlane, ArrayView<byte> uPlane, ArrayView<byte> vPlane,
        ArrayView<byte> yRecon, ArrayView<byte> uRecon, ArrayView<byte> vRecon,
        ArrayView<short> y4Coefs, ArrayView<short> y2Coefs,
        ArrayView<short> uCoefs, ArrayView<short> vCoefs,
        ArrayView<int> dequant,
        Vp8FrameSeqEncodeBatchParams p, int frameCount, int totalDiagonals)
    {
        int frameIdx = Grid.IdxX;
        int posInDiag = Group.IdxX;
        int mbCols = p.MbCols;
        int mbRows = p.MbRows;

        if (frameIdx >= frameCount) return;

        // Per-frame slot views computed once per thread.
        var fY = yPlane.SubView((long)frameIdx * p.YStride, p.YStride);
        var fU = uPlane.SubView((long)frameIdx * p.UvStride, p.UvStride);
        var fV = vPlane.SubView((long)frameIdx * p.UvStride, p.UvStride);
        var fYR = yRecon.SubView((long)frameIdx * p.YStride, p.YStride);
        var fUR = uRecon.SubView((long)frameIdx * p.UvStride, p.UvStride);
        var fVR = vRecon.SubView((long)frameIdx * p.UvStride, p.UvStride);
        var fY4 = y4Coefs.SubView((long)frameIdx * p.Y4CoefStride, p.Y4CoefStride);
        var fY2 = y2Coefs.SubView((long)frameIdx * p.Y2CoefStride, p.Y2CoefStride);
        var fUC = uCoefs.SubView((long)frameIdx * p.UvCoefStride, p.UvCoefStride);
        var fVC = vCoefs.SubView((long)frameIdx * p.UvCoefStride, p.UvCoefStride);
        var fDQ = dequant.SubView((long)frameIdx * p.DequantStride, p.DequantStride);
        int yStride = mbCols * 16;
        int uvStride = mbCols * 8;
        int y1Dc = fDQ[0]; int y1Ac = fDQ[1];
        int y2Dc = fDQ[2]; int y2Ac = fDQ[3];
        int uvDc = fDQ[4]; int uvAc = fDQ[5];

        for (int d = 0; d < totalDiagonals; d++)
        {
            int rMin = d - (mbCols - 1); if (rMin < 0) rMin = 0;
            int rMax = d; if (rMax > mbRows - 1) rMax = mbRows - 1;
            int diagCount = rMax - rMin + 1;
            int mbRow = rMin + posInDiag;
            int mbCol = d - mbRow;
            bool active = posInDiag < diagCount && mbRow >= 0 && mbRow < mbRows && mbCol >= 0 && mbCol < mbCols;
            if (active)
            {
                EncodeMacroblock(
                    mbRow, mbCol, mbCols, yStride, uvStride,
                    fY, fU, fV,
                    fYR, fUR, fVR,
                    fY4, fY2, fUC, fVC,
                    y1Dc, y1Ac, y2Dc, y2Ac, uvDc, uvAc);
            }
            Group.Barrier();
        }
    }

    /// <summary>
    /// Batch wave-front kernel: each thread is one (frame, mb-on-diagonal).
    /// Per-frame buffer slots are SubView'd at the kernel head; MB encode
    /// helper sees a single-frame view and runs unchanged.
    /// </summary>
    private static void EncodeWavefrontBatchKernel(
        Index1D idx,
        ArrayView<byte> yPlane, ArrayView<byte> uPlane, ArrayView<byte> vPlane,
        ArrayView<byte> yRecon, ArrayView<byte> uRecon, ArrayView<byte> vRecon,
        ArrayView<short> y4Coefs, ArrayView<short> y2Coefs,
        ArrayView<short> uCoefs, ArrayView<short> vCoefs,
        ArrayView<int> dequant,
        Vp8FrameSeqEncodeBatchParams p)
    {
        int g = idx.X;
        int frameIdx = g / p.DiagCount;
        int posOnDiag = g - frameIdx * p.DiagCount;
        int diagD = p.DiagAndRMin >> 16;
        int rMin = p.DiagAndRMin & 0xFFFF;
        int mbRow = rMin + posOnDiag;
        int mbCol = diagD - mbRow;
        if (mbRow < 0 || mbRow >= p.MbRows || mbCol < 0 || mbCol >= p.MbCols) return;

        // Per-frame slot views.
        var fY = yPlane.SubView((long)frameIdx * p.YStride, p.YStride);
        var fU = uPlane.SubView((long)frameIdx * p.UvStride, p.UvStride);
        var fV = vPlane.SubView((long)frameIdx * p.UvStride, p.UvStride);
        var fYR = yRecon.SubView((long)frameIdx * p.YStride, p.YStride);
        var fUR = uRecon.SubView((long)frameIdx * p.UvStride, p.UvStride);
        var fVR = vRecon.SubView((long)frameIdx * p.UvStride, p.UvStride);
        var fY4 = y4Coefs.SubView((long)frameIdx * p.Y4CoefStride, p.Y4CoefStride);
        var fY2 = y2Coefs.SubView((long)frameIdx * p.Y2CoefStride, p.Y2CoefStride);
        var fUC = uCoefs.SubView((long)frameIdx * p.UvCoefStride, p.UvCoefStride);
        var fVC = vCoefs.SubView((long)frameIdx * p.UvCoefStride, p.UvCoefStride);
        var fDQ = dequant.SubView((long)frameIdx * p.DequantStride, p.DequantStride);

        int yStride = p.MbCols * 16;
        int uvStride = p.MbCols * 8;
        int y1Dc = fDQ[0]; int y1Ac = fDQ[1];
        int y2Dc = fDQ[2]; int y2Ac = fDQ[3];
        int uvDc = fDQ[4]; int uvAc = fDQ[5];

        EncodeMacroblock(
            mbRow, mbCol, p.MbCols, yStride, uvStride,
            fY, fU, fV,
            fYR, fUR, fVR,
            fY4, fY2, fUC, fVC,
            y1Dc, y1Ac, y2Dc, y2Ac, uvDc, uvAc);
    }

    /// <summary>Encode one MB end-to-end: predict + transform + quant + recon.</summary>
    private static void EncodeMacroblock(
        int mbRow, int mbCol, int mbCols, int yStride, int uvStride,
        ArrayView<byte> yPlane, ArrayView<byte> uPlane, ArrayView<byte> vPlane,
        ArrayView<byte> yRecon, ArrayView<byte> uRecon, ArrayView<byte> vRecon,
        ArrayView<short> y4Coefs, ArrayView<short> y2Coefs,
        ArrayView<short> uCoefs, ArrayView<short> vCoefs,
        int y1Dc, int y1Ac, int y2Dc, int y2Ac, int uvDc, int uvAc)
    {
        int mbIdx = mbRow * mbCols + mbCol;
        bool haveAbove = mbRow > 0;
        bool haveLeft = mbCol > 0;

        // === Y predictor: DC mode ===
        int yPred = ComputeYDcPredictor(mbRow, mbCol, mbCols, yStride, yRecon, haveAbove, haveLeft);
        int uPred = ComputeUvDcPredictor(mbRow, mbCol, mbCols, uvStride, uRecon, haveAbove, haveLeft);
        int vPred = ComputeUvDcPredictor(mbRow, mbCol, mbCols, uvStride, vRecon, haveAbove, haveLeft);

        // === Y4: per-block FDCT + save DC + zero DC + quantize + save coefs ===
        // y4DcVals[16] holds the pre-Walsh DCs for the Y2 transform.
        // Use 16 ints in registers so ILGPU keeps them out of memory.
        int dc0 = 0, dc1 = 0, dc2 = 0, dc3 = 0;
        int dc4 = 0, dc5 = 0, dc6 = 0, dc7 = 0;
        int dc8 = 0, dc9 = 0, dc10 = 0, dc11 = 0;
        int dc12 = 0, dc13 = 0, dc14 = 0, dc15 = 0;

        long y4McBase = (long)mbIdx * 256;
        for (int by = 0; by < 4; by++)
        {
            for (int bx = 0; bx < 4; bx++)
            {
                int blockIdxInMb = by * 4 + bx;
                long y4BlockBase = y4McBase + (long)blockIdxInMb * 16;

                // Read 4x4 source residual = src - yPred.
                short r00 = ReadResidual(yPlane, mbRow, mbCol, by, bx, 0, 0, yStride, yPred);
                short r01 = ReadResidual(yPlane, mbRow, mbCol, by, bx, 0, 1, yStride, yPred);
                short r02 = ReadResidual(yPlane, mbRow, mbCol, by, bx, 0, 2, yStride, yPred);
                short r03 = ReadResidual(yPlane, mbRow, mbCol, by, bx, 0, 3, yStride, yPred);
                short r10 = ReadResidual(yPlane, mbRow, mbCol, by, bx, 1, 0, yStride, yPred);
                short r11 = ReadResidual(yPlane, mbRow, mbCol, by, bx, 1, 1, yStride, yPred);
                short r12 = ReadResidual(yPlane, mbRow, mbCol, by, bx, 1, 2, yStride, yPred);
                short r13 = ReadResidual(yPlane, mbRow, mbCol, by, bx, 1, 3, yStride, yPred);
                short r20 = ReadResidual(yPlane, mbRow, mbCol, by, bx, 2, 0, yStride, yPred);
                short r21 = ReadResidual(yPlane, mbRow, mbCol, by, bx, 2, 1, yStride, yPred);
                short r22 = ReadResidual(yPlane, mbRow, mbCol, by, bx, 2, 2, yStride, yPred);
                short r23 = ReadResidual(yPlane, mbRow, mbCol, by, bx, 2, 3, yStride, yPred);
                short r30 = ReadResidual(yPlane, mbRow, mbCol, by, bx, 3, 0, yStride, yPred);
                short r31 = ReadResidual(yPlane, mbRow, mbCol, by, bx, 3, 1, yStride, yPred);
                short r32 = ReadResidual(yPlane, mbRow, mbCol, by, bx, 3, 2, yStride, yPred);
                short r33 = ReadResidual(yPlane, mbRow, mbCol, by, bx, 3, 3, yStride, yPred);

                // FDCT 4x4. Pass 1: rows. Pass 2: cols.
                FdctRow(r00, r01, r02, r03, out short s00, out short s01, out short s02, out short s03);
                FdctRow(r10, r11, r12, r13, out short s10, out short s11, out short s12, out short s13);
                FdctRow(r20, r21, r22, r23, out short s20, out short s21, out short s22, out short s23);
                FdctRow(r30, r31, r32, r33, out short s30, out short s31, out short s32, out short s33);

                FdctCol(s00, s10, s20, s30, out short c00, out short c10, out short c20, out short c30);
                FdctCol(s01, s11, s21, s31, out short c01, out short c11, out short c21, out short c31);
                FdctCol(s02, s12, s22, s32, out short c02, out short c12, out short c22, out short c32);
                FdctCol(s03, s13, s23, s33, out short c03, out short c13, out short c23, out short c33);

                // Save DC for Y2.
                int dc = c00;
                if (blockIdxInMb == 0) dc0 = dc;
                else if (blockIdxInMb == 1) dc1 = dc;
                else if (blockIdxInMb == 2) dc2 = dc;
                else if (blockIdxInMb == 3) dc3 = dc;
                else if (blockIdxInMb == 4) dc4 = dc;
                else if (blockIdxInMb == 5) dc5 = dc;
                else if (blockIdxInMb == 6) dc6 = dc;
                else if (blockIdxInMb == 7) dc7 = dc;
                else if (blockIdxInMb == 8) dc8 = dc;
                else if (blockIdxInMb == 9) dc9 = dc;
                else if (blockIdxInMb == 10) dc10 = dc;
                else if (blockIdxInMb == 11) dc11 = dc;
                else if (blockIdxInMb == 12) dc12 = dc;
                else if (blockIdxInMb == 13) dc13 = dc;
                else if (blockIdxInMb == 14) dc14 = dc;
                else dc15 = dc;

                // Y4 quantize: coef[0] := 0 (encoder convention - Y2 carries DC),
                // [1..15] := round(c / y1Ac).
                y4Coefs[y4BlockBase + 0] = 0;
                y4Coefs[y4BlockBase + 1] = QuantS(c01, y1Ac);
                y4Coefs[y4BlockBase + 2] = QuantS(c02, y1Ac);
                y4Coefs[y4BlockBase + 3] = QuantS(c03, y1Ac);
                y4Coefs[y4BlockBase + 4] = QuantS(c10, y1Ac);
                y4Coefs[y4BlockBase + 5] = QuantS(c11, y1Ac);
                y4Coefs[y4BlockBase + 6] = QuantS(c12, y1Ac);
                y4Coefs[y4BlockBase + 7] = QuantS(c13, y1Ac);
                y4Coefs[y4BlockBase + 8] = QuantS(c20, y1Ac);
                y4Coefs[y4BlockBase + 9] = QuantS(c21, y1Ac);
                y4Coefs[y4BlockBase + 10] = QuantS(c22, y1Ac);
                y4Coefs[y4BlockBase + 11] = QuantS(c23, y1Ac);
                y4Coefs[y4BlockBase + 12] = QuantS(c30, y1Ac);
                y4Coefs[y4BlockBase + 13] = QuantS(c31, y1Ac);
                y4Coefs[y4BlockBase + 14] = QuantS(c32, y1Ac);
                y4Coefs[y4BlockBase + 15] = QuantS(c33, y1Ac);
            }
        }

        // === Y2: forward Walsh on the 16 Y4 DCs, then quantize ===
        WalshAndQuantizeY2(
            (short)dc0, (short)dc1, (short)dc2, (short)dc3,
            (short)dc4, (short)dc5, (short)dc6, (short)dc7,
            (short)dc8, (short)dc9, (short)dc10, (short)dc11,
            (short)dc12, (short)dc13, (short)dc14, (short)dc15,
            y2Coefs, (long)mbIdx * 16, y2Dc, y2Ac);

        // === U plane: 4 4x4 blocks, FDCT + quant ===
        long uMcBase = (long)mbIdx * 64;
        for (int by = 0; by < 2; by++)
        {
            for (int bx = 0; bx < 2; bx++)
            {
                int blockIdx = by * 2 + bx;
                long uBlockBase = uMcBase + (long)blockIdx * 16;
                EncodeUvBlock(uPlane, mbRow, mbCol, by, bx, uvStride, uPred,
                    uCoefs, uBlockBase, uvDc, uvAc);
            }
        }
        long vMcBase = (long)mbIdx * 64;
        for (int by = 0; by < 2; by++)
        {
            for (int bx = 0; bx < 2; bx++)
            {
                int blockIdx = by * 2 + bx;
                long vBlockBase = vMcBase + (long)blockIdx * 16;
                EncodeUvBlock(vPlane, mbRow, mbCol, by, bx, uvStride, vPred,
                    vCoefs, vBlockBase, uvDc, uvAc);
            }
        }

        // === Inverse transform + reconstruct (writes back to recon planes) ===
        // 1. Dequantize Y2.
        short y2dq0 = (short)(y2Coefs[(long)mbIdx * 16 + 0] * y2Dc);
        short y2dq1 = (short)(y2Coefs[(long)mbIdx * 16 + 1] * y2Ac);
        short y2dq2 = (short)(y2Coefs[(long)mbIdx * 16 + 2] * y2Ac);
        short y2dq3 = (short)(y2Coefs[(long)mbIdx * 16 + 3] * y2Ac);
        short y2dq4 = (short)(y2Coefs[(long)mbIdx * 16 + 4] * y2Ac);
        short y2dq5 = (short)(y2Coefs[(long)mbIdx * 16 + 5] * y2Ac);
        short y2dq6 = (short)(y2Coefs[(long)mbIdx * 16 + 6] * y2Ac);
        short y2dq7 = (short)(y2Coefs[(long)mbIdx * 16 + 7] * y2Ac);
        short y2dq8 = (short)(y2Coefs[(long)mbIdx * 16 + 8] * y2Ac);
        short y2dq9 = (short)(y2Coefs[(long)mbIdx * 16 + 9] * y2Ac);
        short y2dq10 = (short)(y2Coefs[(long)mbIdx * 16 + 10] * y2Ac);
        short y2dq11 = (short)(y2Coefs[(long)mbIdx * 16 + 11] * y2Ac);
        short y2dq12 = (short)(y2Coefs[(long)mbIdx * 16 + 12] * y2Ac);
        short y2dq13 = (short)(y2Coefs[(long)mbIdx * 16 + 13] * y2Ac);
        short y2dq14 = (short)(y2Coefs[(long)mbIdx * 16 + 14] * y2Ac);
        short y2dq15 = (short)(y2Coefs[(long)mbIdx * 16 + 15] * y2Ac);

        // 2. Inverse Walsh on Y2 -> 16 ints (one DC per Y4 block).
        // Branch on Y2 AC presence: if all Y2[1..15] are zero, libvpx
        // uses a DC-broadcast fast path; else full inverse Walsh.
        int y2InvAc =
            y2dq1 | y2dq2 | y2dq3 | y2dq4 | y2dq5 | y2dq6 | y2dq7 |
            y2dq8 | y2dq9 | y2dq10 | y2dq11 | y2dq12 | y2dq13 | y2dq14 | y2dq15;
        int yInv0, yInv1, yInv2, yInv3, yInv4, yInv5, yInv6, yInv7;
        int yInv8, yInv9, yInv10, yInv11, yInv12, yInv13, yInv14, yInv15;
        if (y2InvAc == 0)
        {
            // libvpx vp8_short_inv_walsh4x4_1: broadcast (dc + 3) >> 3.
            int v = (y2dq0 + 3) >> 3;
            yInv0 = yInv1 = yInv2 = yInv3 = yInv4 = yInv5 = yInv6 = yInv7 =
                yInv8 = yInv9 = yInv10 = yInv11 = yInv12 = yInv13 = yInv14 = yInv15 = v;
        }
        else
        {
            InvWalsh4x4(
                y2dq0, y2dq1, y2dq2, y2dq3,
                y2dq4, y2dq5, y2dq6, y2dq7,
                y2dq8, y2dq9, y2dq10, y2dq11,
                y2dq12, y2dq13, y2dq14, y2dq15,
                out yInv0, out yInv1, out yInv2, out yInv3,
                out yInv4, out yInv5, out yInv6, out yInv7,
                out yInv8, out yInv9, out yInv10, out yInv11,
                out yInv12, out yInv13, out yInv14, out yInv15);
        }

        // 3. Per Y4 block: dequant + inject Y2 inv DC + IDCT + add pred + clip + write recon.
        for (int by = 0; by < 4; by++)
        {
            for (int bx = 0; bx < 4; bx++)
            {
                int blockIdxInMb = by * 4 + bx;
                long y4BlockBase = y4McBase + (long)blockIdxInMb * 16;

                int injectDc;
                if (blockIdxInMb == 0) injectDc = yInv0;
                else if (blockIdxInMb == 1) injectDc = yInv1;
                else if (blockIdxInMb == 2) injectDc = yInv2;
                else if (blockIdxInMb == 3) injectDc = yInv3;
                else if (blockIdxInMb == 4) injectDc = yInv4;
                else if (blockIdxInMb == 5) injectDc = yInv5;
                else if (blockIdxInMb == 6) injectDc = yInv6;
                else if (blockIdxInMb == 7) injectDc = yInv7;
                else if (blockIdxInMb == 8) injectDc = yInv8;
                else if (blockIdxInMb == 9) injectDc = yInv9;
                else if (blockIdxInMb == 10) injectDc = yInv10;
                else if (blockIdxInMb == 11) injectDc = yInv11;
                else if (blockIdxInMb == 12) injectDc = yInv12;
                else if (blockIdxInMb == 13) injectDc = yInv13;
                else if (blockIdxInMb == 14) injectDc = yInv14;
                else injectDc = yInv15;

                IdctAddBlock(
                    y4Coefs, y4BlockBase, y1Dc, y1Ac, injectDc,
                    yPred, yRecon, mbRow, mbCol, by, bx, yStride, isUv: false);
            }
        }

        // 4. Per UV block: dequant + IDCT + add pred + clip + write recon.
        for (int by = 0; by < 2; by++)
        {
            for (int bx = 0; bx < 2; bx++)
            {
                int blockIdx = by * 2 + bx;
                long uBlockBase = uMcBase + (long)blockIdx * 16;
                long vBlockBase = vMcBase + (long)blockIdx * 16;
                IdctAddBlock(
                    uCoefs, uBlockBase, uvDc, uvAc, injectDc: int.MinValue,
                    uPred, uRecon, mbRow, mbCol, by, bx, uvStride, isUv: true);
                IdctAddBlock(
                    vCoefs, vBlockBase, uvDc, uvAc, injectDc: int.MinValue,
                    vPred, vRecon, mbRow, mbCol, by, bx, uvStride, isUv: true);
            }
        }
    }

    // ============================================================
    // Helpers
    // ============================================================

    private static int ComputeYDcPredictor(
        int mbRow, int mbCol, int mbCols, int yStride,
        ArrayView<byte> yRecon, bool haveAbove, bool haveLeft)
    {
        if (haveAbove && haveLeft)
        {
            int sum = 0;
            for (int c = 0; c < 16; c++)
                sum += yRecon[(long)(mbRow * 16 - 1) * yStride + mbCol * 16 + c];
            for (int r = 0; r < 16; r++)
                sum += yRecon[(long)(mbRow * 16 + r) * yStride + mbCol * 16 - 1];
            return (sum + 16) >> 5;
        }
        else if (haveAbove)
        {
            int sum = 0;
            for (int c = 0; c < 16; c++)
                sum += yRecon[(long)(mbRow * 16 - 1) * yStride + mbCol * 16 + c];
            return (sum + 8) >> 4;
        }
        else if (haveLeft)
        {
            int sum = 0;
            for (int r = 0; r < 16; r++)
                sum += yRecon[(long)(mbRow * 16 + r) * yStride + mbCol * 16 - 1];
            return (sum + 8) >> 4;
        }
        return 128;
    }

    private static int ComputeUvDcPredictor(
        int mbRow, int mbCol, int mbCols, int uvStride,
        ArrayView<byte> recon, bool haveAbove, bool haveLeft)
    {
        if (haveAbove && haveLeft)
        {
            int sum = 0;
            for (int c = 0; c < 8; c++)
                sum += recon[(long)(mbRow * 8 - 1) * uvStride + mbCol * 8 + c];
            for (int r = 0; r < 8; r++)
                sum += recon[(long)(mbRow * 8 + r) * uvStride + mbCol * 8 - 1];
            return (sum + 8) >> 4;
        }
        else if (haveAbove)
        {
            int sum = 0;
            for (int c = 0; c < 8; c++)
                sum += recon[(long)(mbRow * 8 - 1) * uvStride + mbCol * 8 + c];
            return (sum + 4) >> 3;
        }
        else if (haveLeft)
        {
            int sum = 0;
            for (int r = 0; r < 8; r++)
                sum += recon[(long)(mbRow * 8 + r) * uvStride + mbCol * 8 - 1];
            return (sum + 4) >> 3;
        }
        return 128;
    }

    private static short ReadResidual(
        ArrayView<byte> plane, int mbRow, int mbCol, int by, int bx, int r, int c,
        int stride, int pred)
    {
        long off = (long)(mbRow * 16 + by * 4 + r) * stride + mbCol * 16 + bx * 4 + c;
        return (short)(plane[off] - pred);
    }

    /// <summary>VP8 forward DCT row pass. Bit-exact to Vp8ForwardTransform.ShortFdct4x4 row half.</summary>
    private static void FdctRow(
        short s0, short s1, short s2, short s3,
        out short t0, out short t1, out short t2, out short t3)
    {
        int a1 = (s0 + s3) * 8;
        int b1 = (s1 + s2) * 8;
        int c1 = (s1 - s2) * 8;
        int d1 = (s0 - s3) * 8;
        t0 = (short)(a1 + b1);
        t2 = (short)(a1 - b1);
        t1 = (short)((c1 * 2217 + d1 * 5352 + 14500) >> 12);
        t3 = (short)((d1 * 2217 - c1 * 5352 + 7500) >> 12);
    }

    /// <summary>VP8 forward DCT column pass.</summary>
    private static void FdctCol(
        short s0, short s1, short s2, short s3,
        out short t0, out short t1, out short t2, out short t3)
    {
        int a1 = s0 + s3;
        int b1 = s1 + s2;
        int c1 = s1 - s2;
        int d1 = s0 - s3;
        t0 = (short)((a1 + b1 + 7) >> 4);
        t2 = (short)((a1 - b1 + 7) >> 4);
        t1 = (short)(((c1 * 2217 + d1 * 5352 + 12000) >> 16) + (d1 != 0 ? 1 : 0));
        t3 = (short)((d1 * 2217 - c1 * 5352 + 51000) >> 16);
    }

    /// <summary>Round-half-toward-zero division. Mirror of Vp8ForwardQuantizer.RoundedDivide.</summary>
    private static short QuantS(int value, int divisor)
    {
        if (value >= 0)
            return (short)((value + divisor / 2) / divisor);
        return (short)(-(((-value) + divisor / 2) / divisor));
    }

    /// <summary>Walsh transform on 16 Y2 DCs, then quantize, write to y2Coefs.</summary>
    private static void WalshAndQuantizeY2(
        short i00, short i01, short i02, short i03,
        short i10, short i11, short i12, short i13,
        short i20, short i21, short i22, short i23,
        short i30, short i31, short i32, short i33,
        ArrayView<short> y2Coefs, long y2Base,
        int y2Dc, int y2Ac)
    {
        // Pass 1: rows.
        WalshRow(i00, i01, i02, i03, out short s00, out short s01, out short s02, out short s03);
        WalshRow(i10, i11, i12, i13, out short s10, out short s11, out short s12, out short s13);
        WalshRow(i20, i21, i22, i23, out short s20, out short s21, out short s22, out short s23);
        WalshRow(i30, i31, i32, i33, out short s30, out short s31, out short s32, out short s33);

        // Pass 2: columns.
        WalshCol(s00, s10, s20, s30, out short o00, out short o10, out short o20, out short o30);
        WalshCol(s01, s11, s21, s31, out short o01, out short o11, out short o21, out short o31);
        WalshCol(s02, s12, s22, s32, out short o02, out short o12, out short o22, out short o32);
        WalshCol(s03, s13, s23, s33, out short o03, out short o13, out short o23, out short o33);

        // Quantize and store.
        y2Coefs[y2Base + 0] = QuantS(o00, y2Dc);
        y2Coefs[y2Base + 1] = QuantS(o01, y2Ac);
        y2Coefs[y2Base + 2] = QuantS(o02, y2Ac);
        y2Coefs[y2Base + 3] = QuantS(o03, y2Ac);
        y2Coefs[y2Base + 4] = QuantS(o10, y2Ac);
        y2Coefs[y2Base + 5] = QuantS(o11, y2Ac);
        y2Coefs[y2Base + 6] = QuantS(o12, y2Ac);
        y2Coefs[y2Base + 7] = QuantS(o13, y2Ac);
        y2Coefs[y2Base + 8] = QuantS(o20, y2Ac);
        y2Coefs[y2Base + 9] = QuantS(o21, y2Ac);
        y2Coefs[y2Base + 10] = QuantS(o22, y2Ac);
        y2Coefs[y2Base + 11] = QuantS(o23, y2Ac);
        y2Coefs[y2Base + 12] = QuantS(o30, y2Ac);
        y2Coefs[y2Base + 13] = QuantS(o31, y2Ac);
        y2Coefs[y2Base + 14] = QuantS(o32, y2Ac);
        y2Coefs[y2Base + 15] = QuantS(o33, y2Ac);
    }

    private static void WalshRow(
        short s0, short s1, short s2, short s3,
        out short t0, out short t1, out short t2, out short t3)
    {
        int a1 = (s0 + s2) * 4;
        int d1 = (s1 + s3) * 4;
        int c1 = (s1 - s3) * 4;
        int b1 = (s0 - s2) * 4;
        t0 = (short)(a1 + d1 + (a1 != 0 ? 1 : 0));
        t1 = (short)(b1 + c1);
        t2 = (short)(b1 - c1);
        t3 = (short)(a1 - d1);
    }

    private static void WalshCol(
        short s0, short s1, short s2, short s3,
        out short t0, out short t1, out short t2, out short t3)
    {
        int a1 = s0 + s2;
        int d1 = s1 + s3;
        int c1 = s1 - s3;
        int b1 = s0 - s2;
        int a2 = a1 + d1;
        int b2 = b1 + c1;
        int c2 = b1 - c1;
        int d2 = a1 - d1;
        a2 += a2 < 0 ? 1 : 0;
        b2 += b2 < 0 ? 1 : 0;
        c2 += c2 < 0 ? 1 : 0;
        d2 += d2 < 0 ? 1 : 0;
        t0 = (short)((a2 + 3) >> 3);
        t1 = (short)((b2 + 3) >> 3);
        t2 = (short)((c2 + 3) >> 3);
        t3 = (short)((d2 + 3) >> 3);
    }

    /// <summary>Inverse Walsh: 16 ins -> 16 outs. Bit-exact to Vp8InverseTransform.ShortInvWalsh4x4.</summary>
    private static void InvWalsh4x4(
        short i00, short i01, short i02, short i03,
        short i10, short i11, short i12, short i13,
        short i20, short i21, short i22, short i23,
        short i30, short i31, short i32, short i33,
        out int o00, out int o01, out int o02, out int o03,
        out int o10, out int o11, out int o12, out int o13,
        out int o20, out int o21, out int o22, out int o23,
        out int o30, out int o31, out int o32, out int o33)
    {
        // Column pass.
        InvWalshCol(i00, i10, i20, i30, out short s00, out short s10, out short s20, out short s30);
        InvWalshCol(i01, i11, i21, i31, out short s01, out short s11, out short s21, out short s31);
        InvWalshCol(i02, i12, i22, i32, out short s02, out short s12, out short s22, out short s32);
        InvWalshCol(i03, i13, i23, i33, out short s03, out short s13, out short s23, out short s33);

        // Row pass with +3 round, >>3 shift.
        InvWalshRowFinal(s00, s01, s02, s03, out o00, out o01, out o02, out o03);
        InvWalshRowFinal(s10, s11, s12, s13, out o10, out o11, out o12, out o13);
        InvWalshRowFinal(s20, s21, s22, s23, out o20, out o21, out o22, out o23);
        InvWalshRowFinal(s30, s31, s32, s33, out o30, out o31, out o32, out o33);
    }

    private static void InvWalshCol(
        short i0, short i1, short i2, short i3,
        out short o0, out short o1, out short o2, out short o3)
    {
        int a1 = i0 + i3;
        int b1 = i1 + i2;
        int c1 = i1 - i2;
        int d1 = i0 - i3;
        o0 = (short)(a1 + b1);
        o1 = (short)(c1 + d1);
        o2 = (short)(a1 - b1);
        o3 = (short)(d1 - c1);
    }

    private static void InvWalshRowFinal(
        short s0, short s1, short s2, short s3,
        out int o0, out int o1, out int o2, out int o3)
    {
        int a1 = s0 + s3;
        int b1 = s1 + s2;
        int c1 = s1 - s2;
        int d1 = s0 - s3;
        int a2 = a1 + b1;
        int b2 = c1 + d1;
        int c2 = a1 - b1;
        int d2 = d1 - c1;
        o0 = (a2 + 3) >> 3;
        o1 = (b2 + 3) >> 3;
        o2 = (c2 + 3) >> 3;
        o3 = (d2 + 3) >> 3;
    }

    /// <summary>UV block: read 4x4 residual, FDCT, quantize, write to coefs.</summary>
    private static void EncodeUvBlock(
        ArrayView<byte> plane, int mbRow, int mbCol, int by, int bx, int uvStride, int pred,
        ArrayView<short> coefs, long blockBase, int dcQ, int acQ)
    {
        // Read 4x4 source residual.
        // For UV, base position in plane is (mbRow*8 + by*4, mbCol*8 + bx*4).
        short r00 = ReadUvResidual(plane, mbRow, mbCol, by, bx, 0, 0, uvStride, pred);
        short r01 = ReadUvResidual(plane, mbRow, mbCol, by, bx, 0, 1, uvStride, pred);
        short r02 = ReadUvResidual(plane, mbRow, mbCol, by, bx, 0, 2, uvStride, pred);
        short r03 = ReadUvResidual(plane, mbRow, mbCol, by, bx, 0, 3, uvStride, pred);
        short r10 = ReadUvResidual(plane, mbRow, mbCol, by, bx, 1, 0, uvStride, pred);
        short r11 = ReadUvResidual(plane, mbRow, mbCol, by, bx, 1, 1, uvStride, pred);
        short r12 = ReadUvResidual(plane, mbRow, mbCol, by, bx, 1, 2, uvStride, pred);
        short r13 = ReadUvResidual(plane, mbRow, mbCol, by, bx, 1, 3, uvStride, pred);
        short r20 = ReadUvResidual(plane, mbRow, mbCol, by, bx, 2, 0, uvStride, pred);
        short r21 = ReadUvResidual(plane, mbRow, mbCol, by, bx, 2, 1, uvStride, pred);
        short r22 = ReadUvResidual(plane, mbRow, mbCol, by, bx, 2, 2, uvStride, pred);
        short r23 = ReadUvResidual(plane, mbRow, mbCol, by, bx, 2, 3, uvStride, pred);
        short r30 = ReadUvResidual(plane, mbRow, mbCol, by, bx, 3, 0, uvStride, pred);
        short r31 = ReadUvResidual(plane, mbRow, mbCol, by, bx, 3, 1, uvStride, pred);
        short r32 = ReadUvResidual(plane, mbRow, mbCol, by, bx, 3, 2, uvStride, pred);
        short r33 = ReadUvResidual(plane, mbRow, mbCol, by, bx, 3, 3, uvStride, pred);

        // FDCT.
        FdctRow(r00, r01, r02, r03, out short s00, out short s01, out short s02, out short s03);
        FdctRow(r10, r11, r12, r13, out short s10, out short s11, out short s12, out short s13);
        FdctRow(r20, r21, r22, r23, out short s20, out short s21, out short s22, out short s23);
        FdctRow(r30, r31, r32, r33, out short s30, out short s31, out short s32, out short s33);
        FdctCol(s00, s10, s20, s30, out short c00, out short c10, out short c20, out short c30);
        FdctCol(s01, s11, s21, s31, out short c01, out short c11, out short c21, out short c31);
        FdctCol(s02, s12, s22, s32, out short c02, out short c12, out short c22, out short c32);
        FdctCol(s03, s13, s23, s33, out short c03, out short c13, out short c23, out short c33);

        // Quantize + store.
        coefs[blockBase + 0] = QuantS(c00, dcQ);
        coefs[blockBase + 1] = QuantS(c01, acQ);
        coefs[blockBase + 2] = QuantS(c02, acQ);
        coefs[blockBase + 3] = QuantS(c03, acQ);
        coefs[blockBase + 4] = QuantS(c10, acQ);
        coefs[blockBase + 5] = QuantS(c11, acQ);
        coefs[blockBase + 6] = QuantS(c12, acQ);
        coefs[blockBase + 7] = QuantS(c13, acQ);
        coefs[blockBase + 8] = QuantS(c20, acQ);
        coefs[blockBase + 9] = QuantS(c21, acQ);
        coefs[blockBase + 10] = QuantS(c22, acQ);
        coefs[blockBase + 11] = QuantS(c23, acQ);
        coefs[blockBase + 12] = QuantS(c30, acQ);
        coefs[blockBase + 13] = QuantS(c31, acQ);
        coefs[blockBase + 14] = QuantS(c32, acQ);
        coefs[blockBase + 15] = QuantS(c33, acQ);
    }

    private static short ReadUvResidual(
        ArrayView<byte> plane, int mbRow, int mbCol, int by, int bx, int r, int c,
        int uvStride, int pred)
    {
        long off = (long)(mbRow * 8 + by * 4 + r) * uvStride + mbCol * 8 + bx * 4 + c;
        return (short)(plane[off] - pred);
    }

    /// <summary>
    /// Dequantize one block's quantized coefs, run IDCT, add predictor,
    /// clip, and write 4x4 recon pixels into the recon plane at the
    /// block's position.
    /// </summary>
    /// <param name="injectDc">If isUv=false and injectDc != int.MinValue, override coef[0] with this Y2-derived DC. Otherwise use the block's own quantized coef[0] times DC quantizer.</param>
    private static void IdctAddBlock(
        ArrayView<short> coefs, long blockBase, int dcQ, int acQ, int injectDc,
        int pred, ArrayView<byte> reconPlane,
        int mbRow, int mbCol, int by, int bx, int stride, bool isUv)
    {
        // Dequantize.
        int q0 = coefs[blockBase + 0];
        int dq0 = (injectDc != int.MinValue) ? injectDc : (q0 * dcQ);
        int dq1 = coefs[blockBase + 1] * acQ;
        int dq2 = coefs[blockBase + 2] * acQ;
        int dq3 = coefs[blockBase + 3] * acQ;
        int dq4 = coefs[blockBase + 4] * acQ;
        int dq5 = coefs[blockBase + 5] * acQ;
        int dq6 = coefs[blockBase + 6] * acQ;
        int dq7 = coefs[blockBase + 7] * acQ;
        int dq8 = coefs[blockBase + 8] * acQ;
        int dq9 = coefs[blockBase + 9] * acQ;
        int dq10 = coefs[blockBase + 10] * acQ;
        int dq11 = coefs[blockBase + 11] * acQ;
        int dq12 = coefs[blockBase + 12] * acQ;
        int dq13 = coefs[blockBase + 13] * acQ;
        int dq14 = coefs[blockBase + 14] * acQ;
        int dq15 = coefs[blockBase + 15] * acQ;

        // IDCT 4x4 - column pass then row pass with +4/>>3 shift.
        // Column pass on input arranged as 4x4 (row-major).
        IdctCol((short)dq0, (short)dq4, (short)dq8, (short)dq12,
            out short s00, out short s10, out short s20, out short s30);
        IdctCol((short)dq1, (short)dq5, (short)dq9, (short)dq13,
            out short s01, out short s11, out short s21, out short s31);
        IdctCol((short)dq2, (short)dq6, (short)dq10, (short)dq14,
            out short s02, out short s12, out short s22, out short s32);
        IdctCol((short)dq3, (short)dq7, (short)dq11, (short)dq15,
            out short s03, out short s13, out short s23, out short s33);

        // Row pass + predict-add + clip + write to recon plane.
        IdctRowAddRecon(s00, s01, s02, s03, pred, reconPlane,
            mbRow, mbCol, by, bx, 0, stride, isUv);
        IdctRowAddRecon(s10, s11, s12, s13, pred, reconPlane,
            mbRow, mbCol, by, bx, 1, stride, isUv);
        IdctRowAddRecon(s20, s21, s22, s23, pred, reconPlane,
            mbRow, mbCol, by, bx, 2, stride, isUv);
        IdctRowAddRecon(s30, s31, s32, s33, pred, reconPlane,
            mbRow, mbCol, by, bx, 3, stride, isUv);
    }

    /// <summary>1D IDCT column pass (no shift). Bit-exact to Vp8InverseTransform.</summary>
    private static void IdctCol(
        short i0, short i1, short i2, short i3,
        out short o0, out short o1, out short o2, out short o3)
    {
        int a1 = i0 + i2;
        int b1 = i0 - i2;
        int temp1 = (i1 * SinpiSqrt2) >> 16;
        int temp2 = i3 + ((i3 * CospiSqrt2Minus1) >> 16);
        int c1 = temp1 - temp2;
        temp1 = i1 + ((i1 * CospiSqrt2Minus1) >> 16);
        temp2 = (i3 * SinpiSqrt2) >> 16;
        int d1 = temp1 + temp2;
        o0 = (short)(a1 + d1);
        o3 = (short)(a1 - d1);
        o1 = (short)(b1 + c1);
        o2 = (short)(b1 - c1);
    }

    /// <summary>1D IDCT row pass with +4/&gt;&gt;3 + predict-add-clip + write to recon plane at MB position.</summary>
    private static void IdctRowAddRecon(
        short i0, short i1, short i2, short i3,
        int pred, ArrayView<byte> reconPlane,
        int mbRow, int mbCol, int by, int bx, int rowInBlock,
        int stride, bool isUv)
    {
        int a1 = i0 + i2;
        int b1 = i0 - i2;
        int temp1 = (i1 * SinpiSqrt2) >> 16;
        int temp2 = i3 + ((i3 * CospiSqrt2Minus1) >> 16);
        int c1 = temp1 - temp2;
        temp1 = i1 + ((i1 * CospiSqrt2Minus1) >> 16);
        temp2 = (i3 * SinpiSqrt2) >> 16;
        int d1 = temp1 + temp2;
        int r0 = (a1 + d1 + 4) >> 3;
        int r3 = (a1 - d1 + 4) >> 3;
        int r1 = (b1 + c1 + 4) >> 3;
        int r2 = (b1 - c1 + 4) >> 3;

        // Write to recon plane at (mbRow*16 + by*4 + rowInBlock, mbCol*16 + bx*4) for Y,
        // or (mbRow*8 + by*4 + rowInBlock, mbCol*8 + bx*4) for UV.
        long reconRow = isUv
            ? (long)(mbRow * 8 + by * 4 + rowInBlock) * stride + mbCol * 8 + bx * 4
            : (long)(mbRow * 16 + by * 4 + rowInBlock) * stride + mbCol * 16 + bx * 4;
        reconPlane[reconRow + 0] = ClipAdd(pred, r0);
        reconPlane[reconRow + 1] = ClipAdd(pred, r1);
        reconPlane[reconRow + 2] = ClipAdd(pred, r2);
        reconPlane[reconRow + 3] = ClipAdd(pred, r3);
    }

    private static byte ClipAdd(int p, int r)
    {
        int a = p + r;
        if (a < 0) return 0;
        if (a > 255) return 255;
        return (byte)a;
    }

    /// <summary>Release kernel resources.</summary>
    public void Dispose() { /* auto-grouped */ }
}
