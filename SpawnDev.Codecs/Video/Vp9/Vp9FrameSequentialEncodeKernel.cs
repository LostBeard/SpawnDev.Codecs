// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 v1 multi-MB sequential encoder kernel. Single-thread-per-frame
// GPU kernel that processes every 16x16 macroblock in row-major order,
// running the full forward + inverse pipeline inline so the recon
// plane stays GPU-resident across the entire frame walk.
//
// Per macroblock:
//   Y 16x16  : edges + DC_PRED + residual + FDCT 16x16 + quant
//              + save coefs + dequant + IDCT 16x16 + add-pred -> recon
//   U  8x8   : same shape with 8x8 transform + uv quantizers
//   V  8x8   : same as U on the V plane
//
// Wave-front per-diagonal dispatch: every MB on a given anti-diagonal
// (r+c=d) is independent because DC_PRED only reads MB(r-1,c) and
// MB(r,c-1) - both of which are on diagonal d-1 and finished before
// diagonal d launches. The host driver dispatches one kernel per
// diagonal 0..mbCols+mbRows-2 in order. Peak parallelism is
// min(mbCols, mbRows); for typical 1920x1080 that's 67 (with mbCols=120,
// mbRows=68).
//
// v1 simplifications (mirror Vp9KeyframeEncoder.EncodeKeyFrame):
//   - Profile 0, YUV 4:2:0, width + height multiples of 16
//   - Y intra mode = DC_PRED for every block, transform = Tx16x16
//   - UV intra mode = DC_PRED for every block, transform = Tx8x8
//   - tx_mode = Allow32x32 (means luma uses Tx16x16 - no per-block
//     tx_size signalling)
//   - Single tile, LF disabled, segmentation disabled, default coef
//     probs (no per-frame updates)
//
// The kernel saves quantized coefs to global GPU buffers (yCoefs +
// uCoefs + vCoefs). After this kernel runs, a future
// Vp9FrameEntropyKernel will process those coefs into the bool-coded
// tile bitstream via Vp9BlockCoefEncoderGpu.
//
// Edge defaults for unavailable neighbors mirror libvpx
// (build_intra_predictors_high): the DC predictor variants Top-Only
// / Left-Only / None handle the unavailable side internally - we
// only populate the available side's edge buffer and pick the
// matching variant.

using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>Per-frame slot strides for the batch wave-front sequential encode kernel.</summary>
public struct Vp9FrameSeqEncodeBatchParams
{
    /// <summary>Y plane bytes per frame (yLen).</summary>
    public int YStride;
    /// <summary>UV plane bytes per frame (uvLen).</summary>
    public int UvStride;
    /// <summary>Y coefs per frame (mbCount*256).</summary>
    public int YCoefStride;
    /// <summary>UV coefs per frame (mbCount*64).</summary>
    public int UvCoefStride;
    /// <summary>Dequant ints per frame (4).</summary>
    public int DequantStride;
    /// <summary>mbCols.</summary>
    public int MbCols;
    /// <summary>mbRows.</summary>
    public int MbRows;
    /// <summary>diagD packed in high 16 bits, rMin in low 16.</summary>
    public int DiagAndRMin;
    /// <summary>Number of MBs on this diagonal.</summary>
    public int DiagCount;
}

/// <summary>
/// VP9 v1 multi-MB sequential encoder kernel. Single thread per frame;
/// processes all 16x16 macroblocks in row-major order with inline math
/// composing the per-block GPU helpers. Saves quantized coefs for the
/// downstream entropy kernel; updates recon planes in place.
/// </summary>
public sealed class Vp9FrameSequentialEncodeKernel : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Action<
        Index1D,
        ArrayView<byte>, ArrayView<byte>, ArrayView<byte>,
        ArrayView<byte>, ArrayView<byte>, ArrayView<byte>,
        ArrayView<short>, ArrayView<short>, ArrayView<short>,
        ArrayView<int>,
        int, int, int> _kernel;

    private readonly Action<
        Index1D,
        ArrayView<byte>, ArrayView<byte>, ArrayView<byte>,
        ArrayView<byte>, ArrayView<byte>, ArrayView<byte>,
        ArrayView<short>, ArrayView<short>, ArrayView<short>,
        ArrayView<int>,
        Vp9FrameSeqEncodeBatchParams> _batchKernel;

    /// <summary>
    /// Single-dispatch wave-front: gridSize=numFrames blocks, each block
    /// has peakDiag threads, processes one full frame's wave-front using
    /// Group.Barrier between diagonals. Eliminates per-diagonal launch
    /// overhead.
    /// </summary>
    private readonly Action<
        KernelConfig,
        ArrayView<byte>, ArrayView<byte>, ArrayView<byte>,
        ArrayView<byte>, ArrayView<byte>, ArrayView<byte>,
        ArrayView<short>, ArrayView<short>, ArrayView<short>,
        ArrayView<int>,
        Vp9FrameSeqEncodeBatchParams, int, int> _singleDispatchBatchKernel;

    /// <summary>Compile.</summary>
    public Vp9FrameSequentialEncodeKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<byte>, ArrayView<byte>, ArrayView<byte>,
            ArrayView<byte>, ArrayView<byte>, ArrayView<byte>,
            ArrayView<short>, ArrayView<short>, ArrayView<short>,
            ArrayView<int>,
            int, int, int>(EncodeWavefrontKernel);
        _batchKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<byte>, ArrayView<byte>, ArrayView<byte>,
            ArrayView<byte>, ArrayView<byte>, ArrayView<byte>,
            ArrayView<short>, ArrayView<short>, ArrayView<short>,
            ArrayView<int>,
            Vp9FrameSeqEncodeBatchParams>(EncodeWavefrontBatchKernel);
        _singleDispatchBatchKernel = accelerator.LoadStreamKernel<
            ArrayView<byte>, ArrayView<byte>, ArrayView<byte>,
            ArrayView<byte>, ArrayView<byte>, ArrayView<byte>,
            ArrayView<short>, ArrayView<short>, ArrayView<short>,
            ArrayView<int>,
            Vp9FrameSeqEncodeBatchParams, int, int>(EncodeWavefrontSingleDispatchKernel);
    }

    /// <summary>
    /// Encode all MBs in a frame.
    /// </summary>
    /// <param name="yPlane">Source Y plane (mbRows*16 rows of mbCols*16 bytes).</param>
    /// <param name="uPlane">Source U plane (mbRows*8 rows of mbCols*8 bytes).</param>
    /// <param name="vPlane">Source V plane.</param>
    /// <param name="yRecon">Recon Y plane (in-out; caller pre-fills with arbitrary content - the kernel overwrites with prediction + residual).</param>
    /// <param name="uRecon">Recon U plane.</param>
    /// <param name="vRecon">Recon V plane.</param>
    /// <param name="yCoefs">Output: mbCount*256 quantized coefs.</param>
    /// <param name="uCoefs">Output: mbCount*64 quantized coefs.</param>
    /// <param name="vCoefs">Output: mbCount*64 quantized coefs.</param>
    /// <param name="dequant">4 ints: [Y_DC, Y_AC, UV_DC, UV_AC] - the libvpx dequantizer values that drive both the forward quantizer and the dequantizer step.</param>
    /// <param name="mbCols">Macroblock columns = width / 16.</param>
    /// <param name="mbRows">Macroblock rows = height / 16.</param>
    public void Run(
        ArrayView<byte> yPlane, ArrayView<byte> uPlane, ArrayView<byte> vPlane,
        ArrayView<byte> yRecon, ArrayView<byte> uRecon, ArrayView<byte> vRecon,
        ArrayView<short> yCoefs, ArrayView<short> uCoefs, ArrayView<short> vCoefs,
        ArrayView<int> dequant,
        int mbCols, int mbRows)
    {
        if (mbCols <= 0) throw new ArgumentOutOfRangeException(nameof(mbCols));
        if (mbRows <= 0) throw new ArgumentOutOfRangeException(nameof(mbRows));
        if (dequant.Length < 4) throw new ArgumentException("dequant must hold 4 ints (Y_DC, Y_AC, UV_DC, UV_AC).", nameof(dequant));

        long mbCount = (long)mbCols * mbRows;
        if (yCoefs.Length < mbCount * 256) throw new ArgumentException("yCoefs too short.", nameof(yCoefs));
        if (uCoefs.Length < mbCount * 64) throw new ArgumentException("uCoefs too short.", nameof(uCoefs));
        if (vCoefs.Length < mbCount * 64) throw new ArgumentException("vCoefs too short.", nameof(vCoefs));

        int totalDiagonals = mbCols + mbRows - 1;
        for (int d = 0; d < totalDiagonals; d++)
        {
            int rMin = d - (mbCols - 1); if (rMin < 0) rMin = 0;
            int rMax = d; if (rMax > mbRows - 1) rMax = mbRows - 1;
            int diagCount = rMax - rMin + 1;
            int diagAndRMin = (d << 16) | (rMin & 0xFFFF);
            _kernel(diagCount,
                yPlane, uPlane, vPlane,
                yRecon, uRecon, vRecon,
                yCoefs, uCoefs, vCoefs,
                dequant, mbCols, mbRows, diagAndRMin);
        }
    }

    /// <summary>
    /// Batch wave-front: encode N frames concurrently. 47 dispatches each
    /// at extent = numFrames * diagCount; thread = (frame, mb-on-diagonal).
    /// </summary>
    public void RunBatch(
        ArrayView<byte> yPlane, ArrayView<byte> uPlane, ArrayView<byte> vPlane,
        ArrayView<byte> yRecon, ArrayView<byte> uRecon, ArrayView<byte> vRecon,
        ArrayView<short> yCoefs, ArrayView<short> uCoefs, ArrayView<short> vCoefs,
        ArrayView<int> dequant,
        int mbCols, int mbRows, int frameCount,
        int yStride, int uvStride,
        int yCoefStride, int uvCoefStride, int dequantStride)
    {
        if (frameCount <= 0) throw new ArgumentOutOfRangeException(nameof(frameCount));
        int totalDiagonals = mbCols + mbRows - 1;
        for (int d = 0; d < totalDiagonals; d++)
        {
            int rMin = d - (mbCols - 1); if (rMin < 0) rMin = 0;
            int rMax = d; if (rMax > mbRows - 1) rMax = mbRows - 1;
            int diagCount = rMax - rMin + 1;
            int diagAndRMin = (d << 16) | (rMin & 0xFFFF);
            var p = new Vp9FrameSeqEncodeBatchParams
            {
                YStride = yStride,
                UvStride = uvStride,
                YCoefStride = yCoefStride,
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
                yCoefs, uCoefs, vCoefs,
                dequant, p);
        }
    }

    /// <summary>
    /// Wave-front kernel: each thread encodes one MB on the supplied
    /// diagonal. Per-thread LocalMemory scratch is reused only by that
    /// thread for that one MB; no cross-thread sharing on the diagonal.
    /// </summary>
    private static void EncodeWavefrontKernel(
        Index1D idx,
        ArrayView<byte> yPlane, ArrayView<byte> uPlane, ArrayView<byte> vPlane,
        ArrayView<byte> yRecon, ArrayView<byte> uRecon, ArrayView<byte> vRecon,
        ArrayView<short> yCoefs, ArrayView<short> uCoefs, ArrayView<short> vCoefs,
        ArrayView<int> dequant,
        int mbCols, int mbRows, int diagAndRMin)
    {
        int diagD = diagAndRMin >> 16;
        int rMin = diagAndRMin & 0xFFFF;
        int yStride = mbCols * 16;
        int uvStride = mbCols * 8;

        int yDcQ = dequant[0];
        int yAcQ = dequant[1];
        int uvDcQ = dequant[2];
        int uvAcQ = dequant[3];

        int mbRow = rMin + idx.X;
        int mbCol = diagD - mbRow;
        if (mbRow < 0 || mbRow >= mbRows || mbCol < 0 || mbCol >= mbCols) return;

        // Per-thread scratch (each thread on the diagonal gets its own copy).
        var residual = LocalMemory.Allocate<short>(256);
        var coefsInt = LocalMemory.Allocate<int>(256);
        var coefsShort = LocalMemory.Allocate<short>(256);
        var fdctScratch = LocalMemory.Allocate<int>(256);
        var idctScratch = LocalMemory.Allocate<int>(256);
        var aboveLuma = LocalMemory.Allocate<byte>(16);
        var leftLuma = LocalMemory.Allocate<byte>(16);
        var aboveChroma = LocalMemory.Allocate<byte>(8);
        var leftChroma = LocalMemory.Allocate<byte>(8);

        EncodeMacroblock(
            mbRow, mbCol, mbCols, yStride, uvStride,
            yPlane, uPlane, vPlane,
            yRecon, uRecon, vRecon,
            yCoefs, uCoefs, vCoefs,
            yDcQ, yAcQ, uvDcQ, uvAcQ,
            residual, coefsInt, coefsShort,
            fdctScratch, idctScratch,
            aboveLuma, leftLuma, aboveChroma, leftChroma);
    }

    /// <summary>
    /// Try the single-dispatch wave-front: one block per frame, peakDiag
    /// threads per block, all 47 diagonals folded into one kernel via
    /// Group.Barrier.
    /// </summary>
    public bool TryRunBatchSingleDispatch(
        ArrayView<byte> yPlane, ArrayView<byte> uPlane, ArrayView<byte> vPlane,
        ArrayView<byte> yRecon, ArrayView<byte> uRecon, ArrayView<byte> vRecon,
        ArrayView<short> yCoefs, ArrayView<short> uCoefs, ArrayView<short> vCoefs,
        ArrayView<int> dequant,
        int mbCols, int mbRows, int frameCount,
        int yStride, int uvStride,
        int yCoefStride, int uvCoefStride, int dequantStride)
    {
        if (frameCount <= 0) throw new ArgumentOutOfRangeException(nameof(frameCount));
        int peakDiag = mbCols < mbRows ? mbCols : mbRows;
        int totalDiagonals = mbCols + mbRows - 1;
        if (peakDiag > 1024) return false;

        var p = new Vp9FrameSeqEncodeBatchParams
        {
            YStride = yStride,
            UvStride = uvStride,
            YCoefStride = yCoefStride,
            UvCoefStride = uvCoefStride,
            DequantStride = dequantStride,
            MbCols = mbCols,
            MbRows = mbRows,
            DiagAndRMin = 0,
            DiagCount = peakDiag,
        };
        _singleDispatchBatchKernel(
            new KernelConfig(frameCount, peakDiag),
            yPlane, uPlane, vPlane,
            yRecon, uRecon, vRecon,
            yCoefs, uCoefs, vCoefs,
            dequant, p, frameCount, totalDiagonals);
        return true;
    }

    private static void EncodeWavefrontSingleDispatchKernel(
        ArrayView<byte> yPlane, ArrayView<byte> uPlane, ArrayView<byte> vPlane,
        ArrayView<byte> yRecon, ArrayView<byte> uRecon, ArrayView<byte> vRecon,
        ArrayView<short> yCoefs, ArrayView<short> uCoefs, ArrayView<short> vCoefs,
        ArrayView<int> dequant,
        Vp9FrameSeqEncodeBatchParams p, int frameCount, int totalDiagonals)
    {
        int frameIdx = Grid.IdxX;
        int posInDiag = Group.IdxX;
        int mbCols = p.MbCols;
        int mbRows = p.MbRows;
        if (frameIdx >= frameCount) return;

        var fY = yPlane.SubView((long)frameIdx * p.YStride, p.YStride);
        var fU = uPlane.SubView((long)frameIdx * p.UvStride, p.UvStride);
        var fV = vPlane.SubView((long)frameIdx * p.UvStride, p.UvStride);
        var fYR = yRecon.SubView((long)frameIdx * p.YStride, p.YStride);
        var fUR = uRecon.SubView((long)frameIdx * p.UvStride, p.UvStride);
        var fVR = vRecon.SubView((long)frameIdx * p.UvStride, p.UvStride);
        var fYC = yCoefs.SubView((long)frameIdx * p.YCoefStride, p.YCoefStride);
        var fUC = uCoefs.SubView((long)frameIdx * p.UvCoefStride, p.UvCoefStride);
        var fVC = vCoefs.SubView((long)frameIdx * p.UvCoefStride, p.UvCoefStride);
        var fDQ = dequant.SubView((long)frameIdx * p.DequantStride, p.DequantStride);

        int yStride = mbCols * 16;
        int uvStride = mbCols * 8;
        int yDcQ = fDQ[0]; int yAcQ = fDQ[1];
        int uvDcQ = fDQ[2]; int uvAcQ = fDQ[3];

        var residual = LocalMemory.Allocate<short>(256);
        var coefsInt = LocalMemory.Allocate<int>(256);
        var coefsShort = LocalMemory.Allocate<short>(256);
        var fdctScratch = LocalMemory.Allocate<int>(256);
        var idctScratch = LocalMemory.Allocate<int>(256);
        var aboveLuma = LocalMemory.Allocate<byte>(16);
        var leftLuma = LocalMemory.Allocate<byte>(16);
        var aboveChroma = LocalMemory.Allocate<byte>(8);
        var leftChroma = LocalMemory.Allocate<byte>(8);

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
                    fYC, fUC, fVC,
                    yDcQ, yAcQ, uvDcQ, uvAcQ,
                    residual, coefsInt, coefsShort,
                    fdctScratch, idctScratch,
                    aboveLuma, leftLuma, aboveChroma, leftChroma);
            }
            Group.Barrier();
        }
    }

    /// <summary>Batch wave-front kernel: thread = (frame, mb-on-diag).</summary>
    private static void EncodeWavefrontBatchKernel(
        Index1D idx,
        ArrayView<byte> yPlane, ArrayView<byte> uPlane, ArrayView<byte> vPlane,
        ArrayView<byte> yRecon, ArrayView<byte> uRecon, ArrayView<byte> vRecon,
        ArrayView<short> yCoefs, ArrayView<short> uCoefs, ArrayView<short> vCoefs,
        ArrayView<int> dequant,
        Vp9FrameSeqEncodeBatchParams p)
    {
        int g = idx.X;
        int frameIdx = g / p.DiagCount;
        int posOnDiag = g - frameIdx * p.DiagCount;
        int diagD = p.DiagAndRMin >> 16;
        int rMin = p.DiagAndRMin & 0xFFFF;
        int mbRow = rMin + posOnDiag;
        int mbCol = diagD - mbRow;
        if (mbRow < 0 || mbRow >= p.MbRows || mbCol < 0 || mbCol >= p.MbCols) return;

        var fY = yPlane.SubView((long)frameIdx * p.YStride, p.YStride);
        var fU = uPlane.SubView((long)frameIdx * p.UvStride, p.UvStride);
        var fV = vPlane.SubView((long)frameIdx * p.UvStride, p.UvStride);
        var fYR = yRecon.SubView((long)frameIdx * p.YStride, p.YStride);
        var fUR = uRecon.SubView((long)frameIdx * p.UvStride, p.UvStride);
        var fVR = vRecon.SubView((long)frameIdx * p.UvStride, p.UvStride);
        var fYC = yCoefs.SubView((long)frameIdx * p.YCoefStride, p.YCoefStride);
        var fUC = uCoefs.SubView((long)frameIdx * p.UvCoefStride, p.UvCoefStride);
        var fVC = vCoefs.SubView((long)frameIdx * p.UvCoefStride, p.UvCoefStride);
        var fDQ = dequant.SubView((long)frameIdx * p.DequantStride, p.DequantStride);

        int yStride = p.MbCols * 16;
        int uvStride = p.MbCols * 8;
        int yDcQ = fDQ[0]; int yAcQ = fDQ[1];
        int uvDcQ = fDQ[2]; int uvAcQ = fDQ[3];

        var residual = LocalMemory.Allocate<short>(256);
        var coefsInt = LocalMemory.Allocate<int>(256);
        var coefsShort = LocalMemory.Allocate<short>(256);
        var fdctScratch = LocalMemory.Allocate<int>(256);
        var idctScratch = LocalMemory.Allocate<int>(256);
        var aboveLuma = LocalMemory.Allocate<byte>(16);
        var leftLuma = LocalMemory.Allocate<byte>(16);
        var aboveChroma = LocalMemory.Allocate<byte>(8);
        var leftChroma = LocalMemory.Allocate<byte>(8);

        EncodeMacroblock(
            mbRow, mbCol, p.MbCols, yStride, uvStride,
            fY, fU, fV,
            fYR, fUR, fVR,
            fYC, fUC, fVC,
            yDcQ, yAcQ, uvDcQ, uvAcQ,
            residual, coefsInt, coefsShort,
            fdctScratch, idctScratch,
            aboveLuma, leftLuma, aboveChroma, leftChroma);
    }

    /// <summary>Encode a single 16x16 macroblock (Y + U + V).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void EncodeMacroblock(
        int mbRow, int mbCol, int mbCols,
        int yStride, int uvStride,
        ArrayView<byte> yPlane, ArrayView<byte> uPlane, ArrayView<byte> vPlane,
        ArrayView<byte> yRecon, ArrayView<byte> uRecon, ArrayView<byte> vRecon,
        ArrayView<short> yCoefs, ArrayView<short> uCoefs, ArrayView<short> vCoefs,
        int yDcQ, int yAcQ, int uvDcQ, int uvAcQ,
        ArrayView<short> residual, ArrayView<int> coefsInt, ArrayView<short> coefsShort,
        ArrayView<int> fdctScratch, ArrayView<int> idctScratch,
        ArrayView<byte> aboveLuma, ArrayView<byte> leftLuma,
        ArrayView<byte> aboveChroma, ArrayView<byte> leftChroma)
    {
        long mbIdx = (long)mbRow * mbCols + mbCol;
        long yBase = (long)mbRow * 16 * yStride + mbCol * 16;
        long uvBase = (long)mbRow * 8 * uvStride + mbCol * 8;

        int topAvail = mbRow > 0 ? 1 : 0;
        int leftAvail = mbCol > 0 ? 1 : 0;
        int variant = ComputeVariant(topAvail, leftAvail);

        EncodeLumaBlock16x16(
            yBase, yStride,
            yPlane, yRecon, yCoefs, mbIdx,
            yDcQ, yAcQ,
            topAvail, leftAvail, variant,
            residual, coefsInt, coefsShort, fdctScratch, idctScratch,
            aboveLuma, leftLuma);

        EncodeChromaBlock8x8(
            uvBase, uvStride,
            uPlane, uRecon, uCoefs, mbIdx,
            uvDcQ, uvAcQ,
            topAvail, leftAvail, variant,
            residual, coefsInt, coefsShort, fdctScratch, idctScratch,
            aboveChroma, leftChroma);

        EncodeChromaBlock8x8(
            uvBase, uvStride,
            vPlane, vRecon, vCoefs, mbIdx,
            uvDcQ, uvAcQ,
            topAvail, leftAvail, variant,
            residual, coefsInt, coefsShort, fdctScratch, idctScratch,
            aboveChroma, leftChroma);
    }

    /// <summary>Encode one Y 16x16 block end-to-end.</summary>
    private static void EncodeLumaBlock16x16(
        long yBase, int yStride,
        ArrayView<byte> ySrc, ArrayView<byte> yRecon,
        ArrayView<short> yCoefs, long mbIdx,
        int dcQ, int acQ,
        int topAvail, int leftAvail, int variant,
        ArrayView<short> residual, ArrayView<int> coefsInt, ArrayView<short> coefsShort,
        ArrayView<int> fdctScratch, ArrayView<int> idctScratch,
        ArrayView<byte> aboveBuf, ArrayView<byte> leftBuf)
    {
        // Build above edge from recon's row above (when available).
        if (topAvail != 0)
        {
            for (int i = 0; i < 16; i++)
                aboveBuf[i] = yRecon[yBase - yStride + i];
        }
        // Build left edge from recon's column to the left (when available).
        if (leftAvail != 0)
        {
            for (int r = 0; r < 16; r++)
                leftBuf[r] = yRecon[yBase + (long)r * yStride - 1];
        }

        // DC predict into recon (writes 16x16 = 256 bytes at yRecon[yBase..]).
        Vp9DcPredictorGpu.Predict(
            aboveBuf, 0, leftBuf, 0,
            yRecon, yBase, yStride,
            16, variant);

        // Residual = src - pred. Pred is in recon now.
        for (int r = 0; r < 16; r++)
        {
            for (int c = 0; c < 16; c++)
            {
                int s = ySrc[yBase + (long)r * yStride + c];
                int p = yRecon[yBase + (long)r * yStride + c];
                residual[r * 16 + c] = (short)(s - p);
            }
        }

        // FDCT 16x16: residual (short) -> coefsInt (int).
        Vp9ForwardDct16x16Gpu.Forward16x16(residual, 0, 16, coefsInt, 0, fdctScratch);

        // Forward quantize in place on coefsInt.
        Vp9ForwardQuantizerGpu.QuantizeBlock(coefsInt, 0, 256, dcQ, acQ);

        // Save quantized coefs to global yCoefs[mbIdx*256..] and stage
        // a short copy in coefsShort for the inverse path.
        long yCoefBase = mbIdx * 256;
        for (int i = 0; i < 256; i++)
        {
            short q = (short)coefsInt[i];
            yCoefs[yCoefBase + i] = q;
            coefsShort[i] = q;
        }

        // Dequant in place on coefsShort.
        Vp9DequantBlockGpu.DequantizeBlock(coefsShort, 0, 256, dcQ, acQ);

        // IDCT 16x16 + add residual to recon (in place).
        Vp9Idct16x16Gpu.Idct16x16(coefsShort, 0, yRecon, yBase, yStride, idctScratch);
    }

    /// <summary>Encode one chroma 8x8 block (U or V) end-to-end.</summary>
    private static void EncodeChromaBlock8x8(
        long uvBase, int uvStride,
        ArrayView<byte> src, ArrayView<byte> recon,
        ArrayView<short> coefsOut, long mbIdx,
        int dcQ, int acQ,
        int topAvail, int leftAvail, int variant,
        ArrayView<short> residual, ArrayView<int> coefsInt, ArrayView<short> coefsShort,
        ArrayView<int> fdctScratch, ArrayView<int> idctScratch,
        ArrayView<byte> aboveBuf, ArrayView<byte> leftBuf)
    {
        if (topAvail != 0)
        {
            for (int i = 0; i < 8; i++)
                aboveBuf[i] = recon[uvBase - uvStride + i];
        }
        if (leftAvail != 0)
        {
            for (int r = 0; r < 8; r++)
                leftBuf[r] = recon[uvBase + (long)r * uvStride - 1];
        }

        Vp9DcPredictorGpu.Predict(
            aboveBuf, 0, leftBuf, 0,
            recon, uvBase, uvStride,
            8, variant);

        for (int r = 0; r < 8; r++)
        {
            for (int c = 0; c < 8; c++)
            {
                int s = src[uvBase + (long)r * uvStride + c];
                int p = recon[uvBase + (long)r * uvStride + c];
                residual[r * 8 + c] = (short)(s - p);
            }
        }

        Vp9ForwardDct8x8Gpu.Forward8x8(residual, 0, 8, coefsInt, 0, fdctScratch);
        Vp9ForwardQuantizerGpu.QuantizeBlock(coefsInt, 0, 64, dcQ, acQ);

        long coefBase = mbIdx * 64;
        for (int i = 0; i < 64; i++)
        {
            short q = (short)coefsInt[i];
            coefsOut[coefBase + i] = q;
            coefsShort[i] = q;
        }

        Vp9DequantBlockGpu.DequantizeBlock(coefsShort, 0, 64, dcQ, acQ);
        Vp9Idct8x8Gpu.Idct8x8(coefsShort, 0, recon, uvBase, uvStride, idctScratch);
    }

    /// <summary>
    /// Map (topAvail, leftAvail) to the matching <see cref="Vp9DcVariant"/>
    /// integer. Returns: 0 = Both, 1 = TopOnly, 2 = LeftOnly, 3 = None.
    /// </summary>
    private static int ComputeVariant(int topAvail, int leftAvail)
    {
        if (topAvail != 0 && leftAvail != 0) return (int)Vp9DcVariant.Both;
        if (topAvail != 0) return (int)Vp9DcVariant.TopOnly;
        if (leftAvail != 0) return (int)Vp9DcVariant.LeftOnly;
        return (int)Vp9DcVariant.None;
    }

    /// <summary>Release kernel resources.</summary>
    public void Dispose() { /* auto-grouped */ }
}
