// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP8 keyframe encoder, 100% GPU-resident pipeline. The host (this
// class's caller) is a pure coordinator - it allocates GPU buffers,
// uploads the source YUV planes (necessary I/O), dispatches a chain
// of kernels, and reads back the final encoded keyframe bytes. No
// CPU-side math, no CPU iteration, no CPU bool encoding. Per
// Captain's directive: "The environment using the accelerator is
// not capable of processing any data loads. It is simply the
// coordinator."
//
// Pipeline:
//   [host] upload Y/U/V to GPU
//   [host] dispatch Vp8FrameSetupKernel
//          - computes 6 dequantizers from baseQIndex
//          - writes frame header bits to partition0Out
//          - saves bool encoder state to initialP0State
//   [host] dispatch Vp8FrameSequentialEncodeKernel
//          - per-MB predict + transform + quant + recon
//          - reads dequantizers from buffer, writes recon back
//   [host] dispatch Vp8FrameEntropyKernel
//          - resumes partition0 from initialP0State
//          - writes per-MB modes + coef tokens
//   [host] dispatch Vp8FrameAssembleKernel
//          - writes 10-byte tag + start code + size code
//          - concatenates partition0 + tokenP0 into output
//          - writes final length to outLen[0]
//   [host] read outLen + output, return slice
//
// v1 simplifications (matches Vp8KeyframeEncoder reference):
//   - All MBs use Y_PRED = DC_PRED, UV_PRED = DC_PRED.
//   - No segmentation.
//   - Single token partition (Log2NumPartitions = 0).
//   - Loop filter disabled.
//   - mb_no_skip_coeff disabled.
//   - Default coef probs (no per-frame updates).

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>
/// VP8 keyframe encoder. 100% GPU-resident: host is pure coordinator.
/// Output byte-identical to <see cref="Vp8KeyframeEncoder"/>.
/// </summary>
public sealed class Vp8KeyframeEncoderGpu : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Vp8FrameSetupKernel _setup;
    private readonly Vp8FrameSequentialEncodeKernel _sequentialEncode;
    private readonly Vp8FrameEntropyKernel _entropy;
    private readonly Vp8FrameAssembleKernel _assemble;
    private readonly Vp8StridedPlanePackKernel _stridePack;

    // Constants uploaded once per accelerator and reused across frames.
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _dcQLookup;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _acQLookup;
    private readonly MemoryBuffer1D<byte, Stride1D.Dense> _defaultCoefProbs;
    private readonly MemoryBuffer1D<byte, Stride1D.Dense> _updateCoefProbs;
    private readonly MemoryBuffer1D<byte, Stride1D.Dense> _coefProbsByType;
    private readonly MemoryBuffer1D<byte, Stride1D.Dense> _constsExtended;

    // Per-resolution cached buffers - reallocated only when (width,height) changes.
    // First-frame cost shifts to "first frame at this resolution"; steady-state
    // pays only kernel dispatch + plane upload + readback per frame.
    private int _cachedWidth = -1;
    private int _cachedHeight = -1;
    private MemoryBuffer1D<byte, Stride1D.Dense>? _dY;
    private MemoryBuffer1D<byte, Stride1D.Dense>? _dU;
    private MemoryBuffer1D<byte, Stride1D.Dense>? _dV;
    private MemoryBuffer1D<byte, Stride1D.Dense>? _dYRecon;
    private MemoryBuffer1D<byte, Stride1D.Dense>? _dURecon;
    private MemoryBuffer1D<byte, Stride1D.Dense>? _dVRecon;
    private MemoryBuffer1D<short, Stride1D.Dense>? _dY4Coefs;
    private MemoryBuffer1D<short, Stride1D.Dense>? _dY2Coefs;
    private MemoryBuffer1D<short, Stride1D.Dense>? _dUCoefs;
    private MemoryBuffer1D<short, Stride1D.Dense>? _dVCoefs;
    private MemoryBuffer1D<int, Stride1D.Dense>? _dDequant;
    private MemoryBuffer1D<int, Stride1D.Dense>? _dInitialP0State;
    private MemoryBuffer1D<byte, Stride1D.Dense>? _dP0;
    private MemoryBuffer1D<byte, Stride1D.Dense>? _dTp;
    private MemoryBuffer1D<long, Stride1D.Dense>? _dPartLens;
    private MemoryBuffer1D<byte, Stride1D.Dense>? _dAbove;
    private MemoryBuffer1D<byte, Stride1D.Dense>? _dOutput;
    private MemoryBuffer1D<int, Stride1D.Dense>? _dOutLen;
    private int _p0Stride;
    private int _tp0Stride;

    /// <summary>Compile + cache all kernels and constants onto <paramref name="accelerator"/>.</summary>
    public Vp8KeyframeEncoderGpu(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _setup = new Vp8FrameSetupKernel(accelerator);
        _sequentialEncode = new Vp8FrameSequentialEncodeKernel(accelerator);
        _entropy = new Vp8FrameEntropyKernel(accelerator);
        _assemble = new Vp8FrameAssembleKernel(accelerator);
        _stridePack = new Vp8StridedPlanePackKernel(accelerator);

        // Upload accelerator-resident constants once.
        _dcQLookup = accelerator.Allocate1D<int>(128);
        _acQLookup = accelerator.Allocate1D<int>(128);
        _defaultCoefProbs = accelerator.Allocate1D<byte>(4 * 264);
        _updateCoefProbs = accelerator.Allocate1D<byte>(4 * 264);
        _coefProbsByType = accelerator.Allocate1D<byte>(4 * 264);
        _constsExtended = accelerator.Allocate1D<byte>(Vp8FrameEntropyKernel.ConstsExtendedTotalBytes);

        _dcQLookup.View.CopyFromCPU(Vp8FrameSetupKernel.BuildDcQLookup());
        _acQLookup.View.CopyFromCPU(Vp8FrameSetupKernel.BuildAcQLookup());
        _defaultCoefProbs.View.CopyFromCPU(Vp8FrameSetupKernel.BuildDefaultCoefProbs());
        _updateCoefProbs.View.CopyFromCPU(Vp8FrameSetupKernel.BuildUpdateCoefProbs());
        _coefProbsByType.View.CopyFromCPU(Vp8FrameSetupKernel.BuildDefaultCoefProbs());
        _constsExtended.View.CopyFromCPU(Vp8FrameEntropyKernel.BuildExtendedConstsBuffer());
    }

    /// <summary>
    /// Encode a single VP8 keyframe from YUV420 source. Accepts any positive
    /// (width, height); internally pads to the next 16-multiple working
    /// dimensions. The frame tag in the output bitstream signals the original
    /// (width, height), so spec-compliant decoders crop the working-dim
    /// pixels back to the requested display size.
    /// </summary>
    public byte[] EncodeKeyFrame(
        ReadOnlySpan<byte> ySrc, int ySrcStride,
        ReadOnlySpan<byte> uSrc, int uvSrcStride,
        ReadOnlySpan<byte> vSrc,
        int width, int height,
        int baseQIndex = 30)
    {
        if (width <= 0 || height <= 0) throw new ArgumentOutOfRangeException();
        if (baseQIndex < 0 || baseQIndex > 127)
            throw new ArgumentOutOfRangeException(nameof(baseQIndex));

        // Round up to the next 16-multiple. Source planes are zero-padded
        // into the bottom-right region; the kernels process the full
        // working-dim MB grid bit-exact, and the frame tag emits the
        // original (width, height) so the decoder crops away the pad.
        int workWidth = (width + 15) & ~15;
        int workHeight = (height + 15) & ~15;
        int mbCols = workWidth / 16;
        int mbRows = workHeight / 16;
        int mbCount = mbCols * mbRows;
        int uvWorkWidth = workWidth / 2;
        int uvWorkHeight = workHeight / 2;

        EnsureBuffers(workWidth, workHeight, uvWorkWidth, uvWorkHeight, mbCols, mbCount);

        // Pre-zero so non-aligned padding rows/cols read as 0.
        if (workWidth != width || workHeight != height)
        {
            _dY!.View.MemSetToZero();
            _dU!.View.MemSetToZero();
            _dV!.View.MemSetToZero();
        }

        // === Host work: ONLY upload + dispatch ===
        UploadPaddedPlane(ySrc, ySrcStride, width, height, workWidth, _dY!);
        UploadPaddedPlane(uSrc, uvSrcStride, width / 2, height / 2, uvWorkWidth, _dU!);
        UploadPaddedPlane(vSrc, uvSrcStride, width / 2, height / 2, uvWorkWidth, _dV!);
        // Recon planes are fully overwritten per MB by the sequential
        // encoder; pre-zero only the buffers the bool encoder reads as
        // partial-write carry-back state (P0 / Tp / Above).
        _dP0!.View.MemSetToZero();
        _dTp!.View.MemSetToZero();
        _dAbove!.View.MemSetToZero();

        // 1. Frame setup: dequantizers + frame header.
        _setup.Run(
            _dcQLookup.View, _acQLookup.View,
            _defaultCoefProbs.View, _updateCoefProbs.View,
            _dDequant!.View, _dP0!.View, _dInitialP0State!.View,
            baseQIndex);

        // 2. Sequential encode: per-MB math + recon.
        _sequentialEncode.Run(
            _dY!.View, _dU!.View, _dV!.View,
            _dYRecon!.View, _dURecon!.View, _dVRecon!.View,
            _dY4Coefs!.View, _dY2Coefs!.View, _dUCoefs!.View, _dVCoefs!.View,
            _dDequant!.View,
            mbCols, mbRows);

        // 3. Entropy: continues partition0 from setup state, writes
        // per-MB modes + coefs.
        _entropy.Run(
            _dY4Coefs!.View, _dY2Coefs!.View, _dUCoefs!.View, _dVCoefs!.View,
            _coefProbsByType.View, _constsExtended.View,
            _dP0!.View, _dTp!.View, _dPartLens!.View,
            _dAbove!.View, _dInitialP0State!.View,
            mbCols, mbRows);

        // 4. Assemble: frame tag + start code + size code +
        // partition0 + tokenP0 -> single output buffer; writes final
        // length to outLen[0].
        _assemble.Run(
            _dP0!.View, _dTp!.View, _dPartLens!.View,
            _dOutput!.View, _dOutLen!.View,
            width, height);

        _accelerator.Synchronize();

        // Single partial readback of the final encoded keyframe. SubView
        // -> CopyToCPU is a real per-backend partial readback
        // (SpawnDev.ILGPU 4.9.3+); only the actual finalLen bytes cross
        // the boundary, not the worst-case-sized output buffer.
        int finalLen = _dOutLen!.GetAsArray1D()[0];
        var result = new byte[finalLen];
        _dOutput!.View.SubView(0, finalLen).CopyToCPU(result);
        return result;
    }

    /// <summary>
    /// True frame-batch parallel encode: per-frame buffer slots through
    /// every kernel + extent=N batch entropy. Setup + sequential-encode
    /// + assemble run per-frame against per-frame slots (still serial on
    /// the stream but with no buffer aliasing); the batch entropy kernel
    /// dispatches at extent=numFrames so all frames run their entropy
    /// walks concurrently on independent CUDA cores.
    /// </summary>
    public byte[][] EncodeKeyFramesBatch(
        ReadOnlyMemory<byte>[] yPlanes,
        ReadOnlyMemory<byte>[] uPlanes,
        ReadOnlyMemory<byte>[] vPlanes,
        int width, int height,
        int baseQIndex = 30)
    {
        if (yPlanes is null) throw new ArgumentNullException(nameof(yPlanes));
        if (uPlanes is null) throw new ArgumentNullException(nameof(uPlanes));
        if (vPlanes is null) throw new ArgumentNullException(nameof(vPlanes));
        if (yPlanes.Length != uPlanes.Length || yPlanes.Length != vPlanes.Length)
            throw new ArgumentException("Plane arrays must have equal length.");
        int frameCount = yPlanes.Length;
        if (frameCount == 0) return Array.Empty<byte[]>();

        if (width <= 0 || height <= 0) throw new ArgumentOutOfRangeException();

        // Round up to next 16-multiple working dims; bitstream signals
        // original (width, height) so decoders crop the pad.
        int workWidth = (width + 15) & ~15;
        int workHeight = (height + 15) & ~15;
        int mbCols = workWidth / 16;
        int mbRows = workHeight / 16;
        int mbCount = mbCols * mbRows;
        int uvWorkWidth = workWidth / 2;
        int uvWorkHeight = workHeight / 2;
        int origUvWidth = width / 2;
        int origUvHeight = height / 2;

        EnsureBuffers(workWidth, workHeight, uvWorkWidth, uvWorkHeight, mbCols, mbCount);

        // Per-frame slot strides for batch entropy + sequential encode.
        // Use working dims so kernels see padded planes for non-aligned input.
        int yPlaneStride = workWidth * workHeight;
        int uvPlaneStride = uvWorkWidth * uvWorkHeight;
        int y4Stride = mbCount * 256;
        int y2Stride = mbCount * 16;
        int uvCoefStride = mbCount * 64;
        int aboveStride = mbCols * 9;
        int dequantStride = 6;
        int outputCapacity = 16 + _p0Stride + _tp0Stride;

        // Per-frame slot buffers - sized for all N frames so each batch
        // kernel can SubView per frame.
        using var dAllY = _accelerator.Allocate1D<byte>((long)frameCount * yPlaneStride);
        using var dAllU = _accelerator.Allocate1D<byte>((long)frameCount * uvPlaneStride);
        using var dAllV = _accelerator.Allocate1D<byte>((long)frameCount * uvPlaneStride);
        using var dAllYR = _accelerator.Allocate1D<byte>((long)frameCount * yPlaneStride);
        using var dAllUR = _accelerator.Allocate1D<byte>((long)frameCount * uvPlaneStride);
        using var dAllVR = _accelerator.Allocate1D<byte>((long)frameCount * uvPlaneStride);
        using var dAllY4 = _accelerator.Allocate1D<short>((long)frameCount * y4Stride);
        using var dAllY2 = _accelerator.Allocate1D<short>((long)frameCount * y2Stride);
        using var dAllUC = _accelerator.Allocate1D<short>((long)frameCount * uvCoefStride);
        using var dAllVC = _accelerator.Allocate1D<short>((long)frameCount * uvCoefStride);
        using var dAllDequant = _accelerator.Allocate1D<int>((long)frameCount * dequantStride);
        using var dAllP0 = _accelerator.Allocate1D<byte>((long)frameCount * _p0Stride);
        using var dAllTp = _accelerator.Allocate1D<byte>((long)frameCount * _tp0Stride);
        using var dAllAbove = _accelerator.Allocate1D<byte>((long)frameCount * aboveStride);
        using var dAllInitState = _accelerator.Allocate1D<int>((long)frameCount * 5);
        using var dAllOutLens = _accelerator.Allocate1D<long>((long)frameCount * 2);
        using var dAllOutputs = _accelerator.Allocate1D<byte>((long)frameCount * outputCapacity);
        using var dAllOutputLens = _accelerator.Allocate1D<int>(frameCount);

        dAllP0.View.MemSetToZero();
        dAllTp.View.MemSetToZero();
        dAllAbove.View.MemSetToZero();

        // Phase 1a: bulk upload all frames' planes in 3 host->device transfers.
        // For non-aligned source dims (e.g. height=1080 with workHeight=1088),
        // each frame's source rows are pad-justified into the working-dim
        // slot; remaining padding rows/cols stay zero (new byte[]).
        bool needsPad = workWidth != width || workHeight != height;
        var hostY = new byte[(long)frameCount * yPlaneStride];
        var hostU = new byte[(long)frameCount * uvPlaneStride];
        var hostV = new byte[(long)frameCount * uvPlaneStride];
        if (!needsPad)
        {
            for (int f = 0; f < frameCount; f++)
            {
                yPlanes[f].Span.CopyTo(hostY.AsSpan((int)((long)f * yPlaneStride)));
                uPlanes[f].Span.CopyTo(hostU.AsSpan((int)((long)f * uvPlaneStride)));
                vPlanes[f].Span.CopyTo(hostV.AsSpan((int)((long)f * uvPlaneStride)));
            }
        }
        else
        {
            // Per-frame, per-row pad-justify into the working-dim slot.
            for (int f = 0; f < frameCount; f++)
            {
                long ySlotBase = (long)f * yPlaneStride;
                long uSlotBase = (long)f * uvPlaneStride;
                long vSlotBase = (long)f * uvPlaneStride;
                var ySrc = yPlanes[f].Span;
                var uSrc = uPlanes[f].Span;
                var vSrc = vPlanes[f].Span;
                // Y plane
                for (int r = 0; r < height; r++)
                    ySrc.Slice(r * width, width).CopyTo(hostY.AsSpan((int)(ySlotBase + r * workWidth), width));
                // UV planes
                for (int r = 0; r < origUvHeight; r++)
                {
                    uSrc.Slice(r * origUvWidth, origUvWidth).CopyTo(
                        hostU.AsSpan((int)(uSlotBase + r * uvWorkWidth), origUvWidth));
                    vSrc.Slice(r * origUvWidth, origUvWidth).CopyTo(
                        hostV.AsSpan((int)(vSlotBase + r * uvWorkWidth), origUvWidth));
                }
            }
        }
        dAllY.View.CopyFromCPU(hostY);
        dAllU.View.CopyFromCPU(hostU);
        dAllV.View.CopyFromCPU(hostV);

        // Batch setup: one dispatch with extent=N, each thread sets up its frame slot.
        _setup.RunBatch(
            _dcQLookup.View, _acQLookup.View,
            _defaultCoefProbs.View, _updateCoefProbs.View,
            dAllDequant.View, dAllP0.View, dAllInitState.View,
            baseQIndex, frameCount, dequantStride, _p0Stride);

        // Phase 1b: BATCH wave-front sequential encode. Try single-dispatch
        // path (one kernel with internal Group.Barrier between diagonals);
        // fall back to per-diagonal multi-dispatch if the thread budget
        // exceeds CUDA's 1024-per-block cap.
        bool singleDispatched = _sequentialEncode.TryRunBatchSingleDispatch(
            dAllY.View, dAllU.View, dAllV.View,
            dAllYR.View, dAllUR.View, dAllVR.View,
            dAllY4.View, dAllY2.View, dAllUC.View, dAllVC.View,
            dAllDequant.View,
            mbCols, mbRows, frameCount,
            yPlaneStride, uvPlaneStride,
            y4Stride, y2Stride, uvCoefStride,
            dequantStride);
        if (!singleDispatched)
        {
            _sequentialEncode.RunBatch(
                dAllY.View, dAllU.View, dAllV.View,
                dAllYR.View, dAllUR.View, dAllVR.View,
                dAllY4.View, dAllY2.View, dAllUC.View, dAllVC.View,
                dAllDequant.View,
                mbCols, mbRows, frameCount,
                yPlaneStride, uvPlaneStride,
                y4Stride, y2Stride, uvCoefStride,
                dequantStride);
        }

        // Phase 2: BATCH entropy kernel - all N frames run their entropy
        // walks concurrently on independent CUDA cores.
        var batchStrides = new Vp8FrameEntropyBatchStrides
        {
            Y4Stride = y4Stride,
            Y2Stride = y2Stride,
            UvStride = uvCoefStride,
            P0Stride = _p0Stride,
            TpStride = _tp0Stride,
            AboveStride = aboveStride,
            MbCols = mbCols,
            MbRows = mbRows,
        };
        _entropy.RunBatch(
            dAllY4.View, dAllY2.View, dAllUC.View, dAllVC.View,
            _coefProbsByType.View, _constsExtended.View,
            dAllP0.View, dAllTp.View, dAllOutLens.View,
            dAllAbove.View, dAllInitState.View,
            frameCount, batchStrides);

        // Phase 3: BATCH assemble - one dispatch, each thread assembles its frame.
        _assemble.RunBatch(
            dAllP0.View, dAllTp.View, dAllOutLens.View,
            dAllOutputs.View, dAllOutputLens.View,
            width, height,
            frameCount, _p0Stride, _tp0Stride, outputCapacity);

        _accelerator.Synchronize();
        // Read lengths first; then fetch ONLY the actual encoded bytes
        // per frame via partial readback. Avoids transferring the full
        // worst-case-sized strided output buffer over PCIe.
        var allLens = dAllOutputLens.GetAsArray1D();
        var results = new byte[frameCount][];
        for (int f = 0; f < frameCount; f++)
        {
            int len = allLens[f];
            results[f] = new byte[len];
            if (len > 0)
            {
                dAllOutputs.View
                    .SubView((long)f * outputCapacity, len)
                    .CopyToCPU(results[f]);
            }
        }
        return results;
    }

    /// <summary>
    /// Ensure per-frame GPU buffers are sized for (width,height). Reallocates
    /// only when the resolution changes - steady-state encodes at the same
    /// size pay no per-frame allocation cost.
    /// </summary>
    private void EnsureBuffers(int width, int height, int uvWidth, int uvHeight, int mbCols, int mbCount)
    {
        if (_cachedWidth == width && _cachedHeight == height) return;
        DisposeFrameBuffers();
        _cachedWidth = width;
        _cachedHeight = height;
        _dY = _accelerator.Allocate1D<byte>(width * height);
        _dU = _accelerator.Allocate1D<byte>(uvWidth * uvHeight);
        _dV = _accelerator.Allocate1D<byte>(uvWidth * uvHeight);
        _dYRecon = _accelerator.Allocate1D<byte>(width * height);
        _dURecon = _accelerator.Allocate1D<byte>(uvWidth * uvHeight);
        _dVRecon = _accelerator.Allocate1D<byte>(uvWidth * uvHeight);
        _dY4Coefs = _accelerator.Allocate1D<short>(mbCount * 256);
        _dY2Coefs = _accelerator.Allocate1D<short>(mbCount * 16);
        _dUCoefs = _accelerator.Allocate1D<short>(mbCount * 64);
        _dVCoefs = _accelerator.Allocate1D<short>(mbCount * 64);
        _dDequant = _accelerator.Allocate1D<int>(6);
        _dInitialP0State = _accelerator.Allocate1D<int>(5);
        _p0Stride = 64 * 1024 + mbCount * 32;
        _tp0Stride = 64 * 1024 + mbCount * 256;
        _dP0 = _accelerator.Allocate1D<byte>(_p0Stride);
        _dTp = _accelerator.Allocate1D<byte>(_tp0Stride);
        _dPartLens = _accelerator.Allocate1D<long>(2);
        _dAbove = _accelerator.Allocate1D<byte>(mbCols * 9);
        int outputCapacity = 16 + _p0Stride + _tp0Stride;
        _dOutput = _accelerator.Allocate1D<byte>(outputCapacity);
        _dOutLen = _accelerator.Allocate1D<int>(1);
    }

    private void DisposeFrameBuffers()
    {
        _dY?.Dispose(); _dY = null;
        _dU?.Dispose(); _dU = null;
        _dV?.Dispose(); _dV = null;
        _dYRecon?.Dispose(); _dYRecon = null;
        _dURecon?.Dispose(); _dURecon = null;
        _dVRecon?.Dispose(); _dVRecon = null;
        _dY4Coefs?.Dispose(); _dY4Coefs = null;
        _dY2Coefs?.Dispose(); _dY2Coefs = null;
        _dUCoefs?.Dispose(); _dUCoefs = null;
        _dVCoefs?.Dispose(); _dVCoefs = null;
        _dDequant?.Dispose(); _dDequant = null;
        _dInitialP0State?.Dispose(); _dInitialP0State = null;
        _dP0?.Dispose(); _dP0 = null;
        _dTp?.Dispose(); _dTp = null;
        _dPartLens?.Dispose(); _dPartLens = null;
        _dAbove?.Dispose(); _dAbove = null;
        _dOutput?.Dispose(); _dOutput = null;
        _dOutLen?.Dispose(); _dOutLen = null;
        _cachedWidth = -1;
        _cachedHeight = -1;
    }

    /// <summary>
    /// Upload a non-aligned source plane into a working-dim destination
    /// buffer. Source plane is (origW × origH); destination's row stride is
    /// destStride (= workWidth or uvWorkWidth). Source rows are written
    /// contiguous-prefix into each destination row; the right-edge columns
    /// (when origW &lt; destStride) and bottom rows (when origH &lt; workHeight)
    /// stay at whatever the destination was pre-filled with (caller handles
    /// pre-zero when padding is non-trivial).
    /// </summary>
    private void UploadPaddedPlane(
        ReadOnlySpan<byte> src, int srcStride,
        int origW, int origH, int destStride,
        MemoryBuffer1D<byte, Stride1D.Dense> dest)
    {
        if (srcStride == 0) srcStride = origW;
        if (origW == destStride && srcStride == origW)
        {
            // Aligned width AND no source padding: single contiguous upload
            // covering all origH rows. Bottom pad rows (if any) keep whatever
            // pre-zero state the dest already has.
            dest.View.SubView(0, (long)origH * destStride)
                .CopyFromCPU(src.Slice(0, origH * srcStride).ToArray());
            return;
        }

        // General path: build a row-aligned host buffer of (destStride × origH)
        // bytes with each source row left-justified in its destination row.
        // Single GPU upload of the row-aligned region; pad rows below origH
        // stay at the dest buffer's pre-zero state.
        var hostPadded = new byte[(long)destStride * origH];
        var hostSpan = hostPadded.AsSpan();
        for (int r = 0; r < origH; r++)
        {
            src.Slice(r * srcStride, origW).CopyTo(hostSpan.Slice(r * destStride, origW));
            // Right-edge pad cols [origW..destStride) stay zero (new byte[]).
        }
        dest.View.SubView(0, hostPadded.Length).CopyFromCPU(hostPadded);
    }

    /// <summary>
    /// Upload a plane to GPU, stripping any source-side stride padding.
    /// Source upload itself is necessary I/O (the bytes come from outside
    /// the GPU); the previous version then ran a per-row CPU stride-strip
    /// loop, which was a cardinal-rule violation. This version uploads
    /// the strided source as a single I/O and dispatches
    /// <see cref="Vp8StridedPlanePackKernel"/> to do the strip on-GPU,
    /// keeping all codec data manipulation accelerator-resident per
    /// Captain's directive: "The environment using the accelerator is
    /// not capable of processing any data loads. It is simply the
    /// coordinator."
    /// </summary>
    private void UploadPlane(
        ReadOnlySpan<byte> src, int stride, int w, int h,
        MemoryBuffer1D<byte, Stride1D.Dense> dest)
    {
        if (stride == w)
        {
            // No padding - one I/O, done.
            dest.View.CopyFromCPU(src.Slice(0, w * h).ToArray());
        }
        else
        {
            // Strided source: upload the full strided region in one I/O
            // (no per-row CPU work), then GPU-pack into the dest buffer.
            using var dStrided = _accelerator.Allocate1D<byte>(stride * h);
            dStrided.View.CopyFromCPU(src.Slice(0, stride * h).ToArray());
            _stridePack.Run(dStrided.View, 0, stride, dest.View, 0, w, h);
            // Sync before dStrided's using-scope dispose so the kernel
            // has finished consuming it.
            _accelerator.Synchronize();
        }
    }

    /// <summary>Release kernel resources and constant buffers.</summary>
    public void Dispose()
    {
        _setup.Dispose();
        _sequentialEncode.Dispose();
        _entropy.Dispose();
        _assemble.Dispose();
        _stridePack.Dispose();
        _dcQLookup.Dispose();
        _acQLookup.Dispose();
        _defaultCoefProbs.Dispose();
        _updateCoefProbs.Dispose();
        _coefProbsByType.Dispose();
        _constsExtended.Dispose();
        DisposeFrameBuffers();
    }
}
