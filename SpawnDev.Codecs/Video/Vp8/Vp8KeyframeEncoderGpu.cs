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

    /// <summary>Encode a single VP8 keyframe from YUV420 source.</summary>
    public byte[] EncodeKeyFrame(
        ReadOnlySpan<byte> ySrc, int ySrcStride,
        ReadOnlySpan<byte> uSrc, int uvSrcStride,
        ReadOnlySpan<byte> vSrc,
        int width, int height,
        int baseQIndex = 30)
    {
        if (width <= 0 || height <= 0) throw new ArgumentOutOfRangeException();
        if ((width & 15) != 0 || (height & 15) != 0)
            throw new ArgumentException("Width and height must be multiples of 16 (v1)");
        if (baseQIndex < 0 || baseQIndex > 127)
            throw new ArgumentOutOfRangeException(nameof(baseQIndex));

        int mbCols = width / 16;
        int mbRows = height / 16;
        int mbCount = mbCols * mbRows;
        int uvWidth = width / 2;
        int uvHeight = height / 2;

        EnsureBuffers(width, height, uvWidth, uvHeight, mbCols, mbCount);

        // === Host work: ONLY upload + dispatch ===
        UploadPlane(ySrc, ySrcStride, width, height, _dY!);
        UploadPlane(uSrc, uvSrcStride, uvWidth, uvHeight, _dU!);
        UploadPlane(vSrc, uvSrcStride, uvWidth, uvHeight, _dV!);
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
        if ((width & 15) != 0 || (height & 15) != 0)
            throw new ArgumentException("Width and height must be multiples of 16 (v1)");

        int mbCols = width / 16;
        int mbRows = height / 16;
        int mbCount = mbCols * mbRows;
        int uvWidth = width / 2;
        int uvHeight = height / 2;

        EnsureBuffers(width, height, uvWidth, uvHeight, mbCols, mbCount);

        // Per-frame slot strides for batch entropy + sequential encode.
        int yPlaneStride = width * height;
        int uvPlaneStride = uvWidth * uvHeight;
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

        // Phase 1a: bulk upload all frames' planes in 3 host->device transfers
        // (Y batch, U batch, V batch) instead of 3*frameCount small uploads.
        // PCIe4 memcpy is memory-bound; minimizing the number of CopyFromCPU
        // calls collapses driver overhead.
        var hostY = new byte[(long)frameCount * yPlaneStride];
        var hostU = new byte[(long)frameCount * uvPlaneStride];
        var hostV = new byte[(long)frameCount * uvPlaneStride];
        for (int f = 0; f < frameCount; f++)
        {
            yPlanes[f].Span.CopyTo(hostY.AsSpan((int)((long)f * yPlaneStride)));
            uPlanes[f].Span.CopyTo(hostU.AsSpan((int)((long)f * uvPlaneStride)));
            vPlanes[f].Span.CopyTo(hostV.AsSpan((int)((long)f * uvPlaneStride)));
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
        var allLens = dAllOutputLens.GetAsArray1D();
        var allBytes = dAllOutputs.GetAsArray1D();
        var results = new byte[frameCount][];
        for (int f = 0; f < frameCount; f++)
        {
            int len = allLens[f];
            results[f] = new byte[len];
            Array.Copy(allBytes, (long)f * outputCapacity, results[f], 0, len);
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
