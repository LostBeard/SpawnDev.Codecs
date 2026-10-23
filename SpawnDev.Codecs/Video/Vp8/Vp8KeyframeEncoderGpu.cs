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
// API shape (the template for every codec in this library): three overloads per entry point,
// all funnelling into the ArrayView1D one.
//   - ArrayView1D : planes already on the accelerator in, encoded bytes written to a caller
//                   buffer, returns only the length (a scalar). No bulk data crosses the host.
//   - TypedArray  : browser. JS planes go to the GPU via IBrowserMemoryBuffer.CopyFromJS and the
//                   result comes back as a JS Uint8Array. The bytes never enter the .NET heap.
//   - ReadOnlyMemory<byte> : managed. Desktop and small inputs; a terminal CPU sink/source.
//
// Pipeline:
//   [host] upload Y/U/V to GPU (as-is, strided; no host row loops)
//   [host] dispatch PlanePadKernel x3
//          - strided source -> macroblock-aligned working plane, zero pad
//   [host] dispatch Vp8FrameSetupKernel
//          - computes 6 dequantizers from baseQIndex
//          - writes frame header bits to partition0Out
//          - saves bool encoder state to initialP0State
//   [host] dispatch Vp8FrameSequentialEncodeKernel (one launch per wave-front diagonal, or the
//          Group.Barrier form in launches of DiagonalsPerLaunch diagonals)
//          - per-MB predict + transform + quant + recon
//          - reads dequantizers from buffer, writes recon back
//   [host] dispatch Vp8FrameEntropyKernel (resumable launches of MbsPerLaunch macroblocks)
//          - resumes partition0 from initialP0State
//          - writes per-MB modes + coef tokens
//   [host] dispatch Vp8FrameAssembleKernel (one thread per output byte)
//          - writes 10-byte tag + start code + size code
//          - concatenates partition0 + tokenP0 into the caller's output
//          - writes final length to outLen[0]
//   [host] read the 4-byte length
//
// v1 simplifications (matches Vp8KeyframeEncoder reference):
//   - All MBs use Y_PRED = DC_PRED, UV_PRED = DC_PRED.
//   - No segmentation.
//   - Single token partition (Log2NumPartitions = 0).
//   - Loop filter disabled.
//   - mb_no_skip_coeff disabled.
//   - Default coef probs (no per-frame updates).

using System.Runtime.Versioning;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>
/// VP8 keyframe encoder. 100% GPU-resident: host is pure coordinator.
/// Output byte-identical to the CPU <c>Vp8KeyframeEncoder</c> reference.
/// </summary>
/// <remarks>
/// Not thread-safe: one in-flight encode per instance (the per-resolution working buffers and the
/// staging buffers are shared across calls).
/// </remarks>
public sealed class Vp8KeyframeEncoderGpu : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Vp8FrameSetupKernel _setup;
    private readonly Vp8FrameSequentialEncodeKernel _sequentialEncode;
    private readonly Vp8FrameEntropyKernel _entropy;
    private readonly Vp8FrameAssembleKernel _assemble;
    private readonly PlanePadKernel _pad;

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
    private MemoryBuffer1D<int, Stride1D.Dense>? _dOutLen;
    private int _p0Stride;
    private int _tp0Stride;

    // Grow-only staging for the managed and TypedArray overloads (the ArrayView1D overload needs
    // none: its caller already owns the device buffers). Reallocated only at the start of a call,
    // when every earlier call has already awaited SynchronizeAsync.
    private readonly StageBuffer<byte> _stageY;
    private readonly StageBuffer<byte> _stageU;
    private readonly StageBuffer<byte> _stageV;
    private readonly StageBuffer<byte> _stageOut;
    private readonly StageBuffer<int> _stageLens;

    /// <summary>Compile + cache all kernels and constants onto <paramref name="accelerator"/>.</summary>
    public Vp8KeyframeEncoderGpu(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _setup = new Vp8FrameSetupKernel(accelerator);
        _sequentialEncode = new Vp8FrameSequentialEncodeKernel(accelerator);
        _entropy = new Vp8FrameEntropyKernel(accelerator);
        _assemble = new Vp8FrameAssembleKernel(accelerator);
        _pad = new PlanePadKernel(accelerator);
        _stageY = new StageBuffer<byte>(accelerator);
        _stageU = new StageBuffer<byte>(accelerator);
        _stageV = new StageBuffer<byte>(accelerator);
        _stageOut = new StageBuffer<byte>(accelerator);
        _stageLens = new StageBuffer<int>(accelerator);

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

    // ------------------------------------------------------------------
    // Geometry shared by every overload
    // ------------------------------------------------------------------

    /// <summary>Chroma plane dimension for a luma dimension (4:2:0, rounds up).</summary>
    private static int ChromaDim(int lumaDim) => (lumaDim + 1) / 2;

    private static void WorkingDims(int width, int height, out int workWidth, out int workHeight)
    {
        // Round up to the next 16-multiple. The kernels process the full working-dim MB grid
        // bit-exact; the frame tag emits the original (width, height) so the decoder crops the pad.
        workWidth = (width + 15) & ~15;
        workHeight = (height + 15) & ~15;
    }

    // Partition capacities. Partition 0 holds 4 mode bools per macroblock after the ~1100-bool header. The token
    // partition of white noise at quantizer 0 measured ~890 bytes per macroblock on the CPU reference (the old
    // 256 bytes per macroblock overflowed for detailed content at low quantizers, e.g. 320x180 noise at q 4);
    // a stream that still outgrows its buffer is reported (length -1, an exception from the managed overloads)
    // instead of being written past the end.
    private static int P0Capacity(int mbCount) => 64 * 1024 + mbCount * 32;
    private static int TokenCapacity(int mbCount) => 64 * 1024 + mbCount * 1024;

    /// <summary>
    /// Size in bytes of the output buffer one encoded keyframe needs for a
    /// <paramref name="width"/> x <paramref name="height"/> picture. The ArrayView1D overloads
    /// require an output view at least this long (and the batch overload strides its slots by it).
    /// </summary>
    public static int GetMaxEncodedSize(int width, int height)
    {
        if (width <= 0 || height <= 0) throw new ArgumentOutOfRangeException();
        WorkingDims(width, height, out int workWidth, out int workHeight);
        int mbCount = (workWidth / 16) * (workHeight / 16);
        return 16 + P0Capacity(mbCount) + TokenCapacity(mbCount);
    }

    // ------------------------------------------------------------------
    // ArrayView1D overloads: the implementation
    // ------------------------------------------------------------------

    /// <summary>
    /// Encode a single VP8 keyframe from YUV420 planes that are already on the accelerator.
    /// Accepts any positive (width, height); internally pads to the next 16-multiple working
    /// dimensions. Nothing but the 4-byte result length crosses to the host.
    /// </summary>
    /// <param name="ySrc">Luma plane.</param>
    /// <param name="ySrcStride">Bytes per luma row (0 = width).</param>
    /// <param name="uSrc">Cb plane, ((width+1)/2) x ((height+1)/2).</param>
    /// <param name="uvSrcStride">Bytes per chroma row for both chroma planes (0 = tight).</param>
    /// <param name="vSrc">Cr plane, same layout as <paramref name="uSrc"/>.</param>
    /// <param name="width">Picture width in pixels.</param>
    /// <param name="height">Picture height in pixels.</param>
    /// <param name="output">Receives the encoded keyframe; at least <see cref="GetMaxEncodedSize"/> bytes.</param>
    /// <param name="baseQIndex">VP8 base quantizer index, 0..127.</param>
    /// <returns>The number of bytes written to <paramref name="output"/>.</returns>
    public async Task<int> EncodeKeyFrameAsync(
        ArrayView1D<byte, Stride1D.Dense> ySrc, int ySrcStride,
        ArrayView1D<byte, Stride1D.Dense> uSrc, int uvSrcStride,
        ArrayView1D<byte, Stride1D.Dense> vSrc,
        int width, int height,
        ArrayView1D<byte, Stride1D.Dense> output,
        int baseQIndex = 30)
    {
        if (width <= 0 || height <= 0) throw new ArgumentOutOfRangeException();
        if (baseQIndex < 0 || baseQIndex > 127)
            throw new ArgumentOutOfRangeException(nameof(baseQIndex));
        if (output.Length < GetMaxEncodedSize(width, height))
            throw new ArgumentException(
                $"output must be at least GetMaxEncodedSize({width}, {height}) = {GetMaxEncodedSize(width, height)} bytes.",
                nameof(output));

        WorkingDims(width, height, out int workWidth, out int workHeight);
        int mbCols = workWidth / 16;
        int mbRows = workHeight / 16;
        int mbCount = mbCols * mbRows;
        int uvWorkWidth = workWidth / 2;
        int uvWorkHeight = workHeight / 2;
        int uvWidth = ChromaDim(width);
        int uvHeight = ChromaDim(height);
        if (ySrcStride == 0) ySrcStride = width;
        if (uvSrcStride == 0) uvSrcStride = uvWidth;

        EnsureBuffers(workWidth, workHeight, uvWorkWidth, uvWorkHeight, mbCols, mbCount);

        // === Host work: ONLY dispatch ===
        // Strided source -> working plane with zero pad, on the accelerator. Every working byte is
        // written, so there is no pre-clear.
        _pad.Run(ySrc, 0, ySrcStride, width, height, _dY!.View, workWidth, workHeight);
        _pad.Run(uSrc, 0, uvSrcStride, uvWidth, uvHeight, _dU!.View, uvWorkWidth, uvWorkHeight);
        _pad.Run(vSrc, 0, uvSrcStride, uvWidth, uvHeight, _dV!.View, uvWorkWidth, uvWorkHeight);
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
        await _sequentialEncode.RunAsync(
            _dY!.View, _dU!.View, _dV!.View,
            _dYRecon!.View, _dURecon!.View, _dVRecon!.View,
            _dY4Coefs!.View, _dY2Coefs!.View, _dUCoefs!.View, _dVCoefs!.View,
            _dDequant!.View,
            mbCols, mbRows);

        // 3. Entropy: continues partition0 from setup state, writes
        // per-MB modes + coefs.
        await _entropy.RunAsync(
            _dY4Coefs!.View, _dY2Coefs!.View, _dUCoefs!.View, _dVCoefs!.View,
            _coefProbsByType.View, _constsExtended.View,
            _dP0!.View, _dTp!.View, _dPartLens!.View,
            _dAbove!.View, _dInitialP0State!.View,
            mbCols, mbRows);

        // 4. Assemble: frame tag + start code + size code +
        // partition0 + tokenP0 -> the caller's output; writes final
        // length to outLen[0].
        _assemble.Run(
            _dP0!.View, _dTp!.View, _dPartLens!.View,
            output, _dOutLen!.View,
            width, height);

        await _accelerator.SynchronizeAsync();

        // The only readback: the 4-byte length scalar.
        return CheckLength((await _dOutLen!.CopyToHostAsync())[0]);
    }

    /// <summary>
    /// True frame-batch parallel encode of frames that are already on the accelerator: per-frame
    /// buffer slots through every kernel + extent=N batch entropy. Setup + sequential-encode +
    /// assemble run per-frame against per-frame slots (still serial on the stream but with no
    /// buffer aliasing); the batch entropy kernel dispatches at extent=numFrames so all frames run
    /// their entropy walks concurrently on independent CUDA cores. Only the per-frame lengths
    /// (4 bytes each) are written to <paramref name="lengths"/>; the host reads nothing here.
    /// </summary>
    /// <param name="ySrc">Frame-major tightly packed luma planes: frame f at <c>f * width * height</c>.</param>
    /// <param name="uSrc">Frame-major tightly packed Cb planes: frame f at <c>f * uvWidth * uvHeight</c>
    /// with uvWidth = (width+1)/2, uvHeight = (height+1)/2.</param>
    /// <param name="vSrc">Frame-major tightly packed Cr planes, same layout as <paramref name="uSrc"/>.</param>
    /// <param name="outputs">Frame f's bitstream is written at <c>f * GetMaxEncodedSize(width, height)</c>;
    /// at least <c>frameCount * GetMaxEncodedSize</c> bytes.</param>
    /// <param name="lengths">Receives each frame's encoded length (-1 if a partition outgrew its buffer); at least <paramref name="frameCount"/> ints.</param>
    /// <param name="frameCount">Number of frames.</param>
    /// <param name="width">Picture width in pixels.</param>
    /// <param name="height">Picture height in pixels.</param>
    /// <param name="baseQIndex">VP8 base quantizer index, 0..127.</param>
    public async Task EncodeKeyFramesBatchAsync(
        ArrayView1D<byte, Stride1D.Dense> ySrc,
        ArrayView1D<byte, Stride1D.Dense> uSrc,
        ArrayView1D<byte, Stride1D.Dense> vSrc,
        int frameCount, int width, int height,
        ArrayView1D<byte, Stride1D.Dense> outputs,
        ArrayView1D<int, Stride1D.Dense> lengths,
        int baseQIndex = 30)
    {
        if (frameCount <= 0) throw new ArgumentOutOfRangeException(nameof(frameCount));
        if (width <= 0 || height <= 0) throw new ArgumentOutOfRangeException();
        if (baseQIndex < 0 || baseQIndex > 127)
            throw new ArgumentOutOfRangeException(nameof(baseQIndex));

        WorkingDims(width, height, out int workWidth, out int workHeight);
        int mbCols = workWidth / 16;
        int mbRows = workHeight / 16;
        int mbCount = mbCols * mbRows;
        int uvWorkWidth = workWidth / 2;
        int uvWorkHeight = workHeight / 2;
        int uvWidth = ChromaDim(width);
        int uvHeight = ChromaDim(height);

        int outputCapacity = GetMaxEncodedSize(width, height);
        if (outputs.Length < (long)frameCount * outputCapacity)
            throw new ArgumentException("outputs must hold frameCount * GetMaxEncodedSize bytes.", nameof(outputs));
        if (lengths.Length < frameCount)
            throw new ArgumentException("lengths must hold frameCount ints.", nameof(lengths));
        long ySrcPlane = (long)width * height;
        long uvSrcPlane = (long)uvWidth * uvHeight;
        if (ySrc.Length < frameCount * ySrcPlane)
            throw new ArgumentException("ySrc must hold frameCount tightly packed luma planes.", nameof(ySrc));
        if (uSrc.Length < frameCount * uvSrcPlane)
            throw new ArgumentException("uSrc must hold frameCount tightly packed Cb planes.", nameof(uSrc));
        if (vSrc.Length < frameCount * uvSrcPlane)
            throw new ArgumentException("vSrc must hold frameCount tightly packed Cr planes.", nameof(vSrc));

        // The batch path allocates its own slots; it shares only the per-resolution stride
        // constants with the single-frame path.
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

        dAllP0.View.MemSetToZero();
        dAllTp.View.MemSetToZero();
        dAllAbove.View.MemSetToZero();

        // Phase 1a: pad every frame's planes into its working-dim slot on the accelerator. Every
        // working byte of every slot is written (zero pad included), so there is no host staging
        // array and no host row loop.
        var padPacer = new LaunchPacer(_accelerator);
        for (int f = 0; f < frameCount; f++)
        {
            _pad.Run(ySrc, f * ySrcPlane, width, width, height,
                dAllY.View.SubView((long)f * yPlaneStride, yPlaneStride), workWidth, workHeight);
            await padPacer.LaunchedAsync();
            _pad.Run(uSrc, f * uvSrcPlane, uvWidth, uvWidth, uvHeight,
                dAllU.View.SubView((long)f * uvPlaneStride, uvPlaneStride), uvWorkWidth, uvWorkHeight);
            await padPacer.LaunchedAsync();
            _pad.Run(vSrc, f * uvSrcPlane, uvWidth, uvWidth, uvHeight,
                dAllV.View.SubView((long)f * uvPlaneStride, uvPlaneStride), uvWorkWidth, uvWorkHeight);
            await padPacer.LaunchedAsync();
        }

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
        bool singleDispatched = await _sequentialEncode.TryRunBatchSingleDispatchAsync(
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
            await _sequentialEncode.RunBatchAsync(
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
        await _entropy.RunBatchAsync(
            dAllY4.View, dAllY2.View, dAllUC.View, dAllVC.View,
            _coefProbsByType.View, _constsExtended.View,
            dAllP0.View, dAllTp.View, dAllOutLens.View,
            dAllAbove.View, dAllInitState.View,
            frameCount, batchStrides);

        // Phase 3: BATCH assemble - one dispatch, each thread assembles its frame straight into
        // the caller's outputs/lengths.
        _assemble.RunBatch(
            dAllP0.View, dAllTp.View, dAllOutLens.View,
            outputs, lengths,
            width, height,
            frameCount, _p0Stride, _tp0Stride, outputCapacity);

        // Drain before the using-scoped slot buffers dispose: on the browser backends a dispatch
        // that still references a freed buffer reads freed memory.
        await _accelerator.SynchronizeAsync();
    }

    // ------------------------------------------------------------------
    // Managed overloads
    // ------------------------------------------------------------------

    /// <summary>
    /// Encode a single VP8 keyframe from managed YUV420 planes. Accepts any positive
    /// (width, height); the planes are uploaded as-is (strides honoured) and padded on the
    /// accelerator. Returns the encoded keyframe as a managed array (a terminal CPU sink; in a
    /// browser prefer the <see cref="TypedArray"/> overload).
    /// </summary>
    public async Task<byte[]> EncodeKeyFrameAsync(
        ReadOnlyMemory<byte> ySrc, int ySrcStride,
        ReadOnlyMemory<byte> uSrc, int uvSrcStride,
        ReadOnlyMemory<byte> vSrc,
        int width, int height,
        int baseQIndex = 30)
    {
        if (width <= 0 || height <= 0) throw new ArgumentOutOfRangeException();
        int uvWidth = ChromaDim(width);
        int uvHeight = ChromaDim(height);
        if (ySrcStride == 0) ySrcStride = width;
        if (uvSrcStride == 0) uvSrcStride = uvWidth;

        DeviceTransfer.UploadPlane(_stageY, ySrc, ySrcStride, width, height, nameof(ySrc));
        DeviceTransfer.UploadPlane(_stageU, uSrc, uvSrcStride, uvWidth, uvHeight, nameof(uSrc));
        DeviceTransfer.UploadPlane(_stageV, vSrc, uvSrcStride, uvWidth, uvHeight, nameof(vSrc));
        _stageOut.Ensure(GetMaxEncodedSize(width, height));

        int len = await EncodeKeyFrameAsync(
            _stageY.View, ySrcStride, _stageU.View, uvSrcStride, _stageV.View,
            width, height, _stageOut.View, baseQIndex);
        return await DeviceTransfer.DownloadAsync(_stageOut.View, len);
    }

    /// <summary>
    /// Batch-encode managed keyframes. Every plane is tightly packed (no stride): luma
    /// <c>width * height</c>, chroma <c>((width+1)/2) * ((height+1)/2)</c>. Each plane is uploaded
    /// straight into the device staging buffer (no host-side concatenation).
    /// </summary>
    public async Task<byte[][]> EncodeKeyFramesBatchAsync(
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
        if (frameCount == 0) return System.Array.Empty<byte[]>();
        if (width <= 0 || height <= 0) throw new ArgumentOutOfRangeException();

        long yPlane = (long)width * height;
        long uvPlane = (long)ChromaDim(width) * ChromaDim(height);
        int cap = GetMaxEncodedSize(width, height);
        _stageY.Ensure(frameCount * yPlane);
        _stageU.Ensure(frameCount * uvPlane);
        _stageV.Ensure(frameCount * uvPlane);
        _stageOut.Ensure((long)frameCount * cap);
        _stageLens.Ensure(frameCount);

        for (int f = 0; f < frameCount; f++)
        {
            DeviceTransfer.UploadPlaneSlot(_stageY, f * yPlane, yPlane, yPlanes[f], nameof(yPlanes));
            DeviceTransfer.UploadPlaneSlot(_stageU, f * uvPlane, uvPlane, uPlanes[f], nameof(uPlanes));
            DeviceTransfer.UploadPlaneSlot(_stageV, f * uvPlane, uvPlane, vPlanes[f], nameof(vPlanes));
        }

        await EncodeKeyFramesBatchAsync(
            _stageY.View, _stageU.View, _stageV.View,
            frameCount, width, height, _stageOut.View, _stageLens.View, baseQIndex);

        // Lengths first (4 bytes each); then ONLY the actual encoded bytes per frame via partial
        // readback, never the worst-case-sized slot.
        var allLens = await _stageLens.View.SubView(0, frameCount).CopyToHostAsync();
        var results = new byte[frameCount][];
        for (int f = 0; f < frameCount; f++)
        {
            int len = CheckLength(allLens[f]);
            results[f] = len > 0
                ? await DeviceTransfer.DownloadAsync(_stageOut.View.SubView((long)f * cap, len), len)
                : System.Array.Empty<byte>();
        }
        return results;
    }

    // ------------------------------------------------------------------
    // TypedArray overloads (browser: bulk bytes stay in JS)
    // ------------------------------------------------------------------

    /// <summary>
    /// Encode a single VP8 keyframe from JS typed arrays (browser accelerators only). The planes
    /// go to the GPU with <c>CopyFromJS</c> and the keyframe comes back as a JS
    /// <see cref="Uint8Array"/> (the caller disposes it); no bulk byte enters the .NET heap.
    /// </summary>
    /// <exception cref="NotSupportedException">The accelerator is not a browser accelerator.</exception>
    [SupportedOSPlatform("browser")]
    public async Task<Uint8Array> EncodeKeyFrameAsync(
        TypedArray ySrc, int ySrcStride,
        TypedArray uSrc, int uvSrcStride,
        TypedArray vSrc,
        int width, int height,
        int baseQIndex = 30)
    {
        ArgumentNullException.ThrowIfNull(ySrc);
        ArgumentNullException.ThrowIfNull(uSrc);
        ArgumentNullException.ThrowIfNull(vSrc);
        if (width <= 0 || height <= 0) throw new ArgumentOutOfRangeException();

        _stageY.Ensure(ySrc.ByteLength);
        _stageU.Ensure(uSrc.ByteLength);
        _stageV.Ensure(vSrc.ByteLength);
        _stageOut.Ensure(GetMaxEncodedSize(width, height));
        DeviceTransfer.Upload(_stageY.View, ySrc);
        DeviceTransfer.Upload(_stageU.View, uSrc);
        DeviceTransfer.Upload(_stageV.View, vSrc);

        int len = await EncodeKeyFrameAsync(
            _stageY.View, ySrcStride, _stageU.View, uvSrcStride, _stageV.View,
            width, height, _stageOut.View, baseQIndex);
        return await DeviceTransfer.DownloadUint8ArrayAsync(_stageOut.View, len);
    }

    /// <summary>
    /// Batch-encode keyframes from JS typed arrays (browser accelerators only), one tightly
    /// packed plane per frame per component. Each returned <see cref="Uint8Array"/> is owned (and
    /// disposed) by the caller.
    /// </summary>
    /// <exception cref="NotSupportedException">The accelerator is not a browser accelerator.</exception>
    [SupportedOSPlatform("browser")]
    public async Task<Uint8Array[]> EncodeKeyFramesBatchAsync(
        TypedArray[] yPlanes,
        TypedArray[] uPlanes,
        TypedArray[] vPlanes,
        int width, int height,
        int baseQIndex = 30)
    {
        if (yPlanes is null) throw new ArgumentNullException(nameof(yPlanes));
        if (uPlanes is null) throw new ArgumentNullException(nameof(uPlanes));
        if (vPlanes is null) throw new ArgumentNullException(nameof(vPlanes));
        if (yPlanes.Length != uPlanes.Length || yPlanes.Length != vPlanes.Length)
            throw new ArgumentException("Plane arrays must have equal length.");
        int frameCount = yPlanes.Length;
        if (frameCount == 0) return System.Array.Empty<Uint8Array>();
        if (width <= 0 || height <= 0) throw new ArgumentOutOfRangeException();

        long yPlane = (long)width * height;
        long uvPlane = (long)ChromaDim(width) * ChromaDim(height);
        int cap = GetMaxEncodedSize(width, height);
        _stageY.Ensure(frameCount * yPlane);
        _stageU.Ensure(frameCount * uvPlane);
        _stageV.Ensure(frameCount * uvPlane);
        _stageOut.Ensure((long)frameCount * cap);
        _stageLens.Ensure(frameCount);

        for (int f = 0; f < frameCount; f++)
        {
            DeviceTransfer.Upload(_stageY.View.SubView(f * yPlane, yPlane), yPlanes[f]);
            DeviceTransfer.Upload(_stageU.View.SubView(f * uvPlane, uvPlane), uPlanes[f]);
            DeviceTransfer.Upload(_stageV.View.SubView(f * uvPlane, uvPlane), vPlanes[f]);
        }

        await EncodeKeyFramesBatchAsync(
            _stageY.View, _stageU.View, _stageV.View,
            frameCount, width, height, _stageOut.View, _stageLens.View, baseQIndex);

        var allLens = await _stageLens.View.SubView(0, frameCount).CopyToHostAsync();
        foreach (int l in allLens) CheckLength(l);
        var results = new Uint8Array[frameCount];
        for (int f = 0; f < frameCount; f++)
            results[f] = await DeviceTransfer.DownloadUint8ArrayAsync(
                _stageOut.View.SubView((long)f * cap, allLens[f]), allLens[f]);
        return results;
    }

    /// <summary>
    /// The assemble kernel reports -1 for a frame whose partition outgrew its buffer (the bool encoder never
    /// writes past a buffer; see <see cref="TokenCapacity"/>).
    /// </summary>
    private static int CheckLength(int length)
    {
        if (length < 0)
            throw new InvalidOperationException("VP8 encode: a partition outgrew its buffer; the frame was not assembled.");
        return length;
    }

    // ------------------------------------------------------------------
    // Staging + working buffers
    // ------------------------------------------------------------------

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
        _p0Stride = P0Capacity(mbCount);
        _tp0Stride = TokenCapacity(mbCount);
        _dP0 = _accelerator.Allocate1D<byte>(_p0Stride);
        _dTp = _accelerator.Allocate1D<byte>(_tp0Stride);
        _dPartLens = _accelerator.Allocate1D<long>(2);
        _dAbove = _accelerator.Allocate1D<byte>(mbCols * 9);
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
        _dOutLen?.Dispose(); _dOutLen = null;
        _cachedWidth = -1;
        _cachedHeight = -1;
    }

    /// <summary>Release kernel resources and constant buffers.</summary>
    public void Dispose()
    {
        _setup.Dispose();
        _sequentialEncode.Dispose();
        _entropy.Dispose();
        _assemble.Dispose();
        _pad.Dispose();
        _dcQLookup.Dispose();
        _acQLookup.Dispose();
        _defaultCoefProbs.Dispose();
        _updateCoefProbs.Dispose();
        _coefProbsByType.Dispose();
        _constsExtended.Dispose();
        DisposeFrameBuffers();
        _stageY.Dispose();
        _stageU.Dispose();
        _stageV.Dispose();
        _stageOut.Dispose();
        _stageLens.Dispose();
    }
}
