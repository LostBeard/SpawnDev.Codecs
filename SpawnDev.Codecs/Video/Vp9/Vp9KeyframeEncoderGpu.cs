// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 v1 keyframe encoder, integration class. 100% ILGPU - the host
// is a pure coordinator: alloc GPU buffers, upload YUV source +
// constant tables, dispatch the kernel chain in order, read back
// the final frame bytes. Zero CPU math, zero CPU iteration over
// codec data, zero CPU bool encoding, zero CPU bitstream assembly.
//
// API shape (same as Vp8KeyframeEncoderGpu): three overloads per entry point, all funnelling
// into the ArrayView1D one.
//   - ArrayView1D : planes already on the accelerator in, encoded bytes assembled straight into a
//                   caller-owned output view, returns only the length (a scalar).
//   - TypedArray  : browser. JS planes -> GPU via IBrowserMemoryBuffer.CopyFromJS, result back as a
//                   JS Uint8Array. The bytes never enter the .NET heap.
//   - ReadOnlyMemory<byte> : managed. Desktop and small inputs; a terminal CPU sink/source.
//
// Kernel chain:
//   0. PlanePadKernel                    - tight source plane -> 64-aligned working plane, zero
//                                          pad, on the accelerator (no host row loops).
//   1. Vp9DequantizerComputeKernel       - compute Y/UV dequantizers
//                                          from baseQIndex.
//   2. Vp9FrameCompressedHeaderKernel    - emit tx_mode + no-update
//                                          gates; produces the
//                                          first_partition_size payload.
//   3. Vp9FrameSequentialEncodeKernel    - per-MB predict + FDCT +
//                                          quant + dequant + IDCT +
//                                          recon. Saves quantized
//                                          coefs to per-plane buffers.
//   4. Vp9FrameEntropyKernel             - walks SBs in z-order,
//                                          emits partition + skip +
//                                          mode + coef tokens to the
//                                          tile bool stream.
//   5. Vp9FrameUncompressedHeaderKernel  - emit raw-bit header now
//                                          that we know firstPartitionSize
//                                          (compressed header byte count).
//   6. Vp9FrameAssembleKernel            - concat uncompressed +
//                                          compressed + tile bytes
//                                          into the caller's output buffer.
//
// V1 simplifications (mirror Vp9KeyframeEncoder.EncodeKeyFrame):
//   - Profile 0, YUV 4:2:0
//   - Any positive width + height (padded to the 64-pixel superblock grid on the accelerator).
//     v1 entropy kernel caps at 512-pixel width.
//   - All Y blocks DC_PRED + Tx16x16; all UV blocks DC_PRED + Tx8x8.
//   - tx_mode = Allow32x32, single tile, LF off, segmentation off,
//     default coef probs.
//   - Skip flag hardcoded 0.

using System.Runtime.Versioning;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// VP9 v1 keyframe encoder integration class. Runs the 6-kernel
/// chain on the supplied accelerator and produces the final encoded
/// frame bytes. Host is a pure coordinator.
/// </summary>
/// <remarks>
/// Not thread-safe: one in-flight encode per instance (the per-resolution working buffers and the
/// staging buffers are shared across calls).
/// </remarks>
public sealed class Vp9KeyframeEncoderGpu : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly PlanePadKernel _pad;
    private readonly Vp9DequantizerComputeKernel _dequantKernel;
    private readonly Vp9FrameCompressedHeaderKernel _compressedHeaderKernel;
    private readonly Vp9FrameSequentialEncodeKernel _sequentialKernel;
    private readonly Vp9FrameEntropyKernel _entropyKernel;
    private readonly Vp9FrameUncompressedHeaderKernel _uncompressedHeaderKernel;
    private readonly Vp9FrameAssembleKernel _assembleKernel;

    private readonly MemoryBuffer1D<short, global::ILGPU.Stride1D.Dense> _dDcQLookup;
    private readonly MemoryBuffer1D<short, global::ILGPU.Stride1D.Dense> _dAcQLookup;
    private readonly MemoryBuffer1D<byte, global::ILGPU.Stride1D.Dense> _dByteConsts;
    private readonly MemoryBuffer1D<ushort, global::ILGPU.Stride1D.Dense> _dUshortConsts;

    // Per-resolution cached buffers - reallocated only when (width,height) changes.
    // Steady-state encoding at the same size pays only kernel dispatch + plane
    // upload + readback per frame.
    private int _cachedWidth = -1;
    private int _cachedHeight = -1;
    private MemoryBuffer1D<byte, global::ILGPU.Stride1D.Dense>? _dY;
    private MemoryBuffer1D<byte, global::ILGPU.Stride1D.Dense>? _dU;
    private MemoryBuffer1D<byte, global::ILGPU.Stride1D.Dense>? _dV;
    private MemoryBuffer1D<byte, global::ILGPU.Stride1D.Dense>? _dYRecon;
    private MemoryBuffer1D<byte, global::ILGPU.Stride1D.Dense>? _dURecon;
    private MemoryBuffer1D<byte, global::ILGPU.Stride1D.Dense>? _dVRecon;
    private MemoryBuffer1D<short, global::ILGPU.Stride1D.Dense>? _dYCoefs;
    private MemoryBuffer1D<short, global::ILGPU.Stride1D.Dense>? _dUCoefs;
    private MemoryBuffer1D<short, global::ILGPU.Stride1D.Dense>? _dVCoefs;
    private MemoryBuffer1D<int, global::ILGPU.Stride1D.Dense>? _dDequant;
    private MemoryBuffer1D<byte, global::ILGPU.Stride1D.Dense>? _dCompressedHeader;
    private MemoryBuffer1D<long, global::ILGPU.Stride1D.Dense>? _dCompressedHeaderLen;
    private MemoryBuffer1D<byte, global::ILGPU.Stride1D.Dense>? _dTile;
    private MemoryBuffer1D<long, global::ILGPU.Stride1D.Dense>? _dTileLen;
    private MemoryBuffer1D<byte, global::ILGPU.Stride1D.Dense>? _dUncompressedHeader;
    private MemoryBuffer1D<long, global::ILGPU.Stride1D.Dense>? _dUncompressedHeaderLen;
    private MemoryBuffer1D<long, global::ILGPU.Stride1D.Dense>? _dOutFrameLen;

    // Grow-only staging for the managed and TypedArray overloads.
    private readonly StageBuffer<byte> _stageY;
    private readonly StageBuffer<byte> _stageU;
    private readonly StageBuffer<byte> _stageV;
    private readonly StageBuffer<byte> _stageOut;
    private readonly StageBuffer<long> _stageLens;

    /// <summary>
    /// Compile every kernel + upload one-time-cached constant tables
    /// (DC/AC quantizer lookup + packed entropy / scan / neighbor
    /// constants).
    /// </summary>
    public Vp9KeyframeEncoderGpu(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;

        _pad = new PlanePadKernel(accelerator);
        _dequantKernel = new Vp9DequantizerComputeKernel(accelerator);
        _compressedHeaderKernel = new Vp9FrameCompressedHeaderKernel(accelerator);
        _sequentialKernel = new Vp9FrameSequentialEncodeKernel(accelerator);
        _entropyKernel = new Vp9FrameEntropyKernel(accelerator);
        _uncompressedHeaderKernel = new Vp9FrameUncompressedHeaderKernel(accelerator);
        _assembleKernel = new Vp9FrameAssembleKernel(accelerator);

        _stageY = new StageBuffer<byte>(accelerator);
        _stageU = new StageBuffer<byte>(accelerator);
        _stageV = new StageBuffer<byte>(accelerator);
        _stageOut = new StageBuffer<byte>(accelerator);
        _stageLens = new StageBuffer<long>(accelerator);

        _dDcQLookup = accelerator.Allocate1D<short>(256);
        _dAcQLookup = accelerator.Allocate1D<short>(256);
        _dByteConsts = accelerator.Allocate1D<byte>(Vp9KeyframeConstantsGpu.ByteConstsTotalBytes);
        _dUshortConsts = accelerator.Allocate1D<ushort>(Vp9KeyframeConstantsGpu.UshortConstsTotalEntries);

        _dDcQLookup.View.CopyFromCPU(Vp9DequantizerComputeKernel.BuildDcQLookup());
        _dAcQLookup.View.CopyFromCPU(Vp9DequantizerComputeKernel.BuildAcQLookup());
        _dByteConsts.View.CopyFromCPU(Vp9KeyframeConstantsGpu.BuildByteConstsBuffer());
        _dUshortConsts.View.CopyFromCPU(Vp9KeyframeConstantsGpu.BuildUshortConstsBuffer());
    }

    // ------------------------------------------------------------------
    // Geometry shared by every overload
    // ------------------------------------------------------------------

    /// <summary>Chroma plane dimension for a luma dimension (4:2:0, rounds up).</summary>
    private static int ChromaDim(int lumaDim) => (lumaDim + 1) / 2;

    private static void WorkingDims(int width, int height, out int workWidth, out int workHeight)
    {
        // Round up to the next 64-multiple (superblock grid). The bitstream's frame_width_minus_1 /
        // frame_height_minus_1 fields signal the original (width, height) so decoders crop the pad.
        workWidth = (width + 63) & ~63;
        workHeight = (height + 63) & ~63;
    }

    private static long WorstCaseTile(int mbCount) => mbCount * 1024L + 256L;

    /// <summary>
    /// Size in bytes of the output buffer one encoded keyframe needs for a
    /// <paramref name="width"/> x <paramref name="height"/> picture. The ArrayView1D overloads
    /// require an output view at least this long (and the batch overload strides its slots by it).
    /// </summary>
    public static int GetMaxEncodedSize(int width, int height)
    {
        if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width));
        if (height <= 0) throw new ArgumentOutOfRangeException(nameof(height));
        WorkingDims(width, height, out int workWidth, out int workHeight);
        int mbCount = (workWidth >> 4) * (workHeight >> 4);
        return checked((int)(WorstCaseTile(mbCount) + 128L));
    }

    private static void ValidateArgs(int width, int height, int baseQIndex)
    {
        if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width));
        if (height <= 0) throw new ArgumentOutOfRangeException(nameof(height));
        if (baseQIndex <= 0 || baseQIndex > 255)
            throw new ArgumentOutOfRangeException(nameof(baseQIndex),
                "baseQIndex must be in [1, 255].");
    }

    // ------------------------------------------------------------------
    // ArrayView1D overloads: the implementation
    // ------------------------------------------------------------------

    /// <summary>
    /// Encode a single VP9 v1 keyframe from tightly packed YUV420 planes that are already on the
    /// accelerator. Accepts any positive (width, height); internally pads to the 64-pixel
    /// superblock grid. Nothing but the length scalar crosses to the host.
    /// </summary>
    /// <param name="yPlane">Luma plane, <c>width * height</c> bytes.</param>
    /// <param name="uPlane">Cb plane, <c>((width+1)/2) * ((height+1)/2)</c> bytes.</param>
    /// <param name="vPlane">Cr plane, same layout as <paramref name="uPlane"/>.</param>
    /// <param name="width">Picture width in pixels.</param>
    /// <param name="height">Picture height in pixels.</param>
    /// <param name="output">Receives the encoded frame; at least <see cref="GetMaxEncodedSize"/> bytes.</param>
    /// <param name="baseQIndex">VP9 base quantizer index, 1..255.</param>
    /// <returns>The number of bytes written to <paramref name="output"/>.</returns>
    public async Task<int> EncodeKeyFrameAsync(
        ArrayView1D<byte, Stride1D.Dense> yPlane,
        ArrayView1D<byte, Stride1D.Dense> uPlane,
        ArrayView1D<byte, Stride1D.Dense> vPlane,
        int width, int height,
        ArrayView1D<byte, Stride1D.Dense> output,
        int baseQIndex = 30)
    {
        await EncodeCoreAsync(yPlane, uPlane, vPlane, width, height, output, baseQIndex);
        return (int)(await _dOutFrameLen!.CopyToHostAsync())[0];
    }

    /// <summary>
    /// Runs the kernel chain and drains it. Leaves the encoded length in <c>_dOutFrameLen</c> and
    /// the recon planes in <c>_d*Recon</c> (read by <see cref="EncodeKeyFrameWithReconAsync"/>).
    /// </summary>
    private async Task EncodeCoreAsync(
        ArrayView1D<byte, Stride1D.Dense> yPlane,
        ArrayView1D<byte, Stride1D.Dense> uPlane,
        ArrayView1D<byte, Stride1D.Dense> vPlane,
        int width, int height,
        ArrayView1D<byte, Stride1D.Dense> output,
        int baseQIndex)
    {
        ValidateArgs(width, height, baseQIndex);

        WorkingDims(width, height, out int workWidth, out int workHeight);
        int mbCols = workWidth >> 4;
        int mbRows = workHeight >> 4;
        int yLen = workWidth * workHeight;
        int uvLen = yLen / 4;
        int mbCount = mbCols * mbRows;
        int uvWidth = ChromaDim(width);
        int uvHeight = ChromaDim(height);
        int uvWorkWidth = workWidth / 2;
        int uvWorkHeight = workHeight / 2;

        if (yPlane.Length < (long)width * height) throw new ArgumentException("yPlane too short.", nameof(yPlane));
        if (uPlane.Length < (long)uvWidth * uvHeight) throw new ArgumentException("uPlane too short.", nameof(uPlane));
        if (vPlane.Length < (long)uvWidth * uvHeight) throw new ArgumentException("vPlane too short.", nameof(vPlane));
        if (output.Length < GetMaxEncodedSize(width, height))
            throw new ArgumentException(
                $"output must be at least GetMaxEncodedSize({width}, {height}) = {GetMaxEncodedSize(width, height)} bytes.",
                nameof(output));

        // ---- 1. Ensure per-frame GPU buffers (cached across same-resolution calls) ----
        EnsureBuffers(workWidth, workHeight, yLen, uvLen, mbCount);

        // ---- Pre-zero the bool-encoder carry-back targets. Recon planes and coefs are fully
        // overwritten by the sequential kernel; the assembled output is written by assignment.
        _dCompressedHeader!.View.MemSetToZero();
        _dCompressedHeaderLen!.View.MemSetToZero();
        _dTile!.View.MemSetToZero();
        _dTileLen!.View.MemSetToZero();
        _dUncompressedHeader!.View.MemSetToZero();
        _dUncompressedHeaderLen!.View.MemSetToZero();
        _dOutFrameLen!.View.MemSetToZero();

        // Tight source plane -> 64-aligned working plane with zero pad, on the accelerator. Every
        // working byte is written, so there is no pre-clear and no host row loop.
        _pad.Run(yPlane, 0, width, width, height, _dY!.View, workWidth, workHeight);
        _pad.Run(uPlane, 0, uvWidth, uvWidth, uvHeight, _dU!.View, uvWorkWidth, uvWorkHeight);
        _pad.Run(vPlane, 0, uvWidth, uvWidth, uvHeight, _dV!.View, uvWorkWidth, uvWorkHeight);

        // ---- 2. Dispatch dequantizer compute kernel ----
        // y_dc_delta / uv_dc_delta / uv_ac_delta = 0 in v1.
        _dequantKernel.Run(_dDcQLookup.View, _dAcQLookup.View, _dDequant!.View,
            baseQIndex, 0, 0, 0, 0);

        // ---- 3. Compressed header ----
        _compressedHeaderKernel.Run(_dCompressedHeader!.View, _dCompressedHeaderLen!.View);

        // Display mi dims: the sequential encoder and the entropy walker both need them to handle SBs
        // that straddle the frame boundary spec-correctly (e.g. 1920x1080: bottom-edge SB16 at
        // miRow=134 emits PARTITION_HORZ + top BLOCK_16X8 leaf).
        int displayMiCols = (width + 7) >> 3;
        int displayMiRows = (height + 7) >> 3;

        // ---- 4. Sequential encode (forward + inverse pipeline) ----
        _sequentialKernel.Run(
            _dY!.View, _dU!.View, _dV!.View,
            _dYRecon!.View, _dURecon!.View, _dVRecon!.View,
            _dYCoefs!.View, _dUCoefs!.View, _dVCoefs!.View,
            _dDequant!.View,
            mbCols, mbRows,
            frameMiRows: displayMiRows);

        // ---- 5. Entropy ----
        _entropyKernel.Run(
            _dYCoefs!.View, _dUCoefs!.View, _dVCoefs!.View,
            _dTile!.View, _dTileLen!.View,
            _dByteConsts.View, _dUshortConsts.View,
            mbCols, mbRows,
            displayMiCols, displayMiRows);

        // ---- 6. Uncompressed header (GPU-resident: reads compressedLen
        // directly from the buffer the compressedHeaderKernel wrote to;
        // no host sync). Bitstream signals ORIGINAL display dims; the
        // entropy walker handles boundary-straddling SBs spec-correctly
        // via forced-split partitions (no CDF emit at boundaries). ----
        _uncompressedHeaderKernel.Run(
            _dUncompressedHeader!.View, _dUncompressedHeaderLen!.View,
            _dCompressedHeaderLen!.View,
            width, height, baseQIndex);

        // ---- 7. Assemble (GPU-resident: reads all 3 lens from views,
        // no host sync between header emit and concat) straight into the caller's output ----
        _assembleKernel.Run(
            _dUncompressedHeader!.View, _dCompressedHeader!.View, _dTile!.View,
            output, _dOutFrameLen!.View,
            _dUncompressedHeaderLen!.View, _dCompressedHeaderLen!.View, _dTileLen!.View);

        // ---- 8. Single end-of-frame sync ----
        await _accelerator.SynchronizeAsync();
    }

    /// <summary>
    /// Frame-batch parallel VP9 encoder for frames already on the accelerator. Same architecture
    /// as VP8 batch: per-frame buffer slots through every kernel, batch wave-front + batch entropy
    /// run all N frames concurrently. Only the per-frame lengths (written to
    /// <paramref name="lengths"/>) are produced; the host reads nothing here.
    /// </summary>
    /// <param name="yPlanes">Frame-major tightly packed luma planes: frame f at <c>f * width * height</c>.</param>
    /// <param name="uPlanes">Frame-major tightly packed Cb planes: frame f at <c>f * uvWidth * uvHeight</c>
    /// with uvWidth = (width+1)/2, uvHeight = (height+1)/2.</param>
    /// <param name="vPlanes">Frame-major tightly packed Cr planes, same layout as <paramref name="uPlanes"/>.</param>
    /// <param name="frameCount">Number of frames.</param>
    /// <param name="width">Picture width in pixels.</param>
    /// <param name="height">Picture height in pixels.</param>
    /// <param name="outputs">Frame f is assembled at <c>f * GetMaxEncodedSize(width, height)</c>; at least
    /// <c>frameCount * GetMaxEncodedSize</c> bytes.</param>
    /// <param name="lengths">Receives each frame's encoded length; at least <paramref name="frameCount"/> longs.</param>
    /// <param name="baseQIndex">VP9 base quantizer index, 1..255.</param>
    public async Task EncodeKeyFramesBatchAsync(
        ArrayView1D<byte, Stride1D.Dense> yPlanes,
        ArrayView1D<byte, Stride1D.Dense> uPlanes,
        ArrayView1D<byte, Stride1D.Dense> vPlanes,
        int frameCount, int width, int height,
        ArrayView1D<byte, Stride1D.Dense> outputs,
        ArrayView1D<long, Stride1D.Dense> lengths,
        int baseQIndex = 30)
    {
        if (frameCount <= 0) throw new ArgumentOutOfRangeException(nameof(frameCount));
        ValidateArgs(width, height, baseQIndex);

        WorkingDims(width, height, out int workWidth, out int workHeight);
        int uvWidth = ChromaDim(width);
        int uvHeight = ChromaDim(height);
        int uvWorkWidth = workWidth / 2;
        int uvWorkHeight = workHeight / 2;
        int mbCols = workWidth >> 4;
        int mbRows = workHeight >> 4;
        int yLen = workWidth * workHeight;
        int uvLen = yLen / 4;
        int mbCount = mbCols * mbRows;

        long ySrcPlane = (long)width * height;
        long uvSrcPlane = (long)uvWidth * uvHeight;
        int frameCapacity = GetMaxEncodedSize(width, height);
        if (yPlanes.Length < frameCount * ySrcPlane)
            throw new ArgumentException("yPlanes must hold frameCount tightly packed luma planes.", nameof(yPlanes));
        if (uPlanes.Length < frameCount * uvSrcPlane)
            throw new ArgumentException("uPlanes must hold frameCount tightly packed Cb planes.", nameof(uPlanes));
        if (vPlanes.Length < frameCount * uvSrcPlane)
            throw new ArgumentException("vPlanes must hold frameCount tightly packed Cr planes.", nameof(vPlanes));
        if (outputs.Length < (long)frameCount * frameCapacity)
            throw new ArgumentException("outputs must hold frameCount * GetMaxEncodedSize bytes.", nameof(outputs));
        if (lengths.Length < frameCount)
            throw new ArgumentException("lengths must hold frameCount longs.", nameof(lengths));

        int yCoefStride = mbCount * 256;
        int uvCoefStride = mbCount * 64;
        int dequantStride = 4;
        long worstCaseTile = WorstCaseTile(mbCount);

        // Per-frame slot buffers.
        using var dAllY = _accelerator.Allocate1D<byte>((long)frameCount * yLen);
        using var dAllU = _accelerator.Allocate1D<byte>((long)frameCount * uvLen);
        using var dAllV = _accelerator.Allocate1D<byte>((long)frameCount * uvLen);
        using var dAllYR = _accelerator.Allocate1D<byte>((long)frameCount * yLen);
        using var dAllUR = _accelerator.Allocate1D<byte>((long)frameCount * uvLen);
        using var dAllVR = _accelerator.Allocate1D<byte>((long)frameCount * uvLen);
        using var dAllYC = _accelerator.Allocate1D<short>((long)frameCount * yCoefStride);
        using var dAllUC = _accelerator.Allocate1D<short>((long)frameCount * uvCoefStride);
        using var dAllVC = _accelerator.Allocate1D<short>((long)frameCount * uvCoefStride);
        using var dAllDQ = _accelerator.Allocate1D<int>((long)frameCount * dequantStride);
        using var dAllCH = _accelerator.Allocate1D<byte>((long)frameCount * 64);
        using var dAllCHLen = _accelerator.Allocate1D<long>(frameCount);
        using var dAllTile = _accelerator.Allocate1D<byte>((long)frameCount * worstCaseTile);
        using var dAllTileLen = _accelerator.Allocate1D<long>(frameCount);
        using var dAllUH = _accelerator.Allocate1D<byte>((long)frameCount * 32);
        using var dAllUHLen = _accelerator.Allocate1D<long>(frameCount);

        dAllCH.View.MemSetToZero();
        dAllCHLen.View.MemSetToZero();
        dAllTile.View.MemSetToZero();
        dAllTileLen.View.MemSetToZero();
        dAllUH.View.MemSetToZero();
        dAllUHLen.View.MemSetToZero();

        // Phase 1a: pad every frame's planes into its working-dim slot on the accelerator. Every
        // working byte of every slot is written (zero pad included): no host staging array, no host
        // row loop.
        for (int f = 0; f < frameCount; f++)
        {
            _pad.Run(yPlanes, f * ySrcPlane, width, width, height,
                dAllY.View.SubView((long)f * yLen, yLen), workWidth, workHeight);
            _pad.Run(uPlanes, f * uvSrcPlane, uvWidth, uvWidth, uvHeight,
                dAllU.View.SubView((long)f * uvLen, uvLen), uvWorkWidth, uvWorkHeight);
            _pad.Run(vPlanes, f * uvSrcPlane, uvWidth, uvWidth, uvHeight,
                dAllV.View.SubView((long)f * uvLen, uvLen), uvWorkWidth, uvWorkHeight);
        }

        // Phase 1b: BATCH dequantizer + compressed header.
        _dequantKernel.RunBatch(_dDcQLookup.View, _dAcQLookup.View, dAllDQ.View,
            baseQIndex, 0, 0, 0, 0,
            frameCount, dequantStride);
        _compressedHeaderKernel.RunBatch(
            dAllCH.View, dAllCHLen.View,
            frameCount, 64);

        // Phase 2: BATCH wave-front sequential encode. Single-dispatch path
        // collapses 47 per-diagonal launches into one kernel via Group.Barrier.
        // displayMiRows drives boundary forced-split MB encoding (top BLOCK_16X8
        // with 2 Tx8x8 luma + 2 Tx4x4 chroma) at the bottom-edge MB row.
        int seqFrameMiRows = (height + 7) >> 3;
        bool wavefrontSingleDispatched = _sequentialKernel.TryRunBatchSingleDispatch(
            dAllY.View, dAllU.View, dAllV.View,
            dAllYR.View, dAllUR.View, dAllVR.View,
            dAllYC.View, dAllUC.View, dAllVC.View,
            dAllDQ.View,
            mbCols, mbRows, frameCount,
            yLen, uvLen,
            yCoefStride, uvCoefStride, dequantStride,
            seqFrameMiRows);
        if (!wavefrontSingleDispatched)
        {
            _sequentialKernel.RunBatch(
                dAllY.View, dAllU.View, dAllV.View,
                dAllYR.View, dAllUR.View, dAllVR.View,
                dAllYC.View, dAllUC.View, dAllVC.View,
                dAllDQ.View,
                mbCols, mbRows, frameCount,
                yLen, uvLen,
                yCoefStride, uvCoefStride, dequantStride,
                seqFrameMiRows);
        }

        // Phase 3: BATCH entropy. Display mi dims drive boundary forced-split.
        var entropyStrides = new Vp9FrameEntropyBatchStrides
        {
            YCoefStride = yCoefStride,
            UvCoefStride = uvCoefStride,
            OutBufStride = (int)worstCaseTile,
            MbCols = mbCols,
            MbRows = mbRows,
            FrameMiCols = (width + 7) >> 3,
            FrameMiRows = (height + 7) >> 3,
        };
        _entropyKernel.RunBatch(
            dAllYC.View, dAllUC.View, dAllVC.View,
            dAllTile.View, dAllTileLen.View,
            _dByteConsts.View, _dUshortConsts.View,
            frameCount, entropyStrides);

        // Phase 4: BATCH uncompressed header + assemble. Bitstream signals
        // original display dims; entropy walker handles boundary forced-split.
        _uncompressedHeaderKernel.RunBatch(
            dAllUH.View, dAllUHLen.View, dAllCHLen.View,
            width, height, baseQIndex,
            frameCount, 32);

        var assembleStrides = new Vp9AssembleBatchStrides
        {
            UhStride = 32,
            ChStride = 64,
            TileStride = (int)worstCaseTile,
            OutStride = frameCapacity,
        };
        _assembleKernel.RunBatch(
            dAllUH.View, dAllCH.View, dAllTile.View, outputs, lengths,
            dAllUHLen.View, dAllCHLen.View, dAllTileLen.View,
            frameCount, assembleStrides);

        // Drain before the using-scoped slot buffers dispose: on the browser backends a dispatch
        // that still references a freed buffer reads freed memory.
        await _accelerator.SynchronizeAsync();
    }

    // ------------------------------------------------------------------
    // Managed overloads
    // ------------------------------------------------------------------

    /// <summary>
    /// Encode a single VP9 v1 keyframe from managed, tightly packed YUV420 planes. Returns the
    /// complete VP9 frame bytes (uncompressed header + compressed header + tile data) as a managed
    /// array (a terminal CPU sink; in a browser prefer the <see cref="TypedArray"/> overload).
    /// </summary>
    public async Task<byte[]> EncodeKeyFrameAsync(
        ReadOnlyMemory<byte> yPlane, ReadOnlyMemory<byte> uPlane, ReadOnlyMemory<byte> vPlane,
        int width, int height,
        int baseQIndex = 30)
    {
        ValidateArgs(width, height, baseQIndex);
        int uvWidth = ChromaDim(width), uvHeight = ChromaDim(height);
        DeviceTransfer.UploadPlane(_stageY, yPlane, width, width, height, nameof(yPlane));
        DeviceTransfer.UploadPlane(_stageU, uPlane, uvWidth, uvWidth, uvHeight, nameof(uPlane));
        DeviceTransfer.UploadPlane(_stageV, vPlane, uvWidth, uvWidth, uvHeight, nameof(vPlane));
        _stageOut.Ensure(GetMaxEncodedSize(width, height));

        int len = await EncodeKeyFrameAsync(
            _stageY.View, _stageU.View, _stageV.View, width, height, _stageOut.View, baseQIndex);
        return await DeviceTransfer.DownloadAsync(_stageOut.View, len);
    }

    /// <summary>
    /// Encode + return both the encoded bytes AND the encoder's
    /// internal recon planes. The recon is the same data the
    /// downstream decoder must reconstruct from the encoded bytes -
    /// useful for self-consistency tests of the encoder/decoder
    /// kernel chain. The recon planes are the encoder's working planes (padded to the 64-pixel
    /// grid); this is a diagnostic/test API and reads them back to managed memory.
    /// </summary>
    public async Task<(byte[] bytes, byte[] yRecon, byte[] uRecon, byte[] vRecon)>
        EncodeKeyFrameWithReconAsync(
        ReadOnlyMemory<byte> yPlane, ReadOnlyMemory<byte> uPlane, ReadOnlyMemory<byte> vPlane,
        int width, int height,
        int baseQIndex = 30)
    {
        ValidateArgs(width, height, baseQIndex);
        int uvWidth = ChromaDim(width), uvHeight = ChromaDim(height);
        DeviceTransfer.UploadPlane(_stageY, yPlane, width, width, height, nameof(yPlane));
        DeviceTransfer.UploadPlane(_stageU, uPlane, uvWidth, uvWidth, uvHeight, nameof(uPlane));
        DeviceTransfer.UploadPlane(_stageV, vPlane, uvWidth, uvWidth, uvHeight, nameof(vPlane));
        _stageOut.Ensure(GetMaxEncodedSize(width, height));

        await EncodeCoreAsync(_stageY.View, _stageU.View, _stageV.View, width, height, _stageOut.View, baseQIndex);
        int len = (int)(await _dOutFrameLen!.CopyToHostAsync())[0];
        var bytes = await DeviceTransfer.DownloadAsync(_stageOut.View, len);
        var yRecon = await _dYRecon!.CopyToHostAsync();
        var uRecon = await _dURecon!.CopyToHostAsync();
        var vRecon = await _dVRecon!.CopyToHostAsync();
        return (bytes, yRecon, uRecon, vRecon);
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
        ValidateArgs(width, height, baseQIndex);

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

        // Lengths first (8 bytes each); then ONLY the actual encoded bytes per frame via partial
        // readback, never the worst-case-sized slot.
        var lens = await _stageLens.View.SubView(0, frameCount).CopyToHostAsync();
        var results = new byte[frameCount][];
        for (int f = 0; f < frameCount; f++)
        {
            int len = (int)lens[f];
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
    /// Encode a single VP9 keyframe from JS typed arrays (browser accelerators only), tightly
    /// packed planes. The planes go to the GPU with <c>CopyFromJS</c> and the frame comes back as
    /// a JS <see cref="Uint8Array"/> (the caller disposes it); no bulk byte enters the .NET heap.
    /// </summary>
    /// <exception cref="NotSupportedException">The accelerator is not a browser accelerator.</exception>
    [SupportedOSPlatform("browser")]
    public async Task<Uint8Array> EncodeKeyFrameAsync(
        TypedArray yPlane, TypedArray uPlane, TypedArray vPlane,
        int width, int height,
        int baseQIndex = 30)
    {
        ArgumentNullException.ThrowIfNull(yPlane);
        ArgumentNullException.ThrowIfNull(uPlane);
        ArgumentNullException.ThrowIfNull(vPlane);
        ValidateArgs(width, height, baseQIndex);

        _stageY.Ensure(yPlane.ByteLength);
        _stageU.Ensure(uPlane.ByteLength);
        _stageV.Ensure(vPlane.ByteLength);
        _stageOut.Ensure(GetMaxEncodedSize(width, height));
        DeviceTransfer.Upload(_stageY.View, yPlane);
        DeviceTransfer.Upload(_stageU.View, uPlane);
        DeviceTransfer.Upload(_stageV.View, vPlane);

        int len = await EncodeKeyFrameAsync(
            _stageY.View, _stageU.View, _stageV.View, width, height, _stageOut.View, baseQIndex);
        return await DeviceTransfer.DownloadUint8ArrayAsync(_stageOut.View, len);
    }

    /// <summary>
    /// Batch-encode keyframes from JS typed arrays (browser accelerators only), one tightly packed
    /// plane per frame per component. Each returned <see cref="Uint8Array"/> is owned (and
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
        ValidateArgs(width, height, baseQIndex);

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

        var lens = await _stageLens.View.SubView(0, frameCount).CopyToHostAsync();
        var results = new Uint8Array[frameCount];
        for (int f = 0; f < frameCount; f++)
            results[f] = await DeviceTransfer.DownloadUint8ArrayAsync(
                _stageOut.View.SubView((long)f * cap, lens[f]), lens[f]);
        return results;
    }

    // ------------------------------------------------------------------
    // Working buffers
    // ------------------------------------------------------------------

    /// <summary>
    /// Ensure per-frame GPU buffers are sized for (width,height). Reallocates
    /// only when the resolution changes.
    /// </summary>
    private void EnsureBuffers(int width, int height, int yLen, int uvLen, int mbCount)
    {
        if (_cachedWidth == width && _cachedHeight == height) return;
        DisposeFrameBuffers();
        _cachedWidth = width;
        _cachedHeight = height;
        _dY = _accelerator.Allocate1D<byte>(yLen);
        _dU = _accelerator.Allocate1D<byte>(uvLen);
        _dV = _accelerator.Allocate1D<byte>(uvLen);
        _dYRecon = _accelerator.Allocate1D<byte>(yLen);
        _dURecon = _accelerator.Allocate1D<byte>(uvLen);
        _dVRecon = _accelerator.Allocate1D<byte>(uvLen);
        _dYCoefs = _accelerator.Allocate1D<short>((long)mbCount * 256);
        _dUCoefs = _accelerator.Allocate1D<short>((long)mbCount * 64);
        _dVCoefs = _accelerator.Allocate1D<short>((long)mbCount * 64);
        _dDequant = _accelerator.Allocate1D<int>(4);
        _dCompressedHeader = _accelerator.Allocate1D<byte>(64);
        _dCompressedHeaderLen = _accelerator.Allocate1D<long>(1);
        _dTile = _accelerator.Allocate1D<byte>(WorstCaseTile(mbCount));
        _dTileLen = _accelerator.Allocate1D<long>(1);
        _dUncompressedHeader = _accelerator.Allocate1D<byte>(32);
        _dUncompressedHeaderLen = _accelerator.Allocate1D<long>(1);
        _dOutFrameLen = _accelerator.Allocate1D<long>(1);
    }

    private void DisposeFrameBuffers()
    {
        _dY?.Dispose(); _dY = null;
        _dU?.Dispose(); _dU = null;
        _dV?.Dispose(); _dV = null;
        _dYRecon?.Dispose(); _dYRecon = null;
        _dURecon?.Dispose(); _dURecon = null;
        _dVRecon?.Dispose(); _dVRecon = null;
        _dYCoefs?.Dispose(); _dYCoefs = null;
        _dUCoefs?.Dispose(); _dUCoefs = null;
        _dVCoefs?.Dispose(); _dVCoefs = null;
        _dDequant?.Dispose(); _dDequant = null;
        _dCompressedHeader?.Dispose(); _dCompressedHeader = null;
        _dCompressedHeaderLen?.Dispose(); _dCompressedHeaderLen = null;
        _dTile?.Dispose(); _dTile = null;
        _dTileLen?.Dispose(); _dTileLen = null;
        _dUncompressedHeader?.Dispose(); _dUncompressedHeader = null;
        _dUncompressedHeaderLen?.Dispose(); _dUncompressedHeaderLen = null;
        _dOutFrameLen?.Dispose(); _dOutFrameLen = null;
        _cachedWidth = -1;
        _cachedHeight = -1;
    }

    /// <summary>Release every resource the encoder owns.</summary>
    public void Dispose()
    {
        _pad.Dispose();
        _dequantKernel.Dispose();
        _compressedHeaderKernel.Dispose();
        _sequentialKernel.Dispose();
        _entropyKernel.Dispose();
        _uncompressedHeaderKernel.Dispose();
        _assembleKernel.Dispose();

        _dDcQLookup.Dispose();
        _dAcQLookup.Dispose();
        _dByteConsts.Dispose();
        _dUshortConsts.Dispose();
        DisposeFrameBuffers();
        _stageY.Dispose();
        _stageU.Dispose();
        _stageV.Dispose();
        _stageOut.Dispose();
        _stageLens.Dispose();
    }
}
