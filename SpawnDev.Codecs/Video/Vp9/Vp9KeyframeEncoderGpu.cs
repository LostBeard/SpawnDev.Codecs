// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 v1 keyframe encoder, integration class. 100% ILGPU - the host
// is a pure coordinator: alloc GPU buffers, upload YUV source +
// constant tables, dispatch the kernel chain in order, read back
// the final frame bytes. Zero CPU math, zero CPU iteration over
// codec data, zero CPU bool encoding, zero CPU bitstream assembly.
//
// Kernel chain:
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
//                                          into final output buffer.
//
// V1 simplifications (mirror Vp9KeyframeEncoder.EncodeKeyFrame):
//   - Profile 0, YUV 4:2:0
//   - Width + height multiples of 64 (single tile, integer-SB grid).
//     v1 entropy kernel caps at 512-pixel width.
//   - All Y blocks DC_PRED + Tx16x16; all UV blocks DC_PRED + Tx8x8.
//   - tx_mode = Allow32x32, single tile, LF off, segmentation off,
//     default coef probs.
//   - Skip flag hardcoded 0.

using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// VP9 v1 keyframe encoder integration class. Runs the 6-kernel
/// chain on the supplied accelerator and returns the final encoded
/// frame bytes. Host is a pure coordinator.
/// </summary>
public sealed class Vp9KeyframeEncoderGpu : IDisposable
{
    private readonly Accelerator _accelerator;
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
    private MemoryBuffer1D<byte, global::ILGPU.Stride1D.Dense>? _dOutFrame;
    private MemoryBuffer1D<long, global::ILGPU.Stride1D.Dense>? _dOutFrameLen;

    /// <summary>
    /// Compile every kernel + upload one-time-cached constant tables
    /// (DC/AC quantizer lookup + packed entropy / scan / neighbor
    /// constants).
    /// </summary>
    public Vp9KeyframeEncoderGpu(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;

        _dequantKernel = new Vp9DequantizerComputeKernel(accelerator);
        _compressedHeaderKernel = new Vp9FrameCompressedHeaderKernel(accelerator);
        _sequentialKernel = new Vp9FrameSequentialEncodeKernel(accelerator);
        _entropyKernel = new Vp9FrameEntropyKernel(accelerator);
        _uncompressedHeaderKernel = new Vp9FrameUncompressedHeaderKernel(accelerator);
        _assembleKernel = new Vp9FrameAssembleKernel(accelerator);

        _dDcQLookup = accelerator.Allocate1D<short>(256);
        _dAcQLookup = accelerator.Allocate1D<short>(256);
        _dByteConsts = accelerator.Allocate1D<byte>(Vp9KeyframeConstantsGpu.ByteConstsTotalBytes);
        _dUshortConsts = accelerator.Allocate1D<ushort>(Vp9KeyframeConstantsGpu.UshortConstsTotalEntries);

        _dDcQLookup.View.CopyFromCPU(Vp9DequantizerComputeKernel.BuildDcQLookup());
        _dAcQLookup.View.CopyFromCPU(Vp9DequantizerComputeKernel.BuildAcQLookup());
        _dByteConsts.View.CopyFromCPU(Vp9KeyframeConstantsGpu.BuildByteConstsBuffer());
        _dUshortConsts.View.CopyFromCPU(Vp9KeyframeConstantsGpu.BuildUshortConstsBuffer());
    }

    /// <summary>
    /// Encode a single VP9 v1 keyframe from YUV420 source. Returns
    /// the complete VP9 frame bytes (uncompressed header + compressed
    /// header + tile data).
    /// </summary>
    public Task<byte[]> EncodeKeyFrameAsync(
        byte[] yPlane, byte[] uPlane, byte[] vPlane,
        int width, int height,
        int baseQIndex = 30)
    {
        // Fast path: caller doesn't want recon planes. Skip the three
        // extra device->host readbacks (yRecon/uRecon/vRecon) - those
        // are pure overhead in the encode-only path.
        return EncodeKeyFrameInternalAsync(yPlane, uPlane, vPlane, width, height, baseQIndex, returnRecon: false);
    }

    /// <summary>
    /// Encode + return both the encoded bytes AND the encoder's
    /// internal recon planes. The recon is the same data the
    /// downstream decoder must reconstruct from the encoded bytes -
    /// useful for self-consistency tests of the encoder/decoder
    /// kernel chain.
    /// </summary>
    public async Task<(byte[] bytes, byte[] yRecon, byte[] uRecon, byte[] vRecon)>
        EncodeKeyFrameWithReconAsync(
        byte[] yPlane, byte[] uPlane, byte[] vPlane,
        int width, int height,
        int baseQIndex = 30)
    {
        // The recon path returns full Y/U/V; fall through to the unified internal.
        var bytes = await EncodeKeyFrameInternalAsync(yPlane, uPlane, vPlane, width, height, baseQIndex, returnRecon: true);
        // After EncodeKeyFrameInternalAsync(returnRecon:true) the recon planes
        // are populated on the host; pull them from the helper-set fields.
        return (bytes, _lastYRecon!, _lastURecon!, _lastVRecon!);
    }

    private byte[]? _lastYRecon;
    private byte[]? _lastURecon;
    private byte[]? _lastVRecon;

    private async Task<byte[]> EncodeKeyFrameInternalAsync(
        byte[] yPlane, byte[] uPlane, byte[] vPlane,
        int width, int height,
        int baseQIndex,
        bool returnRecon)
    {
        if (yPlane is null) throw new ArgumentNullException(nameof(yPlane));
        if (uPlane is null) throw new ArgumentNullException(nameof(uPlane));
        if (vPlane is null) throw new ArgumentNullException(nameof(vPlane));
        if (width <= 0 || (width & 63) != 0)
            throw new ArgumentOutOfRangeException(nameof(width),
                "v1 GPU encoder requires width that is a positive multiple of 64.");
        if (height <= 0 || (height & 63) != 0)
            throw new ArgumentOutOfRangeException(nameof(height),
                "v1 GPU encoder requires height that is a positive multiple of 64.");
        if (baseQIndex <= 0 || baseQIndex > 255)
            throw new ArgumentOutOfRangeException(nameof(baseQIndex),
                "baseQIndex must be in [1, 255].");

        int mbCols = width >> 4;
        int mbRows = height >> 4;
        int yLen = width * height;
        int uvLen = yLen / 4;
        int mbCount = mbCols * mbRows;

        if (yPlane.Length < yLen) throw new ArgumentException("yPlane too short.", nameof(yPlane));
        if (uPlane.Length < uvLen) throw new ArgumentException("uPlane too short.", nameof(uPlane));
        if (vPlane.Length < uvLen) throw new ArgumentException("vPlane too short.", nameof(vPlane));

        // ---- 1. Ensure per-frame GPU buffers (cached across same-resolution calls) ----
        EnsureBuffers(width, height, yLen, uvLen, mbCount);

        // ---- Pre-zero output buffers so the bool encoder's carry-back pass
        // reads stable bytes. Recon planes are fully overwritten by the
        // sequential kernel; coefs are fully overwritten too. Headers /
        // tile / outframe are bool-encoder carry-back targets and need
        // pre-zero each frame.
        _dCompressedHeader!.View.MemSetToZero();
        _dCompressedHeaderLen!.View.MemSetToZero();
        _dTile!.View.MemSetToZero();
        _dTileLen!.View.MemSetToZero();
        _dUncompressedHeader!.View.MemSetToZero();
        _dUncompressedHeaderLen!.View.MemSetToZero();
        _dOutFrame!.View.MemSetToZero();
        _dOutFrameLen!.View.MemSetToZero();

        _dY!.View.CopyFromCPU(yPlane);
        _dU!.View.CopyFromCPU(uPlane);
        _dV!.View.CopyFromCPU(vPlane);

        // ---- 2. Dispatch dequantizer compute kernel ----
        // y_dc_delta / uv_dc_delta / uv_ac_delta = 0 in v1.
        _dequantKernel.Run(_dDcQLookup.View, _dAcQLookup.View, _dDequant!.View,
            baseQIndex, 0, 0, 0, 0);

        // ---- 3. Compressed header ----
        _compressedHeaderKernel.Run(_dCompressedHeader!.View, _dCompressedHeaderLen!.View);

        // ---- 4. Sequential encode (forward + inverse pipeline) ----
        _sequentialKernel.Run(
            _dY!.View, _dU!.View, _dV!.View,
            _dYRecon!.View, _dURecon!.View, _dVRecon!.View,
            _dYCoefs!.View, _dUCoefs!.View, _dVCoefs!.View,
            _dDequant!.View,
            mbCols, mbRows);

        // ---- 5. Entropy ----
        _entropyKernel.Run(
            _dYCoefs!.View, _dUCoefs!.View, _dVCoefs!.View,
            _dTile!.View, _dTileLen!.View,
            _dByteConsts.View, _dUshortConsts.View,
            mbCols, mbRows);

        // ---- 6. Uncompressed header (GPU-resident: reads compressedLen
        // directly from the buffer the compressedHeaderKernel wrote to;
        // no host sync) ----
        _uncompressedHeaderKernel.Run(
            _dUncompressedHeader!.View, _dUncompressedHeaderLen!.View,
            _dCompressedHeaderLen!.View,
            width, height, baseQIndex);

        // ---- 7. Assemble (GPU-resident: reads all 3 lens from views,
        // no host sync between header emit and concat) ----
        _assembleKernel.Run(
            _dUncompressedHeader!.View, _dCompressedHeader!.View, _dTile!.View,
            _dOutFrame!.View, _dOutFrameLen!.View,
            _dUncompressedHeaderLen!.View, _dCompressedHeaderLen!.View, _dTileLen!.View);

        // ---- 8. Single end-of-frame sync + readback ----
        await _accelerator.SynchronizeAsync();
        long outFrameLen = (await _dOutFrameLen!.CopyToHostAsync())[0];
        // Real per-backend partial readback (SpawnDev.ILGPU 4.9.3+).
        var result = await _dOutFrame!.View.SubView(0, outFrameLen).CopyToHostAsync();

        if (returnRecon)
        {
            _lastYRecon = await _dYRecon!.CopyToHostAsync();
            _lastURecon = await _dURecon!.CopyToHostAsync();
            _lastVRecon = await _dVRecon!.CopyToHostAsync();
        }

        return result;
    }

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
        long worstCaseTile = mbCount * 1024L + 256L;
        long worstCaseFrame = worstCaseTile + 128L;
        _dCompressedHeader = _accelerator.Allocate1D<byte>(64);
        _dCompressedHeaderLen = _accelerator.Allocate1D<long>(1);
        _dTile = _accelerator.Allocate1D<byte>(worstCaseTile);
        _dTileLen = _accelerator.Allocate1D<long>(1);
        _dUncompressedHeader = _accelerator.Allocate1D<byte>(32);
        _dUncompressedHeaderLen = _accelerator.Allocate1D<long>(1);
        _dOutFrame = _accelerator.Allocate1D<byte>(worstCaseFrame);
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
        _dOutFrame?.Dispose(); _dOutFrame = null;
        _dOutFrameLen?.Dispose(); _dOutFrameLen = null;
        _cachedWidth = -1;
        _cachedHeight = -1;
    }

    /// <summary>
    /// Frame-batch parallel VP9 encoder. Same architecture as VP8 batch:
    /// per-frame buffer slots through every kernel, batch wave-front +
    /// batch entropy run all N frames concurrently.
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
        if (frameCount == 0) return Array.Empty<byte[]>();
        if (width <= 0 || (width & 63) != 0)
            throw new ArgumentOutOfRangeException(nameof(width));
        if (height <= 0 || (height & 63) != 0)
            throw new ArgumentOutOfRangeException(nameof(height));

        int mbCols = width >> 4;
        int mbRows = height >> 4;
        int yLen = width * height;
        int uvLen = yLen / 4;
        int mbCount = mbCols * mbRows;

        int yCoefStride = mbCount * 256;
        int uvCoefStride = mbCount * 64;
        int dequantStride = 4;
        long worstCaseTile = mbCount * 1024L + 256L;
        long worstCaseFrame = worstCaseTile + 128L;

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
        using var dAllOut = _accelerator.Allocate1D<byte>((long)frameCount * worstCaseFrame);
        using var dAllOutLen = _accelerator.Allocate1D<long>(frameCount);

        dAllCH.View.MemSetToZero();
        dAllCHLen.View.MemSetToZero();
        dAllTile.View.MemSetToZero();
        dAllTileLen.View.MemSetToZero();
        dAllUH.View.MemSetToZero();
        dAllUHLen.View.MemSetToZero();
        dAllOut.View.MemSetToZero();
        dAllOutLen.View.MemSetToZero();

        // Phase 1a: bulk upload all source planes (3 host->device transfers).
        var hostY = new byte[(long)frameCount * yLen];
        var hostU = new byte[(long)frameCount * uvLen];
        var hostV = new byte[(long)frameCount * uvLen];
        for (int f = 0; f < frameCount; f++)
        {
            yPlanes[f].Span.CopyTo(hostY.AsSpan((int)((long)f * yLen)));
            uPlanes[f].Span.CopyTo(hostU.AsSpan((int)((long)f * uvLen)));
            vPlanes[f].Span.CopyTo(hostV.AsSpan((int)((long)f * uvLen)));
        }
        dAllY.View.CopyFromCPU(hostY);
        dAllU.View.CopyFromCPU(hostU);
        dAllV.View.CopyFromCPU(hostV);

        // Phase 1b: BATCH dequantizer + compressed header.
        _dequantKernel.RunBatch(_dDcQLookup.View, _dAcQLookup.View, dAllDQ.View,
            baseQIndex, 0, 0, 0, 0,
            frameCount, dequantStride);
        _compressedHeaderKernel.RunBatch(
            dAllCH.View, dAllCHLen.View,
            frameCount, 64);

        // Phase 2: BATCH wave-front sequential encode. Single-dispatch path
        // collapses 47 per-diagonal launches into one kernel via Group.Barrier.
        bool wavefrontSingleDispatched = _sequentialKernel.TryRunBatchSingleDispatch(
            dAllY.View, dAllU.View, dAllV.View,
            dAllYR.View, dAllUR.View, dAllVR.View,
            dAllYC.View, dAllUC.View, dAllVC.View,
            dAllDQ.View,
            mbCols, mbRows, frameCount,
            yLen, uvLen,
            yCoefStride, uvCoefStride, dequantStride);
        if (!wavefrontSingleDispatched)
        {
            _sequentialKernel.RunBatch(
                dAllY.View, dAllU.View, dAllV.View,
                dAllYR.View, dAllUR.View, dAllVR.View,
                dAllYC.View, dAllUC.View, dAllVC.View,
                dAllDQ.View,
                mbCols, mbRows, frameCount,
                yLen, uvLen,
                yCoefStride, uvCoefStride, dequantStride);
        }

        // Phase 3: BATCH entropy.
        var entropyStrides = new Vp9FrameEntropyBatchStrides
        {
            YCoefStride = yCoefStride,
            UvCoefStride = uvCoefStride,
            OutBufStride = (int)worstCaseTile,
            MbCols = mbCols,
            MbRows = mbRows,
        };
        _entropyKernel.RunBatch(
            dAllYC.View, dAllUC.View, dAllVC.View,
            dAllTile.View, dAllTileLen.View,
            _dByteConsts.View, _dUshortConsts.View,
            frameCount, entropyStrides);

        // Phase 4: BATCH uncompressed header + assemble.
        _uncompressedHeaderKernel.RunBatch(
            dAllUH.View, dAllUHLen.View, dAllCHLen.View,
            width, height, baseQIndex,
            frameCount, 32);

        var assembleStrides = new Vp9AssembleBatchStrides
        {
            UhStride = 32,
            ChStride = 64,
            TileStride = (int)worstCaseTile,
            OutStride = (int)worstCaseFrame,
        };
        _assembleKernel.RunBatch(
            dAllUH.View, dAllCH.View, dAllTile.View, dAllOut.View, dAllOutLen.View,
            dAllUHLen.View, dAllCHLen.View, dAllTileLen.View,
            frameCount, assembleStrides);

        await _accelerator.SynchronizeAsync();
        var outLensHost = await dAllOutLen.CopyToHostAsync();
        var outBytesHost = await dAllOut.CopyToHostAsync();
        var results = new byte[frameCount][];
        for (int f = 0; f < frameCount; f++)
        {
            int len = (int)outLensHost[f];
            results[f] = new byte[len];
            Array.Copy(outBytesHost, (long)f * worstCaseFrame, results[f], 0, len);
        }
        return results;
    }

    /// <summary>Release every resource the encoder owns.</summary>
    public void Dispose()
    {
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
    }
}
