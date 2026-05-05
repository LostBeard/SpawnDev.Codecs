// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 v1 keyframe encoder, GPU integration class. Symmetric to
// Vp8KeyframeEncoderGpu / Vp9KeyframeEncoderGpu. The host is a pure
// coordinator: alloc + upload + dispatch + readback. Single GPU
// thread runs the entire EncodeSingleTile pipeline bit-exact vs the
// CPU Av1KeyframeEncoder reference.
//
// V1 phase (this commit):
//   - EncodeSingleTileAsync: returns raw range-coder tile bytes for a
//     YUV 4:2:0 8-bit frame at width + height multiples of 64.
//
// V2 phase (follow-up):
//   - Build TD OBU + SH OBU + Frame OBU wrap kernels and an
//     Av1FrameAssembleKernel mirroring the CPU EncodeKeyFrame path,
//     producing the full keyframe byte stream entirely on GPU.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>
/// AV1 v1 keyframe encoder integration class. V1 phase ships the
/// raw tile bytes; full OBU framing lands in V2.
/// </summary>
public sealed class Av1KeyframeEncoderGpu : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Av1FrameSequentialEncodeKernel _frameKernel;

    private readonly MemoryBuffer1D<byte, global::ILGPU.Stride1D.Dense> _dByteConsts;
    private readonly MemoryBuffer1D<ushort, global::ILGPU.Stride1D.Dense> _dUshortConsts;
    private readonly MemoryBuffer1D<short, global::ILGPU.Stride1D.Dense> _dDcAcQuant;

    // Per-resolution cached buffers - reallocated only when (width,height) changes.
    // Steady-state encoding pays only kernel dispatch + plane upload + readback per frame.
    private int _cachedWidth = -1;
    private int _cachedHeight = -1;
    private MemoryBuffer1D<byte, global::ILGPU.Stride1D.Dense>? _dSrc;
    private MemoryBuffer1D<byte, global::ILGPU.Stride1D.Dense>? _dRecon;
    private MemoryBuffer1D<byte, global::ILGPU.Stride1D.Dense>? _dTile;
    private MemoryBuffer1D<long, global::ILGPU.Stride1D.Dense>? _dTileLen;
    private MemoryBuffer1D<byte, global::ILGPU.Stride1D.Dense>? _dScratchByte;
    private MemoryBuffer1D<int, global::ILGPU.Stride1D.Dense>? _dScratchInt;
    private MemoryBuffer1D<short, global::ILGPU.Stride1D.Dense>? _dScratchShort;
    private int _cachedScratchByteLen;

    /// <summary>Construct an encoder bound to <paramref name="accelerator"/>.
    /// Uploads the constant tables once.</summary>
    public Av1KeyframeEncoderGpu(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;

        _frameKernel = new Av1FrameSequentialEncodeKernel(accelerator);

        _dByteConsts = accelerator.Allocate1D<byte>(Av1KeyframeConstantsGpu.ByteConstsTotalBytes);
        _dUshortConsts = accelerator.Allocate1D<ushort>(Av1KeyframeConstantsGpu.UshortConstsTotalEntries);
        _dByteConsts.View.CopyFromCPU(Av1KeyframeConstantsGpu.BuildByteConstsBuffer());
        _dUshortConsts.View.CopyFromCPU(Av1KeyframeConstantsGpu.BuildUshortConstsBuffer());

        // DC[0..256) + AC[256..512) lookup tables for 8-bit content.
        var dcAc = new short[512];
        for (int i = 0; i < 256; i++)
        {
            dcAc[i] = Av1DequantTables.DcLookup8[i];
            dcAc[256 + i] = Av1DequantTables.AcLookup8[i];
        }
        _dDcAcQuant = accelerator.Allocate1D<short>(512);
        _dDcAcQuant.View.CopyFromCPU(dcAc);
    }

    /// <summary>
    /// Encode + return both the encoded keyframe bytes AND the
    /// encoder's internal recon planes. The recon is the same data
    /// the downstream decoder must reconstruct - useful for
    /// self-consistency tests of the encoder/decoder pair.
    /// </summary>
    public async Task<(byte[] bytes, byte[] yRecon, byte[] uRecon, byte[] vRecon)>
        EncodeKeyFrameWithReconAsync(
        byte[] yPlane, byte[] uPlane, byte[] vPlane,
        int width, int height,
        int baseQIndex = 32)
    {
        var (tileBytes, yRecon, uRecon, vRecon) = await EncodeSingleTileWithReconAsync(
            yPlane, uPlane, vPlane, width, height, baseQIndex);
        var fullBytes = Av1KeyframeEncoder.EncodeKeyFrameWithExternalTile(
            width, height, baseQIndex, tileBytes);
        return (fullBytes, yRecon, uRecon, vRecon);
    }

    /// <summary>
    /// Run the GPU walker on the supplied YUV 4:2:0 frame and return
    /// the raw range-coder tile bytes (the same bytes the CPU
    /// Av1KeyframeEncoder.EncodeSingleTile produces).
    /// </summary>
    public Task<byte[]> EncodeSingleTileAsync(
        byte[] yPlane, byte[] uPlane, byte[] vPlane,
        int width, int height,
        int baseQIndex = 32)
    {
        // Fast path: skip recon readback (3 device->host transfers per
        // frame the encode-only caller doesn't need).
        return EncodeSingleTileInternalAsync(yPlane, uPlane, vPlane, width, height, baseQIndex, returnRecon: false);
    }

    /// <summary>
    /// Encode one keyframe tile and return the encoded bytes alongside
    /// the encoder's internal recon planes.
    /// </summary>
    public async Task<(byte[] tileBytes, byte[] yRecon, byte[] uRecon, byte[] vRecon)>
        EncodeSingleTileWithReconAsync(
        byte[] yPlane, byte[] uPlane, byte[] vPlane,
        int width, int height,
        int baseQIndex = 32)
    {
        var bytes = await EncodeSingleTileInternalAsync(yPlane, uPlane, vPlane, width, height, baseQIndex, returnRecon: true);
        return (bytes, _lastYRecon!, _lastURecon!, _lastVRecon!);
    }

    private byte[]? _lastYRecon;
    private byte[]? _lastURecon;
    private byte[]? _lastVRecon;

    private async Task<byte[]> EncodeSingleTileInternalAsync(
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

        int yLen = width * height;
        int uvLen = yLen / 4;
        int srcLen = yLen + uvLen + uvLen;

        if (yPlane.Length < yLen) throw new ArgumentException("yPlane too short.", nameof(yPlane));
        if (uPlane.Length < uvLen) throw new ArgumentException("uPlane too short.", nameof(uPlane));
        if (vPlane.Length < uvLen) throw new ArgumentException("vPlane too short.", nameof(vPlane));

        // Frame mi-units. v1: 4-px per mi unit.
        int frameMiCols = ((width + 7) >> 3) << 1;
        int frameMiRows = ((height + 7) >> 3) << 1;

        // ---- Build params struct (scratchByte layout) ----
        var p = new Av1FrameSeqEncodeParams
        {
            Width = width,
            Height = height,
            BaseQIndex = baseQIndex,
            YPlaneOff = 0,
            UPlaneOff = yLen,
            VPlaneOff = yLen + uvLen,
            FrameMiCols = frameMiCols,
            FrameMiRows = frameMiRows,
        };

        int byteOff = 0;
        p.AboveEntropyOff = byteOff; byteOff += 3 * frameMiCols;
        p.LeftEntropyOff = byteOff;  byteOff += 3 * 32;
        p.AbovePartOff = byteOff;    byteOff += frameMiCols;
        p.LeftPartOff = byteOff;     byteOff += 32;
        p.AboveYModeOff = byteOff;   byteOff += frameMiCols;
        p.LeftYModeOff = byteOff;    byteOff += 32;
        p.AboveSkipOff = byteOff;    byteOff += frameMiCols;
        p.LeftSkipOff = byteOff;     byteOff += 32;
        p.EdgeAboveOff = byteOff;    byteOff += 33;
        p.EdgeLeftOff = byteOff;     byteOff += 33;
        p.PredictOff = byteOff;      byteOff += 256;
        p.LevelsOff = byteOff;       byteOff += 1384;

        int scratchByteLen = byteOff;
        int scratchIntLen = Av1FrameSequentialEncodeKernel.MinScratchIntLength;
        int scratchShortLen = 256;

        // Worst-case tile bytes. AV1 keyframe entropy varies with content; for
        // a v1 encoder hitting random YUV with default CDFs, a generous
        // 2 KB / 16x16 leaf is safe (we measure ~150-300 bytes / leaf typical).
        int leaves = (width >> 4) * (height >> 4);
        long worstCaseTile = leaves * 2048L + 256L;

        // ---- Ensure per-frame GPU buffers (cached across same-resolution calls) ----
        EnsureBuffers(width, height, srcLen, worstCaseTile, scratchByteLen, scratchIntLen, scratchShortLen);

        // ---- Upload sources + zero outputs ----
        // Three direct uploads to subviews of dSrc - no host-side packed
        // buffer, no Buffer.BlockCopy iteration over input pixels.
        _dSrc!.View.SubView(0, yLen).CopyFromCPU(yPlane);
        _dSrc!.View.SubView(yLen, uvLen).CopyFromCPU(uPlane);
        _dSrc!.View.SubView(yLen + uvLen, uvLen).CopyFromCPU(vPlane);

        // Pre-zero output + scratch buffers per frame (kernel reads them
        // as carry-back state). Recon is fully overwritten per pixel so
        // skip its zero-fill.
        _dTile!.View.MemSetToZero();
        _dTileLen!.View.MemSetToZero();
        _dScratchByte!.View.MemSetToZero();
        _dScratchInt!.View.MemSetToZero();
        _dScratchShort!.View.MemSetToZero();

        // ---- Dispatch the walker ----
        _frameKernel.Run(
            _dSrc!.View, _dRecon!.View,
            _dTile!.View, _dTileLen!.View,
            _dByteConsts.View, _dUshortConsts.View,
            _dDcAcQuant.View,
            _dScratchByte!.View, _dScratchInt!.View, _dScratchShort!.View,
            p);

        await _accelerator.SynchronizeAsync();

        // ---- Read back tile bytes ----
        // SubView -> CopyToHostAsync is a real per-backend partial readback
        // (SpawnDev.ILGPU 4.9.3+); only the slice's bytes cross the boundary.
        long tileLen = (await _dTileLen!.CopyToHostAsync())[0];
        var tileResult = await _dTile!.View.SubView(0, tileLen).CopyToHostAsync();

        if (returnRecon)
        {
            _lastYRecon = await _dRecon!.View.SubView(0, yLen).CopyToHostAsync();
            _lastURecon = await _dRecon!.View.SubView(yLen, uvLen).CopyToHostAsync();
            _lastVRecon = await _dRecon!.View.SubView(yLen + uvLen, uvLen).CopyToHostAsync();
        }

        return tileResult;
    }

    /// <summary>
    /// Ensure per-frame GPU buffers are sized for (width,height). Reallocates
    /// only when the resolution changes.
    /// </summary>
    private void EnsureBuffers(int width, int height, int srcLen, long worstCaseTile,
        int scratchByteLen, int scratchIntLen, int scratchShortLen)
    {
        if (_cachedWidth == width && _cachedHeight == height) return;
        DisposeFrameBuffers();
        _cachedWidth = width;
        _cachedHeight = height;
        _cachedScratchByteLen = scratchByteLen;
        _dSrc = _accelerator.Allocate1D<byte>(srcLen);
        _dRecon = _accelerator.Allocate1D<byte>(srcLen);
        _dTile = _accelerator.Allocate1D<byte>(worstCaseTile);
        _dTileLen = _accelerator.Allocate1D<long>(1);
        _dScratchByte = _accelerator.Allocate1D<byte>(scratchByteLen);
        _dScratchInt = _accelerator.Allocate1D<int>(scratchIntLen);
        _dScratchShort = _accelerator.Allocate1D<short>(scratchShortLen);
    }

    private void DisposeFrameBuffers()
    {
        _dSrc?.Dispose(); _dSrc = null;
        _dRecon?.Dispose(); _dRecon = null;
        _dTile?.Dispose(); _dTile = null;
        _dTileLen?.Dispose(); _dTileLen = null;
        _dScratchByte?.Dispose(); _dScratchByte = null;
        _dScratchInt?.Dispose(); _dScratchInt = null;
        _dScratchShort?.Dispose(); _dScratchShort = null;
        _cachedWidth = -1;
        _cachedHeight = -1;
    }

    /// <summary>
    /// Encode a full AV1 v1 keyframe (TD OBU + SH OBU + Frame OBU) and
    /// return the complete byte stream. The codec-data bits (entropy,
    /// transforms, quantization, recon) come from the GPU walker; the
    /// OBU framing (TD + SH + Frame OBU header bytes) is produced by
    /// Av1ObuWriter and the existing Av1KeyframeEncoder helpers, which
    /// only do metadata struct setup + bit packing of fixed config
    /// (allowed under the CARDINAL rule's "metadata struct setup"
    /// allowance).
    ///
    /// Output is bit-exact against Av1KeyframeEncoder.EncodeKeyFrame.
    /// </summary>
    public async Task<byte[]> EncodeKeyFrameAsync(
        byte[] yPlane, byte[] uPlane, byte[] vPlane,
        int width, int height,
        int baseQIndex = 32)
    {
        // 1. GPU produces the entropy-coded tile bytes (the codec-data part).
        byte[] tileBytes = await EncodeSingleTileAsync(
            yPlane, uPlane, vPlane, width, height, baseQIndex);

        // 2. Use the existing CPU Av1KeyframeEncoder helpers for OBU framing.
        //    These produce TD OBU + SH OBU + Frame OBU(uncompressed_header + tileBytes).
        //    Av1KeyframeEncoder.EncodeKeyFrame is the reference; we replicate it
        //    here so we can substitute the GPU tile bytes.
        return Av1KeyframeEncoder.EncodeKeyFrameWithExternalTile(
            width, height, baseQIndex, tileBytes);
    }

    /// <summary>
    /// Frame-batch parallel AV1 keyframe encoder. Per-frame slots through
    /// the kernel; extent=N dispatch runs all frames concurrently. Each
    /// thread has its own range encoder state, scratch buffers, and tile
    /// output slot.
    /// </summary>
    public async Task<byte[][]> EncodeKeyFramesBatchAsync(
        ReadOnlyMemory<byte>[] yPlanes,
        ReadOnlyMemory<byte>[] uPlanes,
        ReadOnlyMemory<byte>[] vPlanes,
        int width, int height,
        int baseQIndex = 32)
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

        int yLen = width * height;
        int uvLen = yLen / 4;
        int srcLen = yLen + uvLen + uvLen;

        int frameMiCols = ((width + 7) >> 3) << 1;
        int frameMiRows = ((height + 7) >> 3) << 1;

        var p = new Av1FrameSeqEncodeParams
        {
            Width = width,
            Height = height,
            BaseQIndex = baseQIndex,
            YPlaneOff = 0,
            UPlaneOff = yLen,
            VPlaneOff = yLen + uvLen,
            FrameMiCols = frameMiCols,
            FrameMiRows = frameMiRows,
        };
        int byteOff = 0;
        p.AboveEntropyOff = byteOff; byteOff += 3 * frameMiCols;
        p.LeftEntropyOff = byteOff;  byteOff += 3 * 32;
        p.AbovePartOff = byteOff;    byteOff += frameMiCols;
        p.LeftPartOff = byteOff;     byteOff += 32;
        p.AboveYModeOff = byteOff;   byteOff += frameMiCols;
        p.LeftYModeOff = byteOff;    byteOff += 32;
        p.AboveSkipOff = byteOff;    byteOff += frameMiCols;
        p.LeftSkipOff = byteOff;     byteOff += 32;
        p.EdgeAboveOff = byteOff;    byteOff += 33;
        p.EdgeLeftOff = byteOff;     byteOff += 33;
        p.PredictOff = byteOff;      byteOff += 256;
        p.LevelsOff = byteOff;       byteOff += 1384;
        int scratchByteLen = byteOff;
        int scratchIntLen = Av1FrameSequentialEncodeKernel.MinScratchIntLength;
        int scratchShortLen = 256;

        int leaves = (width >> 4) * (height >> 4);
        int worstCaseTile = leaves * 2048 + 256;

        // Per-frame slot buffers.
        using var dAllSrc = _accelerator.Allocate1D<byte>((long)frameCount * srcLen);
        using var dAllRecon = _accelerator.Allocate1D<byte>((long)frameCount * srcLen);
        using var dAllTile = _accelerator.Allocate1D<byte>((long)frameCount * worstCaseTile);
        using var dAllTileLen = _accelerator.Allocate1D<long>(frameCount);
        using var dAllScratchByte = _accelerator.Allocate1D<byte>((long)frameCount * scratchByteLen);
        using var dAllScratchInt = _accelerator.Allocate1D<int>((long)frameCount * scratchIntLen);
        using var dAllScratchShort = _accelerator.Allocate1D<short>((long)frameCount * scratchShortLen);

        dAllTile.View.MemSetToZero();
        dAllTileLen.View.MemSetToZero();
        dAllScratchByte.View.MemSetToZero();
        dAllScratchInt.View.MemSetToZero();
        dAllScratchShort.View.MemSetToZero();

        // Bulk upload all frames' Y/U/V planes into per-frame slots.
        var hostSrc = new byte[(long)frameCount * srcLen];
        for (int f = 0; f < frameCount; f++)
        {
            int baseOff = f * srcLen;
            yPlanes[f].Span.CopyTo(hostSrc.AsSpan(baseOff));
            uPlanes[f].Span.CopyTo(hostSrc.AsSpan(baseOff + yLen));
            vPlanes[f].Span.CopyTo(hostSrc.AsSpan(baseOff + yLen + uvLen));
        }
        dAllSrc.View.CopyFromCPU(hostSrc);

        var strides = new Av1FrameBatchStrides
        {
            SrcStride = srcLen,
            ReconStride = srcLen,
            TileStride = worstCaseTile,
            ScratchByteStride = scratchByteLen,
            ScratchIntStride = scratchIntLen,
            ScratchShortStride = scratchShortLen,
        };

        // Single batch dispatch encodes all N frames in parallel.
        _frameKernel.RunBatch(
            dAllSrc.View, dAllRecon.View,
            dAllTile.View, dAllTileLen.View,
            _dByteConsts.View, _dUshortConsts.View, _dDcAcQuant.View,
            dAllScratchByte.View, dAllScratchInt.View, dAllScratchShort.View,
            p, frameCount, strides);

        await _accelerator.SynchronizeAsync();
        // Read lengths first; partial-readback only the actual tile bytes
        // per frame.
        var tileLensHost = await dAllTileLen.CopyToHostAsync();
        var results = new byte[frameCount][];
        for (int f = 0; f < frameCount; f++)
        {
            int tileLen = (int)tileLensHost[f];
            byte[] tileSlot = tileLen > 0
                ? await dAllTile.View.SubView((long)f * worstCaseTile, tileLen).CopyToHostAsync()
                : Array.Empty<byte>();
            results[f] = Av1KeyframeEncoder.EncodeKeyFrameWithExternalTile(
                width, height, baseQIndex, tileSlot);
        }
        return results;
    }

    /// <summary>Release every resource the encoder owns.</summary>
    public void Dispose()
    {
        _frameKernel.Dispose();
        _dByteConsts.Dispose();
        _dUshortConsts.Dispose();
        _dDcAcQuant.Dispose();
        DisposeFrameBuffers();
    }
}
