// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 v1 keyframe encoder, GPU integration class. Symmetric to Vp8KeyframeEncoderGpu / Vp9KeyframeEncoderGpu. The
// host is a pure coordinator: alloc + upload + dispatch + readback.
//
//   Av1FrameSequentialEncodeKernel : one GPU thread per frame runs the EncodeSingleTile pipeline (predict, transform,
//                                    quantize, entropy code) bit-exact vs the CPU Av1KeyframeEncoder reference, in
//                                    bounded resumable launches of LeavesPerLaunch 16x16 leaves.
//   Av1ObuAssembleKernel           : writes the Temporal Delimiter + Sequence Header OBUs, the Frame OBU header, the
//                                    leb128 size (which depends on the GPU-produced tile length), the uncompressed
//                                    frame header and the tile bytes into the caller's output. The only host-built
//                                    bytes are a small fixed blob (TD + SH + frame header, under 128 bytes of metadata).
//
// API shape (same as the VP8 / VP9 encoders): three overloads per entry point, all funnelling into the ArrayView1D one.
//   - ArrayView1D : planes already on the accelerator, temporal unit assembled into a caller-owned view, returns the length.
//   - TypedArray  : browser. JS planes in via CopyFromJS, JS Uint8Array out; nothing enters the .NET heap.
//   - ReadOnlyMemory<byte> : managed. Desktop and small inputs; a terminal CPU sink/source.
//
// V1 limits (named, not hidden): width and height must be multiples of 64 (boundary superblocks need the walker's
// forced-split syntax: tracked as task #23); 8-bit 4:2:0; DC prediction only; no loop filter / CDEF / restoration.

using System.Runtime.Versioning;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>
/// AV1 v1 keyframe encoder integration class.
/// </summary>
/// <remarks>Not thread-safe: one in-flight encode per instance.</remarks>
public sealed class Av1KeyframeEncoderGpu : IDisposable
{
    private const int BlobCapacity = 1024;

    private readonly Accelerator _accelerator;
    private readonly Av1FrameSequentialEncodeKernel _frameKernel;
    private readonly Av1ObuAssembleKernel _obu;

    private readonly MemoryBuffer1D<byte, global::ILGPU.Stride1D.Dense> _dByteConsts;
    private readonly MemoryBuffer1D<ushort, global::ILGPU.Stride1D.Dense> _dUshortConsts;
    private readonly MemoryBuffer1D<short, global::ILGPU.Stride1D.Dense> _dDcAcQuant;
    private readonly MemoryBuffer1D<byte, global::ILGPU.Stride1D.Dense> _blob;
    private readonly MemoryBuffer1D<int, global::ILGPU.Stride1D.Dense> _outLen;

    // Per-resolution cached buffers - reallocated only when (width,height) changes.
    private int _cachedWidth = -1;
    private int _cachedHeight = -1;
    private MemoryBuffer1D<byte, global::ILGPU.Stride1D.Dense>? _dSrc;
    private MemoryBuffer1D<byte, global::ILGPU.Stride1D.Dense>? _dRecon;
    private MemoryBuffer1D<byte, global::ILGPU.Stride1D.Dense>? _dTile;
    private MemoryBuffer1D<long, global::ILGPU.Stride1D.Dense>? _dTileLen;
    private MemoryBuffer1D<byte, global::ILGPU.Stride1D.Dense>? _dScratchByte;
    private MemoryBuffer1D<int, global::ILGPU.Stride1D.Dense>? _dScratchInt;
    private MemoryBuffer1D<short, global::ILGPU.Stride1D.Dense>? _dScratchShort;

    // Grow-only staging for the managed and TypedArray overloads.
    private readonly StageBuffer<byte> _stageY;
    private readonly StageBuffer<byte> _stageU;
    private readonly StageBuffer<byte> _stageV;
    private readonly StageBuffer<byte> _stageOut;
    private readonly StageBuffer<int> _stageLens;

    /// <summary>Construct an encoder bound to <paramref name="accelerator"/>. Uploads the constant tables once.</summary>
    public Av1KeyframeEncoderGpu(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;

        _frameKernel = new Av1FrameSequentialEncodeKernel(accelerator);
        _obu = new Av1ObuAssembleKernel(accelerator);

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

        _blob = accelerator.Allocate1D<byte>(BlobCapacity);
        _outLen = accelerator.Allocate1D<int>(1);
        _stageY = new StageBuffer<byte>(accelerator);
        _stageU = new StageBuffer<byte>(accelerator);
        _stageV = new StageBuffer<byte>(accelerator);
        _stageOut = new StageBuffer<byte>(accelerator);
        _stageLens = new StageBuffer<int>(accelerator);
    }

    // ------------------------------------------------------------------
    // Geometry
    // ------------------------------------------------------------------

    private static void RequireSupportedSize(int width, int height)
    {
        if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width));
        if (height <= 0) throw new ArgumentOutOfRangeException(nameof(height));
        // Spec-compliant boundary forced-split at non-aligned dims (e.g. 1920x1080) is tracked as task #23.
        if ((width & 63) != 0 || (height & 63) != 0)
            throw new NotSupportedException(
                $"AV1 v1 GPU encoder requires width and height to be multiples of 64; got {width}x{height}. " +
                "Non-aligned support requires walker boundary forced-split work (tracked task #23).");
    }

    private static long WorstCaseTile(int width, int height) => (long)(width >> 4) * (height >> 4) * 2048L + 256L;

    /// <summary>
    /// Size in bytes of the output buffer one encoded keyframe needs for a <paramref name="width"/> x
    /// <paramref name="height"/> picture (the batch overload strides its slots by it).
    /// </summary>
    public static int GetMaxEncodedSize(int width, int height)
    {
        RequireSupportedSize(width, height);
        return checked((int)(WorstCaseTile(width, height) + BlobCapacity + 16));
    }

    /// <summary>The walker's scratch layout for a frame size (offsets go into the kernel params).</summary>
    private static Av1FrameSeqEncodeParams BuildParams(
        int width, int height, int baseQIndex, out int scratchByteLen, out int scratchIntLen, out int scratchShortLen)
    {
        int yLen = width * height;
        int uvLen = yLen / 4;
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
        scratchByteLen = byteOff;
        scratchIntLen = Av1FrameSequentialEncodeKernel.MinScratchIntLength;
        scratchShortLen = 256;
        return p;
    }

    private static void ValidatePlanes(
        ArrayView1D<byte, global::ILGPU.Stride1D.Dense> y, ArrayView1D<byte, global::ILGPU.Stride1D.Dense> u,
        ArrayView1D<byte, global::ILGPU.Stride1D.Dense> v, int width, int height, int baseQIndex, long frames = 1)
    {
        RequireSupportedSize(width, height);
        if (baseQIndex <= 0 || baseQIndex > 255)
            throw new ArgumentOutOfRangeException(nameof(baseQIndex), "baseQIndex must be in [1, 255].");
        long yLen = (long)width * height;
        if (y.Length < frames * yLen) throw new ArgumentException("yPlane too short.", nameof(y));
        if (u.Length < frames * (yLen / 4)) throw new ArgumentException("uPlane too short.", nameof(u));
        if (v.Length < frames * (yLen / 4)) throw new ArgumentException("vPlane too short.", nameof(v));
    }

    // ------------------------------------------------------------------
    // ArrayView1D overloads: the implementation
    // ------------------------------------------------------------------

    /// <summary>
    /// Encode a keyframe from tightly packed YUV 4:2:0 planes that are already on the accelerator into a
    /// complete AV1 temporal unit (TD + SH + Frame OBU) in <paramref name="output"/> (at least
    /// <see cref="GetMaxEncodedSize"/> bytes). Only the length scalar crosses to the host. Returns the length.
    /// </summary>
    /// <param name="yPlane">Luma plane, <c>width * height</c> bytes.</param>
    /// <param name="uPlane">Cb plane, <c>width * height / 4</c> bytes.</param>
    /// <param name="vPlane">Cr plane, <c>width * height / 4</c> bytes.</param>
    /// <param name="width">Picture width, a multiple of 64.</param>
    /// <param name="height">Picture height, a multiple of 64.</param>
    /// <param name="output">Receives the temporal unit.</param>
    /// <param name="baseQIndex">AV1 base quantizer index, 1..255.</param>
    public async Task<int> EncodeKeyFrameAsync(
        ArrayView1D<byte, global::ILGPU.Stride1D.Dense> yPlane,
        ArrayView1D<byte, global::ILGPU.Stride1D.Dense> uPlane,
        ArrayView1D<byte, global::ILGPU.Stride1D.Dense> vPlane,
        int width, int height,
        ArrayView1D<byte, global::ILGPU.Stride1D.Dense> output,
        int baseQIndex = 32)
    {
        ValidatePlanes(yPlane, uPlane, vPlane, width, height, baseQIndex);
        if (output.Length < GetMaxEncodedSize(width, height))
            throw new ArgumentException($"output must be at least GetMaxEncodedSize({width}, {height}) bytes.", nameof(output));

        await RunTileAsync(yPlane, uPlane, vPlane, width, height, baseQIndex);
        AssembleOne(width, height, baseQIndex, _dTile!.View, _dTileLen!.View, output, _outLen.View, (int)WorstCaseTile(width, height));
        await _accelerator.SynchronizeAsync();
        return (await _outLen.CopyToHostAsync())[0];
    }

    /// <summary>Dispatches the OBU assembly for one frame (blob upload + kernel). No synchronization.</summary>
    private void AssembleOne(
        int width, int height, int baseQIndex,
        ArrayView1D<byte, global::ILGPU.Stride1D.Dense> tile, ArrayView1D<long, global::ILGPU.Stride1D.Dense> tileLen,
        ArrayView1D<byte, global::ILGPU.Stride1D.Dense> output, ArrayView1D<int, global::ILGPU.Stride1D.Dense> outLen,
        int maxTile)
    {
        byte[] blob = Av1KeyframeFraming.BuildFramingBlob(width, height, baseQIndex, out int aLen, out int bLen);
        if (blob.Length > BlobCapacity) throw new InvalidOperationException("AV1 framing blob exceeds its buffer (internal error).");
        DeviceTransfer.Upload(_blob.View, blob);
        _obu.Run(_blob.View, tile, tileLen, output, outLen, aLen, bLen, aLen + 1 + 5 + bLen + maxTile);
    }

    /// <summary>
    /// Runs the walker on the planes and leaves the raw range-coder tile in <c>_dTile</c> (length in
    /// <c>_dTileLen</c>) and the recon planes in <c>_dRecon</c>. Drains the stream before returning.
    /// </summary>
    private async Task RunTileAsync(
        ArrayView1D<byte, global::ILGPU.Stride1D.Dense> yPlane,
        ArrayView1D<byte, global::ILGPU.Stride1D.Dense> uPlane,
        ArrayView1D<byte, global::ILGPU.Stride1D.Dense> vPlane,
        int width, int height, int baseQIndex)
    {
        int yLen = width * height;
        int uvLen = yLen / 4;
        int srcLen = yLen + uvLen + uvLen;
        var p = BuildParams(width, height, baseQIndex, out int scratchByteLen, out int scratchIntLen, out int scratchShortLen);
        long worstCaseTile = WorstCaseTile(width, height);
        EnsureBuffers(width, height, srcLen, worstCaseTile, scratchByteLen, scratchIntLen, scratchShortLen);

        // Planes -> the walker's contiguous Y|U|V source, device to device.
        _dSrc!.View.SubView(0, yLen).CopyFrom(yPlane.SubView(0, yLen));
        _dSrc.View.SubView(yLen, uvLen).CopyFrom(uPlane.SubView(0, uvLen));
        _dSrc.View.SubView(yLen + uvLen, uvLen).CopyFrom(vPlane.SubView(0, uvLen));

        // Pre-zero output buffers per frame (kernel reads them as carry-back state; the walker zeroes its own
        // scratchByte / scratchInt). Recon is fully
        // overwritten per pixel so skip its zero-fill.
        _dTile!.View.MemSetToZero();
        _dTileLen!.View.MemSetToZero();
        _dScratchShort!.View.MemSetToZero();

        await _frameKernel.RunAsync(
            _dSrc.View, _dRecon!.View,
            _dTile.View, _dTileLen.View,
            _dByteConsts.View, _dUshortConsts.View,
            _dDcAcQuant.View,
            _dScratchByte.View, _dScratchInt.View, _dScratchShort.View,
            p);
        await _accelerator.SynchronizeAsync();
    }

    /// <summary>
    /// Frame-batch parallel AV1 keyframe encoder for frames already on the accelerator. Per-frame slots through the
    /// kernel; extent=N dispatch runs all frames concurrently (each thread has its own range encoder state, scratch
    /// buffers and tile slot). Only the per-frame lengths (written to <paramref name="lengths"/>) are produced.
    /// </summary>
    /// <param name="yPlanes">Frame-major tightly packed luma planes: frame f at <c>f * width * height</c>.</param>
    /// <param name="uPlanes">Frame-major tightly packed Cb planes: frame f at <c>f * width * height / 4</c>.</param>
    /// <param name="vPlanes">Frame-major tightly packed Cr planes, same layout as <paramref name="uPlanes"/>.</param>
    /// <param name="frameCount">Number of frames.</param>
    /// <param name="width">Picture width, a multiple of 64.</param>
    /// <param name="height">Picture height, a multiple of 64.</param>
    /// <param name="outputs">Frame f's temporal unit is written at <c>f * GetMaxEncodedSize(width, height)</c>.</param>
    /// <param name="lengths">Receives each frame's length; at least <paramref name="frameCount"/> ints.</param>
    /// <param name="baseQIndex">AV1 base quantizer index, 1..255.</param>
    public async Task EncodeKeyFramesBatchAsync(
        ArrayView1D<byte, global::ILGPU.Stride1D.Dense> yPlanes,
        ArrayView1D<byte, global::ILGPU.Stride1D.Dense> uPlanes,
        ArrayView1D<byte, global::ILGPU.Stride1D.Dense> vPlanes,
        int frameCount, int width, int height,
        ArrayView1D<byte, global::ILGPU.Stride1D.Dense> outputs,
        ArrayView1D<int, global::ILGPU.Stride1D.Dense> lengths,
        int baseQIndex = 32)
    {
        if (frameCount <= 0) throw new ArgumentOutOfRangeException(nameof(frameCount));
        ValidatePlanes(yPlanes, uPlanes, vPlanes, width, height, baseQIndex, frameCount);
        int cap = GetMaxEncodedSize(width, height);
        if (outputs.Length < (long)frameCount * cap)
            throw new ArgumentException("outputs must hold frameCount * GetMaxEncodedSize bytes.", nameof(outputs));
        if (lengths.Length < frameCount) throw new ArgumentException("lengths must hold frameCount ints.", nameof(lengths));

        int yLen = width * height;
        int uvLen = yLen / 4;
        int srcLen = yLen + uvLen + uvLen;
        var p = BuildParams(width, height, baseQIndex, out int scratchByteLen, out int scratchIntLen, out int scratchShortLen);
        int worstCaseTile = checked((int)WorstCaseTile(width, height));

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
        dAllScratchShort.View.MemSetToZero();

        // Planes -> per-frame Y|U|V slots, device to device.
        for (int f = 0; f < frameCount; f++)
        {
            long slot = (long)f * srcLen;
            dAllSrc.View.SubView(slot, yLen).CopyFrom(yPlanes.SubView((long)f * yLen, yLen));
            dAllSrc.View.SubView(slot + yLen, uvLen).CopyFrom(uPlanes.SubView((long)f * uvLen, uvLen));
            dAllSrc.View.SubView(slot + yLen + uvLen, uvLen).CopyFrom(vPlanes.SubView((long)f * uvLen, uvLen));
        }

        var strides = new Av1FrameBatchStrides
        {
            SrcStride = srcLen,
            ReconStride = srcLen,
            TileStride = worstCaseTile,
            ScratchByteStride = scratchByteLen,
            ScratchIntStride = scratchIntLen,
            ScratchShortStride = scratchShortLen,
        };

        // Batch launches encode all N frames in parallel (LeavesPerLaunch leaves of every frame per launch).
        await _frameKernel.RunBatchAsync(
            dAllSrc.View, dAllRecon.View,
            dAllTile.View, dAllTileLen.View,
            _dByteConsts.View, _dUshortConsts.View, _dDcAcQuant.View,
            dAllScratchByte.View, dAllScratchInt.View, dAllScratchShort.View,
            p, frameCount, strides);

        // One shared framing blob; one assemble dispatch per frame straight into the caller's slot (paced).
        byte[] blob = Av1KeyframeFraming.BuildFramingBlob(width, height, baseQIndex, out int aLen, out int bLen);
        DeviceTransfer.Upload(_blob.View, blob);
        var obuPacer = new LaunchPacer(_accelerator);
        for (int f = 0; f < frameCount; f++)
        {
            _obu.Run(
                _blob.View,
                dAllTile.View.SubView((long)f * worstCaseTile, worstCaseTile),
                dAllTileLen.View.SubView(f, 1),
                outputs.SubView((long)f * cap, cap),
                lengths.SubView(f, 1),
                aLen, bLen, aLen + 1 + 5 + bLen + worstCaseTile);
            await obuPacer.LaunchedAsync();
        }

        // Drain before the using-scoped slot buffers dispose: on the browser backends a dispatch that still
        // references a freed buffer reads freed memory.
        await _accelerator.SynchronizeAsync();
    }

    // ------------------------------------------------------------------
    // Managed overloads
    // ------------------------------------------------------------------

    private void StagePlanes(ReadOnlyMemory<byte> y, ReadOnlyMemory<byte> u, ReadOnlyMemory<byte> v, int width, int height)
    {
        RequireSupportedSize(width, height);
        long yLen = (long)width * height;
        DeviceTransfer.UploadPlane(_stageY, y, width, width, height, nameof(y));
        DeviceTransfer.UploadPlane(_stageU, u, width / 2, width / 2, height / 2, nameof(u));
        DeviceTransfer.UploadPlane(_stageV, v, width / 2, width / 2, height / 2, nameof(v));
        _stageOut.Ensure(GetMaxEncodedSize(width, height));
        _ = yLen;
    }

    /// <summary>
    /// Encode a full AV1 v1 keyframe (TD OBU + SH OBU + Frame OBU) from managed planes and return the byte stream
    /// as a managed array (a terminal CPU sink; in a browser prefer the <see cref="TypedArray"/> overload).
    /// </summary>
    public async Task<byte[]> EncodeKeyFrameAsync(
        ReadOnlyMemory<byte> yPlane, ReadOnlyMemory<byte> uPlane, ReadOnlyMemory<byte> vPlane,
        int width, int height,
        int baseQIndex = 32)
    {
        StagePlanes(yPlane, uPlane, vPlane, width, height);
        int n = await EncodeKeyFrameAsync(_stageY.View, _stageU.View, _stageV.View, width, height, _stageOut.View, baseQIndex);
        return await DeviceTransfer.DownloadAsync(_stageOut.View, n);
    }

    /// <summary>
    /// Run the GPU walker on managed planes and return the raw range-coder tile bytes (the bytes the CPU
    /// <c>Av1KeyframeEncoder.EncodeSingleTile</c> produces) - a test / diagnostic entry point.
    /// </summary>
    public async Task<byte[]> EncodeSingleTileAsync(
        ReadOnlyMemory<byte> yPlane, ReadOnlyMemory<byte> uPlane, ReadOnlyMemory<byte> vPlane,
        int width, int height,
        int baseQIndex = 32)
    {
        ValidateManagedQ(baseQIndex);
        StagePlanes(yPlane, uPlane, vPlane, width, height);
        await RunTileAsync(_stageY.View, _stageU.View, _stageV.View, width, height, baseQIndex);
        long tileLen = (await _dTileLen!.CopyToHostAsync())[0];
        return await DeviceTransfer.DownloadAsync(_dTile!.View, tileLen);
    }

    /// <summary>
    /// Like <see cref="EncodeSingleTileAsync"/> but also returns the encoder's internal recon planes - the data the
    /// downstream decoder must reconstruct (self-consistency tests of the encoder / decoder pair).
    /// </summary>
    public async Task<(byte[] tileBytes, byte[] yRecon, byte[] uRecon, byte[] vRecon)>
        EncodeSingleTileWithReconAsync(
        ReadOnlyMemory<byte> yPlane, ReadOnlyMemory<byte> uPlane, ReadOnlyMemory<byte> vPlane,
        int width, int height,
        int baseQIndex = 32)
    {
        var tile = await EncodeSingleTileAsync(yPlane, uPlane, vPlane, width, height, baseQIndex);
        int yLen = width * height, uvLen = yLen / 4;
        var y = await DeviceTransfer.DownloadAsync(_dRecon!.View.SubView(0, yLen), yLen);
        var u = await DeviceTransfer.DownloadAsync(_dRecon.View.SubView(yLen, uvLen), uvLen);
        var v = await DeviceTransfer.DownloadAsync(_dRecon.View.SubView(yLen + uvLen, uvLen), uvLen);
        return (tile, y, u, v);
    }

    /// <summary>
    /// Encode + return both the encoded keyframe bytes AND the encoder's internal recon planes (managed planes in,
    /// managed out; a diagnostic entry point).
    /// </summary>
    public async Task<(byte[] bytes, byte[] yRecon, byte[] uRecon, byte[] vRecon)>
        EncodeKeyFrameWithReconAsync(
        ReadOnlyMemory<byte> yPlane, ReadOnlyMemory<byte> uPlane, ReadOnlyMemory<byte> vPlane,
        int width, int height,
        int baseQIndex = 32)
    {
        ValidateManagedQ(baseQIndex);
        StagePlanes(yPlane, uPlane, vPlane, width, height);
        int n = await EncodeKeyFrameAsync(_stageY.View, _stageU.View, _stageV.View, width, height, _stageOut.View, baseQIndex);
        var bytes = await DeviceTransfer.DownloadAsync(_stageOut.View, n);
        int yLen = width * height, uvLen = yLen / 4;
        var y = await DeviceTransfer.DownloadAsync(_dRecon!.View.SubView(0, yLen), yLen);
        var u = await DeviceTransfer.DownloadAsync(_dRecon.View.SubView(yLen, uvLen), uvLen);
        var v = await DeviceTransfer.DownloadAsync(_dRecon.View.SubView(yLen + uvLen, uvLen), uvLen);
        return (bytes, y, u, v);
    }

    private static void ValidateManagedQ(int baseQIndex)
    {
        if (baseQIndex <= 0 || baseQIndex > 255)
            throw new ArgumentOutOfRangeException(nameof(baseQIndex), "baseQIndex must be in [1, 255].");
    }

    /// <summary>
    /// Batch-encode managed keyframes. Every plane is tightly packed. Each plane is uploaded straight into the
    /// device staging buffer (no host-side concatenation); each result is a complete temporal unit.
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
        if (frameCount == 0) return System.Array.Empty<byte[]>();
        RequireSupportedSize(width, height);

        long yPlane = (long)width * height;
        long uvPlane = yPlane / 4;
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
            _stageY.View, _stageU.View, _stageV.View, frameCount, width, height, _stageOut.View, _stageLens.View, baseQIndex);

        var lens = await _stageLens.View.SubView(0, frameCount).CopyToHostAsync();
        var results = new byte[frameCount][];
        for (int f = 0; f < frameCount; f++)
            results[f] = lens[f] > 0
                ? await DeviceTransfer.DownloadAsync(_stageOut.View.SubView((long)f * cap, lens[f]), lens[f])
                : System.Array.Empty<byte>();
        return results;
    }

    // ------------------------------------------------------------------
    // TypedArray overloads (browser: bulk bytes stay in JS)
    // ------------------------------------------------------------------

    /// <summary>
    /// Encode a keyframe from JS typed arrays (browser accelerators only). The planes go to the GPU with
    /// <c>CopyFromJS</c> and the temporal unit comes back as a JS <see cref="Uint8Array"/> (the caller disposes it);
    /// no bulk byte enters the .NET heap.
    /// </summary>
    /// <exception cref="NotSupportedException">The accelerator is not a browser accelerator, or the size is unsupported.</exception>
    [SupportedOSPlatform("browser")]
    public async Task<Uint8Array> EncodeKeyFrameAsync(
        TypedArray yPlane, TypedArray uPlane, TypedArray vPlane,
        int width, int height,
        int baseQIndex = 32)
    {
        ArgumentNullException.ThrowIfNull(yPlane);
        ArgumentNullException.ThrowIfNull(uPlane);
        ArgumentNullException.ThrowIfNull(vPlane);
        RequireSupportedSize(width, height);
        _stageY.Ensure(yPlane.ByteLength);
        _stageU.Ensure(uPlane.ByteLength);
        _stageV.Ensure(vPlane.ByteLength);
        _stageOut.Ensure(GetMaxEncodedSize(width, height));
        DeviceTransfer.Upload(_stageY.View, yPlane);
        DeviceTransfer.Upload(_stageU.View, uPlane);
        DeviceTransfer.Upload(_stageV.View, vPlane);
        int n = await EncodeKeyFrameAsync(_stageY.View, _stageU.View, _stageV.View, width, height, _stageOut.View, baseQIndex);
        return await DeviceTransfer.DownloadUint8ArrayAsync(_stageOut.View, n);
    }

    /// <summary>
    /// Batch-encode keyframes from JS typed arrays (browser accelerators only), one tightly packed plane per frame
    /// per component. Each returned <see cref="Uint8Array"/> is owned (and disposed) by the caller.
    /// </summary>
    [SupportedOSPlatform("browser")]
    public async Task<Uint8Array[]> EncodeKeyFramesBatchAsync(
        TypedArray[] yPlanes, TypedArray[] uPlanes, TypedArray[] vPlanes,
        int width, int height,
        int baseQIndex = 32)
    {
        if (yPlanes is null) throw new ArgumentNullException(nameof(yPlanes));
        if (uPlanes is null) throw new ArgumentNullException(nameof(uPlanes));
        if (vPlanes is null) throw new ArgumentNullException(nameof(vPlanes));
        if (yPlanes.Length != uPlanes.Length || yPlanes.Length != vPlanes.Length)
            throw new ArgumentException("Plane arrays must have equal length.");
        int frameCount = yPlanes.Length;
        if (frameCount == 0) return System.Array.Empty<Uint8Array>();
        RequireSupportedSize(width, height);

        long yPlane = (long)width * height;
        long uvPlane = yPlane / 4;
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
            _stageY.View, _stageU.View, _stageV.View, frameCount, width, height, _stageOut.View, _stageLens.View, baseQIndex);

        var lens = await _stageLens.View.SubView(0, frameCount).CopyToHostAsync();
        var results = new Uint8Array[frameCount];
        for (int f = 0; f < frameCount; f++)
            results[f] = await DeviceTransfer.DownloadUint8ArrayAsync(_stageOut.View.SubView((long)f * cap, lens[f]), lens[f]);
        return results;
    }

    // ------------------------------------------------------------------
    // Working buffers
    // ------------------------------------------------------------------

    /// <summary>Ensure per-frame GPU buffers are sized for (width,height). Reallocates only when the resolution changes.</summary>
    private void EnsureBuffers(int width, int height, int srcLen, long worstCaseTile,
        int scratchByteLen, int scratchIntLen, int scratchShortLen)
    {
        if (_cachedWidth == width && _cachedHeight == height) return;
        DisposeFrameBuffers();
        _cachedWidth = width;
        _cachedHeight = height;
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

    /// <summary>Release every resource the encoder owns.</summary>
    public void Dispose()
    {
        _frameKernel.Dispose();
        _obu.Dispose();
        _dByteConsts.Dispose();
        _dUshortConsts.Dispose();
        _dDcAcQuant.Dispose();
        _blob.Dispose();
        _outLen.Dispose();
        DisposeFrameBuffers();
        _stageY.Dispose();
        _stageU.Dispose();
        _stageV.Dispose();
        _stageOut.Dispose();
        _stageLens.Dispose();
    }
}
