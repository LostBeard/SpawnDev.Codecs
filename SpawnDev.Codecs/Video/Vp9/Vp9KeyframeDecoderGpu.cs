// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 v1 keyframe decoder, integration class. 100% ILGPU - the host
// is a pure coordinator: parse the uncompressed header on CPU
// (metadata extraction only - allowed under the cardinal rule),
// upload the encoded frame bytes, dispatch the kernel chain, read
// back the recon planes.
//
// API shape (same as Vp8KeyframeDecoderGpu): three overloads, all funnelling into the
// ArrayView1D one.
//   - ArrayView1D : encoded frame already on the accelerator, planes written into the caller's
//                   Y/U/V views. Only the uncompressed header bytes cross to the host.
//   - TypedArray  : browser. JS bytes in, JS Uint8Array planes out; nothing enters the .NET heap.
//   - ReadOnlyMemory<byte> : managed. Desktop and small inputs; planes come back as byte[].
//
// Kernel chain:
//   1. Vp9DequantizerComputeKernel - compute Y/UV dequantizers from
//                                    baseQIndex extracted by host.
//   2. Vp9KeyframeDecodeKernel    - parse tile bool stream + walk
//                                    SBs in z-order + decode per-MB
//                                    skip + modes + coefs + recon.
//
// Host responsibilities (allowed):
//   - Parse uncompressed header (raw bits, structural metadata).
//   - Extract width / height / baseQIndex / first_partition_size /
//     uncompressed_header_size as scalar metadata.
//   - Allocate GPU buffers sized to the metadata.
//   - Upload encoded frame bytes.
//   - Dispatch the 2 kernels.
//
// KNOWN GAP (open, not hidden): the decode kernel walks a whole superblock grid and has no
// display-size parameters, so it cannot parse the forced-split partitions the encoder emits for a
// superblock that straddles the picture edge. Sizes that are not multiples of 64 are therefore
// rejected until the kernel learns the boundary syntax (a kernel feature, not a crop).

using System.Runtime.Versioning;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// Decoded VP9 frame: Y, U, V planes plus the frame dimensions.
/// </summary>
public readonly record struct Vp9DecodedFrame(
    byte[] YPlane, byte[] UPlane, byte[] VPlane,
    int Width, int Height);

/// <summary>
/// Decoded VP9 frame as JS typed arrays (browser). The caller disposes it; the planes never entered
/// the .NET heap.
/// </summary>
[SupportedOSPlatform("browser")]
public sealed class Vp9DecodedFrameJS : IDisposable
{
    /// <summary>Luma plane, <c>Width * Height</c> bytes.</summary>
    public Uint8Array YPlane { get; }
    /// <summary>Cb plane, <c>Width * Height / 4</c> bytes.</summary>
    public Uint8Array UPlane { get; }
    /// <summary>Cr plane, <c>Width * Height / 4</c> bytes.</summary>
    public Uint8Array VPlane { get; }
    /// <summary>Picture width in pixels.</summary>
    public int Width { get; }
    /// <summary>Picture height in pixels.</summary>
    public int Height { get; }

    internal Vp9DecodedFrameJS(Uint8Array y, Uint8Array u, Uint8Array v, int width, int height)
    {
        YPlane = y; UPlane = u; VPlane = v; Width = width; Height = height;
    }

    /// <summary>Dispose the three JS arrays.</summary>
    public void Dispose()
    {
        YPlane.Dispose();
        UPlane.Dispose();
        VPlane.Dispose();
    }
}

/// <summary>
/// VP9 v1 keyframe decoder integration class. Decodes a complete
/// VP9 frame produced by <see cref="Vp9KeyframeEncoderGpu"/> (or
/// the CPU <c>Vp9KeyframeEncoder</c> reference) and returns the recon planes.
/// </summary>
/// <remarks>
/// Not thread-safe: one in-flight decode per instance (the staging buffer is shared across calls).
/// </remarks>
public sealed class Vp9KeyframeDecoderGpu : IDisposable
{
    // The uncompressed header of a keyframe is a few tens of bytes; this is a generous upper bound
    // for the header-only readback (metadata, far below the "tiny scalar" ceiling for bulk data).
    private const int HeaderReadbackBytes = 256;

    private readonly Accelerator _accelerator;
    private readonly Vp9DequantizerComputeKernel _dequantKernel;
    private readonly Vp9KeyframeDecodeKernel _decodeKernel;

    private readonly MemoryBuffer1D<short, global::ILGPU.Stride1D.Dense> _dDcQLookup;
    private readonly MemoryBuffer1D<short, global::ILGPU.Stride1D.Dense> _dAcQLookup;
    private readonly MemoryBuffer1D<byte, global::ILGPU.Stride1D.Dense> _dByteConsts;
    private readonly MemoryBuffer1D<ushort, global::ILGPU.Stride1D.Dense> _dUshortConsts;

    // Grow-only staging for the encoded bytes of the managed and TypedArray overloads.
    private readonly StageBuffer<byte> _stageIn;

    /// <summary>
    /// Compile every kernel + upload one-time-cached constant tables.
    /// </summary>
    public Vp9KeyframeDecoderGpu(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;

        _dequantKernel = new Vp9DequantizerComputeKernel(accelerator);
        _decodeKernel = new Vp9KeyframeDecodeKernel(accelerator);
        _stageIn = new StageBuffer<byte>(accelerator);

        _dDcQLookup = accelerator.Allocate1D<short>(256);
        _dAcQLookup = accelerator.Allocate1D<short>(256);
        _dByteConsts = accelerator.Allocate1D<byte>(Vp9KeyframeConstantsGpu.ByteConstsTotalBytes);
        _dUshortConsts = accelerator.Allocate1D<ushort>(Vp9KeyframeConstantsGpu.UshortConstsTotalEntries);

        _dDcQLookup.View.CopyFromCPU(Vp9DequantizerComputeKernel.BuildDcQLookup());
        _dAcQLookup.View.CopyFromCPU(Vp9DequantizerComputeKernel.BuildAcQLookup());
        _dByteConsts.View.CopyFromCPU(Vp9KeyframeConstantsGpu.BuildByteConstsBuffer());
        _dUshortConsts.View.CopyFromCPU(Vp9KeyframeConstantsGpu.BuildUshortConstsBuffer());
    }

    /// <summary>Luma plane size in bytes for a decoded <paramref name="width"/> x <paramref name="height"/> frame.</summary>
    public static long GetLumaPlaneSize(int width, int height) => (long)width * height;

    /// <summary>Chroma plane size in bytes (each of Cb, Cr) for a decoded <paramref name="width"/> x <paramref name="height"/> frame.</summary>
    public static long GetChromaPlaneSize(int width, int height) => (long)width * height / 4;

    // ------------------------------------------------------------------
    // ArrayView1D overloads: the implementation
    // ------------------------------------------------------------------

    /// <summary>
    /// Read and parse the uncompressed header of an encoded keyframe that is on the accelerator
    /// (a header-sized readback). Use the returned size to allocate the output planes for
    /// <see cref="DecodeKeyFrameAsync(ArrayView1D{byte, Stride1D.Dense}, int, ArrayView1D{byte, Stride1D.Dense}, ArrayView1D{byte, Stride1D.Dense}, ArrayView1D{byte, Stride1D.Dense})"/>.
    /// </summary>
    public async Task<Vp9UncompressedHeader> ReadFrameHeaderAsync(ArrayView1D<byte, Stride1D.Dense> encoded)
    {
        long n = Math.Min(encoded.Length, HeaderReadbackBytes);
        if (n < 1) throw new ArgumentException("VP9 frame is empty.", nameof(encoded));
        byte[] header = await DeviceTransfer.DownloadAsync(encoded, n);
        return ParseKeyFrameHeader(header);
    }

    /// <summary>
    /// Decode a VP9 v1 keyframe that is already on the accelerator into caller-owned plane views.
    /// Only the uncompressed header crosses to the host.
    /// </summary>
    /// <param name="encoded">The encoded frame (may be longer than <paramref name="encodedLength"/>).</param>
    /// <param name="encodedLength">Number of valid bytes in <paramref name="encoded"/>.</param>
    /// <param name="yOut">Receives the luma plane; at least <see cref="GetLumaPlaneSize"/> bytes.</param>
    /// <param name="uOut">Receives the Cb plane; at least <see cref="GetChromaPlaneSize"/> bytes.</param>
    /// <param name="vOut">Receives the Cr plane; at least <see cref="GetChromaPlaneSize"/> bytes.</param>
    /// <returns>The parsed uncompressed header (width, height, base q index, ...).</returns>
    public async Task<Vp9UncompressedHeader> DecodeKeyFrameAsync(
        ArrayView1D<byte, Stride1D.Dense> encoded, int encodedLength,
        ArrayView1D<byte, Stride1D.Dense> yOut,
        ArrayView1D<byte, Stride1D.Dense> uOut,
        ArrayView1D<byte, Stride1D.Dense> vOut)
    {
        if (encodedLength <= 0 || encodedLength > encoded.Length)
            throw new ArgumentOutOfRangeException(nameof(encodedLength));
        var complete = await ReadFrameHeaderAsync(encoded.SubView(0, encodedLength));
        await DecodeCoreAsync(encoded, encodedLength, complete, yOut, uOut, vOut);
        return complete;
    }

    private static Vp9UncompressedHeader ParseKeyFrameHeader(ReadOnlySpan<byte> header)
    {
        // Host-side metadata extraction. The complete uncompressed header parser pulls scalar
        // fields (width, height, baseQ, first_partition_size, uncompressed_header_size) from raw
        // bits. No bool-coded data is touched on the CPU.
        var complete = Vp9CompleteUncompressedHeaderParser.Parse(header, refFrameSizes: null!);
        if (complete.FrameHeader.FrameType != Vp9FrameType.Key)
            throw new ArgumentException("v1 GPU decoder handles keyframes only.");
        int width = complete.FrameHeader.FrameWidth;
        int height = complete.FrameHeader.FrameHeight;
        if ((width & 63) != 0 || (height & 63) != 0)
            throw new ArgumentException(
                "v1 GPU decoder requires width + height multiples of 64 (the decode kernel does not " +
                "yet parse the forced-split partitions of a superblock that straddles the picture edge).");
        return complete;
    }

    private async Task DecodeCoreAsync(
        ArrayView1D<byte, Stride1D.Dense> encoded, int encodedLength, Vp9UncompressedHeader complete,
        ArrayView1D<byte, Stride1D.Dense> yOut,
        ArrayView1D<byte, Stride1D.Dense> uOut,
        ArrayView1D<byte, Stride1D.Dense> vOut)
    {
        int width = complete.FrameHeader.FrameWidth;
        int height = complete.FrameHeader.FrameHeight;
        int baseQIndex = complete.Quantization.BaseQIndex;
        int uncompressedHeaderSize = complete.UncompressedHeaderSizeBytes;
        int firstPartitionSize = complete.FirstPartitionSize;
        int tileStartOffset = uncompressedHeaderSize + firstPartitionSize;
        int tileLength = encodedLength - tileStartOffset;
        if (tileLength <= 0)
            throw new InvalidDataException("VP9 frame has no tile data.");

        long yBytes = GetLumaPlaneSize(width, height);
        long uvBytes = GetChromaPlaneSize(width, height);
        if (yOut.Length < yBytes) throw new ArgumentException($"yOut must hold {yBytes} bytes.", nameof(yOut));
        if (uOut.Length < uvBytes) throw new ArgumentException($"uOut must hold {uvBytes} bytes.", nameof(uOut));
        if (vOut.Length < uvBytes) throw new ArgumentException($"vOut must hold {uvBytes} bytes.", nameof(vOut));

        int mbCols = width >> 4;
        int mbRows = height >> 4;

        using var dDequant = _accelerator.Allocate1D<int>(4);

        // Pre-zero the recon planes via GPU-side memset (avoids host allocation + bus transfer of
        // zeros). The decode writes straight into the caller's views.
        var yTarget = yOut.SubView(0, yBytes);
        var uTarget = uOut.SubView(0, uvBytes);
        var vTarget = vOut.SubView(0, uvBytes);
        yTarget.MemSetToZero();
        uTarget.MemSetToZero();
        vTarget.MemSetToZero();

        // ---- 1. Compute dequantizers ----
        // V1: y_dc_delta / uv_dc_delta / uv_ac_delta = 0.
        _dequantKernel.Run(_dDcQLookup.View, _dAcQLookup.View, dDequant.View,
            baseQIndex, 0, 0, 0, 0);

        // ---- 2. Decode kernel ----
        // SubView passes the tile slice without copy - the kernel sees an ArrayView<byte> of
        // length tileLength starting at the tile's first byte, so it never reads the
        // uncompressed/compressed header bytes at [0..tileStartOffset).
        await _decodeKernel.RunAsync(
            encoded.SubView(tileStartOffset, tileLength),
            yTarget, uTarget, vTarget,
            dDequant.View,
            _dByteConsts.View, _dUshortConsts.View,
            tileLength, mbCols, mbRows);

        // Drain before dDequant disposes: on the browser backends a dispatch that still references
        // a freed buffer reads freed memory.
        await _accelerator.SynchronizeAsync();
    }

    // ------------------------------------------------------------------
    // Managed overload
    // ------------------------------------------------------------------

    /// <summary>
    /// Decode a single VP9 v1 keyframe from the supplied frame bytes.
    /// Returns the Y/U/V recon planes + frame dimensions (managed in, managed out: a terminal CPU
    /// source/sink; in a browser prefer the <see cref="TypedArray"/> overload).
    /// </summary>
    public async Task<Vp9DecodedFrame> DecodeKeyFrameAsync(ReadOnlyMemory<byte> frameBytes)
    {
        if (frameBytes.IsEmpty) throw new ArgumentException("VP9 frame is empty.", nameof(frameBytes));

        var complete = StageManaged(frameBytes);
        int width = complete.FrameHeader.FrameWidth;
        int height = complete.FrameHeader.FrameHeight;
        using var dY = _accelerator.Allocate1D<byte>(GetLumaPlaneSize(width, height));
        using var dU = _accelerator.Allocate1D<byte>(GetChromaPlaneSize(width, height));
        using var dV = _accelerator.Allocate1D<byte>(GetChromaPlaneSize(width, height));
        await DecodeCoreAsync(_stageIn.View, frameBytes.Length, complete, dY.View, dU.View, dV.View);

        // CopyToHostAsync returns fresh byte[] arrays sized to the GPU buffers; handed directly to
        // the result record.
        var yRecon = await DeviceTransfer.DownloadAsync(dY.View, dY.Length);
        var uRecon = await DeviceTransfer.DownloadAsync(dU.View, dU.Length);
        var vRecon = await DeviceTransfer.DownloadAsync(dV.View, dV.Length);
        return new Vp9DecodedFrame(yRecon, uRecon, vRecon, width, height);
    }

    /// <summary>Not async: it holds a span. Uploads the frame into the staging buffer and parses its header.</summary>
    private Vp9UncompressedHeader StageManaged(ReadOnlyMemory<byte> frameBytes)
    {
        var complete = ParseKeyFrameHeader(frameBytes.Span.Slice(0, (int)Math.Min(frameBytes.Length, HeaderReadbackBytes)));
        _stageIn.Ensure(frameBytes.Length);
        DeviceTransfer.Upload(_stageIn.View, frameBytes.Span);
        return complete;
    }

    // ------------------------------------------------------------------
    // TypedArray overload (browser: bulk bytes stay in JS)
    // ------------------------------------------------------------------

    /// <summary>
    /// Decode a VP9 v1 keyframe held in a JS typed array (browser accelerators only). The bytes go
    /// to the GPU with <c>CopyFromJS</c> and the planes come back as JS <see cref="Uint8Array"/>s;
    /// none of it enters the .NET heap. The caller disposes the result.
    /// </summary>
    /// <exception cref="NotSupportedException">The accelerator is not a browser accelerator.</exception>
    [SupportedOSPlatform("browser")]
    public async Task<Vp9DecodedFrameJS> DecodeKeyFrameAsync(TypedArray frameBytes)
    {
        ArgumentNullException.ThrowIfNull(frameBytes);
        long length = frameBytes.ByteLength;
        if (length < 1) throw new ArgumentException("VP9 frame is empty.", nameof(frameBytes));
        _stageIn.Ensure(length);
        DeviceTransfer.Upload(_stageIn.View, frameBytes);

        var complete = await ReadFrameHeaderAsync(_stageIn.View.SubView(0, length));
        int width = complete.FrameHeader.FrameWidth;
        int height = complete.FrameHeader.FrameHeight;
        using var dY = _accelerator.Allocate1D<byte>(GetLumaPlaneSize(width, height));
        using var dU = _accelerator.Allocate1D<byte>(GetChromaPlaneSize(width, height));
        using var dV = _accelerator.Allocate1D<byte>(GetChromaPlaneSize(width, height));
        await DecodeCoreAsync(_stageIn.View, (int)length, complete, dY.View, dU.View, dV.View);

        var y = await DeviceTransfer.DownloadUint8ArrayAsync(dY.View, dY.Length);
        var u = await DeviceTransfer.DownloadUint8ArrayAsync(dU.View, dU.Length);
        var v = await DeviceTransfer.DownloadUint8ArrayAsync(dV.View, dV.Length);
        return new Vp9DecodedFrameJS(y, u, v, width, height);
    }

    /// <summary>Release every resource the decoder owns.</summary>
    public void Dispose()
    {
        _dequantKernel.Dispose();
        _decodeKernel.Dispose();
        _dDcQLookup.Dispose();
        _dAcQLookup.Dispose();
        _dByteConsts.Dispose();
        _dUshortConsts.Dispose();
        _stageIn.Dispose();
    }
}
