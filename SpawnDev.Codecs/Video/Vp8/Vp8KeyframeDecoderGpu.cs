// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP8 keyframe decoder, 100% GPU-resident pipeline. Symmetric
// companion to Vp8KeyframeEncoderGpu. The host is a pure coordinator:
// allocates GPU buffers, uploads encoded bytes, dispatches the decode
// kernel, reads back recon planes. No CPU-side bool decoding, no
// CPU-side header parsing of the compressed body, no CPU-side
// per-MB iteration.
//
// API shape: three overloads, all funnelling into the ArrayView1D one (see Vp8KeyframeEncoderGpu).
//   - ArrayView1D : encoded frame already on the accelerator, decoded planes written into the
//                   caller's Y/U/V views. Only the 10-byte uncompressed header crosses to the host.
//   - TypedArray  : browser. JS bytes in, JS Uint8Array planes out; nothing enters the .NET heap.
//   - ReadOnlyMemory<byte> : managed. Desktop and small inputs; planes come back as byte[].
//
// v1 simplifications (matches what Vp8KeyframeEncoderGpu produces):
//   - All MBs Y_PRED = DC_PRED, UV_PRED = DC_PRED
//   - Single token partition (npart = 1)
//   - No segmentation, no loop filter, no skip-coef flag
//   - Default coef probs
//
// Host-side parses ONLY the 10-byte uncompressed VP8 frame tag to
// extract width/height/first-partition-size (those fields ARE plain bytes per
// RFC 6386 sec 9.1 - tag is uncompressed by spec; reading raw int
// fields out of a 10-byte buffer is metadata extraction, not
// codec-data processing). Everything else is decoded on the GPU.
//
// Any positive picture size is accepted. The kernels decode the whole macroblock grid; when the
// signalled size is not a multiple of 16 the grid is cropped to the signalled size on the
// accelerator (Vp8StridedPlanePackKernel), exactly the inverse of the encoder's pad.

using System.Runtime.Versioning;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>
/// VP8 keyframe decoder using the GPU-resident pipeline. Reverses
/// what <see cref="Vp8KeyframeEncoderGpu"/> produces back to YUV
/// recon planes.
/// </summary>
/// <remarks>
/// Not thread-safe: one in-flight decode per instance (the staging buffer is shared across calls).
/// </remarks>
public sealed class Vp8KeyframeDecoderGpu : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Vp8FrameSetupKernel _setup;
    private readonly Vp8KeyframeDecodeKernel _decode;
    private readonly Vp8StridedPlanePackKernel _crop;

    // Constants uploaded once per accelerator and reused.
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _dcQLookup;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _acQLookup;
    private readonly MemoryBuffer1D<byte, Stride1D.Dense> _defaultCoefProbs;
    private readonly MemoryBuffer1D<byte, Stride1D.Dense> _updateCoefProbs;
    private readonly MemoryBuffer1D<byte, Stride1D.Dense> _coefProbsByType;
    private readonly MemoryBuffer1D<byte, Stride1D.Dense> _constsExtended;

    // Grow-only staging for the encoded bytes of the managed and TypedArray overloads.
    private readonly StageBuffer<byte> _stageIn;

    /// <summary>Decoded keyframe result: recon planes + dimensions.</summary>
    public sealed record DecodedFrame(
        byte[] YPlane, byte[] UPlane, byte[] VPlane,
        int Width, int Height);

    /// <summary>
    /// Decoded keyframe as JS typed arrays (browser). The caller disposes it; the planes never
    /// entered the .NET heap.
    /// </summary>
    [SupportedOSPlatform("browser")]
    public sealed class DecodedFrameJS : IDisposable
    {
        /// <summary>Luma plane, <c>Width * Height</c> bytes.</summary>
        public Uint8Array YPlane { get; }
        /// <summary>Cb plane, <c>ChromaWidth * ChromaHeight</c> bytes.</summary>
        public Uint8Array UPlane { get; }
        /// <summary>Cr plane, <c>ChromaWidth * ChromaHeight</c> bytes.</summary>
        public Uint8Array VPlane { get; }
        /// <summary>Picture width in pixels.</summary>
        public int Width { get; }
        /// <summary>Picture height in pixels.</summary>
        public int Height { get; }

        internal DecodedFrameJS(Uint8Array y, Uint8Array u, Uint8Array v, int width, int height)
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

    /// <summary>Compile + cache kernels and constants onto <paramref name="accelerator"/>.</summary>
    public Vp8KeyframeDecoderGpu(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _setup = new Vp8FrameSetupKernel(accelerator);
        _decode = new Vp8KeyframeDecodeKernel(accelerator);
        _crop = new Vp8StridedPlanePackKernel(accelerator);
        _stageIn = new StageBuffer<byte>(accelerator);

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

    /// <summary>Luma plane size in bytes for a decoded <paramref name="width"/> x <paramref name="height"/> frame.</summary>
    public static long GetLumaPlaneSize(int width, int height) => (long)width * height;

    /// <summary>Chroma plane size in bytes (each of Cb, Cr) for a decoded <paramref name="width"/> x <paramref name="height"/> frame.</summary>
    public static long GetChromaPlaneSize(int width, int height) => (long)ChromaDim(width) * ChromaDim(height);

    private static int ChromaDim(int lumaDim) => (lumaDim + 1) / 2;

    // ------------------------------------------------------------------
    // ArrayView1D overloads: the implementation
    // ------------------------------------------------------------------

    /// <summary>
    /// Read the uncompressed frame tag of an encoded keyframe that is on the accelerator (a
    /// 10-byte readback). Use the returned size to allocate the output planes for
    /// <see cref="DecodeKeyFrameAsync(ArrayView1D{byte, Stride1D.Dense}, int, int, ArrayView1D{byte, Stride1D.Dense}, ArrayView1D{byte, Stride1D.Dense}, ArrayView1D{byte, Stride1D.Dense})"/>.
    /// </summary>
    public async Task<Vp8FrameTag> ReadFrameTagAsync(ArrayView1D<byte, Stride1D.Dense> encoded)
    {
        if (encoded.Length < 10)
            throw new ArgumentException("VP8 keyframe must be at least 10 bytes (uncompressed tag).", nameof(encoded));
        byte[] header = await DeviceTransfer.DownloadAsync(encoded, 10);
        return ParseKeyFrameTag(header);
    }

    /// <summary>
    /// Decode an encoded VP8 keyframe that is already on the accelerator into caller-owned plane
    /// views. Only the 10-byte header crosses to the host.
    /// </summary>
    /// <param name="encoded">The encoded keyframe (may be longer than <paramref name="encodedLength"/>).</param>
    /// <param name="encodedLength">Number of valid bytes in <paramref name="encoded"/>.</param>
    /// <param name="baseQIndex">The base quantizer index the frame was encoded with.</param>
    /// <param name="yOut">Receives the luma plane, tightly packed; at least <see cref="GetLumaPlaneSize"/> bytes.</param>
    /// <param name="uOut">Receives the Cb plane, tightly packed; at least <see cref="GetChromaPlaneSize"/> bytes.</param>
    /// <param name="vOut">Receives the Cr plane, tightly packed; at least <see cref="GetChromaPlaneSize"/> bytes.</param>
    /// <returns>The parsed frame tag (width, height, ...).</returns>
    public async Task<Vp8FrameTag> DecodeKeyFrameAsync(
        ArrayView1D<byte, Stride1D.Dense> encoded, int encodedLength, int baseQIndex,
        ArrayView1D<byte, Stride1D.Dense> yOut,
        ArrayView1D<byte, Stride1D.Dense> uOut,
        ArrayView1D<byte, Stride1D.Dense> vOut)
    {
        if (encodedLength < 10)
            throw new ArgumentException("VP8 keyframe must be at least 10 bytes (uncompressed tag).", nameof(encodedLength));
        if (encodedLength > encoded.Length)
            throw new ArgumentOutOfRangeException(nameof(encodedLength));
        var tag = await ReadFrameTagAsync(encoded);
        await DecodeCoreAsync(encoded, encodedLength, tag, baseQIndex, yOut, uOut, vOut);
        return tag;
    }

    private static Vp8FrameTag ParseKeyFrameTag(ReadOnlySpan<byte> header)
    {
        var tag = Vp8FrameTagParser.Parse(header);
        if (!tag.IsKeyFrame) throw new InvalidDataException("Not a key frame.");
        if (tag.Width is not > 0 || tag.Height is not > 0)
            throw new InvalidDataException("VP8 keyframe signals a zero-sized picture.");
        return tag;
    }

    private async Task DecodeCoreAsync(
        ArrayView1D<byte, Stride1D.Dense> encoded, int encodedLength, Vp8FrameTag tag, int baseQIndex,
        ArrayView1D<byte, Stride1D.Dense> yOut,
        ArrayView1D<byte, Stride1D.Dense> uOut,
        ArrayView1D<byte, Stride1D.Dense> vOut)
    {
        // Per RFC 6386 sec 9.1 these are plain packed integer fields - metadata extraction, not
        // codec-data processing.
        int width = tag.Width!.Value;
        int height = tag.Height!.Value;
        int workWidth = (width + 15) & ~15;
        int workHeight = (height + 15) & ~15;
        int mbCols = workWidth / 16;
        int mbRows = workHeight / 16;
        int uvWidth = ChromaDim(width);
        int uvHeight = ChromaDim(height);
        int uvWorkWidth = workWidth / 2;
        int uvWorkHeight = workHeight / 2;

        long yBytes = GetLumaPlaneSize(width, height);
        long uvBytes = GetChromaPlaneSize(width, height);
        if (yOut.Length < yBytes) throw new ArgumentException($"yOut must hold {yBytes} bytes.", nameof(yOut));
        if (uOut.Length < uvBytes) throw new ArgumentException($"uOut must hold {uvBytes} bytes.", nameof(uOut));
        if (vOut.Length < uvBytes) throw new ArgumentException($"vOut must hold {uvBytes} bytes.", nameof(vOut));

        int p0Offset = 10;
        int p0Len = tag.FirstPartitionSize;
        int tp0Offset = 10 + p0Len;
        int tp0Len = encodedLength - tp0Offset;
        if (tp0Len < 0)
            throw new InvalidDataException("First partition size exceeds frame length.");

        // Slice partition0 + tokenP0 are the same buffer at different ranges.
        // Pass the whole encoded buffer + ranges via streamRanges array.
        using var dDequant = _accelerator.Allocate1D<int>(6);
        using var dInitialP0State = _accelerator.Allocate1D<int>(5);
        // partition0Out is reused as a scratch for Setup's frame-header
        // bool emit; Decode doesn't read partition0Out, only the
        // subset of the encoded buffer at p0Offset..p0Offset+p0Len.
        // To keep the constant-table pipeline clean, we run Setup
        // anyway to compute dequantizers; the partition0Out it emits
        // is discarded for decode. (Setup is fast.)
        int p0SetupCapacity = 64 * 1024 + mbCols * mbRows * 32;
        using var dP0Setup = _accelerator.Allocate1D<byte>(p0SetupCapacity);
        dP0Setup.View.MemSetToZero();

        // A picture that is already macroblock-aligned decodes straight into the caller's planes.
        // Otherwise the grid is decoded into working planes and cropped on the accelerator.
        bool aligned = width == workWidth && height == workHeight;
        MemoryBuffer1D<byte, Stride1D.Dense>? wY = null, wU = null, wV = null;
        try
        {
            ArrayView1D<byte, Stride1D.Dense> yTarget, uTarget, vTarget;
            if (aligned)
            {
                yTarget = yOut.SubView(0, yBytes);
                uTarget = uOut.SubView(0, uvBytes);
                vTarget = vOut.SubView(0, uvBytes);
            }
            else
            {
                wY = _accelerator.Allocate1D<byte>((long)workWidth * workHeight);
                wU = _accelerator.Allocate1D<byte>((long)uvWorkWidth * uvWorkHeight);
                wV = _accelerator.Allocate1D<byte>((long)uvWorkWidth * uvWorkHeight);
                yTarget = wY.View; uTarget = wU.View; vTarget = wV.View;
            }
            yTarget.MemSetToZero();
            uTarget.MemSetToZero();
            vTarget.MemSetToZero();

            using var dAbove = _accelerator.Allocate1D<byte>(mbCols * 9);
            dAbove.View.MemSetToZero();

            using var dStreamRanges = _accelerator.Allocate1D<int>(4);
            dStreamRanges.View.CopyFromCPU(new int[] { p0Offset, p0Len, tp0Offset, tp0Len });

            // Step 1: Compute dequantizers. (Setup also writes header bits
            // into dP0Setup but we discard those - the encoded bytes
            // already contain the header.)
            _setup.Run(
                _dcQLookup.View, _acQLookup.View,
                _defaultCoefProbs.View, _updateCoefProbs.View,
                dDequant.View, dP0Setup.View, dInitialP0State.View,
                baseQIndex);

            // Step 2: Decode the frame.
            _decode.Run(
                encoded, encoded,
                _coefProbsByType.View, _constsExtended.View,
                yTarget, uTarget, vTarget,
                dDequant.View, dAbove.View, dStreamRanges.View,
                mbCols, mbRows);

            // Step 3: crop the macroblock grid to the signalled picture size.
            if (!aligned)
            {
                _crop.Run(wY!.View, 0, workWidth, yOut, 0, width, height);
                _crop.Run(wU!.View, 0, uvWorkWidth, uOut, 0, uvWidth, uvHeight);
                _crop.Run(wV!.View, 0, uvWorkWidth, vOut, 0, uvWidth, uvHeight);
            }

            // Drain before the scoped buffers dispose: on the browser backends a dispatch that still
            // references a freed buffer reads freed memory.
            await _accelerator.SynchronizeAsync();
        }
        finally
        {
            wY?.Dispose();
            wU?.Dispose();
            wV?.Dispose();
        }
    }

    // ------------------------------------------------------------------
    // Managed overload
    // ------------------------------------------------------------------

    /// <summary>
    /// Decode an encoded VP8 keyframe back to YUV recon planes (managed in, managed out: a
    /// terminal CPU source/sink; in a browser prefer the <see cref="TypedArray"/> overload).
    /// </summary>
    public async Task<DecodedFrame> DecodeKeyFrameAsync(ReadOnlyMemory<byte> encoded, int baseQIndex)
    {
        if (encoded.Length < 10)
            throw new ArgumentException("VP8 keyframe must be at least 10 bytes (uncompressed tag).", nameof(encoded));

        var tag = StageManaged(encoded);
        int width = tag.Width!.Value;
        int height = tag.Height!.Value;
        using var dY = _accelerator.Allocate1D<byte>(GetLumaPlaneSize(width, height));
        using var dU = _accelerator.Allocate1D<byte>(GetChromaPlaneSize(width, height));
        using var dV = _accelerator.Allocate1D<byte>(GetChromaPlaneSize(width, height));
        await DecodeCoreAsync(_stageIn.View, encoded.Length, tag, baseQIndex, dY.View, dU.View, dV.View);

        // Single readback per plane - the only host-side work besides
        // dispatch is moving the decoded recon back to host memory.
        var yPlane = await DeviceTransfer.DownloadAsync(dY.View, dY.Length);
        var uPlane = await DeviceTransfer.DownloadAsync(dU.View, dU.Length);
        var vPlane = await DeviceTransfer.DownloadAsync(dV.View, dV.Length);
        return new DecodedFrame(yPlane, uPlane, vPlane, width, height);
    }

    /// <summary>Not async: it holds a span. Uploads the frame into the staging buffer and parses the tag.</summary>
    private Vp8FrameTag StageManaged(ReadOnlyMemory<byte> encoded)
    {
        var tag = ParseKeyFrameTag(encoded.Span.Slice(0, 10));
        _stageIn.Ensure(encoded.Length);
        DeviceTransfer.Upload(_stageIn.View, encoded.Span);
        return tag;
    }

    // ------------------------------------------------------------------
    // TypedArray overload (browser: bulk bytes stay in JS)
    // ------------------------------------------------------------------

    /// <summary>
    /// Decode an encoded VP8 keyframe held in a JS typed array (browser accelerators only). The
    /// bytes go to the GPU with <c>CopyFromJS</c> and the planes come back as JS
    /// <see cref="Uint8Array"/>s; none of it enters the .NET heap. The caller disposes the result.
    /// </summary>
    /// <exception cref="NotSupportedException">The accelerator is not a browser accelerator.</exception>
    [SupportedOSPlatform("browser")]
    public async Task<DecodedFrameJS> DecodeKeyFrameAsync(TypedArray encoded, int baseQIndex)
    {
        ArgumentNullException.ThrowIfNull(encoded);
        long length = encoded.ByteLength;
        if (length < 10)
            throw new ArgumentException("VP8 keyframe must be at least 10 bytes (uncompressed tag).", nameof(encoded));
        _stageIn.Ensure(length);
        DeviceTransfer.Upload(_stageIn.View, encoded);

        var tag = await ReadFrameTagAsync(_stageIn.View);
        int width = tag.Width!.Value;
        int height = tag.Height!.Value;
        using var dY = _accelerator.Allocate1D<byte>(GetLumaPlaneSize(width, height));
        using var dU = _accelerator.Allocate1D<byte>(GetChromaPlaneSize(width, height));
        using var dV = _accelerator.Allocate1D<byte>(GetChromaPlaneSize(width, height));
        await DecodeCoreAsync(_stageIn.View, (int)length, tag, baseQIndex, dY.View, dU.View, dV.View);

        var y = await DeviceTransfer.DownloadUint8ArrayAsync(dY.View, dY.Length);
        var u = await DeviceTransfer.DownloadUint8ArrayAsync(dU.View, dU.Length);
        var v = await DeviceTransfer.DownloadUint8ArrayAsync(dV.View, dV.Length);
        return new DecodedFrameJS(y, u, v, width, height);
    }

    /// <summary>Release kernel resources and constant buffers.</summary>
    public void Dispose()
    {
        _setup.Dispose();
        _decode.Dispose();
        _crop.Dispose();
        _dcQLookup.Dispose();
        _acQLookup.Dispose();
        _defaultCoefProbs.Dispose();
        _updateCoefProbs.Dispose();
        _coefProbsByType.Dispose();
        _constsExtended.Dispose();
        _stageIn.Dispose();
    }
}
