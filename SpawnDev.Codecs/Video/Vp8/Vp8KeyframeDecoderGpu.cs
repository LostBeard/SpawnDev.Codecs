// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP8 keyframe decoder, 100% GPU-resident pipeline. Symmetric
// companion to Vp8KeyframeEncoderGpu. The host is a pure coordinator:
// allocates GPU buffers, uploads encoded bytes, dispatches the decode
// kernel, reads back recon planes. No CPU-side bool decoding, no
// CPU-side header parsing of the compressed body, no CPU-side
// per-MB iteration.
//
// v1 simplifications (matches what Vp8KeyframeEncoderGpu produces):
//   - All MBs Y_PRED = DC_PRED, UV_PRED = DC_PRED
//   - Single token partition (npart = 1)
//   - No segmentation, no loop filter, no skip-coef flag
//   - Default coef probs
//
// Host-side parses ONLY the 10-byte uncompressed VP8 frame tag to
// extract width/height/baseQIndex (those fields ARE plain bytes per
// RFC 6386 sec 9.1 - tag is uncompressed by spec; reading raw int
// fields out of a 10-byte buffer is metadata extraction, not
// codec-data processing). Everything else is decoded on the GPU.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>
/// VP8 keyframe decoder using the GPU-resident pipeline. Reverses
/// what <see cref="Vp8KeyframeEncoderGpu"/> produces back to YUV
/// recon planes.
/// </summary>
public sealed class Vp8KeyframeDecoderGpu : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Vp8FrameSetupKernel _setup;
    private readonly Vp8KeyframeDecodeKernel _decode;

    // Constants uploaded once per accelerator and reused.
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _dcQLookup;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _acQLookup;
    private readonly MemoryBuffer1D<byte, Stride1D.Dense> _defaultCoefProbs;
    private readonly MemoryBuffer1D<byte, Stride1D.Dense> _updateCoefProbs;
    private readonly MemoryBuffer1D<byte, Stride1D.Dense> _coefProbsByType;
    private readonly MemoryBuffer1D<byte, Stride1D.Dense> _constsExtended;

    /// <summary>Decoded keyframe result: recon planes + dimensions.</summary>
    public sealed record DecodedFrame(
        byte[] YPlane, byte[] UPlane, byte[] VPlane,
        int Width, int Height);

    /// <summary>Compile + cache kernels and constants onto <paramref name="accelerator"/>.</summary>
    public Vp8KeyframeDecoderGpu(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _setup = new Vp8FrameSetupKernel(accelerator);
        _decode = new Vp8KeyframeDecodeKernel(accelerator);

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

    /// <summary>Decode an encoded VP8 keyframe back to YUV recon planes.</summary>
    public DecodedFrame DecodeKeyFrame(ReadOnlySpan<byte> encoded, int baseQIndex)
    {
        if (encoded.Length < 10)
            throw new ArgumentException("VP8 keyframe must be at least 10 bytes (uncompressed tag).", nameof(encoded));

        // Parse the 10-byte uncompressed tag to extract width / height /
        // first_partition_size. Per RFC 6386 sec 9.1 these are plain
        // packed integer fields - metadata extraction, not codec-data
        // processing.
        uint tag0 = encoded[0]; uint tag1 = encoded[1]; uint tag2 = encoded[2];
        uint tagBits = tag0 | (tag1 << 8) | (tag2 << 16);
        bool isKeyFrame = (tagBits & 1u) == 0;
        if (!isKeyFrame) throw new InvalidDataException("Not a key frame.");
        int firstPartitionSize = (int)((tagBits >> 5) & 0x7FFFFu);
        // Bytes [3..6) are the start code; [6..8) horiz_size_code; [8..10) vert_size_code.
        if (encoded[3] != 0x9D || encoded[4] != 0x01 || encoded[5] != 0x2A)
            throw new InvalidDataException("Missing VP8 start code.");
        int horizSize = encoded[6] | (encoded[7] << 8);
        int vertSize = encoded[8] | (encoded[9] << 8);
        int width = horizSize & 0x3FFF;
        int height = vertSize & 0x3FFF;
        if ((width & 15) != 0 || (height & 15) != 0)
            throw new NotSupportedException("v1 supports multiple-of-16 dimensions only.");

        int mbCols = width / 16;
        int mbRows = height / 16;
        int uvWidth = width / 2;
        int uvHeight = height / 2;
        int p0Offset = 10;
        int p0Len = firstPartitionSize;
        int tp0Offset = 10 + p0Len;
        int tp0Len = encoded.Length - tp0Offset;
        if (tp0Len < 0)
            throw new InvalidDataException("First partition size exceeds frame length.");

        // GPU buffers.
        using var dEncoded = _accelerator.Allocate1D<byte>(encoded.Length);
        dEncoded.View.CopyFromCPU(encoded.ToArray());

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

        using var dYRecon = _accelerator.Allocate1D<byte>(width * height);
        using var dURecon = _accelerator.Allocate1D<byte>(uvWidth * uvHeight);
        using var dVRecon = _accelerator.Allocate1D<byte>(uvWidth * uvHeight);
        dYRecon.View.MemSetToZero();
        dURecon.View.MemSetToZero();
        dVRecon.View.MemSetToZero();

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
            dEncoded.View, dEncoded.View,
            _coefProbsByType.View, _constsExtended.View,
            dYRecon.View, dURecon.View, dVRecon.View,
            dDequant.View, dAbove.View, dStreamRanges.View,
            mbCols, mbRows);

        _accelerator.Synchronize();

        // Single readback per plane - the only host-side work besides
        // dispatch is moving the decoded recon back to host memory.
        var yPlane = dYRecon.GetAsArray1D();
        var uPlane = dURecon.GetAsArray1D();
        var vPlane = dVRecon.GetAsArray1D();
        return new DecodedFrame(yPlane, uPlane, vPlane, width, height);
    }

    /// <summary>Release kernel resources and constant buffers.</summary>
    public void Dispose()
    {
        _setup.Dispose();
        _decode.Dispose();
        _dcQLookup.Dispose();
        _acQLookup.Dispose();
        _defaultCoefProbs.Dispose();
        _updateCoefProbs.Dispose();
        _coefProbsByType.Dispose();
        _constsExtended.Dispose();
    }
}
