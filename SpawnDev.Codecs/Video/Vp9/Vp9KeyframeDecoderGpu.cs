// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 v1 keyframe decoder, integration class. 100% ILGPU - the host
// is a pure coordinator: parse the uncompressed header on CPU
// (metadata extraction only - allowed under the cardinal rule),
// upload the encoded frame bytes, dispatch the kernel chain, read
// back the recon planes.
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
//   - Read back the 3 recon planes.

using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// Decoded VP9 frame: Y, U, V planes plus the frame dimensions.
/// </summary>
public readonly record struct Vp9DecodedFrame(
    byte[] YPlane, byte[] UPlane, byte[] VPlane,
    int Width, int Height);

/// <summary>
/// VP9 v1 keyframe decoder integration class. Decodes a complete
/// VP9 frame produced by <see cref="Vp9KeyframeEncoderGpu"/> (or
/// <see cref="Vp9KeyframeEncoder"/>) and returns the recon planes.
/// </summary>
public sealed class Vp9KeyframeDecoderGpu : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Vp9DequantizerComputeKernel _dequantKernel;
    private readonly Vp9KeyframeDecodeKernel _decodeKernel;

    private readonly MemoryBuffer1D<short, global::ILGPU.Stride1D.Dense> _dDcQLookup;
    private readonly MemoryBuffer1D<short, global::ILGPU.Stride1D.Dense> _dAcQLookup;
    private readonly MemoryBuffer1D<byte, global::ILGPU.Stride1D.Dense> _dByteConsts;
    private readonly MemoryBuffer1D<ushort, global::ILGPU.Stride1D.Dense> _dUshortConsts;

    /// <summary>
    /// Compile every kernel + upload one-time-cached constant tables.
    /// </summary>
    public Vp9KeyframeDecoderGpu(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;

        _dequantKernel = new Vp9DequantizerComputeKernel(accelerator);
        _decodeKernel = new Vp9KeyframeDecodeKernel(accelerator);

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
    /// Decode a single VP9 v1 keyframe from the supplied frame bytes.
    /// Returns the Y/U/V recon planes + frame dimensions.
    /// </summary>
    public async Task<Vp9DecodedFrame> DecodeKeyFrameAsync(byte[] frameBytes)
    {
        ArgumentNullException.ThrowIfNull(frameBytes);

        // Host-side metadata extraction. The complete uncompressed
        // header parser pulls scalar fields (width, height, baseQ,
        // first_partition_size, uncompressed_header_size) from raw
        // bits. No bool-coded data is touched on the CPU.
        var complete = Vp9CompleteUncompressedHeaderParser.Parse(
            frameBytes, refFrameSizes: null!);
        var header = complete.FrameHeader;

        if (header.FrameType != Vp9FrameType.Key)
            throw new ArgumentException("v1 GPU decoder handles keyframes only.", nameof(frameBytes));

        int width = header.FrameWidth;
        int height = header.FrameHeight;
        if ((width & 63) != 0 || (height & 63) != 0)
            throw new ArgumentException(
                "v1 GPU decoder requires width + height multiples of 64.",
                nameof(frameBytes));

        int baseQIndex = complete.Quantization.BaseQIndex;
        int uncompressedHeaderSize = complete.UncompressedHeaderSizeBytes;
        int firstPartitionSize = complete.FirstPartitionSize;
        int tileStartOffset = uncompressedHeaderSize + firstPartitionSize;
        int tileLength = frameBytes.Length - tileStartOffset;
        if (tileLength <= 0)
            throw new InvalidDataException("VP9 frame has no tile data.");

        int mbCols = width >> 4;
        int mbRows = height >> 4;
        int yLen = width * height;
        int uvLen = yLen / 4;

        // ---- Allocate GPU buffers ----
        // dFrame holds the full encoded frame, single bulk upload. The
        // decode kernel sees a SubView starting at tileStartOffset, so
        // it never reads the uncompressed/compressed header bytes
        // sitting at frame offset [0..tileStartOffset). Allocating the
        // full frame instead of just the tile costs a few extra bytes
        // (uncompressed header + compressed header are ~tens of bytes)
        // and eliminates the per-decode Array.Copy of codec data on
        // the host side per cardinal rule.
        using var dFrame = _accelerator.Allocate1D<byte>(frameBytes.Length);
        using var dYRecon = _accelerator.Allocate1D<byte>(yLen);
        using var dURecon = _accelerator.Allocate1D<byte>(uvLen);
        using var dVRecon = _accelerator.Allocate1D<byte>(uvLen);
        using var dDequant = _accelerator.Allocate1D<int>(4);

        // Single bulk upload of the entire encoded frame. No host-side
        // slice copy, no Array.Copy iteration on codec data.
        dFrame.View.CopyFromCPU(frameBytes);
        // Pre-zero the recon planes via GPU-side memset (avoids host
        // allocation + bus transfer of zeros).
        dYRecon.View.MemSetToZero();
        dURecon.View.MemSetToZero();
        dVRecon.View.MemSetToZero();

        // ---- 1. Compute dequantizers ----
        // V1: y_dc_delta / uv_dc_delta / uv_ac_delta = 0.
        _dequantKernel.Run(_dDcQLookup.View, _dAcQLookup.View, dDequant.View,
            baseQIndex, 0, 0, 0, 0);

        // ---- 2. Decode kernel ----
        // SubView passes the tile slice without copy - the kernel sees
        // an ArrayView<byte> of length tileLength starting at the tile's
        // first byte, exactly the contract dTile.View used to provide.
        _decodeKernel.Run(
            dFrame.View.SubView(tileStartOffset, tileLength),
            dYRecon.View, dURecon.View, dVRecon.View,
            dDequant.View,
            _dByteConsts.View, _dUshortConsts.View,
            tileLength, mbCols, mbRows);

        await _accelerator.SynchronizeAsync();

        // ---- 3. Read back recon ----
        // CopyToHostAsync returns fresh byte[] arrays sized to the GPU
        // buffers (yLen / uvLen / uvLen). No CPU iteration on codec data
        // needed - we hand them directly to the result record.
        var yRecon = await dYRecon.CopyToHostAsync();
        var uRecon = await dURecon.CopyToHostAsync();
        var vRecon = await dVRecon.CopyToHostAsync();

        return new Vp9DecodedFrame(yRecon, uRecon, vRecon, width, height);
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
    }
}
