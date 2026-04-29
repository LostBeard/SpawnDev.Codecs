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
    public async Task<byte[]> EncodeKeyFrameAsync(
        byte[] yPlane, byte[] uPlane, byte[] vPlane,
        int width, int height,
        int baseQIndex = 30)
    {
        var (bytes, _, _, _) = await EncodeKeyFrameWithReconAsync(
            yPlane, uPlane, vPlane, width, height, baseQIndex);
        return bytes;
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

        // ---- 1. Allocate per-frame GPU buffers + upload sources ----
        using var dY = _accelerator.Allocate1D<byte>(yLen);
        using var dU = _accelerator.Allocate1D<byte>(uvLen);
        using var dV = _accelerator.Allocate1D<byte>(uvLen);
        using var dYRecon = _accelerator.Allocate1D<byte>(yLen);
        using var dURecon = _accelerator.Allocate1D<byte>(uvLen);
        using var dVRecon = _accelerator.Allocate1D<byte>(uvLen);
        using var dYCoefs = _accelerator.Allocate1D<short>((long)mbCount * 256);
        using var dUCoefs = _accelerator.Allocate1D<short>((long)mbCount * 64);
        using var dVCoefs = _accelerator.Allocate1D<short>((long)mbCount * 64);
        using var dDequant = _accelerator.Allocate1D<int>(4);

        // Worst-case output sizes. Compressed header is tiny (< 64
        // bytes); uncompressed header < 32 bytes. Tile data scales
        // with content entropy: random YUV bytes produce non-trivial
        // residuals after DC_PRED that quantize to many Cat3..Cat6
        // tokens. 1024 bytes per MB (~ 8192 bits, vs ~ 1500 worst-case
        // bits per MB measured) is a safe overestimate that still
        // bounds GPU buffer allocation reasonably.
        long worstCaseTile = mbCount * 1024L + 256L;
        long worstCaseFrame = worstCaseTile + 128L; // + header room
        using var dCompressedHeader = _accelerator.Allocate1D<byte>(64);
        using var dCompressedHeaderLen = _accelerator.Allocate1D<long>(1);
        using var dTile = _accelerator.Allocate1D<byte>(worstCaseTile);
        using var dTileLen = _accelerator.Allocate1D<long>(1);
        using var dUncompressedHeader = _accelerator.Allocate1D<byte>(32);
        using var dUncompressedHeaderLen = _accelerator.Allocate1D<long>(1);
        using var dOutFrame = _accelerator.Allocate1D<byte>(worstCaseFrame);
        using var dOutFrameLen = _accelerator.Allocate1D<long>(1);

        // Pre-zero output buffers so the bool encoder's carry-back
        // pass reads stable bytes.
        var zeroBytes64 = new byte[64];
        var zeroBytes32 = new byte[32];
        var zeroLong1 = new long[1];
        dCompressedHeader.View.CopyFromCPU(zeroBytes64);
        dCompressedHeaderLen.View.CopyFromCPU(zeroLong1);
        dTile.View.CopyFromCPU(new byte[worstCaseTile]);
        dTileLen.View.CopyFromCPU(zeroLong1);
        dUncompressedHeader.View.CopyFromCPU(zeroBytes32);
        dUncompressedHeaderLen.View.CopyFromCPU(zeroLong1);
        dOutFrame.View.CopyFromCPU(new byte[worstCaseFrame]);
        dOutFrameLen.View.CopyFromCPU(zeroLong1);

        dY.View.CopyFromCPU(yPlane);
        dU.View.CopyFromCPU(uPlane);
        dV.View.CopyFromCPU(vPlane);
        // Pre-fill recon to zero (sequential kernel overwrites every
        // pixel with prediction + residual but the carry-back path
        // for partial writes wants stable starting state).
        dYRecon.View.CopyFromCPU(new byte[yLen]);
        dURecon.View.CopyFromCPU(new byte[uvLen]);
        dVRecon.View.CopyFromCPU(new byte[uvLen]);

        // ---- 2. Dispatch dequantizer compute kernel ----
        // y_dc_delta / uv_dc_delta / uv_ac_delta = 0 in v1.
        _dequantKernel.Run(_dDcQLookup.View, _dAcQLookup.View, dDequant.View,
            baseQIndex, 0, 0, 0, 0);

        // ---- 3. Compressed header ----
        _compressedHeaderKernel.Run(dCompressedHeader.View, dCompressedHeaderLen.View);

        // ---- 4. Sequential encode (forward + inverse pipeline) ----
        _sequentialKernel.Run(
            dY.View, dU.View, dV.View,
            dYRecon.View, dURecon.View, dVRecon.View,
            dYCoefs.View, dUCoefs.View, dVCoefs.View,
            dDequant.View,
            mbCols, mbRows);

        // ---- 5. Entropy ----
        _entropyKernel.Run(
            dYCoefs.View, dUCoefs.View, dVCoefs.View,
            dTile.View, dTileLen.View,
            _dByteConsts.View, _dUshortConsts.View,
            mbCols, mbRows);

        // We need compressedHeaderLen on the host to seed the
        // uncompressed header's first_partition_size field.
        await _accelerator.SynchronizeAsync();
        long compressedLen = (await dCompressedHeaderLen.CopyToHostAsync())[0];

        // ---- 6. Uncompressed header ----
        _uncompressedHeaderKernel.Run(
            dUncompressedHeader.View, dUncompressedHeaderLen.View,
            width, height, baseQIndex, (int)compressedLen);

        // ---- 7. Read back lengths needed by Assemble ----
        await _accelerator.SynchronizeAsync();
        long uncompressedLen = (await dUncompressedHeaderLen.CopyToHostAsync())[0];
        long tileLen = (await dTileLen.CopyToHostAsync())[0];

        // ---- 8. Assemble ----
        _assembleKernel.Run(
            dUncompressedHeader.View, dCompressedHeader.View, dTile.View,
            dOutFrame.View, dOutFrameLen.View,
            (int)uncompressedLen, (int)compressedLen, (int)tileLen);

        await _accelerator.SynchronizeAsync();
        long outFrameLen = (await dOutFrameLen.CopyToHostAsync())[0];
        // Real per-backend partial readback (SpawnDev.ILGPU 4.9.3+).
        var result = await dOutFrame.View.SubView(0, outFrameLen).CopyToHostAsync();

        // Read back recon planes for self-consistency testing.
        // CopyToHostAsync returns fresh byte[] arrays sized to dY/dU/dV
        // (yLen / uvLen / uvLen). No host-side iteration over codec data
        // needed - hand them straight to the result tuple.
        var yReconBuf = await dYRecon.CopyToHostAsync();
        var uReconBuf = await dURecon.CopyToHostAsync();
        var vReconBuf = await dVRecon.CopyToHostAsync();

        return (result, yReconBuf, uReconBuf, vReconBuf);
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
    }
}
