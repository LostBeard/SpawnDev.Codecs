// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 v1 keyframe GPU decoder integration class. Symmetric to
// Av1KeyframeEncoderGpu. Takes the entropy-coded tile bytes (e.g.
// from Av1KeyframeEncoderGpu.EncodeSingleTileAsync) and reconstructs
// the YUV 4:2:0 frame entirely on GPU via the
// Av1FrameSequentialDecodeKernel walker.
//
// V1 phase:
//   - DecodeSingleTileAsync: tile bytes -> recon Y/U/V planes.
//
// V2 phase (follow-up):
//   - DecodeKeyFrameAsync: full TD/SH/Frame OBU stream parsing on GPU.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>
/// AV1 v1 keyframe GPU decoder integration class. Decodes raw tile
/// bytes (codec-data) into YUV 4:2:0 recon planes.
/// </summary>
public sealed class Av1KeyframeDecoderGpu : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Av1FrameSequentialDecodeKernel _frameKernel;

    private readonly MemoryBuffer1D<byte, global::ILGPU.Stride1D.Dense> _dByteConsts;
    private readonly MemoryBuffer1D<ushort, global::ILGPU.Stride1D.Dense> _dUshortConsts;
    private readonly MemoryBuffer1D<short, global::ILGPU.Stride1D.Dense> _dDcAcQuant;

    /// <summary>Construct decoder bound to <paramref name="accelerator"/>.</summary>
    public Av1KeyframeDecoderGpu(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _frameKernel = new Av1FrameSequentialDecodeKernel(accelerator);

        _dByteConsts = accelerator.Allocate1D<byte>(Av1KeyframeConstantsGpu.ByteConstsTotalBytes);
        _dUshortConsts = accelerator.Allocate1D<ushort>(Av1KeyframeConstantsGpu.UshortConstsTotalEntries);
        _dByteConsts.View.CopyFromCPU(Av1KeyframeConstantsGpu.BuildByteConstsBuffer());
        _dUshortConsts.View.CopyFromCPU(Av1KeyframeConstantsGpu.BuildUshortConstsBuffer());

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
    /// Decode raw tile bytes into YUV 4:2:0 recon planes. Returns
    /// (yPlane, uPlane, vPlane) tuples.
    /// </summary>
    public async Task<(byte[] y, byte[] u, byte[] v)> DecodeSingleTileAsync(
        byte[] tileBytes, int width, int height, int baseQIndex = 32)
    {
        if (tileBytes is null) throw new ArgumentNullException(nameof(tileBytes));
        if (width <= 0 || (width & 63) != 0)
            throw new ArgumentOutOfRangeException(nameof(width),
                "v1 GPU decoder requires width that is a positive multiple of 64.");
        if (height <= 0 || (height & 63) != 0)
            throw new ArgumentOutOfRangeException(nameof(height),
                "v1 GPU decoder requires height that is a positive multiple of 64.");
        if (baseQIndex <= 0 || baseQIndex > 255)
            throw new ArgumentOutOfRangeException(nameof(baseQIndex),
                "baseQIndex must be in [1, 255].");

        int yLen = width * height;
        int uvLen = yLen / 4;
        int reconLen = yLen + uvLen + uvLen;

        int frameMiCols = ((width + 7) >> 3) << 1;
        int frameMiRows = ((height + 7) >> 3) << 1;

        var p = new Av1FrameSeqDecodeParams
        {
            Width = width,
            Height = height,
            BaseQIndex = baseQIndex,
            YPlaneOff = 0,
            UPlaneOff = yLen,
            VPlaneOff = yLen + uvLen,
            FrameMiCols = frameMiCols,
            FrameMiRows = frameMiRows,
            TileBytesOffset = 0,
            TileBytesLength = tileBytes.Length,
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
        int scratchIntLen = Av1FrameSequentialDecodeKernel.MinScratchIntLength;

        using var dTile = _accelerator.Allocate1D<byte>(tileBytes.Length);
        using var dRecon = _accelerator.Allocate1D<byte>(reconLen);
        using var dScratchByte = _accelerator.Allocate1D<byte>(scratchByteLen);
        using var dScratchInt = _accelerator.Allocate1D<int>(scratchIntLen);

        dTile.View.CopyFromCPU(tileBytes);
        dRecon.View.CopyFromCPU(new byte[reconLen]);
        dScratchByte.View.CopyFromCPU(new byte[scratchByteLen]);
        dScratchInt.View.CopyFromCPU(new int[scratchIntLen]);

        _frameKernel.Run(
            dTile.View, dRecon.View,
            _dByteConsts.View, _dUshortConsts.View,
            _dDcAcQuant.View,
            dScratchByte.View, dScratchInt.View, p);

        await _accelerator.SynchronizeAsync();

        // Three per-plane partial readbacks of dRecon at the YPlaneOff /
        // UPlaneOff / VPlaneOff offsets. No host-side iteration over
        // codec output bytes at the call site.
        var y = await dRecon.CopyToHostAsync(0, yLen);
        var u = await dRecon.CopyToHostAsync(yLen, uvLen);
        var v = await dRecon.CopyToHostAsync(yLen + uvLen, uvLen);
        return (y, u, v);
    }

    /// <summary>Release every resource the decoder owns.</summary>
    public void Dispose()
    {
        _frameKernel.Dispose();
        _dByteConsts.Dispose();
        _dUshortConsts.Dispose();
        _dDcAcQuant.Dispose();
    }
}
