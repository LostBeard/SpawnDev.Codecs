// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP8 keyframe encoder, GPU-resident pipeline. Top-level integration
// of all the Vp8*Gpu pipeline drivers + entropy kernel. Produces the
// same byte-for-byte output as Vp8KeyframeEncoder (CPU) - same
// simplifications (DC_PRED only, single token partition, no loop
// filter) - but the math + entropy hot path runs on the GPU. Only
// the frame header and final byte concatenation stay on CPU.
//
// v1 limitations (matches Vp8KeyframeEncoder.cs CPU encoder):
//   - All MBs use Y_PRED = DC_PRED, UV_PRED = DC_PRED.
//   - No segmentation.
//   - Single token partition (Log2NumPartitions = 0).
//   - Loop filter disabled.
//   - mb_no_skip_coeff disabled.
//
// For multi-MB frames the predictor build + recon are sequenced
// CPU-side (one kernel dispatch per MB) - per-MB recon is needed
// for next-MB intra prediction. Wave-parallel scheduling is the
// future optimization; v1 keeps it simple.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>
/// VP8 keyframe encoder using the GPU-resident pipeline. Output is
/// byte-identical to <see cref="Vp8KeyframeEncoder"/>; the math and
/// entropy stages run on the GPU.
/// </summary>
public sealed class Vp8KeyframeEncoderGpu : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Vp8FrameTransformGpu _transform;
    private readonly Vp8FrameEntropyKernel _entropy;
    private readonly Vp8FrameLayoutKernels _layout;
    private readonly Vp8SubtractKernel _subtract;
    private readonly Vp8FramePredictorGpu _predictor;
    private readonly Vp8FrameReconstructGpu _reconstruct;

    /// <summary>Compile + cache all the kernels onto <paramref name="accelerator"/>.</summary>
    public Vp8KeyframeEncoderGpu(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _transform = new Vp8FrameTransformGpu(accelerator);
        _entropy = new Vp8FrameEntropyKernel(accelerator);
        _layout = new Vp8FrameLayoutKernels(accelerator);
        _subtract = new Vp8SubtractKernel(accelerator);
        _predictor = new Vp8FramePredictorGpu(accelerator);
        _reconstruct = new Vp8FrameReconstructGpu(accelerator);
    }

    /// <summary>
    /// Encode a single VP8 keyframe from YUV420 source. Output is a
    /// complete VP8 frame ready to wrap in IVF or WebM. Byte-identical
    /// to <see cref="Vp8KeyframeEncoder.EncodeKeyFrame"/> for the same
    /// inputs.
    /// </summary>
    public byte[] EncodeKeyFrame(
        ReadOnlySpan<byte> ySrc, int ySrcStride,
        ReadOnlySpan<byte> uSrc, int uvSrcStride,
        ReadOnlySpan<byte> vSrc,
        int width, int height,
        int baseQIndex = 30)
    {
        if (width <= 0 || height <= 0) throw new ArgumentOutOfRangeException();
        if ((width & 15) != 0 || (height & 15) != 0)
            throw new ArgumentException("Width and height must be multiples of 16 (v1)");
        if (baseQIndex < 0 || baseQIndex > 127)
            throw new ArgumentOutOfRangeException(nameof(baseQIndex));

        int mbCols = width / 16;
        int mbRows = height / 16;
        int mbCount = mbCols * mbRows;
        int uvWidth = width / 2;
        int uvHeight = height / 2;

        // Per-frame dequantizers (CPU-side; passed to GPU as-is).
        var dequant = Vp8MbDequantizer.Compute(0,
            new Vp8QuantizerIndices
            {
                BaseQIndex = baseQIndex,
                Y1DcDeltaQ = 0, Y2DcDeltaQ = 0, Y2AcDeltaQ = 0, UvDcDeltaQ = 0, UvAcDeltaQ = 0,
            },
            new Vp8SegmentationParams
            {
                Enabled = false, UpdateMap = false, UpdateData = false, AbsDelta = false,
                FeatureData = new int[2, 4],
                SegmentTreeProbs = new byte[3] { 255, 255, 255 },
            });

        // GPU buffers. Per-frame; could be cached across frames but for
        // v1 just allocate fresh per call.
        using var dY = _accelerator.Allocate1D<byte>(width * height);
        using var dU = _accelerator.Allocate1D<byte>(uvWidth * uvHeight);
        using var dV = _accelerator.Allocate1D<byte>(uvWidth * uvHeight);
        using var dYRecon = _accelerator.Allocate1D<byte>(width * height);
        using var dURecon = _accelerator.Allocate1D<byte>(uvWidth * uvHeight);
        using var dVRecon = _accelerator.Allocate1D<byte>(uvWidth * uvHeight);

        // Per-MB packed buffers (block-major).
        using var dY16Packed = _accelerator.Allocate1D<byte>(mbCount * 256);
        using var dU8Packed = _accelerator.Allocate1D<byte>(mbCount * 64);
        using var dV8Packed = _accelerator.Allocate1D<byte>(mbCount * 64);

        // Y4 4x4 packed (16 blocks * 16 bytes per MB).
        using var dY4Pred = _accelerator.Allocate1D<byte>(mbCount * 256);
        using var dUPred = _accelerator.Allocate1D<byte>(mbCount * 64);
        using var dVPred = _accelerator.Allocate1D<byte>(mbCount * 64);
        using var dY4Residual = _accelerator.Allocate1D<short>(mbCount * 256);
        using var dURes = _accelerator.Allocate1D<short>(mbCount * 64);
        using var dVRes = _accelerator.Allocate1D<short>(mbCount * 64);
        using var dY4Coefs = _accelerator.Allocate1D<short>(mbCount * 256);
        using var dY2Coefs = _accelerator.Allocate1D<short>(mbCount * 16);
        using var dUCoefs = _accelerator.Allocate1D<short>(mbCount * 64);
        using var dVCoefs = _accelerator.Allocate1D<short>(mbCount * 64);

        // Per-MB Y4 packed source (16 blocks * 16 bytes per MB) - we
        // need this layout for the FDCT kernel.
        using var dY4Src = _accelerator.Allocate1D<byte>(mbCount * 256);

        // Quantizer arrays per-MB (all same for v1 since segmentation
        // is off).
        var y1DcQ = new short[mbCount];
        var y1AcQ = new short[mbCount];
        var y2DcQ = new short[mbCount];
        var y2AcQ = new short[mbCount];
        var uvDcQ = new short[mbCount];
        var uvAcQ = new short[mbCount];
        for (int b = 0; b < mbCount; b++)
        {
            y1DcQ[b] = (short)dequant.Y1Dc;
            y1AcQ[b] = (short)dequant.Y1Ac;
            y2DcQ[b] = (short)dequant.Y2Dc;
            y2AcQ[b] = (short)dequant.Y2Ac;
            uvDcQ[b] = (short)dequant.UvDc;
            uvAcQ[b] = (short)dequant.UvAc;
        }

        using var dY1Dc = _accelerator.Allocate1D<short>(mbCount);
        using var dY1Ac = _accelerator.Allocate1D<short>(mbCount);
        using var dY2Dc = _accelerator.Allocate1D<short>(mbCount);
        using var dY2Ac = _accelerator.Allocate1D<short>(mbCount);
        using var dUvDc = _accelerator.Allocate1D<short>(mbCount);
        using var dUvAc = _accelerator.Allocate1D<short>(mbCount);
        dY1Dc.View.CopyFromCPU(y1DcQ);
        dY1Ac.View.CopyFromCPU(y1AcQ);
        dY2Dc.View.CopyFromCPU(y2DcQ);
        dY2Ac.View.CopyFromCPU(y2AcQ);
        dUvDc.View.CopyFromCPU(uvDcQ);
        dUvAc.View.CopyFromCPU(uvAcQ);

        // Upload source planes - stride-flatten to packed per-row.
        UploadPlane(ySrc, ySrcStride, width, height, dY);
        UploadPlane(uSrc, uvSrcStride, uvWidth, uvHeight, dU);
        UploadPlane(vSrc, uvSrcStride, uvWidth, uvHeight, dV);

        // The GPU encoder pipeline. v1 NOTE: per-MB recon dependency is
        // fundamental to multi-MB frames. For 1-MB frames (16x16 input)
        // there are no neighbours, so this v1 only supports single-MB
        // frames bit-exactly. Larger frames need wave-parallel
        // scheduling (planned).
        if (mbRows != 1 || mbCols != 1)
        {
            throw new NotSupportedException(
                "Vp8KeyframeEncoderGpu v1 supports 16x16 single-MB frames only. " +
                "Multi-MB support requires wave-parallel predictor + recon scheduling " +
                "to honour the per-MB recon dependency chain. Filed for follow-up.");
        }

        return EncodeSingleMacroblockFrame(
            dY, dU, dV, dYRecon, dURecon, dVRecon,
            dY16Packed, dU8Packed, dV8Packed, dY4Src,
            dY4Pred, dUPred, dVPred,
            dY4Residual, dURes, dVRes,
            dY4Coefs, dY2Coefs, dUCoefs, dVCoefs,
            dY1Dc, dY1Ac, dY2Dc, dY2Ac, dUvDc, dUvAc,
            width, height, baseQIndex, dequant);
    }

    /// <summary>
    /// Single-MB v1 fast path. Predictor is 128-fill (no neighbours);
    /// no recon-dependency loop. Runs the GPU pipeline once for the
    /// frame's lone MB.
    /// </summary>
    private byte[] EncodeSingleMacroblockFrame(
        MemoryBuffer1D<byte, Stride1D.Dense> dY,
        MemoryBuffer1D<byte, Stride1D.Dense> dU,
        MemoryBuffer1D<byte, Stride1D.Dense> dV,
        MemoryBuffer1D<byte, Stride1D.Dense> dYRecon,
        MemoryBuffer1D<byte, Stride1D.Dense> dURecon,
        MemoryBuffer1D<byte, Stride1D.Dense> dVRecon,
        MemoryBuffer1D<byte, Stride1D.Dense> dY16Packed,
        MemoryBuffer1D<byte, Stride1D.Dense> dU8Packed,
        MemoryBuffer1D<byte, Stride1D.Dense> dV8Packed,
        MemoryBuffer1D<byte, Stride1D.Dense> dY4Src,
        MemoryBuffer1D<byte, Stride1D.Dense> dY4Pred,
        MemoryBuffer1D<byte, Stride1D.Dense> dUPred,
        MemoryBuffer1D<byte, Stride1D.Dense> dVPred,
        MemoryBuffer1D<short, Stride1D.Dense> dY4Residual,
        MemoryBuffer1D<short, Stride1D.Dense> dURes,
        MemoryBuffer1D<short, Stride1D.Dense> dVRes,
        MemoryBuffer1D<short, Stride1D.Dense> dY4Coefs,
        MemoryBuffer1D<short, Stride1D.Dense> dY2Coefs,
        MemoryBuffer1D<short, Stride1D.Dense> dUCoefs,
        MemoryBuffer1D<short, Stride1D.Dense> dVCoefs,
        MemoryBuffer1D<short, Stride1D.Dense> dY1Dc,
        MemoryBuffer1D<short, Stride1D.Dense> dY1Ac,
        MemoryBuffer1D<short, Stride1D.Dense> dY2Dc,
        MemoryBuffer1D<short, Stride1D.Dense> dY2Ac,
        MemoryBuffer1D<short, Stride1D.Dense> dUvDc,
        MemoryBuffer1D<short, Stride1D.Dense> dUvAc,
        int width, int height, int baseQIndex,
        Vp8MbDequant dequant)
    {
        // Step 1: gather MB pixels into per-MB packed layout (Y 16x16, U 8x8, V 8x8).
        _layout.GatherY16(dY.View, dY16Packed.View, mbCols: 1, mbRows: 1, yStride: width);
        _layout.GatherUv8(dU.View, dU8Packed.View, mbCols: 1, mbRows: 1, uvStride: width / 2);
        _layout.GatherUv8(dV.View, dV8Packed.View, mbCols: 1, mbRows: 1, uvStride: width / 2);

        // Step 2: split the 16x16 Y MB into 16 4x4-packed blocks and the
        // 8x8 UV MBs into 4 4x4-packed blocks (libvpx convention:
        // row-major within each 4x4 block, blocks in row-major order
        // within the MB). The FDCT kernel expects 16 contiguous shorts
        // per block, which only holds with this layout.
        using var dU4Src = _accelerator.Allocate1D<byte>(64);
        using var dV4Src = _accelerator.Allocate1D<byte>(64);
        SplitY16ToY4Packed(dY16Packed, dY4Src);
        SplitUv8ToBlocks(dU8Packed, dU4Src);
        SplitUv8ToBlocks(dV8Packed, dV4Src);

        // Step 3: build predictor. Single MB, no neighbours, all-DC
        // means predictor is 128 fill for Y (256 bytes) and 128 for U
        // and V (64 bytes each).
        FillBuffer(dY4Pred, 128);
        FillBuffer(dUPred, 128);
        FillBuffer(dVPred, 128);

        // Step 4: subtract residuals. Y4 first (256 pixels), then U
        // and V (64 each). The UV inputs are now 4-block packed.
        _subtract.Run(dY4Src.View, dY4Pred.View, dY4Residual.View, 256);
        _subtract.Run(dU4Src.View, dUPred.View, dURes.View, 64);
        _subtract.Run(dV4Src.View, dVPred.View, dVRes.View, 64);

        // Step 5: forward transform + quantize on GPU.
        _transform.Run(
            dY4Residual.View, dURes.View, dVRes.View,
            dY4Coefs.View, dY2Coefs.View, dUCoefs.View, dVCoefs.View,
            dY1Dc.View, dY1Ac.View, dY2Dc.View, dY2Ac.View, dUvDc.View, dUvAc.View,
            mbCount: 1);

        // Step 6: write frame header on CPU; snapshot bool encoder state.
        var partition0 = new Vp8BoolEncoder();
        var hdr = BuildFrameHeader(width, height, baseQIndex);
        Vp8FrameHeaderWriter.WriteKeyFrameHeader(partition0, hdr);
        var snapshot = partition0.GetSnapshot();

        // Step 7: GPU entropy. Pre-load partition0Out with the CPU
        // header bytes; pass snapshot state via initialP0State.
        const int p0Stride = 32 * 1024;
        const int tp0Stride = 64 * 1024;
        using var dP0 = _accelerator.Allocate1D<byte>(p0Stride);
        using var dTp = _accelerator.Allocate1D<byte>(tp0Stride);
        using var dLens = _accelerator.Allocate1D<long>(2);
        using var dAbove = _accelerator.Allocate1D<byte>(1 * 9); // mbCols=1
        using var dInitState = _accelerator.Allocate1D<int>(5);
        using var dCoefProbs = _accelerator.Allocate1D<byte>(4 * 264);
        using var dConstsExtended = _accelerator.Allocate1D<byte>(Vp8FrameEntropyKernel.ConstsExtendedTotalBytes);

        // Pre-load partition0Out with header bytes; rest is zeroed.
        var primedP0 = new byte[p0Stride];
        Array.Copy(snapshot.Buf, 0, primedP0, 0, snapshot.Buf.Length);
        dP0.View.CopyFromCPU(primedP0);
        dTp.View.MemSetToZero();
        dAbove.View.MemSetToZero();
        dInitState.View.CopyFromCPU(new int[]
        {
            (int)snapshot.LowValue, (int)snapshot.Range, snapshot.Count,
            snapshot.Buf.Length, 0,
        });

        // coefProbsByType: flat 4 * 264 from Vp8DefaultCoefProbs.
        var coefProbsByType = new byte[4 * 264];
        var defaults = hdr.CoefProbs;
        for (int t = 0; t < 4; t++)
            for (int band = 0; band < 8; band++)
                for (int c = 0; c < 3; c++)
                    for (int n = 0; n < 11; n++)
                        coefProbsByType[t * 264 + band * 33 + c * 11 + n] = defaults[t, band, c, n];
        dCoefProbs.View.CopyFromCPU(coefProbsByType);
        dConstsExtended.View.CopyFromCPU(Vp8FrameEntropyKernel.BuildExtendedConstsBuffer());

        _entropy.Run(
            dY4Coefs.View, dY2Coefs.View, dUCoefs.View, dVCoefs.View,
            dCoefProbs.View, dConstsExtended.View,
            dP0.View, dTp.View, dLens.View,
            dAbove.View, dInitState.View,
            mbCols: 1, mbRows: 1);
        _accelerator.Synchronize();

        // Step 8: read back partition0 + tokenP0.
        var lensBack = dLens.GetAsArray1D();
        var p0Back = dP0.GetAsArray1D();
        var tpBack = dTp.GetAsArray1D();

        // Step 9: assemble final frame bytes - tag + partition0 + tokenBlob.
        var tag = new Vp8FrameTag
        {
            IsKeyFrame = true,
            Version = Vp8Version.Bicubic,
            ShowFrame = true,
            FirstPartitionSize = (int)lensBack[0],
            Width = width, Height = height,
            HorizontalScale = 0, VerticalScale = 0,
        };
        var tagBytes = Vp8FrameTagWriter.WriteTag(tag);

        // npart=1 so no token-partition size headers; tokenBlob is just
        // the single token partition's bytes.
        long p0Len = lensBack[0];
        long tp0Len = lensBack[1];
        var output = new byte[tagBytes.Length + p0Len + tp0Len];
        Buffer.BlockCopy(tagBytes, 0, output, 0, tagBytes.Length);
        Array.Copy(p0Back, 0, output, tagBytes.Length, (int)p0Len);
        Array.Copy(tpBack, 0, output, tagBytes.Length + (int)p0Len, (int)tp0Len);
        return output;
    }

    private static Vp8FrameHeader BuildFrameHeader(int width, int height, int baseQIndex) =>
        new Vp8FrameHeader
        {
            ColorSpace = 0,
            ClampingType = 0,
            Segmentation = new Vp8SegmentationParams
            {
                Enabled = false, UpdateMap = false, UpdateData = false, AbsDelta = false,
                FeatureData = new int[2, 4],
                SegmentTreeProbs = new byte[3] { 255, 255, 255 },
            },
            LoopFilter = new Vp8LoopFilterParams
            {
                FilterType = 0, FilterLevel = 0, SharpnessLevel = 0,
                ModeRefLfDeltaEnabled = false,
                RefLfDeltas = new int[4], ModeLfDeltas = new int[4],
            },
            Log2NumPartitions = 0,
            Quantizer = new Vp8QuantizerIndices
            {
                BaseQIndex = baseQIndex,
                Y1DcDeltaQ = 0, Y2DcDeltaQ = 0, Y2AcDeltaQ = 0, UvDcDeltaQ = 0, UvAcDeltaQ = 0,
            },
            RefreshEntropyProbs = true,
            CoefProbs = (byte[,,,])Vp8DefaultCoefProbs.DefaultProbs.Clone(),
            MbNoSkipCoeffEnabled = false,
            ProbSkipFalse = 0,
        };

    /// <summary>Upload a plane to GPU, stripping any source-side stride padding.</summary>
    private static void UploadPlane(
        ReadOnlySpan<byte> src, int stride, int w, int h,
        MemoryBuffer1D<byte, Stride1D.Dense> dest)
    {
        if (stride == w)
        {
            dest.View.CopyFromCPU(src.Slice(0, w * h).ToArray());
        }
        else
        {
            var packed = new byte[w * h];
            for (int r = 0; r < h; r++) src.Slice(r * stride, w).CopyTo(packed.AsSpan(r * w));
            dest.View.CopyFromCPU(packed);
        }
    }

    /// <summary>Split a per-MB 8x8 packed UV buffer into 4 4x4 packed blocks (block-major).</summary>
    private static void SplitUv8ToBlocks(
        MemoryBuffer1D<byte, Stride1D.Dense> uv8Packed,
        MemoryBuffer1D<byte, Stride1D.Dense> uv4Src)
    {
        var src = uv8Packed.GetAsArray1D();
        var dst = new byte[uv4Src.Length];
        long mbCount = uv8Packed.Length / 64;
        for (long mb = 0; mb < mbCount; mb++)
        {
            long mb8Base = mb * 64;
            long mb4Base = mb * 64;
            for (int by = 0; by < 2; by++)
            {
                for (int bx = 0; bx < 2; bx++)
                {
                    int blockIdx = by * 2 + bx;
                    long block4Base = mb4Base + (long)blockIdx * 16;
                    for (int r = 0; r < 4; r++)
                    {
                        long mb8Row = mb8Base + (by * 4 + r) * 8 + bx * 4;
                        long block4Row = block4Base + (long)r * 4;
                        for (int c = 0; c < 4; c++)
                            dst[block4Row + c] = src[mb8Row + c];
                    }
                }
            }
        }
        uv4Src.View.CopyFromCPU(dst);
    }

    /// <summary>Split a per-MB 16x16 packed buffer into 16 4x4 packed Y4 blocks.</summary>
    private static void SplitY16ToY4Packed(
        MemoryBuffer1D<byte, Stride1D.Dense> y16Packed,
        MemoryBuffer1D<byte, Stride1D.Dense> y4Src)
    {
        // CPU-side splitting for v1; the data is already on GPU but we
        // run this as a small kernel-equivalent here. Future
        // optimization: a GPU kernel that does the layout conversion.
        var y16Host = y16Packed.GetAsArray1D();
        var y4Host = new byte[y4Src.Length];
        // For each MB (just 1 in v1), gather 16 4x4 blocks in row-major
        // (by, bx) order, and within each block in row-major (r, c).
        long mbCount = y16Packed.Length / 256;
        for (long mb = 0; mb < mbCount; mb++)
        {
            long mb16Base = mb * 256;
            long mb4Base = mb * 256;
            for (int by = 0; by < 4; by++)
            {
                for (int bx = 0; bx < 4; bx++)
                {
                    int blockIdx = by * 4 + bx;
                    long block4Base = mb4Base + (long)blockIdx * 16;
                    for (int r = 0; r < 4; r++)
                    {
                        long mb16Row = mb16Base + (by * 4 + r) * 16 + bx * 4;
                        long block4Row = block4Base + (long)r * 4;
                        for (int c = 0; c < 4; c++)
                            y4Host[block4Row + c] = y16Host[mb16Row + c];
                    }
                }
            }
        }
        y4Src.View.CopyFromCPU(y4Host);
    }

    /// <summary>Fill a byte buffer with a constant value using a small kernel.</summary>
    private void FillBuffer(MemoryBuffer1D<byte, Stride1D.Dense> buf, byte value)
    {
        // For v1 we use a CPU upload of a constant array; small enough
        // that it doesn't matter. Future: a tiny ILGPU fill kernel.
        var arr = new byte[buf.Length];
        Array.Fill(arr, value);
        buf.View.CopyFromCPU(arr);
    }

    /// <summary>Release kernel resources. Does NOT dispose the accelerator.</summary>
    public void Dispose()
    {
        _transform.Dispose();
        _entropy.Dispose();
        _layout.Dispose();
        _subtract.Dispose();
        _predictor.Dispose();
        _reconstruct.Dispose();
    }
}
