// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP8 keyframe encoder, 100% GPU-resident pipeline. The host (this
// class's caller) is a pure coordinator - it allocates GPU buffers,
// uploads the source YUV planes (necessary I/O), dispatches a chain
// of kernels, and reads back the final encoded keyframe bytes. No
// CPU-side math, no CPU iteration, no CPU bool encoding. Per
// Captain's directive: "The environment using the accelerator is
// not capable of processing any data loads. It is simply the
// coordinator."
//
// Pipeline:
//   [host] upload Y/U/V to GPU
//   [host] dispatch Vp8FrameSetupKernel
//          - computes 6 dequantizers from baseQIndex
//          - writes frame header bits to partition0Out
//          - saves bool encoder state to initialP0State
//   [host] dispatch Vp8FrameSequentialEncodeKernel
//          - per-MB predict + transform + quant + recon
//          - reads dequantizers from buffer, writes recon back
//   [host] dispatch Vp8FrameEntropyKernel
//          - resumes partition0 from initialP0State
//          - writes per-MB modes + coef tokens
//   [host] dispatch Vp8FrameAssembleKernel
//          - writes 10-byte tag + start code + size code
//          - concatenates partition0 + tokenP0 into output
//          - writes final length to outLen[0]
//   [host] read outLen + output, return slice
//
// v1 simplifications (matches Vp8KeyframeEncoder reference):
//   - All MBs use Y_PRED = DC_PRED, UV_PRED = DC_PRED.
//   - No segmentation.
//   - Single token partition (Log2NumPartitions = 0).
//   - Loop filter disabled.
//   - mb_no_skip_coeff disabled.
//   - Default coef probs (no per-frame updates).

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>
/// VP8 keyframe encoder. 100% GPU-resident: host is pure coordinator.
/// Output byte-identical to <see cref="Vp8KeyframeEncoder"/>.
/// </summary>
public sealed class Vp8KeyframeEncoderGpu : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Vp8FrameSetupKernel _setup;
    private readonly Vp8FrameSequentialEncodeKernel _sequentialEncode;
    private readonly Vp8FrameEntropyKernel _entropy;
    private readonly Vp8FrameAssembleKernel _assemble;
    private readonly Vp8StridedPlanePackKernel _stridePack;

    // Constants uploaded once per accelerator and reused across frames.
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _dcQLookup;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _acQLookup;
    private readonly MemoryBuffer1D<byte, Stride1D.Dense> _defaultCoefProbs;
    private readonly MemoryBuffer1D<byte, Stride1D.Dense> _updateCoefProbs;
    private readonly MemoryBuffer1D<byte, Stride1D.Dense> _coefProbsByType;
    private readonly MemoryBuffer1D<byte, Stride1D.Dense> _constsExtended;

    /// <summary>Compile + cache all kernels and constants onto <paramref name="accelerator"/>.</summary>
    public Vp8KeyframeEncoderGpu(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _setup = new Vp8FrameSetupKernel(accelerator);
        _sequentialEncode = new Vp8FrameSequentialEncodeKernel(accelerator);
        _entropy = new Vp8FrameEntropyKernel(accelerator);
        _assemble = new Vp8FrameAssembleKernel(accelerator);
        _stridePack = new Vp8StridedPlanePackKernel(accelerator);

        // Upload accelerator-resident constants once.
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

    /// <summary>Encode a single VP8 keyframe from YUV420 source.</summary>
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

        // GPU buffers - allocated per call for v1; can be cached as
        // members for repeated encodes at the same resolution.
        using var dY = _accelerator.Allocate1D<byte>(width * height);
        using var dU = _accelerator.Allocate1D<byte>(uvWidth * uvHeight);
        using var dV = _accelerator.Allocate1D<byte>(uvWidth * uvHeight);
        using var dYRecon = _accelerator.Allocate1D<byte>(width * height);
        using var dURecon = _accelerator.Allocate1D<byte>(uvWidth * uvHeight);
        using var dVRecon = _accelerator.Allocate1D<byte>(uvWidth * uvHeight);

        using var dY4Coefs = _accelerator.Allocate1D<short>(mbCount * 256);
        using var dY2Coefs = _accelerator.Allocate1D<short>(mbCount * 16);
        using var dUCoefs = _accelerator.Allocate1D<short>(mbCount * 64);
        using var dVCoefs = _accelerator.Allocate1D<short>(mbCount * 64);

        using var dDequant = _accelerator.Allocate1D<int>(6);
        using var dInitialP0State = _accelerator.Allocate1D<int>(5);

        // Worst-case sized partition buffers. Header + per-MB modes
        // is bounded; tokenP0 grows with mbCount.
        int p0Stride = 64 * 1024 + mbCount * 32;
        int tp0Stride = 64 * 1024 + mbCount * 256;
        using var dP0 = _accelerator.Allocate1D<byte>(p0Stride);
        using var dTp = _accelerator.Allocate1D<byte>(tp0Stride);
        using var dPartLens = _accelerator.Allocate1D<long>(2);
        using var dAbove = _accelerator.Allocate1D<byte>(mbCols * 9);

        int outputCapacity = 16 + p0Stride + tp0Stride;
        using var dOutput = _accelerator.Allocate1D<byte>(outputCapacity);
        using var dOutLen = _accelerator.Allocate1D<int>(1);

        // === Host work: ONLY upload + dispatch ===
        UploadPlane(ySrc, ySrcStride, width, height, dY);
        UploadPlane(uSrc, uvSrcStride, uvWidth, uvHeight, dU);
        UploadPlane(vSrc, uvSrcStride, uvWidth, uvHeight, dV);
        dYRecon.View.MemSetToZero();
        dURecon.View.MemSetToZero();
        dVRecon.View.MemSetToZero();
        dP0.View.MemSetToZero();
        dTp.View.MemSetToZero();
        dAbove.View.MemSetToZero();

        // 1. Frame setup: dequantizers + frame header.
        _setup.Run(
            _dcQLookup.View, _acQLookup.View,
            _defaultCoefProbs.View, _updateCoefProbs.View,
            dDequant.View, dP0.View, dInitialP0State.View,
            baseQIndex);

        // 2. Sequential encode: per-MB math + recon.
        _sequentialEncode.Run(
            dY.View, dU.View, dV.View,
            dYRecon.View, dURecon.View, dVRecon.View,
            dY4Coefs.View, dY2Coefs.View, dUCoefs.View, dVCoefs.View,
            dDequant.View,
            mbCols, mbRows);

        // 3. Entropy: continues partition0 from setup state, writes
        // per-MB modes + coefs.
        _entropy.Run(
            dY4Coefs.View, dY2Coefs.View, dUCoefs.View, dVCoefs.View,
            _coefProbsByType.View, _constsExtended.View,
            dP0.View, dTp.View, dPartLens.View,
            dAbove.View, dInitialP0State.View,
            mbCols, mbRows);

        // 4. Assemble: frame tag + start code + size code +
        // partition0 + tokenP0 -> single output buffer; writes final
        // length to outLen[0].
        _assemble.Run(
            dP0.View, dTp.View, dPartLens.View,
            dOutput.View, dOutLen.View,
            width, height);

        _accelerator.Synchronize();

        // Single readback of the final encoded keyframe.
        var lenArr = dOutLen.GetAsArray1D();
        int finalLen = lenArr[0];
        var outputBuffer = dOutput.GetAsArray1D();
        var result = new byte[finalLen];
        Array.Copy(outputBuffer, 0, result, 0, finalLen);
        return result;
    }

    /// <summary>
    /// Upload a plane to GPU, stripping any source-side stride padding.
    /// Source upload itself is necessary I/O (the bytes come from outside
    /// the GPU); the previous version then ran a per-row CPU stride-strip
    /// loop, which was a cardinal-rule violation. This version uploads
    /// the strided source as a single I/O and dispatches
    /// <see cref="Vp8StridedPlanePackKernel"/> to do the strip on-GPU,
    /// keeping all codec data manipulation accelerator-resident per
    /// Captain's directive: "The environment using the accelerator is
    /// not capable of processing any data loads. It is simply the
    /// coordinator."
    /// </summary>
    private void UploadPlane(
        ReadOnlySpan<byte> src, int stride, int w, int h,
        MemoryBuffer1D<byte, Stride1D.Dense> dest)
    {
        if (stride == w)
        {
            // No padding - one I/O, done.
            dest.View.CopyFromCPU(src.Slice(0, w * h).ToArray());
        }
        else
        {
            // Strided source: upload the full strided region in one I/O
            // (no per-row CPU work), then GPU-pack into the dest buffer.
            using var dStrided = _accelerator.Allocate1D<byte>(stride * h);
            dStrided.View.CopyFromCPU(src.Slice(0, stride * h).ToArray());
            _stridePack.Run(dStrided.View, 0, stride, dest.View, 0, w, h);
            // Sync before dStrided's using-scope dispose so the kernel
            // has finished consuming it.
            _accelerator.Synchronize();
        }
    }

    /// <summary>Release kernel resources and constant buffers.</summary>
    public void Dispose()
    {
        _setup.Dispose();
        _sequentialEncode.Dispose();
        _entropy.Dispose();
        _assemble.Dispose();
        _stridePack.Dispose();
        _dcQLookup.Dispose();
        _acQLookup.Dispose();
        _defaultCoefProbs.Dispose();
        _updateCoefProbs.Dispose();
        _coefProbsByType.Dispose();
        _constsExtended.Dispose();
    }
}
