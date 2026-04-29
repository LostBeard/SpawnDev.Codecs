// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Wrapper kernel that drives Av1CoefEncoderGpu.WriteCoeffsTxb through
// ILGPU - one block per dispatch. Used for cross-backend correctness
// verification against the CPU Av1CoefEncoder reference. The
// upcoming Av1FrameEntropyKernel will call the same helper inline
// from a per-frame walker without dispatching this wrapper.

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>
/// Single-block ILGPU driver for
/// <see cref="Av1CoefEncoderGpu.WriteCoeffsTxb"/>. Encodes ONE
/// transform block at a time so per-block bit-exact verification
/// against the CPU encoder is straightforward.
/// </summary>
public sealed class Av1CoefEncoderGpuKernel : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Action<
        Index1D,
        ArrayView<byte>, ArrayView<byte>, ArrayView<ushort>,
        ArrayView<int>, ArrayView<byte>, ArrayView<long>,
        ArrayView<int>, ArrayView<int>,
        int, int, int, int, int, int> _kernel;

    /// <summary>Compile.</summary>
    public Av1CoefEncoderGpuKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<byte>, ArrayView<byte>, ArrayView<ushort>,
            ArrayView<int>, ArrayView<byte>, ArrayView<long>,
            ArrayView<int>, ArrayView<int>,
            int, int, int, int, int, int>(EncoderKernel);
    }

    /// <summary>
    /// Encode one transform block via Av1CoefEncoderGpu. Single thread
    /// per dispatch.
    /// </summary>
    /// <param name="outBuf">Range encoder output buffer (worst-case sized).</param>
    /// <param name="constsByte">Av1KeyframeConstantsGpu byte buffer.</param>
    /// <param name="constsUshort">Av1KeyframeConstantsGpu ushort buffer.</param>
    /// <param name="coefsRaster">Quantized coef block (length = txW * txH).</param>
    /// <param name="levelsBuf">Padded levels[] scratch.</param>
    /// <param name="outLen">[0] = encoded byte count after Done().</param>
    /// <param name="eobOut">[0] = EOB after encode.</param>
    /// <param name="culLevelOut">[0] = CulLevel after encode.</param>
    public void Run(
        ArrayView<byte> outBuf,
        ArrayView<byte> constsByte,
        ArrayView<ushort> constsUshort,
        ArrayView<int> coefsRaster,
        ArrayView<byte> levelsBuf,
        ArrayView<long> outLen,
        ArrayView<int> eobOut,
        ArrayView<int> culLevelOut,
        int txSize, int plane, int qctx,
        int txbSkipCtx, int dcSignCtx, int qindex)
    {
        _kernel(1, outBuf, constsByte, constsUshort, coefsRaster, levelsBuf, outLen,
            eobOut, culLevelOut,
            txSize, plane, qctx, txbSkipCtx, dcSignCtx, qindex);
    }

    private static void EncoderKernel(
        Index1D _,
        ArrayView<byte> outBuf,
        ArrayView<byte> constsByte,
        ArrayView<ushort> constsUshort,
        ArrayView<int> coefsRaster,
        ArrayView<byte> levelsBuf,
        ArrayView<long> outLen,
        ArrayView<int> eobOut,
        ArrayView<int> culLevelOut,
        int txSize, int plane, int qctx,
        int txbSkipCtx, int dcSignCtx, int qindex)
    {
        var re = Av1RangeEncoderGpu.Init();
        Av1CoefEncoderGpu.WriteCoeffsTxb(
            ref re, outBuf, constsByte, constsUshort,
            coefsRaster, 0, levelsBuf, 0,
            txSize, plane, qctx, txbSkipCtx, dcSignCtx, qindex,
            eobOut, culLevelOut, 0);
        Av1RangeEncoderGpu.Done(ref re, outBuf);
        outLen[0] = re.OutLen;
    }

    /// <summary>Release kernel resources.</summary>
    public void Dispose() { /* auto-grouped */ }
}
