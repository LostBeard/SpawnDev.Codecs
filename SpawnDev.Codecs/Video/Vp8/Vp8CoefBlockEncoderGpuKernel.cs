// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Test/integration kernel exercising Vp8CoefBlockEncoderGpu. One
// thread per stream; each stream encodes its sequence of N coef
// blocks through a private bool encoder state, writing to its own
// output buffer slice. Validates the GPU coef encoder is bit-exact
// to the CPU version across blocks.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>
/// Test kernel: per stream, encode N coef blocks through the GPU bool
/// encoder + coef block encoder. Each block uses ctx + firstCoef from
/// per-block parameter buffers.
/// </summary>
public sealed class Vp8CoefBlockEncoderTestKernel : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Action<
        Index1D,
        ArrayView<short>,
        ArrayView<int>,
        ArrayView<int>,
        ArrayView<byte>,
        ArrayView<byte>,
        ArrayView<byte>,
        ArrayView<long>,
        int, int, int> _kernel;

    /// <summary>Compile.</summary>
    public Vp8CoefBlockEncoderTestKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<short>, ArrayView<int>, ArrayView<int>,
            ArrayView<byte>, ArrayView<byte>, ArrayView<byte>,
            ArrayView<long>,
            int, int, int>(EncodeStreamsKernel);
    }

    /// <summary>
    /// Run the test. Coefs are streamCount * blocksPerStream * 16
    /// shorts. ctxs/firstCoefs are per-block. probsFlat is the
    /// 264-byte block-type table; constsFlat is the 56-byte combined
    /// zigzag+bands+cat3-6 buffer.
    /// </summary>
    public void Run(
        ArrayView<short> coefs,
        ArrayView<int> ctxs,
        ArrayView<int> firstCoefs,
        ArrayView<byte> probsFlat,
        ArrayView<byte> constsFlat,
        ArrayView<byte> outBuf,
        ArrayView<long> outLens,
        int streamCount,
        int blocksPerStream,
        int outBufStride)
    {
        if (streamCount < 0) throw new ArgumentOutOfRangeException(nameof(streamCount));
        if (streamCount == 0) return;
        _kernel(streamCount, coefs, ctxs, firstCoefs, probsFlat, constsFlat,
            outBuf, outLens, streamCount, blocksPerStream, outBufStride);
    }

    private static void EncodeStreamsKernel(
        Index1D streamIdx,
        ArrayView<short> coefs,
        ArrayView<int> ctxs,
        ArrayView<int> firstCoefs,
        ArrayView<byte> probsFlat,
        ArrayView<byte> constsFlat,
        ArrayView<byte> outBuf,
        ArrayView<long> outLens,
        int streamCount,
        int blocksPerStream,
        int outBufStride)
    {
        int idx = streamIdx;
        if (idx >= streamCount) return;
        long coefsBase = (long)idx * blocksPerStream * 16;
        long paramsBase = (long)idx * blocksPerStream;
        long outBase = (long)idx * outBufStride;
        var streamOut = outBuf.SubView(outBase, outBufStride);

        var state = Vp8BoolEncoderGpu.Init();
        for (int b = 0; b < blocksPerStream; b++)
        {
            int ctx = ctxs[paramsBase + b];
            int firstCoef = firstCoefs[paramsBase + b];
            var blockCoefs = coefs.SubView(coefsBase + (long)b * 16, 16);
            Vp8CoefBlockEncoderGpu.Encode(
                ref state, streamOut, probsFlat, constsFlat,
                ctx, firstCoef, blockCoefs);
        }
        Vp8BoolEncoderGpu.Stop(ref state, streamOut);

        outLens[idx] = state.OutLen;
    }

    /// <summary>Release kernel resources.</summary>
    public void Dispose() { /* auto-grouped */ }
}
