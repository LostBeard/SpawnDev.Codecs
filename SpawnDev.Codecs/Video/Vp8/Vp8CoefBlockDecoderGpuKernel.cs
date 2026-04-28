// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Test/integration kernel for Vp8CoefBlockDecoderGpu. One thread per
// stream; each stream decodes a sequence of N coef blocks from its
// slice of the input buffer, writes the decoded coef arrays + EOBs to
// per-stream output slices.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>
/// Test kernel exercising Vp8CoefBlockDecoderGpu.Decode on a stream
/// of N pre-encoded coef blocks. One thread per stream.
/// </summary>
public sealed class Vp8CoefBlockDecoderTestKernel : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Action<
        Index1D,
        ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>,
        ArrayView<byte>, ArrayView<byte>,
        ArrayView<short>, ArrayView<int>,
        int, int, int> _kernel;

    /// <summary>Compile.</summary>
    public Vp8CoefBlockDecoderTestKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>,
            ArrayView<byte>, ArrayView<byte>,
            ArrayView<short>, ArrayView<int>,
            int, int, int>(DecodeStreamsKernel);
    }

    /// <summary>
    /// Decode N coef blocks per stream. inBuf holds the encoded bytes
    /// at offsets `i * inStride`, with each stream's actual length in
    /// inLens. ctxs + firstCoefs are per-block parameters.
    /// </summary>
    public void Run(
        ArrayView<byte> inBuf,
        ArrayView<int> inLens,
        ArrayView<int> ctxs,
        ArrayView<int> firstCoefs,
        ArrayView<byte> probsFlat,
        ArrayView<byte> constsFlat,
        ArrayView<short> coefsOut,
        ArrayView<int> eobsOut,
        int streamCount,
        int blocksPerStream,
        int inStride)
    {
        if (streamCount < 0) throw new ArgumentOutOfRangeException(nameof(streamCount));
        if (streamCount == 0) return;
        _kernel(streamCount, inBuf, inLens, ctxs, firstCoefs,
            probsFlat, constsFlat, coefsOut, eobsOut,
            streamCount, blocksPerStream, inStride);
    }

    private static void DecodeStreamsKernel(
        Index1D streamIdx,
        ArrayView<byte> inBuf,
        ArrayView<int> inLens,
        ArrayView<int> ctxs,
        ArrayView<int> firstCoefs,
        ArrayView<byte> probsFlat,
        ArrayView<byte> constsFlat,
        ArrayView<short> coefsOut,
        ArrayView<int> eobsOut,
        int streamCount,
        int blocksPerStream,
        int inStride)
    {
        int idx = streamIdx;
        if (idx >= streamCount) return;
        int inOffset = idx * inStride;
        int inLen = inLens[idx];
        long paramsBase = (long)idx * blocksPerStream;
        long coefsBase = (long)idx * blocksPerStream * 16;
        long eobsBase = paramsBase;

        var state = Vp8BoolDecoderGpu.Init(inBuf, inOffset, inLen);
        for (int b = 0; b < blocksPerStream; b++)
        {
            int ctx = ctxs[paramsBase + b];
            int firstCoef = firstCoefs[paramsBase + b];
            long blockBase = coefsBase + (long)b * 16;
            int eob = Vp8CoefBlockDecoderGpu.Decode(
                ref state, inBuf, probsFlat, constsFlat,
                ctx, firstCoef, coefsOut, blockBase);
            eobsOut[eobsBase + b] = eob;
        }
    }

    /// <summary>Release kernel resources.</summary>
    public void Dispose() { /* auto-grouped */ }
}
