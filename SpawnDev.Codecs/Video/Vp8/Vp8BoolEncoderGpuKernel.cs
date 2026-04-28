// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Test/integration kernel that exercises Vp8BoolEncoderGpu by
// encoding a sequence of (bit, prob) pairs from a per-stream input
// view into a per-stream output buffer. One thread per stream;
// streams run independently. This is the foundational test for the
// GPU-resident range coder.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>
/// Encodes batches of (bit, prob) pairs through <see cref="Vp8BoolEncoderGpu"/>.
/// One thread per stream; each stream encodes its own (bit, prob)
/// sequence into its own output buffer slice. Used to verify
/// bit-exact agreement with <see cref="Vp8BoolEncoder"/>.
/// </summary>
public sealed class Vp8BoolEncoderTestKernel : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<long>, int, int, int> _kernel;

    /// <summary>Compile the kernel.</summary>
    public Vp8BoolEncoderTestKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<long>, int, int, int>(EncodeStreamsKernel);
    }

    /// <summary>
    /// Encode <paramref name="streamCount"/> independent (bit, prob)
    /// streams. Each stream encodes <paramref name="bitsPerStream"/>
    /// pairs from <paramref name="bits"/> + <paramref name="probs"/>.
    /// Stream i reads bits[i * bitsPerStream + 0..bitsPerStream-1].
    /// Output: stream i's bytes go to outBuf[i * outBufStride + 0..],
    /// with the actual byte count returned in outLens[i].
    /// </summary>
    public void Run(
        ArrayView<byte> bits,
        ArrayView<byte> probs,
        ArrayView<byte> outBuf,
        int outBufStride,
        ArrayView<long> outLens,
        int streamCount,
        int bitsPerStream)
    {
        if (streamCount < 0) throw new ArgumentOutOfRangeException(nameof(streamCount));
        if (streamCount == 0) return;
        if (bitsPerStream < 0) throw new ArgumentOutOfRangeException(nameof(bitsPerStream));
        if (outLens.Length < streamCount)
            throw new ArgumentException("outLens too short.", nameof(outLens));
        if (outBuf.Length < (long)streamCount * outBufStride)
            throw new ArgumentException("outBuf too short for streamCount*outBufStride.", nameof(outBuf));
        _kernel(streamCount, bits, probs, outBuf, outLens, streamCount, bitsPerStream, outBufStride);
    }

    private static void EncodeStreamsKernel(
        Index1D streamIdx,
        ArrayView<byte> bits,
        ArrayView<byte> probs,
        ArrayView<byte> outBuf,
        ArrayView<long> outLens,
        int streamCount,
        int bitsPerStream,
        int outBufStride)
    {
        int idx = streamIdx;
        if (idx >= streamCount) return;
        long bitsBase = (long)idx * bitsPerStream;
        long outBase = (long)idx * outBufStride;
        var streamOut = outBuf.SubView(outBase, outBufStride);

        var state = Vp8BoolEncoderGpu.Init();
        for (int b = 0; b < bitsPerStream; b++)
        {
            int bit = bits[bitsBase + b];
            int prob = probs[bitsBase + b];
            Vp8BoolEncoderGpu.EncodeBool(ref state, streamOut, bit, prob);
        }
        Vp8BoolEncoderGpu.Stop(ref state, streamOut);

        outLens[idx] = state.OutLen;
    }

    /// <summary>Release kernel resources.</summary>
    public void Dispose() { /* auto-grouped */ }
}
