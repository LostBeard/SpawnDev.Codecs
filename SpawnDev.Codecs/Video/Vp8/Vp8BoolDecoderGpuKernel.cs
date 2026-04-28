// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Test/integration kernel for Vp8BoolDecoderGpu. One thread per
// stream; each stream decodes a known sequence of (probability) bits
// from its slice of the input buffer and writes the decoded bits to
// its slice of the output buffer.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>
/// Test kernel that exercises Vp8BoolDecoderGpu by decoding N bits
/// per stream against per-bit probabilities. One thread per stream.
/// </summary>
public sealed class Vp8BoolDecoderTestKernel : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Action<
        Index1D,
        ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<byte>,
        int, int, int, int> _kernel;

    /// <summary>Compile.</summary>
    public Vp8BoolDecoderTestKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<byte>,
            int, int, int, int>(DecodeStreamsKernel);
    }

    /// <summary>
    /// Decode <paramref name="streamCount"/> independent streams.
    /// Each stream reads from inBuf at offset i*inStride, decodes
    /// <paramref name="bitsPerStream"/> bits with the per-bit probs in
    /// probs (length = streamCount * bitsPerStream), writes decoded bits
    /// to outBits (same length).
    /// </summary>
    public void Run(
        ArrayView<byte> inBuf,
        ArrayView<int> inLens,
        ArrayView<int> probs,
        ArrayView<byte> outBits,
        int streamCount,
        int bitsPerStream,
        int inStride)
    {
        if (streamCount < 0) throw new ArgumentOutOfRangeException(nameof(streamCount));
        if (streamCount == 0) return;
        _kernel(streamCount, inBuf, inLens, probs, outBits,
            streamCount, bitsPerStream, inStride, 0);
    }

    private static void DecodeStreamsKernel(
        Index1D streamIdx,
        ArrayView<byte> inBuf,
        ArrayView<int> inLens,
        ArrayView<int> probs,
        ArrayView<byte> outBits,
        int streamCount,
        int bitsPerStream,
        int inStride,
        int unused)
    {
        int idx = streamIdx;
        if (idx >= streamCount) return;
        int inOffset = idx * inStride;
        int inLen = inLens[idx];
        long probsBase = (long)idx * bitsPerStream;
        long bitsBase = probsBase;

        var state = Vp8BoolDecoderGpu.Init(inBuf, inOffset, inLen);
        for (int b = 0; b < bitsPerStream; b++)
        {
            int prob = probs[probsBase + b];
            int bit = Vp8BoolDecoderGpu.DecodeBool(ref state, inBuf, prob);
            outBits[bitsBase + b] = (byte)bit;
        }
    }

    /// <summary>Release kernel resources.</summary>
    public void Dispose() { /* auto-grouped */ }
}
