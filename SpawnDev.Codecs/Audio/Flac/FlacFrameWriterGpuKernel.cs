// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Multi-frame ILGPU driver for FlacFrameWriterGpu. Dispatches with
// extent=frameCount, encoding all frames in parallel into a contiguous
// strided output buffer (one per-frame slot of `outBufStride` bytes each).
// FLAC frames are independent of each other in the spec, so frame-parallel
// is the natural axis (each frame gets its own thread, its own samples slice,
// its own output slice).

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs.Audio.Flac;

/// <summary>
/// Multi-frame FLAC encoder kernel. Each thread encodes one frame
/// independently into its own slot of the strided output buffer.
/// </summary>
public sealed class FlacFrameWriterGpuKernel : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Action<
        Index1D, ArrayView<int>, ArrayView<byte>, ArrayView<long>,
        int, int, int, ulong, int, int> _kernel;

    /// <summary>Compile.</summary>
    public FlacFrameWriterGpuKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, ArrayView<byte>, ArrayView<long>,
            int, int, int, ulong, int, int>(FrameKernel);
    }

    /// <summary>
    /// Encode one FLAC frame on the accelerator (legacy single-frame call).
    /// </summary>
    public void Run(
        ArrayView<int> samples,
        ArrayView<byte> outBuf,
        ArrayView<long> outLen,
        int blockSize, int channels, int bps,
        ulong frameNumber)
    {
        // frameCount=1, samplesStride=0, outBufStride=0 -> single-frame mode.
        _kernel(1, samples, outBuf, outLen,
            blockSize, channels, bps, frameNumber, 0, 0);
    }

    /// <summary>
    /// Batch-encode <paramref name="frameCount"/> FLAC frames in parallel.
    /// Thread <c>i</c> reads from <c>samples[i*samplesStride..]</c>
    /// (samplesStride samples per frame) and writes to
    /// <c>outBuf[i*outBufStride..]</c> with the per-frame byte length
    /// stored in <c>outLen[i]</c>.
    /// </summary>
    public void RunBatch(
        ArrayView<int> samples,
        ArrayView<byte> outBuf,
        ArrayView<long> outLen,
        int blockSize, int channels, int bps,
        ulong startFrameNumber,
        int frameCount,
        int samplesStride,
        int outBufStride)
    {
        if (frameCount <= 0) throw new ArgumentOutOfRangeException(nameof(frameCount));
        _kernel(frameCount, samples, outBuf, outLen,
            blockSize, channels, bps, startFrameNumber,
            samplesStride, outBufStride);
    }

    private static void FrameKernel(
        Index1D idx,
        ArrayView<int> samples,
        ArrayView<byte> outBuf,
        ArrayView<long> outLen,
        int blockSize, int channels, int bps,
        ulong startFrameNumber,
        int samplesStride, int outBufStride)
    {
        int i = idx.X;
        long sampleOff = (long)i * samplesStride;
        long outOff = (long)i * outBufStride;
        long len = FlacFrameWriterGpu.EncodeFrame(
            samples, sampleOff, blockSize, channels, bps, startFrameNumber + (ulong)i,
            outBuf, outOff);
        outLen[i] = len;
    }

    /// <summary>Release kernel resources.</summary>
    public void Dispose() { /* auto-grouped */ }
}
