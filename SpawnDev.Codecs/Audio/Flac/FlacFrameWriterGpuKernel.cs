// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Single-frame ILGPU driver for FlacFrameWriterGpu. One thread per
// dispatch; encodes one FLAC frame into the supplied byte buffer.

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs.Audio.Flac;

/// <summary>
/// Single-frame FLAC encoder kernel. Drives FlacFrameWriterGpu.EncodeFrame.
/// </summary>
public sealed class FlacFrameWriterGpuKernel : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Action<
        Index1D, ArrayView<int>, ArrayView<byte>, ArrayView<long>,
        int, int, int, ulong> _kernel;

    /// <summary>Compile.</summary>
    public FlacFrameWriterGpuKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, ArrayView<byte>, ArrayView<long>,
            int, int, int, ulong>(FrameKernel);
    }

    /// <summary>
    /// Encode one FLAC frame on the accelerator. <paramref name="outLen"/>[0]
    /// receives the encoded byte count.
    /// </summary>
    public void Run(
        ArrayView<int> samples,
        ArrayView<byte> outBuf,
        ArrayView<long> outLen,
        int blockSize, int channels, int bps,
        ulong frameNumber)
    {
        _kernel(1, samples, outBuf, outLen,
            blockSize, channels, bps, frameNumber);
    }

    private static void FrameKernel(
        Index1D _,
        ArrayView<int> samples,
        ArrayView<byte> outBuf,
        ArrayView<long> outLen,
        int blockSize, int channels, int bps,
        ulong frameNumber)
    {
        long len = FlacFrameWriterGpu.EncodeFrame(
            samples, 0, blockSize, channels, bps, frameNumber,
            outBuf, 0);
        outLen[0] = len;
    }

    /// <summary>Release kernel resources.</summary>
    public void Dispose() { /* auto-grouped */ }
}
