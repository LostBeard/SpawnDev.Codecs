// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Single-frame ILGPU driver for FlacFrameReaderGpu. Decodes one
// VERBATIM FLAC frame from the input bytes; writes decoded samples
// to the output buffer.

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs.Audio.Flac;

/// <summary>
/// Single-frame FLAC decoder kernel. Drives FlacFrameReaderGpu.DecodeFrame.
/// </summary>
public sealed class FlacFrameReaderGpuKernel : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Action<
        Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<long>,
        long, int, int, int, int> _kernel;

    /// <summary>Compile.</summary>
    public FlacFrameReaderGpuKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<long>,
            long, int, int, int, int>(FrameKernel);
    }

    /// <summary>
    /// Decode one FLAC frame on the accelerator. Outputs:
    /// <c>samples</c> = decoded samples (channel-major);
    /// <c>statusOut</c>[0] = 0 success or non-zero error code;
    /// <c>frameLen</c>[0] = bytes consumed (0 on error).
    /// </summary>
    public void Run(
        ArrayView<byte> data,
        ArrayView<int> samples,
        ArrayView<int> statusOut,
        ArrayView<long> frameLen,
        long frameBase, int frameLength,
        int blockSize, int channels, int bps)
    {
        _kernel(1, data, samples, statusOut, frameLen,
            frameBase, frameLength, blockSize, channels, bps);
    }

    private static void FrameKernel(
        Index1D _,
        ArrayView<byte> data,
        ArrayView<int> samples,
        ArrayView<int> statusOut,
        ArrayView<long> frameLen,
        long frameBase, int frameLength,
        int blockSize, int channels, int bps)
    {
        long len = FlacFrameReaderGpu.DecodeFrame(
            data, frameBase, frameLength,
            blockSize, channels, bps,
            samples, 0,
            statusOut);
        frameLen[0] = len;
    }

    /// <summary>Release kernel resources.</summary>
    public void Dispose() { /* auto-grouped */ }
}
