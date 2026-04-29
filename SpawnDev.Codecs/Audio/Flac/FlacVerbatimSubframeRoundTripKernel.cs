// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Round-trip kernel for FLAC VERBATIM subframe encoder + decoder GPU
// pair. Encodes one subframe of N samples at given bps then decodes
// it back. Verifies the GPU pair produces decode == encode for any
// signed sample range.

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs.Audio.Flac;

/// <summary>
/// Drives an end-to-end FlacVerbatimSubframeGpu.Encode -&gt; .Decode
/// round-trip on the accelerator.
/// </summary>
public sealed class FlacVerbatimSubframeRoundTripKernel : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Action<
        Index1D,
        ArrayView<int>, ArrayView<int>,
        ArrayView<byte>, ArrayView<long>, ArrayView<int>,
        int, int> _kernel;

    /// <summary>Compile.</summary>
    public FlacVerbatimSubframeRoundTripKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<int>, ArrayView<int>,
            ArrayView<byte>, ArrayView<long>, ArrayView<int>,
            int, int>(RoundTripKernel);
    }

    /// <summary>
    /// Encode + decode one VERBATIM subframe. Outputs:
    /// <c>outLen[0]</c> = encoded byte count;
    /// <c>statusOut[0]</c> = decoder status (1 = success, 0 = header mismatch).
    /// </summary>
    public void Run(
        ArrayView<int> samples, ArrayView<int> decodedSamples,
        ArrayView<byte> scratch, ArrayView<long> outLen, ArrayView<int> statusOut,
        int sampleCount, int bps)
    {
        _kernel(1, samples, decodedSamples, scratch, outLen, statusOut, sampleCount, bps);
    }

    private static void RoundTripKernel(
        Index1D _,
        ArrayView<int> samples, ArrayView<int> decodedSamples,
        ArrayView<byte> scratch, ArrayView<long> outLen, ArrayView<int> statusOut,
        int sampleCount, int bps)
    {
        var w = FlacBitWriterGpu.Init();
        FlacVerbatimSubframeGpu.Encode(ref w, scratch, samples, 0, sampleCount, bps);
        FlacBitWriterGpu.AlignToByte(ref w, scratch);
        outLen[0] = w.OutLen;

        var r = FlacBitReaderGpu.Init((int)w.OutLen);
        statusOut[0] = FlacVerbatimSubframeGpu.Decode(ref r, scratch, decodedSamples, 0, sampleCount, bps);
    }

    /// <summary>Release kernel resources.</summary>
    public void Dispose() { /* auto-grouped */ }
}
