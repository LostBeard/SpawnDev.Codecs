// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Round-trip kernel for FLAC VERBATIM subframe encoder + decoder GPU
// pair. Encodes one subframe of N samples at given bps then decodes
// it back. Verifies the GPU pair produces decode == encode for any
// signed sample range.
//
// Runs the production (parallel) forms: one thread per encoded byte (FlacVerbatimSubframeGpu.PackByteAt), one
// thread for the header check + length, one thread per decoded sample (UnpackSampleAt). Three launches whatever
// the sample count, each thread O(bps) work.

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs.Audio.Flac;

/// <summary>
/// Drives an end-to-end FlacVerbatimSubframeGpu pack -&gt; unpack
/// round-trip on the accelerator.
/// </summary>
public sealed class FlacVerbatimSubframeRoundTripKernel : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<byte>, int, int> _packKernel;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<long>, ArrayView<int>, int, int> _headerKernel;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, int, int> _unpackKernel;

    /// <summary>Compile.</summary>
    public FlacVerbatimSubframeRoundTripKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _packKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, ArrayView<byte>, int, int>(FlacVerbatimPackKernel);
        _headerKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<long>, ArrayView<int>, int, int>(FlacVerbatimHeaderKernel);
        _unpackKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, int, int>(FlacVerbatimUnpackKernel);
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
        int bytes = checked((int)FlacVerbatimSubframeGpu.SubframesBytes(sampleCount, 1, bps));
        _packKernel(bytes, samples, scratch, sampleCount, bps);
        _headerKernel(1, scratch, outLen, statusOut, sampleCount, bps);
        if (sampleCount > 0) _unpackKernel(sampleCount, scratch, decodedSamples, statusOut, sampleCount, bps);
    }

    /// <summary>Encoded byte <c>j</c> of the subframe.</summary>
    private static void FlacVerbatimPackKernel(
        Index1D j, ArrayView<int> samples, ArrayView<byte> scratch, int sampleCount, int bps)
    {
        scratch[j.X] = FlacVerbatimSubframeGpu.PackByteAt(samples, 0, sampleCount, sampleCount, 1, bps, j.X);
    }

    /// <summary>Encoded length and the decoder's header check (1 = VERBATIM, no wasted bits).</summary>
    private static void FlacVerbatimHeaderKernel(
        Index1D _, ArrayView<byte> scratch, ArrayView<long> outLen, ArrayView<int> statusOut, int sampleCount, int bps)
    {
        outLen[0] = FlacVerbatimSubframeGpu.SubframesBytes(sampleCount, 1, bps);
        statusOut[0] = FlacVerbatimSubframeGpu.HeaderOkAt(scratch, 0, sampleCount, bps, 0);
    }

    /// <summary>Decoded sample <c>i</c> (only after a good header, as the serial decoder).</summary>
    private static void FlacVerbatimUnpackKernel(
        Index1D i, ArrayView<byte> scratch, ArrayView<int> decodedSamples, ArrayView<int> statusOut,
        int sampleCount, int bps)
    {
        if (statusOut[0] == 0) return;
        decodedSamples[i.X] = FlacVerbatimSubframeGpu.UnpackSampleAt(scratch, 0, sampleCount, bps, 0, i.X);
    }

    /// <summary>Release kernel resources.</summary>
    public void Dispose() { /* auto-grouped */ }
}
