// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Round-trip kernel for the Vorbis bit reader/writer GPU pair.
// Encodes a sequence of (value, bits) pairs LSB-first via
// VorbisBitWriterGpu then reads them back via VorbisBitReaderGpu.
// Cross-backend verification companion to FlacBitRoundTripKernel.

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs.Audio.Vorbis;

/// <summary>
/// Drives an end-to-end VorbisBitWriterGpu -&gt; VorbisBitReaderGpu
/// round-trip on the accelerator.
/// </summary>
public sealed class VorbisBitRoundTripKernel : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Action<
        Index1D,
        ArrayView<uint>, ArrayView<int>, ArrayView<uint>,
        ArrayView<byte>, ArrayView<long>,
        int> _kernel;

    /// <summary>Compile.</summary>
    public VorbisBitRoundTripKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<uint>, ArrayView<int>, ArrayView<uint>,
            ArrayView<byte>, ArrayView<long>,
            int>(RoundTripKernel);
    }

    /// <summary>Run round-trip on <paramref name="symCount"/> symbols.</summary>
    public void Run(
        ArrayView<uint> inputValues, ArrayView<int> inputBits, ArrayView<uint> decodedValues,
        ArrayView<byte> scratch, ArrayView<long> outLen,
        int symCount)
    {
        _kernel(1, inputValues, inputBits, decodedValues, scratch, outLen, symCount);
    }

    private static void RoundTripKernel(
        Index1D _,
        ArrayView<uint> inputValues, ArrayView<int> inputBits, ArrayView<uint> decodedValues,
        ArrayView<byte> scratch, ArrayView<long> outLen,
        int symCount)
    {
        var w = VorbisBitWriterGpu.Init();
        for (int i = 0; i < symCount; i++)
        {
            VorbisBitWriterGpu.WriteBits(ref w, scratch, inputValues[i], inputBits[i]);
        }
        VorbisBitWriterGpu.Finish(ref w, scratch);
        outLen[0] = w.OutLen;

        var r = VorbisBitReaderGpu.Init((int)w.OutLen);
        for (int i = 0; i < symCount; i++)
        {
            decodedValues[i] = VorbisBitReaderGpu.ReadBits(ref r, scratch, inputBits[i]);
        }
    }

    /// <summary>Release kernel resources.</summary>
    public void Dispose() { /* auto-grouped */ }
}
