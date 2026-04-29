// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Round-trip kernel for FLAC bit writer + reader GPU verification.
// Writes a sequence of unsigned ints (each at a specified bit width)
// then reads them back. Lets the host verify (a) bytes match a CPU
// reference encoded sequence, (b) decoded values match input.

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs.Audio.Flac;

/// <summary>
/// Drives an end-to-end FlacBitWriterGpu -&gt; FlacBitReaderGpu
/// round-trip on the accelerator.
/// </summary>
public sealed class FlacBitRoundTripKernel : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Action<
        Index1D,
        ArrayView<uint>, ArrayView<int>, ArrayView<uint>,
        ArrayView<byte>, ArrayView<long>,
        int> _kernel;

    /// <summary>Compile.</summary>
    public FlacBitRoundTripKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<uint>, ArrayView<int>, ArrayView<uint>,
            ArrayView<byte>, ArrayView<long>,
            int>(RoundTripKernel);
    }

    /// <summary>
    /// Write <paramref name="symCount"/> values from <paramref name="inputValues"/>
    /// each at <paramref name="inputBits"/>[i] bits, then read them back into
    /// <paramref name="decodedValues"/>. <paramref name="scratch"/> receives the
    /// written byte sequence; <paramref name="outLen"/>[0] = bytes written.
    /// </summary>
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
        // Write phase.
        var w = FlacBitWriterGpu.Init();
        for (int i = 0; i < symCount; i++)
        {
            FlacBitWriterGpu.Write(ref w, scratch, inputValues[i], inputBits[i]);
        }
        FlacBitWriterGpu.AlignToByte(ref w, scratch);
        outLen[0] = w.OutLen;

        // Read phase.
        var r = FlacBitReaderGpu.Init((int)w.OutLen);
        for (int i = 0; i < symCount; i++)
        {
            decodedValues[i] = FlacBitReaderGpu.ReadBits(ref r, scratch, inputBits[i]);
        }
    }

    /// <summary>Release kernel resources.</summary>
    public void Dispose() { /* auto-grouped */ }
}
