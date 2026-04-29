// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Test/integration kernel that drives an end-to-end round-trip of
// the AV1 range coder on the accelerator: encodes a sequence of
// binary symbols via Av1RangeEncoderGpu, decodes them back via
// Av1RangeDecoderGpu in the same dispatch, writes the decoded
// values to an output buffer for the host to verify.
//
// Single-thread per dispatch. Encoder + decoder share the same
// scratch byte buffer (encoder writes bytes; decoder reads them).
// The intermediate byte count is reported via outLen.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>
/// Drives an end-to-end Av1RangeEncoderGpu -> Av1RangeDecoderGpu
/// round-trip on the accelerator. Used to verify bit-exact agreement
/// of both halves with the CPU reference.
/// </summary>
public sealed class Av1RangeCoderRoundTripKernel : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Action<
        Index1D,
        ArrayView<int>, ArrayView<uint>, ArrayView<int>,
        ArrayView<byte>, ArrayView<long>,
        int> _kernel;

    /// <summary>Compile.</summary>
    public Av1RangeCoderRoundTripKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<int>, ArrayView<uint>, ArrayView<int>,
            ArrayView<byte>, ArrayView<long>,
            int>(RoundTripKernel);
    }

    /// <summary>
    /// Encode <paramref name="symbolCount"/> binary symbols via the
    /// range encoder, then decode them back via the range decoder.
    /// </summary>
    /// <param name="encodeBits">Input bits (0 or 1) to encode.</param>
    /// <param name="encodeProbs">q15 probabilities of 1 (in (0, 32768)) per symbol.</param>
    /// <param name="decodeBitsOut">Decoded bits written here.</param>
    /// <param name="scratchBytes">Intermediate byte buffer the encoder writes + decoder reads.</param>
    /// <param name="outLen">[0] = encoded byte count.</param>
    public void Run(
        ArrayView<int> encodeBits,
        ArrayView<uint> encodeProbs,
        ArrayView<int> decodeBitsOut,
        ArrayView<byte> scratchBytes,
        ArrayView<long> outLen,
        int symbolCount)
    {
        if (symbolCount < 0) throw new ArgumentOutOfRangeException(nameof(symbolCount));
        if (encodeBits.Length < symbolCount)
            throw new ArgumentException("encodeBits too short.", nameof(encodeBits));
        if (encodeProbs.Length < symbolCount)
            throw new ArgumentException("encodeProbs too short.", nameof(encodeProbs));
        if (decodeBitsOut.Length < symbolCount)
            throw new ArgumentException("decodeBitsOut too short.", nameof(decodeBitsOut));
        if (outLen.Length < 1)
            throw new ArgumentException("outLen too short.", nameof(outLen));
        _kernel(1, encodeBits, encodeProbs, decodeBitsOut, scratchBytes, outLen, symbolCount);
    }

    private static void RoundTripKernel(
        Index1D _,
        ArrayView<int> encodeBits,
        ArrayView<uint> encodeProbs,
        ArrayView<int> decodeBitsOut,
        ArrayView<byte> scratchBytes,
        ArrayView<long> outLen,
        int symbolCount)
    {
        // Encode phase.
        var encState = Av1RangeEncoderGpu.Init();
        for (int i = 0; i < symbolCount; i++)
        {
            int bit = encodeBits[i];
            uint prob = encodeProbs[i];
            Av1RangeEncoderGpu.EncodeBoolQ15(ref encState, scratchBytes, bit, prob);
        }
        Av1RangeEncoderGpu.Done(ref encState, scratchBytes);
        outLen[0] = encState.OutLen;

        // Decode phase.
        var decState = Av1RangeDecoderGpu.Init(scratchBytes, 0, (int)encState.OutLen);
        for (int i = 0; i < symbolCount; i++)
        {
            uint prob = encodeProbs[i];
            int bit = Av1RangeDecoderGpu.DecodeBoolQ15(ref decState, scratchBytes, prob);
            decodeBitsOut[i] = bit;
        }
    }

    /// <summary>Release kernel resources.</summary>
    public void Dispose() { /* auto-grouped */ }
}
