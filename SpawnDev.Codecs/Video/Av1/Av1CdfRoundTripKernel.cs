// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Minimal DecodeCdfQ15 round-trip kernel - encodes a sequence of
// multi-syms via Av1RangeEncoderGpu.EncodeCdfQ15 then decodes them
// back via Av1RangeDecoderGpu.DecodeCdfQ15. Companion to
// Av1RangeCoderRoundTripKernel which only exercises the binary
// EncodeBoolQ15 / DecodeBoolQ15 path.

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>
/// Drives an end-to-end EncodeCdfQ15 -&gt; DecodeCdfQ15 round-trip on
/// the accelerator. Used to isolate any backend-specific issue with
/// the multi-sym CDF path that the binary-only Av1RangeCoderRoundTripKernel
/// won't catch.
/// </summary>
public sealed class Av1CdfRoundTripKernel : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Action<
        Index1D,
        ArrayView<int>, ArrayView<int>, ArrayView<ushort>,
        ArrayView<byte>, ArrayView<long>,
        int, int> _kernel;

    /// <summary>Compile.</summary>
    public Av1CdfRoundTripKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<int>, ArrayView<int>, ArrayView<ushort>,
            ArrayView<byte>, ArrayView<long>,
            int, int>(RoundTripKernel);
    }

    /// <summary>
    /// Encode <paramref name="symCount"/> symbols via EncodeCdfQ15 (each
    /// using the same icdf row), then decode them back via DecodeCdfQ15.
    /// </summary>
    public void Run(
        ArrayView<int> inputSyms, ArrayView<int> decodedSyms,
        ArrayView<ushort> icdf, ArrayView<byte> scratch, ArrayView<long> outLen,
        int symCount, int nsyms)
    {
        _kernel(1, inputSyms, decodedSyms, icdf, scratch, outLen, symCount, nsyms);
    }

    private static void RoundTripKernel(
        Index1D _,
        ArrayView<int> inputSyms, ArrayView<int> decodedSyms,
        ArrayView<ushort> icdf, ArrayView<byte> scratch, ArrayView<long> outLen,
        int symCount, int nsyms)
    {
        var re = Av1RangeEncoderGpu.Init();
        for (int i = 0; i < symCount; i++)
        {
            int sym = inputSyms[i];
            Av1RangeEncoderGpu.EncodeCdfQ15(ref re, scratch, sym, icdf, 0, nsyms);
        }
        Av1RangeEncoderGpu.Done(ref re, scratch);
        outLen[0] = re.OutLen;

        var rd = Av1RangeDecoderGpu.Init(scratch, 0, (int)re.OutLen);
        for (int i = 0; i < symCount; i++)
        {
            int sym = Av1RangeDecoderGpu.DecodeCdfQ15(ref rd, scratch, icdf, 0, nsyms);
            decodedSyms[i] = sym;
        }
    }

    /// <summary>Release kernel resources.</summary>
    public void Dispose() { /* auto-grouped */ }
}
