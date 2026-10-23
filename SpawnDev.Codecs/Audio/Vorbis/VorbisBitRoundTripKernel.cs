// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Round-trip kernel for the Vorbis bit reader/writer GPU pair.
// Encodes a sequence of (value, bits) pairs LSB-first via
// VorbisBitWriterGpu then reads them back via VorbisBitReaderGpu.
// Cross-backend verification companion to FlacBitRoundTripKernel.
//
// Resumable: the writer and reader states live in a small device buffer; each launch writes (then reads) at most
// StepSymbols symbols and continues where the previous launch stopped, so no launch grows with the sequence. The
// launch count follows from the symbol count (no readback); the host paces the launches with LaunchPacer.

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs.Audio.Vorbis;

/// <summary>
/// Drives an end-to-end VorbisBitWriterGpu -&gt; VorbisBitReaderGpu
/// round-trip on the accelerator.
/// </summary>
public sealed class VorbisBitRoundTripKernel : IDisposable
{
    /// <summary>Symbols per write / read launch.</summary>
    public const int StepSymbols = 8192;

    // State slots: writer CurrentByte, BitPos, OutLen; reader BytePos, BitPos.
    private const int SWCur = 0, SWBit = 1, SWLen = 2, SRByte = 3, SRBit = 4, StateInts = 8;

    private readonly Accelerator _accelerator;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _state;
    private readonly Action<Index1D, ArrayView<uint>, ArrayView<int>, ArrayView<byte>, ArrayView<int>, int, int> _writeKernel;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<long>, ArrayView<int>> _finishKernel;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<uint>, ArrayView<byte>, ArrayView<int>, int, int> _readKernel;

    /// <summary>Compile.</summary>
    public VorbisBitRoundTripKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _state = accelerator.Allocate1D<int>(StateInts);
        _writeKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<uint>, ArrayView<int>, ArrayView<byte>, ArrayView<int>, int, int>(VorbisBitRoundTripWriteKernel);
        _finishKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<long>, ArrayView<int>>(VorbisBitRoundTripFinishKernel);
        _readKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, ArrayView<uint>, ArrayView<byte>, ArrayView<int>, int, int>(VorbisBitRoundTripReadKernel);
    }

    /// <summary>
    /// Write <paramref name="symCount"/> values from <paramref name="inputValues"/>
    /// each at <paramref name="inputBits"/>[i] bits, then read them back into
    /// <paramref name="decodedValues"/>. <paramref name="scratch"/> receives the
    /// written byte sequence; <paramref name="outLen"/>[0] = bytes written. Completes when done.
    /// </summary>
    public async Task RunAsync(
        ArrayView<uint> inputValues, ArrayView<int> inputBits, ArrayView<uint> decodedValues,
        ArrayView<byte> scratch, ArrayView<long> outLen,
        int symCount)
    {
        if (symCount < 0) throw new ArgumentOutOfRangeException(nameof(symCount));
        var pacer = new LaunchPacer(_accelerator);
        int steps = Math.Max(1, (symCount + StepSymbols - 1) / StepSymbols);
        for (int k = 0; k < steps; k++)
        {
            _writeKernel(1, inputValues, inputBits, scratch, _state.View, symCount, k * StepSymbols);
            await pacer.LaunchedAsync();
        }
        _finishKernel(1, scratch, outLen, _state.View);
        await pacer.LaunchedAsync();
        for (int k = 0; k < steps; k++)
        {
            _readKernel(1, inputBits, decodedValues, scratch, _state.View, symCount, k * StepSymbols);
            await pacer.LaunchedAsync();
        }
        await _accelerator.SynchronizeAsync();
    }

    /// <summary>Writes symbols [start, start + StepSymbols) (the launch with start 0 starts a fresh writer).</summary>
    private static void VorbisBitRoundTripWriteKernel(
        Index1D _,
        ArrayView<uint> inputValues, ArrayView<int> inputBits, ArrayView<byte> scratch, ArrayView<int> state,
        int symCount, int start)
    {
        var w = VorbisBitWriterGpu.Init();
        if (start != 0)
        {
            w.CurrentByte = state[SWCur];
            w.BitPos = state[SWBit];
            w.OutLen = state[SWLen];
        }
        int end = start + StepSymbols;
        if (end > symCount) end = symCount;
        for (int i = start; i < end; i++)
        {
            VorbisBitWriterGpu.WriteBits(ref w, scratch, inputValues[i], inputBits[i]);
        }
        state[SWCur] = w.CurrentByte;
        state[SWBit] = w.BitPos;
        state[SWLen] = (int)w.OutLen;
    }

    /// <summary>Flushes the writer, stores the length, starts the reader at byte 0.</summary>
    private static void VorbisBitRoundTripFinishKernel(Index1D _, ArrayView<byte> scratch, ArrayView<long> outLen, ArrayView<int> state)
    {
        var w = VorbisBitWriterGpu.Init();
        w.CurrentByte = state[SWCur];
        w.BitPos = state[SWBit];
        w.OutLen = state[SWLen];
        VorbisBitWriterGpu.Finish(ref w, scratch);
        outLen[0] = w.OutLen;
        state[SWLen] = (int)w.OutLen;
        state[SRByte] = 0;
        state[SRBit] = 0;
    }

    /// <summary>Reads symbols [start, start + StepSymbols) back.</summary>
    private static void VorbisBitRoundTripReadKernel(
        Index1D _,
        ArrayView<int> inputBits, ArrayView<uint> decodedValues, ArrayView<byte> scratch, ArrayView<int> state,
        int symCount, int start)
    {
        var r = VorbisBitReaderGpu.Init(state[SWLen]);
        r.BytePos = state[SRByte];
        r.BitPos = state[SRBit];
        int end = start + StepSymbols;
        if (end > symCount) end = symCount;
        for (int i = start; i < end; i++)
        {
            decodedValues[i] = VorbisBitReaderGpu.ReadBits(ref r, scratch, inputBits[i]);
        }
        state[SRByte] = r.BytePos;
        state[SRBit] = r.BitPos;
    }

    /// <summary>Release kernel resources.</summary>
    public void Dispose() => _state.Dispose();
}
