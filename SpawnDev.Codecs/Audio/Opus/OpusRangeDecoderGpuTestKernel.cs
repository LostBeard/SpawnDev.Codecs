// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Test/integration kernel that drives an end-to-end DecodeIcdf
// sequence on the accelerator using OpusRangeDecoderGpu. Used to
// verify bit-exact agreement of the GPU decoder with the CPU
// reference (`OpusRangeDecoder`).
//
// Single thread, resumable: the range decoder state lives in a one-element device buffer, each launch decodes at
// most StepSymbols symbols and continues where the previous launch stopped (the launch with start 0 initialises
// the decoder from the packet), so no launch grows with the symbol count. The launch count follows from the count
// (no readback); the host paces the launches with LaunchPacer.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Audio.Opus;

/// <summary>
/// Drives a sequence of <see cref="OpusRangeDecoderGpu.DecodeIcdf"/>
/// calls on the accelerator. Used to verify the GPU decoder matches
/// the CPU `OpusRangeDecoder` bit-exactly.
/// </summary>
public sealed class OpusRangeDecoderGpuTestKernel : IDisposable
{
    /// <summary>Symbols per launch.</summary>
    public const int StepSymbols = 4096;

    private readonly Accelerator _accelerator;
    private readonly MemoryBuffer1D<OpusRangeDecoderGpuState, Stride1D.Dense> _state;

    private readonly Action<
        Index1D,
        ArrayView<byte>, int, int,
        ArrayView<byte>, int, int,
        ArrayView<int>, int,
        ArrayView<OpusRangeDecoderGpuState>, int> _kernel;

    private readonly Lazy<Action<
        Index1D,
        ArrayView<byte>, int, int,
        int,
        ArrayView<int>, int,
        ArrayView<OpusRangeDecoderGpuState>, int>> _bitLogPKernel;

    private readonly Lazy<Action<
        Index1D,
        ArrayView<byte>, int, int,
        ArrayView<uint>,
        ArrayView<uint>, int,
        ArrayView<OpusRangeDecoderGpuState>, int>> _uintKernel;

    /// <summary>Compile the kernel on the supplied accelerator.</summary>
    public OpusRangeDecoderGpuTestKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _state = accelerator.Allocate1D<OpusRangeDecoderGpuState>(1);
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<byte>, int, int,
            ArrayView<byte>, int, int,
            ArrayView<int>, int,
            ArrayView<OpusRangeDecoderGpuState>, int>(OpusRangeDecodeIcdfKernel);
        _bitLogPKernel = new Lazy<Action<Index1D, ArrayView<byte>, int, int, int, ArrayView<int>, int, ArrayView<OpusRangeDecoderGpuState>, int>>(
            () => accelerator.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView<byte>, int, int,
                int,
                ArrayView<int>, int,
                ArrayView<OpusRangeDecoderGpuState>, int>(OpusRangeDecodeBitLogPKernel));
        _uintKernel = new Lazy<Action<Index1D, ArrayView<byte>, int, int, ArrayView<uint>, ArrayView<uint>, int, ArrayView<OpusRangeDecoderGpuState>, int>>(
            () => accelerator.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView<byte>, int, int,
                ArrayView<uint>,
                ArrayView<uint>, int,
                ArrayView<OpusRangeDecoderGpuState>, int>(OpusRangeDecodeUintKernel));
    }

    private static int Steps(int count) => Math.Max(1, (count + StepSymbols - 1) / StepSymbols);

    /// <summary>
    /// Decode <paramref name="symbolCount"/> uniform-distributed uints, one per
    /// <paramref name="ftPerSymbol"/> entry, into <paramref name="decodedOut"/>. Completes when done.
    /// </summary>
    public async Task RunDecodeUintAsync(
        ArrayView<byte> packet, int packetStart, int packetStorage,
        ArrayView<uint> ftPerSymbol,
        ArrayView<uint> decodedOut, int symbolCount)
    {
        if (symbolCount < 0) throw new ArgumentOutOfRangeException(nameof(symbolCount));
        if (decodedOut.Length < symbolCount)
            throw new ArgumentException("decodedOut too short.", nameof(decodedOut));
        var pacer = new LaunchPacer(_accelerator);
        for (int k = 0; k < Steps(symbolCount); k++)
        {
            _uintKernel.Value(1, packet, packetStart, packetStorage,
                ftPerSymbol, decodedOut, symbolCount, _state.View, k * StepSymbols);
            await pacer.LaunchedAsync();
        }
        await _accelerator.SynchronizeAsync();
    }

    /// <summary>Symbols [start, start + StepSymbols) of the uint sequence.</summary>
    private static void OpusRangeDecodeUintKernel(
        Index1D _,
        ArrayView<byte> packet, int packetStart, int packetStorage,
        ArrayView<uint> ftPerSymbol,
        ArrayView<uint> decodedOut, int symbolCount,
        ArrayView<OpusRangeDecoderGpuState> stateBuf, int start)
    {
        var state = start == 0
            ? OpusRangeDecoderGpu.Init(packet, packetStart, (uint)packetStorage)
            : stateBuf[0];
        int end = start + StepSymbols;
        if (end > symbolCount) end = symbolCount;
        for (int i = start; i < end; i++)
        {
            decodedOut[i] = OpusRangeDecoderGpu.DecodeUint(
                ref state, packet, packetStart, (uint)packetStorage,
                ftPerSymbol[i]);
        }
        stateBuf[0] = state;
    }

    /// <summary>
    /// Decode <paramref name="bitCount"/> bits with fixed log-probability
    /// <paramref name="logp"/> into <paramref name="decodedOut"/> (0 / 1 each). Completes when done.
    /// </summary>
    public async Task RunDecodeBitLogPAsync(
        ArrayView<byte> packet, int packetStart, int packetStorage,
        int logp,
        ArrayView<int> decodedOut, int bitCount)
    {
        if (bitCount < 0) throw new ArgumentOutOfRangeException(nameof(bitCount));
        if (decodedOut.Length < bitCount)
            throw new ArgumentException("decodedOut too short.", nameof(decodedOut));
        var pacer = new LaunchPacer(_accelerator);
        for (int k = 0; k < Steps(bitCount); k++)
        {
            _bitLogPKernel.Value(1, packet, packetStart, packetStorage,
                logp, decodedOut, bitCount, _state.View, k * StepSymbols);
            await pacer.LaunchedAsync();
        }
        await _accelerator.SynchronizeAsync();
    }

    /// <summary>Bits [start, start + StepSymbols) of the logp sequence.</summary>
    private static void OpusRangeDecodeBitLogPKernel(
        Index1D _,
        ArrayView<byte> packet, int packetStart, int packetStorage,
        int logp,
        ArrayView<int> decodedOut, int bitCount,
        ArrayView<OpusRangeDecoderGpuState> stateBuf, int start)
    {
        var state = start == 0
            ? OpusRangeDecoderGpu.Init(packet, packetStart, (uint)packetStorage)
            : stateBuf[0];
        int end = start + StepSymbols;
        if (end > bitCount) end = bitCount;
        for (int i = start; i < end; i++)
        {
            decodedOut[i] = OpusRangeDecoderGpu.DecodeBitLogP(
                ref state, packet, packetStart, (uint)packetStorage, logp);
        }
        stateBuf[0] = state;
    }

    /// <summary>
    /// Decode <paramref name="symbolCount"/> ICDF symbols from
    /// <paramref name="packet"/> using a fixed <paramref name="icdf"/>
    /// table. Outputs land in <paramref name="decodedOut"/>. Completes when done.
    /// </summary>
    public async Task RunAsync(
        ArrayView<byte> packet, int packetStart, int packetStorage,
        ArrayView<byte> icdf, int icdfOffset, int ftb,
        ArrayView<int> decodedOut, int symbolCount)
    {
        if (symbolCount < 0) throw new ArgumentOutOfRangeException(nameof(symbolCount));
        if (decodedOut.Length < symbolCount)
            throw new ArgumentException("decodedOut too short.", nameof(decodedOut));
        var pacer = new LaunchPacer(_accelerator);
        for (int k = 0; k < Steps(symbolCount); k++)
        {
            _kernel(1,
                packet, packetStart, packetStorage,
                icdf, icdfOffset, ftb,
                decodedOut, symbolCount, _state.View, k * StepSymbols);
            await pacer.LaunchedAsync();
        }
        await _accelerator.SynchronizeAsync();
    }

    /// <summary>Symbols [start, start + StepSymbols) of the ICDF sequence.</summary>
    private static void OpusRangeDecodeIcdfKernel(
        Index1D _,
        ArrayView<byte> packet, int packetStart, int packetStorage,
        ArrayView<byte> icdf, int icdfOffset, int ftb,
        ArrayView<int> decodedOut, int symbolCount,
        ArrayView<OpusRangeDecoderGpuState> stateBuf, int start)
    {
        var state = start == 0
            ? OpusRangeDecoderGpu.Init(packet, packetStart, (uint)packetStorage)
            : stateBuf[0];
        int end = start + StepSymbols;
        if (end > symbolCount) end = symbolCount;
        for (int i = start; i < end; i++)
        {
            int s = OpusRangeDecoderGpu.DecodeIcdf(
                ref state, packet, packetStart, (uint)packetStorage,
                icdf, icdfOffset, ftb);
            decodedOut[i] = s;
        }
        stateBuf[0] = state;
    }

    /// <summary>Release.</summary>
    public void Dispose() => _state.Dispose();
}
