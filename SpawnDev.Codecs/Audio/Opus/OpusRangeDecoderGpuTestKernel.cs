// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Test/integration kernel that drives an end-to-end DecodeIcdf
// sequence on the accelerator using OpusRangeDecoderGpu. Used to
// verify bit-exact agreement of the GPU decoder with the CPU
// reference (`OpusRangeDecoder`).
//
// Single-thread per dispatch.

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
    private readonly Action<
        Index1D,
        ArrayView<byte>, int, int,
        ArrayView<byte>, int, int,
        ArrayView<int>, int> _kernel;

    private readonly Action<
        Index1D,
        ArrayView<byte>, int, int,
        int,
        ArrayView<int>, int> _bitLogPKernel;

    private readonly Action<
        Index1D,
        ArrayView<byte>, int, int,
        ArrayView<uint>,
        ArrayView<uint>, int> _uintKernel;

    /// <summary>Compile.</summary>
    public OpusRangeDecoderGpuTestKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<byte>, int, int,
            ArrayView<byte>, int, int,
            ArrayView<int>, int>(DecodeIcdfKernel);
        _bitLogPKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<byte>, int, int,
            int,
            ArrayView<int>, int>(DecodeBitLogPKernel);
        _uintKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<byte>, int, int,
            ArrayView<uint>,
            ArrayView<uint>, int>(DecodeUintKernel);
    }

    /// <summary>
    /// Decode <paramref name="symbolCount"/> uniform integers from
    /// <paramref name="packet"/> via repeated <c>DecodeUint</c> calls.
    /// Each symbol's <c>ft</c> value is read from
    /// <paramref name="ftPerSymbol"/>[i] (so a varying-range stream can be
    /// tested with one dispatch).
    /// </summary>
    public void RunDecodeUint(
        ArrayView<byte> packet, int packetStart, int packetStorage,
        ArrayView<uint> ftPerSymbol,
        ArrayView<uint> decodedOut, int symbolCount)
    {
        if (symbolCount < 0) throw new ArgumentOutOfRangeException(nameof(symbolCount));
        if (decodedOut.Length < symbolCount)
            throw new ArgumentException("decodedOut too short.", nameof(decodedOut));
        _uintKernel(1, packet, packetStart, packetStorage,
            ftPerSymbol, decodedOut, symbolCount);
    }

    private static void DecodeUintKernel(
        Index1D _,
        ArrayView<byte> packet, int packetStart, int packetStorage,
        ArrayView<uint> ftPerSymbol,
        ArrayView<uint> decodedOut, int symbolCount)
    {
        var state = OpusRangeDecoderGpu.Init(packet, packetStart, (uint)packetStorage);
        for (int i = 0; i < symbolCount; i++)
        {
            decodedOut[i] = OpusRangeDecoderGpu.DecodeUint(
                ref state, packet, packetStart, (uint)packetStorage,
                ftPerSymbol[i]);
        }
    }

    /// <summary>
    /// Decode <paramref name="bitCount"/> bits from <paramref name="packet"/>
    /// via repeated <c>DecodeBitLogP</c> calls at the given
    /// <paramref name="logp"/> probability. Decoded bits (0/1) go into
    /// <paramref name="decodedOut"/>. Used by CELT silence-flag /
    /// transient-flag / intra-flag / post-filter-flag bit decodes.
    /// </summary>
    public void RunDecodeBitLogP(
        ArrayView<byte> packet, int packetStart, int packetStorage,
        int logp,
        ArrayView<int> decodedOut, int bitCount)
    {
        if (bitCount < 0) throw new ArgumentOutOfRangeException(nameof(bitCount));
        if (decodedOut.Length < bitCount)
            throw new ArgumentException("decodedOut too short.", nameof(decodedOut));
        _bitLogPKernel(1, packet, packetStart, packetStorage,
            logp, decodedOut, bitCount);
    }

    private static void DecodeBitLogPKernel(
        Index1D _,
        ArrayView<byte> packet, int packetStart, int packetStorage,
        int logp,
        ArrayView<int> decodedOut, int bitCount)
    {
        var state = OpusRangeDecoderGpu.Init(packet, packetStart, (uint)packetStorage);
        for (int i = 0; i < bitCount; i++)
        {
            decodedOut[i] = OpusRangeDecoderGpu.DecodeBitLogP(
                ref state, packet, packetStart, (uint)packetStorage, logp);
        }
    }

    /// <summary>
    /// Decode <paramref name="symbolCount"/> symbols from
    /// <paramref name="packet"/> via repeated <c>DecodeIcdf</c> calls,
    /// using <paramref name="icdf"/>[<paramref name="icdfOffset"/>..]
    /// at <paramref name="ftb"/> bits of precision. Decoded symbols go
    /// into <paramref name="decodedOut"/>.
    /// </summary>
    public void Run(
        ArrayView<byte> packet, int packetStart, int packetStorage,
        ArrayView<byte> icdf, int icdfOffset, int ftb,
        ArrayView<int> decodedOut, int symbolCount)
    {
        if (symbolCount < 0) throw new ArgumentOutOfRangeException(nameof(symbolCount));
        if (decodedOut.Length < symbolCount)
            throw new ArgumentException("decodedOut too short.", nameof(decodedOut));
        _kernel(1,
            packet, packetStart, packetStorage,
            icdf, icdfOffset, ftb,
            decodedOut, symbolCount);
    }

    private static void DecodeIcdfKernel(
        Index1D _,
        ArrayView<byte> packet, int packetStart, int packetStorage,
        ArrayView<byte> icdf, int icdfOffset, int ftb,
        ArrayView<int> decodedOut, int symbolCount)
    {
        var state = OpusRangeDecoderGpu.Init(packet, packetStart, (uint)packetStorage);
        for (int i = 0; i < symbolCount; i++)
        {
            int s = OpusRangeDecoderGpu.DecodeIcdf(
                ref state, packet, packetStart, (uint)packetStorage,
                icdf, icdfOffset, ftb);
            decodedOut[i] = s;
        }
    }

    /// <summary>Release kernel resources.</summary>
    public void Dispose() { /* auto-grouped */ }
}
