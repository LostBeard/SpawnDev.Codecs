// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Test/integration kernel that drives `SilkGainIndicesDecoderGpu.DecodeIndices`
// on the accelerator. Used to verify bit-exact agreement of the GPU port
// with the CPU reference (`SilkGainDecoder.DecodeIndices`).

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// Drives `SilkGainIndicesDecoderGpu.DecodeIndices` on the accelerator.
/// Decodes <c>nbSubfr</c> gain indices from a packet and writes them
/// into a caller-allocated int output buffer.
/// </summary>
public sealed class SilkGainIndicesDecoderGpuTestKernel : IDisposable
{
    private readonly Action<
        Index1D,
        ArrayView<byte>, int, int,
        ArrayView<byte>, ArrayView<byte>, ArrayView<byte>,
        int, int, int,
        ArrayView<int>> _kernel;

    /// <summary>Compile.</summary>
    public SilkGainIndicesDecoderGpuTestKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<byte>, int, int,
            ArrayView<byte>, ArrayView<byte>, ArrayView<byte>,
            int, int, int,
            ArrayView<int>>(GainIndicesKernel);
    }

    /// <summary>Decode <paramref name="nbSubfr"/> gain indices from <paramref name="packet"/>.</summary>
    public void Run(
        ArrayView<byte> packet, int packetStart, int packetStorage,
        ArrayView<byte> gainIcdf,
        ArrayView<byte> deltaGainIcdf,
        ArrayView<byte> uniform8Icdf,
        int signalType, int conditional, int nbSubfr,
        ArrayView<int> output)
    {
        if (output.Length < nbSubfr)
            throw new ArgumentException("output too short.", nameof(output));
        _kernel(1,
            packet, packetStart, packetStorage,
            gainIcdf, deltaGainIcdf, uniform8Icdf,
            signalType, conditional, nbSubfr,
            output);
    }

    private static void GainIndicesKernel(
        Index1D _,
        ArrayView<byte> packet, int packetStart, int packetStorage,
        ArrayView<byte> gainIcdf,
        ArrayView<byte> deltaGainIcdf,
        ArrayView<byte> uniform8Icdf,
        int signalType, int conditional, int nbSubfr,
        ArrayView<int> output)
    {
        var state = OpusRangeDecoderGpu.Init(packet, packetStart, (uint)packetStorage);
        SilkGainIndicesDecoderGpu.DecodeIndices(
            ref state, packet, packetStart, (uint)packetStorage,
            gainIcdf, 0,
            deltaGainIcdf, 0,
            uniform8Icdf, 0,
            signalType, conditional, nbSubfr,
            output, 0);
    }

    /// <summary>Release kernel resources.</summary>
    public void Dispose() { /* auto-grouped */ }
}
