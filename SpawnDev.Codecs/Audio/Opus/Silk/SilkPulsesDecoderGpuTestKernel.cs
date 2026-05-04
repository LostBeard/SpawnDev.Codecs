// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// Drives `SilkPulsesDecoderGpu.Decode` on the accelerator.
/// </summary>
public sealed class SilkPulsesDecoderGpuTestKernel : IDisposable
{
    private readonly Action<
        Index1D,
        ArrayView<byte>, int, int,
        SilkPulsesInputs,
        int, int, int,
        ArrayView<short>> _kernel;

    /// <summary>Compile.</summary>
    public SilkPulsesDecoderGpuTestKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<byte>, int, int,
            SilkPulsesInputs,
            int, int, int,
            ArrayView<short>>(PulsesKernel);
    }

    /// <summary>Decode the signed excitation pulses for one SILK frame.</summary>
    public void Run(
        ArrayView<byte> packet, int packetStart, int packetStorage,
        SilkPulsesInputs inputs,
        int signalType, int quantOffsetType, int frameLength,
        ArrayView<short> pulsesOut)
    {
        _kernel(1, packet, packetStart, packetStorage, inputs,
            signalType, quantOffsetType, frameLength, pulsesOut);
    }

    private static void PulsesKernel(
        Index1D _,
        ArrayView<byte> packet, int packetStart, int packetStorage,
        SilkPulsesInputs inputs,
        int signalType, int quantOffsetType, int frameLength,
        ArrayView<short> pulsesOut)
    {
        var state = OpusRangeDecoderGpu.Init(packet, packetStart, (uint)packetStorage);
        SilkPulsesDecoderGpu.Decode(
            ref state, packet, packetStart, (uint)packetStorage,
            inputs,
            signalType, quantOffsetType, frameLength,
            pulsesOut, 0);
    }

    /// <summary>Release.</summary>
    public void Dispose() { /* auto-grouped */ }
}
