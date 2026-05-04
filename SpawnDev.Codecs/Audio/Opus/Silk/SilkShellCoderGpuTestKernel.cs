// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// Drives `SilkShellCoderGpu.Decode` on the accelerator.
/// </summary>
public sealed class SilkShellCoderGpuTestKernel : IDisposable
{
    private readonly Action<
        Index1D,
        ArrayView<byte>, int, int,
        SilkShellCoderTables,
        int,
        ArrayView<short>> _kernel;

    /// <summary>Compile.</summary>
    public SilkShellCoderGpuTestKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<byte>, int, int,
            SilkShellCoderTables,
            int,
            ArrayView<short>>(ShellCoderKernel);
    }

    /// <summary>Decode 16 pulse magnitudes from the packet.</summary>
    public void Run(
        ArrayView<byte> packet, int packetStart, int packetStorage,
        SilkShellCoderTables tables,
        int pulsesTotal,
        ArrayView<short> pulsesOut)
    {
        if (pulsesOut.Length < SilkShellCoderGpu.ShellCodecFrameLength)
            throw new ArgumentException("pulsesOut too short.", nameof(pulsesOut));
        _kernel(1, packet, packetStart, packetStorage, tables, pulsesTotal, pulsesOut);
    }

    private static void ShellCoderKernel(
        Index1D _,
        ArrayView<byte> packet, int packetStart, int packetStorage,
        SilkShellCoderTables tables,
        int pulsesTotal,
        ArrayView<short> pulsesOut)
    {
        var state = OpusRangeDecoderGpu.Init(packet, packetStart, (uint)packetStorage);
        SilkShellCoderGpu.Decode(
            ref state, packet, packetStart, (uint)packetStorage,
            tables, pulsesTotal,
            pulsesOut, 0);
    }

    /// <summary>Release.</summary>
    public void Dispose() { /* auto-grouped */ }
}
