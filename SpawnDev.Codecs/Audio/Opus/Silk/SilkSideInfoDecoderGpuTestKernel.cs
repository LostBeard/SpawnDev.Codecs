// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Test/integration kernel that drives `SilkSideInfoDecoderGpu`
// `DecodeSignalType` + `DecodeSeed` calls on the accelerator. Used
// to verify bit-exact agreement of the GPU port with the CPU
// reference (`SilkSideInfoDecoder`).
//
// Single-thread per dispatch.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// Drives <see cref="SilkSideInfoDecoderGpu.DecodeSignalType"/> +
/// <see cref="SilkSideInfoDecoderGpu.DecodeSeed"/> on the
/// accelerator. Decodes ONE (signalType, quantOffsetType, seed)
/// triple from the packet and writes it to a 3-int output buffer.
/// </summary>
public sealed class SilkSideInfoDecoderGpuTestKernel : IDisposable
{
    private readonly Action<
        Index1D,
        ArrayView<byte>, int, int,
        ArrayView<byte>, ArrayView<byte>, ArrayView<byte>,
        int,
        ArrayView<int>> _kernel;

    /// <summary>Compile.</summary>
    public SilkSideInfoDecoderGpuTestKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<byte>, int, int,
            ArrayView<byte>, ArrayView<byte>, ArrayView<byte>,
            int,
            ArrayView<int>>(SideInfoKernel);
    }

    /// <summary>
    /// Decode one SILK side-info triple from <paramref name="packet"/>.
    /// </summary>
    /// <param name="packet">Encoded packet bytes.</param>
    /// <param name="packetStart">Offset of the packet in <paramref name="packet"/>.</param>
    /// <param name="packetStorage">Length of the packet in bytes.</param>
    /// <param name="typeOffsetVadIcdf">silk_type_offset_VAD_iCDF (4 bytes).</param>
    /// <param name="typeOffsetNoVadIcdf">silk_type_offset_no_VAD_iCDF (2 bytes).</param>
    /// <param name="uniform4Icdf">silk_uniform4_iCDF (4 bytes).</param>
    /// <param name="useVadTable">1 if VAD path, 0 if no-VAD path.</param>
    /// <param name="output">3-int output buffer:
    /// [0]=signalType, [1]=quantOffsetType, [2]=seed.</param>
    public void Run(
        ArrayView<byte> packet, int packetStart, int packetStorage,
        ArrayView<byte> typeOffsetVadIcdf,
        ArrayView<byte> typeOffsetNoVadIcdf,
        ArrayView<byte> uniform4Icdf,
        int useVadTable,
        ArrayView<int> output)
    {
        if (output.Length < 3)
            throw new ArgumentException("output too short (need 3).", nameof(output));
        _kernel(1,
            packet, packetStart, packetStorage,
            typeOffsetVadIcdf, typeOffsetNoVadIcdf, uniform4Icdf,
            useVadTable,
            output);
    }

    private static void SideInfoKernel(
        Index1D _,
        ArrayView<byte> packet, int packetStart, int packetStorage,
        ArrayView<byte> typeOffsetVadIcdf,
        ArrayView<byte> typeOffsetNoVadIcdf,
        ArrayView<byte> uniform4Icdf,
        int useVadTable,
        ArrayView<int> output)
    {
        var state = OpusRangeDecoderGpu.Init(packet, packetStart, (uint)packetStorage);

        SilkSideInfoDecoderGpu.DecodeSignalType(
            ref state, packet, packetStart, (uint)packetStorage,
            typeOffsetVadIcdf, 0,
            typeOffsetNoVadIcdf, 0,
            useVadTable != 0,
            out int signalType, out int quantOffsetType);

        int seed = SilkSideInfoDecoderGpu.DecodeSeed(
            ref state, packet, packetStart, (uint)packetStorage,
            uniform4Icdf, 0);

        output[0] = signalType;
        output[1] = quantOffsetType;
        output[2] = seed;
    }

    /// <summary>Release kernel resources.</summary>
    public void Dispose() { /* auto-grouped */ }
}
