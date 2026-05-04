// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// Body-struct kernel parameter for SilkLtpIndicesDecoderGpu test kernel.
/// </summary>
public struct SilkLtpIndicesInputs
{
    /// <summary>silk_LTP_per_index_iCDF (3 entries).</summary>
    public ArrayView<byte> LtpPerIndexIcdf;
    /// <summary>Flat-packed LtpGain0+1+2 (8+16+32 = 56 entries).</summary>
    public ArrayView<byte> LtpGainIcdfFlat;
    /// <summary>Offsets into LtpGainIcdfFlat per perIndex: [0, 8, 24].</summary>
    public ArrayView<int> LtpGainOffsets;
    /// <summary>silk_LTP_scale_iCDF (3 entries).</summary>
    public ArrayView<byte> LtpScaleIcdf;
}

/// <summary>
/// Drives `SilkLtpIndicesDecoderGpu.DecodeIndices`.
/// </summary>
public sealed class SilkLtpIndicesDecoderGpuTestKernel : IDisposable
{
    private readonly Action<
        Index1D,
        ArrayView<byte>, int, int,
        SilkLtpIndicesInputs,
        int, int,
        ArrayView<int>> _kernel;

    /// <summary>Compile.</summary>
    public SilkLtpIndicesDecoderGpuTestKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<byte>, int, int,
            SilkLtpIndicesInputs,
            int, int,
            ArrayView<int>>(LtpIndicesKernel);
    }

    /// <summary>Decode the LTP index block. Output: [0]=perIndex, [1]=ltpScaleIndex, [2..2+nbSubfr]=gain indices.</summary>
    public void Run(
        ArrayView<byte> packet, int packetStart, int packetStorage,
        SilkLtpIndicesInputs inputs,
        int conditional, int nbSubfr,
        ArrayView<int> output)
    {
        if (output.Length < 2 + nbSubfr)
            throw new ArgumentException("output too short.", nameof(output));
        _kernel(1,
            packet, packetStart, packetStorage,
            inputs,
            conditional, nbSubfr,
            output);
    }

    private static void LtpIndicesKernel(
        Index1D _,
        ArrayView<byte> packet, int packetStart, int packetStorage,
        SilkLtpIndicesInputs inputs,
        int conditional, int nbSubfr,
        ArrayView<int> output)
    {
        var state = OpusRangeDecoderGpu.Init(packet, packetStart, (uint)packetStorage);
        SilkLtpIndicesDecoderGpu.DecodeIndices(
            ref state, packet, packetStart, (uint)packetStorage,
            inputs.LtpPerIndexIcdf, 0,
            inputs.LtpGainIcdfFlat, 0,
            inputs.LtpGainOffsets, 0,
            inputs.LtpScaleIcdf, 0,
            conditional, nbSubfr,
            output, 0);
    }

    /// <summary>Release.</summary>
    public void Dispose() { /* auto-grouped */ }
}
