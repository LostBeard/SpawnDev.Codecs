// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Test/integration kernel that drives `SilkPitchIndicesDecoderGpu.DecodeIndices`.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// Body-struct kernel parameter for SilkPitchIndicesDecoderGpu test kernel.
/// Plain POD struct so ILGPU's kernel-parameter marshaling can pack it.
/// </summary>
public struct SilkPitchIndicesInputs
{
    /// <summary>silk_pitch_delta_iCDF (21 entries).</summary>
    public ArrayView<byte> PitchDeltaIcdf;
    /// <summary>silk_pitch_lag_iCDF (32 entries).</summary>
    public ArrayView<byte> PitchLagIcdf;
    /// <summary>fs_kHz-resolved Uniform4/6/8 iCDF for the lag LSB.</summary>
    public ArrayView<byte> LagLowBitsIcdf;
    /// <summary>(fs_kHz, nbSubfr)-resolved contour iCDF.</summary>
    public ArrayView<byte> ContourIcdf;
}

/// <summary>
/// Drives `SilkPitchIndicesDecoderGpu.DecodeIndices` on the accelerator.
/// </summary>
public sealed class SilkPitchIndicesDecoderGpuTestKernel : IDisposable
{
    private readonly Action<
        Index1D,
        ArrayView<byte>, int, int,
        SilkPitchIndicesInputs,
        int, int, int, int,
        ArrayView<int>> _kernel;

    /// <summary>Compile.</summary>
    public SilkPitchIndicesDecoderGpuTestKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<byte>, int, int,
            SilkPitchIndicesInputs,
            int, int, int, int,
            ArrayView<int>>(PitchIndicesKernel);
    }

    /// <summary>
    /// Decode (lagIndex, contourIndex). Output [0]=lagIndex, [1]=contourIndex.
    /// </summary>
    public void Run(
        ArrayView<byte> packet, int packetStart, int packetStorage,
        SilkPitchIndicesInputs inputs,
        int fsKHz, int prevLagIndex, int prevSignalTypeWasVoiced, int conditional,
        ArrayView<int> output)
    {
        if (output.Length < 2)
            throw new ArgumentException("output too short (need 2).", nameof(output));
        _kernel(1,
            packet, packetStart, packetStorage,
            inputs,
            fsKHz, prevLagIndex, prevSignalTypeWasVoiced, conditional,
            output);
    }

    private static void PitchIndicesKernel(
        Index1D _,
        ArrayView<byte> packet, int packetStart, int packetStorage,
        SilkPitchIndicesInputs inputs,
        int fsKHz, int prevLagIndex, int prevSignalTypeWasVoiced, int conditional,
        ArrayView<int> output)
    {
        var state = OpusRangeDecoderGpu.Init(packet, packetStart, (uint)packetStorage);
        SilkPitchIndicesDecoderGpu.DecodeIndices(
            ref state, packet, packetStart, (uint)packetStorage,
            inputs.PitchDeltaIcdf, 0,
            inputs.PitchLagIcdf, 0,
            inputs.LagLowBitsIcdf, 0,
            inputs.ContourIcdf, 0,
            fsKHz, prevLagIndex, prevSignalTypeWasVoiced, conditional,
            output, 0);
    }

    /// <summary>Release kernel resources.</summary>
    public void Dispose() { /* auto-grouped */ }
}
