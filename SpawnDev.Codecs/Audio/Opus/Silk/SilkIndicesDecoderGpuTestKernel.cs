// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Test/integration kernel that drives `SilkIndicesDecoderGpu.Decode`.
// The iCDF tables + the byte scratch live in ONE byte buffer (offsets in
// SilkIndicesInputs), so the kernel binds 5 storage buffers (WebGPU allows
// at most 16 per stage; one view per table needed 24).

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// Body-struct kernel parameter for SilkIndicesDecoderGpu: the per-stream codebook + iCDF tables and the byte
/// scratch packed into one byte buffer (<see cref="Bytes"/>, element offsets in the *At fields; build it with
/// <see cref="PackedDeviceBuffer{T}"/>), plus the int and short side buffers.
/// </summary>
public struct SilkIndicesInputs
{
    /// <summary>Every byte table and the predQ8 scratch.</summary>
    public ArrayView<byte> Bytes;
    /// <summary>[0, 8, 24] LTP gain offsets per perIndex.</summary>
    public ArrayView<int> LtpGainOffsets;
    /// <summary>Scratch buffer for ecIx[order].</summary>
    public ArrayView<short> EcIxScratch;
    /// <summary>silk_type_offset_VAD_iCDF.</summary>
    public int TypeOffsetVadIcdfAt;
    /// <summary>silk_type_offset_no_VAD_iCDF.</summary>
    public int TypeOffsetNoVadIcdfAt;
    /// <summary>silk_uniform4_iCDF.</summary>
    public int Uniform4IcdfAt;
    /// <summary>silk_gain_iCDF flat (24 entries).</summary>
    public int GainIcdfAt;
    /// <summary>silk_delta_gain_iCDF.</summary>
    public int DeltaGainIcdfAt;
    /// <summary>silk_uniform8_iCDF.</summary>
    public int Uniform8IcdfAt;
    /// <summary>NLSF codebook Cb1Icdf.</summary>
    public int Cb1IcdfAt;
    /// <summary>NLSF codebook EcIcdf.</summary>
    public int EcIcdfAt;
    /// <summary>NLSF codebook EcSel.</summary>
    public int EcSelAt;
    /// <summary>NLSF codebook PredQ8 source.</summary>
    public int PredQ8SourceAt;
    /// <summary>silk_NLSF_EXT_iCDF.</summary>
    public int NlsfExtIcdfAt;
    /// <summary>silk_NLSF_interpolation_factor_iCDF.</summary>
    public int NlsfInterpolationFactorIcdfAt;
    /// <summary>silk_pitch_delta_iCDF.</summary>
    public int PitchDeltaIcdfAt;
    /// <summary>silk_pitch_lag_iCDF.</summary>
    public int PitchLagIcdfAt;
    /// <summary>fs_kHz-resolved Uniform4/6/8.</summary>
    public int LagLowBitsIcdfAt;
    /// <summary>(fs_kHz, nbSubfr)-resolved contour iCDF.</summary>
    public int ContourIcdfAt;
    /// <summary>silk_LTP_per_index_iCDF.</summary>
    public int LtpPerIndexIcdfAt;
    /// <summary>Flat-packed LtpGain0+1+2.</summary>
    public int LtpGainIcdfFlatAt;
    /// <summary>silk_LTP_scale_iCDF.</summary>
    public int LtpScaleIcdfAt;
    /// <summary>Scratch region for predQ8[order].</summary>
    public int PredQ8ScratchAt;
}

/// <summary>
/// Per-call scalar parameters for SilkIndicesDecoderGpu test kernel.
/// </summary>
public struct SilkIndicesScalars
{
    /// <summary>NLSF codebook NVectors.</summary>
    public int NVectors;
    /// <summary>NLSF codebook order (10 or 16).</summary>
    public int Order;
    /// <summary>Subframe count (2 or 4).</summary>
    public int NbSubfr;
    /// <summary>Internal SILK sample rate (8, 12, or 16).</summary>
    public int FsKHz;
    /// <summary>VAD flag (0/1).</summary>
    public int VadFlag;
    /// <summary>Decode LBRR (0/1).</summary>
    public int DecodeLbrr;
    /// <summary>Conditional coding (0 = independent).</summary>
    public int Conditional;
    /// <summary>Previous frame's pitch lag.</summary>
    public int PrevLagIndex;
    /// <summary>Previous signal-type-was-voiced flag (0/1).</summary>
    public int PrevSignalTypeWasVoiced;
    /// <summary>First frame after reset (0/1).</summary>
    public int FirstFrameAfterReset;
}

/// <summary>
/// Drives `SilkIndicesDecoderGpu.Decode` on the accelerator.
/// </summary>
public sealed class SilkIndicesDecoderGpuTestKernel : IDisposable
{
    private readonly Action<
        Index1D,
        ArrayView<byte>, int, int,
        SilkIndicesInputs,
        SilkIndicesScalars,
        ArrayView<int>> _kernel;

    /// <summary>Compile.</summary>
    public SilkIndicesDecoderGpuTestKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<byte>, int, int,
            SilkIndicesInputs,
            SilkIndicesScalars,
            ArrayView<int>>(SilkIndicesTestKernel);
    }

    /// <summary>
    /// Decode the full SILK side-information block. Output layout per
    /// <see cref="SilkDecodedIndicesLayout"/>; length must be
    /// &gt;= <see cref="SilkDecodedIndicesLayout.TotalSlots"/>.
    /// </summary>
    public void Run(
        ArrayView<byte> packet, int packetStart, int packetStorage,
        SilkIndicesInputs inputs,
        SilkIndicesScalars scalars,
        ArrayView<int> output)
    {
        if (output.Length < SilkDecodedIndicesLayout.TotalSlots)
            throw new ArgumentException(
                $"output too short (need {SilkDecodedIndicesLayout.TotalSlots}).",
                nameof(output));
        _kernel(1, packet, packetStart, packetStorage, inputs, scalars, output);
    }

    private static void SilkIndicesTestKernel(
        Index1D _,
        ArrayView<byte> packet, int packetStart, int packetStorage,
        SilkIndicesInputs inputs,
        SilkIndicesScalars scalars,
        ArrayView<int> output)
    {
        var state = OpusRangeDecoderGpu.Init(packet, packetStart, (uint)packetStorage);
        SilkIndicesDecoderGpu.Decode(
            ref state, packet, packetStart, (uint)packetStorage,
            inputs,
            scalars.NVectors, scalars.Order, scalars.NbSubfr, scalars.FsKHz,
            scalars.VadFlag, scalars.DecodeLbrr, scalars.Conditional,
            scalars.PrevLagIndex, scalars.PrevSignalTypeWasVoiced,
            scalars.FirstFrameAfterReset,
            output, 0);
    }

    /// <summary>Release.</summary>
    public void Dispose() { /* auto-grouped */ }
}
