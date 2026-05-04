// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Test/integration kernel that drives `SilkIndicesDecoderGpu.Decode`.
// Body-struct kernel parameters bundle ~22 iCDF + scratch ArrayViews
// into one logical kernel parameter; scalars bundle into another.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// Body-struct kernel parameter for SilkIndicesDecoderGpu test kernel.
/// Bundles all per-stream codebook + iCDF tables + scratches.
/// </summary>
public struct SilkIndicesInputs
{
    /// <summary>silk_type_offset_VAD_iCDF.</summary>
    public ArrayView<byte> TypeOffsetVadIcdf;
    /// <summary>silk_type_offset_no_VAD_iCDF.</summary>
    public ArrayView<byte> TypeOffsetNoVadIcdf;
    /// <summary>silk_uniform4_iCDF.</summary>
    public ArrayView<byte> Uniform4Icdf;
    /// <summary>silk_gain_iCDF flat (24 entries).</summary>
    public ArrayView<byte> GainIcdf;
    /// <summary>silk_delta_gain_iCDF.</summary>
    public ArrayView<byte> DeltaGainIcdf;
    /// <summary>silk_uniform8_iCDF.</summary>
    public ArrayView<byte> Uniform8Icdf;
    /// <summary>NLSF codebook Cb1Icdf.</summary>
    public ArrayView<byte> Cb1Icdf;
    /// <summary>NLSF codebook EcIcdf.</summary>
    public ArrayView<byte> EcIcdf;
    /// <summary>NLSF codebook EcSel.</summary>
    public ArrayView<byte> EcSel;
    /// <summary>NLSF codebook PredQ8 source.</summary>
    public ArrayView<byte> PredQ8Source;
    /// <summary>silk_NLSF_EXT_iCDF.</summary>
    public ArrayView<byte> NlsfExtIcdf;
    /// <summary>silk_NLSF_interpolation_factor_iCDF.</summary>
    public ArrayView<byte> NlsfInterpolationFactorIcdf;
    /// <summary>silk_pitch_delta_iCDF.</summary>
    public ArrayView<byte> PitchDeltaIcdf;
    /// <summary>silk_pitch_lag_iCDF.</summary>
    public ArrayView<byte> PitchLagIcdf;
    /// <summary>fs_kHz-resolved Uniform4/6/8.</summary>
    public ArrayView<byte> LagLowBitsIcdf;
    /// <summary>(fs_kHz, nbSubfr)-resolved contour iCDF.</summary>
    public ArrayView<byte> ContourIcdf;
    /// <summary>silk_LTP_per_index_iCDF.</summary>
    public ArrayView<byte> LtpPerIndexIcdf;
    /// <summary>Flat-packed LtpGain0+1+2.</summary>
    public ArrayView<byte> LtpGainIcdfFlat;
    /// <summary>[0, 8, 24] LTP gain offsets per perIndex.</summary>
    public ArrayView<int> LtpGainOffsets;
    /// <summary>silk_LTP_scale_iCDF.</summary>
    public ArrayView<byte> LtpScaleIcdf;
    /// <summary>Scratch buffer for ecIx[order].</summary>
    public ArrayView<short> EcIxScratch;
    /// <summary>Scratch buffer for predQ8[order].</summary>
    public ArrayView<byte> PredQ8Scratch;
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
            ArrayView<int>>(IndicesKernel);
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

    private static void IndicesKernel(
        Index1D _,
        ArrayView<byte> packet, int packetStart, int packetStorage,
        SilkIndicesInputs inputs,
        SilkIndicesScalars scalars,
        ArrayView<int> output)
    {
        var state = OpusRangeDecoderGpu.Init(packet, packetStart, (uint)packetStorage);
        SilkIndicesDecoderGpu.Decode(
            ref state, packet, packetStart, (uint)packetStorage,
            inputs.TypeOffsetVadIcdf,
            inputs.TypeOffsetNoVadIcdf,
            inputs.Uniform4Icdf,
            inputs.GainIcdf,
            inputs.DeltaGainIcdf,
            inputs.Uniform8Icdf,
            inputs.Cb1Icdf,
            inputs.EcIcdf,
            inputs.EcSel,
            inputs.PredQ8Source,
            inputs.NlsfExtIcdf,
            inputs.NlsfInterpolationFactorIcdf,
            inputs.PitchDeltaIcdf,
            inputs.PitchLagIcdf,
            inputs.LagLowBitsIcdf,
            inputs.ContourIcdf,
            inputs.LtpPerIndexIcdf,
            inputs.LtpGainIcdfFlat,
            inputs.LtpGainOffsets,
            inputs.LtpScaleIcdf,
            inputs.EcIxScratch,
            inputs.PredQ8Scratch,
            scalars.NVectors, scalars.Order, scalars.NbSubfr, scalars.FsKHz,
            scalars.VadFlag, scalars.DecodeLbrr, scalars.Conditional,
            scalars.PrevLagIndex, scalars.PrevSignalTypeWasVoiced,
            scalars.FirstFrameAfterReset,
            output, 0);
    }

    /// <summary>Release.</summary>
    public void Dispose() { /* auto-grouped */ }
}
