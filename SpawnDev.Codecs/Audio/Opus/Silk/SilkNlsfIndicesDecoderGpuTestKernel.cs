// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Test/integration kernel that drives `SilkNlsfIndicesDecoderGpu.DecodeIndices`
// on the accelerator. Decodes the NLSF index block + writes (order+1) ints
// + the interpolation factor to caller-allocated output.
//
// Uses a body-struct kernel parameter (`SilkNlsfIndicesInputs`): the byte
// tables + scratch live in one byte buffer (offsets in the struct), so the
// kernel binds 4 storage buffers. Caller pre-
// computes the cb1IcdfBaseOffset = (signalType >> 1) * nVectors so the
// struct stays free of derived signal-type / nVectors ints.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// Body-struct kernel parameter for SilkNlsfIndicesDecoderGpu test kernel.
/// Plain POD struct (public fields, no properties / init / required) so
/// ILGPU's kernel-parameter marshaling can pack it.
/// </summary>
public struct SilkNlsfIndicesInputs
{
    /// <summary>The six byte tables and the predQ8 scratch, packed (offsets below; <see cref="PackedDeviceBuffer{T}"/>).</summary>
    public ArrayView<byte> Bytes;
    /// <summary>Scratch for ecIx[order].</summary>
    public ArrayView<short> EcIxScratch;
    /// <summary>NLSF codebook Cb1Icdf offset in <see cref="Bytes"/>.</summary>
    public int Cb1IcdfAt;
    /// <summary>NLSF codebook EcIcdf offset.</summary>
    public int EcIcdfAt;
    /// <summary>NLSF codebook EcSel offset.</summary>
    public int EcSelAt;
    /// <summary>NLSF codebook PredQ8 source offset.</summary>
    public int PredQ8SourceAt;
    /// <summary>silk_NLSF_EXT_iCDF offset.</summary>
    public int NlsfExtIcdfAt;
    /// <summary>silk_NLSF_interpolation_factor_iCDF offset.</summary>
    public int NlsfInterpolationFactorIcdfAt;
    /// <summary>predQ8[order] scratch offset.</summary>
    public int PredQ8ScratchAt;
}

/// <summary>
/// Drives `SilkNlsfIndicesDecoderGpu.DecodeIndices` on the accelerator.
/// </summary>
public sealed class SilkNlsfIndicesDecoderGpuTestKernel : IDisposable
{
    private readonly Action<
        Index1D,
        ArrayView<byte>, int, int,
        SilkNlsfIndicesInputs,
        int, int, int,
        ArrayView<int>> _kernel;

    /// <summary>Compile.</summary>
    public SilkNlsfIndicesDecoderGpuTestKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<byte>, int, int,
            SilkNlsfIndicesInputs,
            int, int, int,
            ArrayView<int>>(SilkNlsfIndicesTestKernel);
    }

    /// <summary>
    /// Decode the NLSF index block. Output layout:
    /// <c>[0..order]</c>      = nlsf indices (order+1 entries: cb1 + per-coef residuals)
    /// <c>[order+1]</c>       = interpolation factor Q2
    ///
    /// Caller pre-computes <paramref name="cb1IcdfBaseOffset"/> =
    /// <c>(signalType &gt;&gt; 1) * nVectors</c>.
    /// </summary>
    public void Run(
        ArrayView<byte> packet, int packetStart, int packetStorage,
        SilkNlsfIndicesInputs inputs,
        int cb1IcdfBaseOffset, int order, int nbSubfr,
        ArrayView<int> output)
    {
        if (output.Length < order + 2)
            throw new ArgumentException("output too short (need order+2).", nameof(output));
        _kernel(1,
            packet, packetStart, packetStorage,
            inputs,
            cb1IcdfBaseOffset, order, nbSubfr,
            output);
    }

    private static void SilkNlsfIndicesTestKernel(
        Index1D _,
        ArrayView<byte> packet, int packetStart, int packetStorage,
        SilkNlsfIndicesInputs inputs,
        int cb1IcdfBaseOffset, int order, int nbSubfr,
        ArrayView<int> output)
    {
        var state = OpusRangeDecoderGpu.Init(packet, packetStart, (uint)packetStorage);
        int interp = SilkNlsfIndicesDecoderGpu.DecodeIndices(
            ref state, packet, packetStart, (uint)packetStorage,
            inputs.Bytes, inputs.Cb1IcdfAt + cb1IcdfBaseOffset,
            inputs.Bytes, inputs.EcIcdfAt,
            inputs.Bytes, inputs.EcSelAt,
            inputs.Bytes, inputs.PredQ8SourceAt,
            inputs.Bytes, inputs.NlsfExtIcdfAt,
            inputs.Bytes, inputs.NlsfInterpolationFactorIcdfAt,
            inputs.EcIxScratch, 0,
            inputs.Bytes, inputs.PredQ8ScratchAt,
            order, nbSubfr,
            output, 0);
        output[order + 1] = interp;
    }

    /// <summary>Release kernel resources.</summary>
    public void Dispose() { /* auto-grouped */ }
}
