// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Test/integration kernel that drives `SilkNlsfIndicesDecoderGpu.DecodeIndices`
// on the accelerator. Decodes the NLSF index block + writes (order+1) ints
// + the interpolation factor to caller-allocated output.
//
// Uses a body-struct kernel parameter (`SilkNlsfIndicesInputs`) to pack
// all per-stream codebook + iCDF table ArrayViews into one logical parameter -
// keeps the kernel under ILGPU's Action<...> generic ceiling. Caller pre-
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
    /// <summary>Codebook Cb1Icdf, length 2 * NVectors.</summary>
    public ArrayView<byte> Cb1Icdf;
    /// <summary>Codebook EcIcdf (9 entries per ecIx slot).</summary>
    public ArrayView<byte> EcIcdf;
    /// <summary>Codebook EcSel bytes (length nVectors * order / 2).</summary>
    public ArrayView<byte> EcSel;
    /// <summary>Codebook PredQ8 source (length 2 * (order - 1)).</summary>
    public ArrayView<byte> PredQ8Source;
    /// <summary>silk_NLSF_EXT_iCDF (7 entries).</summary>
    public ArrayView<byte> NlsfExtIcdf;
    /// <summary>silk_NLSF_interpolation_factor_iCDF (5 entries).</summary>
    public ArrayView<byte> NlsfInterpolationFactorIcdf;
    /// <summary>Scratch buffer for ecIx[order] (length >= MaxLpcOrder).</summary>
    public ArrayView<short> EcIxScratch;
    /// <summary>Scratch buffer for predQ8[order] (length >= MaxLpcOrder).</summary>
    public ArrayView<byte> PredQ8Scratch;
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
            ArrayView<int>>(NlsfIndicesKernel);
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

    private static void NlsfIndicesKernel(
        Index1D _,
        ArrayView<byte> packet, int packetStart, int packetStorage,
        SilkNlsfIndicesInputs inputs,
        int cb1IcdfBaseOffset, int order, int nbSubfr,
        ArrayView<int> output)
    {
        var state = OpusRangeDecoderGpu.Init(packet, packetStart, (uint)packetStorage);
        int interp = SilkNlsfIndicesDecoderGpu.DecodeIndices(
            ref state, packet, packetStart, (uint)packetStorage,
            inputs.Cb1Icdf, cb1IcdfBaseOffset,
            inputs.EcIcdf, 0,
            inputs.EcSel, 0,
            inputs.PredQ8Source, 0,
            inputs.NlsfExtIcdf, 0,
            inputs.NlsfInterpolationFactorIcdf, 0,
            inputs.EcIxScratch, 0,
            inputs.PredQ8Scratch, 0,
            order, nbSubfr,
            output, 0);
        output[order + 1] = interp;
    }

    /// <summary>Release kernel resources.</summary>
    public void Dispose() { /* auto-grouped */ }
}
