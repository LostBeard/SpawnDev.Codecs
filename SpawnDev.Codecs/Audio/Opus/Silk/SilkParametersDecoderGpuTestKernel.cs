// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Test/integration kernel that drives `SilkParametersDecoderGpu.Decode`.
// Body-struct kernel parameter packs all per-stream codebook tables +
// scratches into one logical kernel parameter; another body struct
// holds the in/out state buffers; scalars bundle into another struct.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// Body-struct kernel parameter for SilkParametersDecoderGpu test kernel.
/// Bundles all per-stream codebook + table + scratch ArrayViews.
/// </summary>
public struct SilkParametersInputs
{
    /// <summary>NLSF codebook Cb1NlsfQ8.</summary>
    public ArrayView<byte> Cb1NlsfQ8;
    /// <summary>NLSF codebook Cb1WghtQ9.</summary>
    public ArrayView<short> Cb1WghtQ9;
    /// <summary>NLSF codebook EcSel.</summary>
    public ArrayView<byte> EcSel;
    /// <summary>NLSF codebook PredQ8 source.</summary>
    public ArrayView<byte> PredQ8Source;
    /// <summary>NLSF codebook DeltaMinQ15 array.</summary>
    public ArrayView<short> DeltaMinQ15;
    /// <summary>SilkLsfCosTab.Q12 (length 129).</summary>
    public ArrayView<short> LsfCosTabQ12;
    /// <summary>(fs_kHz, nbSubfr)-resolved pitch contour codebook.</summary>
    public ArrayView<sbyte> ContourCb;
    /// <summary>Flat-packed LTP gain Q7 codebooks (LtpGain0+1+2 ×5 taps each = 280 sbytes).</summary>
    public ArrayView<sbyte> LtpGainTablesFlat;
    /// <summary>[0, 40, 120] - sbyte offsets into LtpGainTablesFlat per perIndex.</summary>
    public ArrayView<int> LtpGainOffsets;
    /// <summary>LtpScalesQ14 lookup [15565, 12288, 8192].</summary>
    public ArrayView<short> LtpScaleQ14Table;
}

/// <summary>
/// In/out state body struct.
/// </summary>
public struct SilkParametersState
{
    /// <summary>Previous frame's NLSFs Q15 (length order). In/out.</summary>
    public ArrayView<short> PrevNlsfQ15;
    /// <summary>Last gain index buffer (length 1). In/out.</summary>
    public ArrayView<int> LastGainIndex;
    /// <summary>Scratch for SilkNlsfDecodeGpu shorts (length >= 3*16).</summary>
    public ArrayView<short> NlsfDecodeScratch;
    /// <summary>Scratch for SilkNlsfDecodeGpu predQ8 (length >= 16).</summary>
    public ArrayView<byte> NlsfDecodePredScratch;
    /// <summary>Scratch for SilkNlsf2AGpu (length >= 65 ints).</summary>
    public ArrayView<int> Nlsf2aScratch;
    /// <summary>Scratch holding (order+1) sbyte NLSF indices.</summary>
    public ArrayView<sbyte> NlsfIndicesScratch;
    /// <summary>Scratch holding nbSubfr sbyte gain indices.</summary>
    public ArrayView<sbyte> GainIndicesScratch;
}

/// <summary>
/// Per-call scalar parameters.
/// </summary>
public struct SilkParametersScalars
{
    /// <summary>Codebook quant step size Q16.</summary>
    public int QuantStepSizeQ16;
    /// <summary>Codebook order (10 or 16).</summary>
    public int Order;
    /// <summary>Subframe count (2 or 4).</summary>
    public int NbSubfr;
    /// <summary>Internal SILK sample rate (8, 12, or 16).</summary>
    public int FsKHz;
    /// <summary>Resolved pitch contour codebook size.</summary>
    public int ContourCbSize;
    /// <summary>Conditional gain coding flag (0 = independent).</summary>
    public int Conditional;
}

/// <summary>
/// Drives `SilkParametersDecoderGpu.Decode` on the accelerator.
/// </summary>
public sealed class SilkParametersDecoderGpuTestKernel : IDisposable
{
    private readonly Action<
        Index1D,
        ArrayView<int>,
        SilkParametersInputs,
        SilkParametersState,
        SilkParametersScalars,
        ArrayView<int>,
        ArrayView<short>> _kernel;

    /// <summary>Compile.</summary>
    public SilkParametersDecoderGpuTestKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<int>,
            SilkParametersInputs,
            SilkParametersState,
            SilkParametersScalars,
            ArrayView<int>,
            ArrayView<short>>(ParametersKernel);
    }

    /// <summary>
    /// Dequantize parameters from the indices buffer. Outputs split between
    /// intOut (gainsQ16 + pitchL) and shortOut (nlsfQ15 + predCoefQ12 +
    /// ltpCoefQ14 + ltpScaleQ14) per <see cref="SilkDecodedParametersLayout"/>.
    /// </summary>
    public void Run(
        ArrayView<int> indicesIn,
        SilkParametersInputs inputs,
        SilkParametersState state,
        SilkParametersScalars scalars,
        ArrayView<int> intOut,
        ArrayView<short> shortOut)
    {
        if (intOut.Length < SilkDecodedParametersLayout.IntTotalSlots)
            throw new ArgumentException(
                $"intOut too short (need {SilkDecodedParametersLayout.IntTotalSlots}).",
                nameof(intOut));
        if (shortOut.Length < SilkDecodedParametersLayout.ShortTotalSlots)
            throw new ArgumentException(
                $"shortOut too short (need {SilkDecodedParametersLayout.ShortTotalSlots}).",
                nameof(shortOut));
        _kernel(1, indicesIn, inputs, state, scalars, intOut, shortOut);
    }

    private static void ParametersKernel(
        Index1D _,
        ArrayView<int> indicesIn,
        SilkParametersInputs inputs,
        SilkParametersState state,
        SilkParametersScalars scalars,
        ArrayView<int> intOut,
        ArrayView<short> shortOut)
    {
        SilkParametersDecoderGpu.Decode(
            indicesIn, 0,
            inputs.Cb1NlsfQ8,
            inputs.Cb1WghtQ9,
            inputs.EcSel,
            inputs.PredQ8Source,
            inputs.DeltaMinQ15,
            inputs.LsfCosTabQ12,
            inputs.ContourCb, scalars.ContourCbSize,
            inputs.LtpGainTablesFlat,
            inputs.LtpGainOffsets,
            inputs.LtpScaleQ14Table,
            state.PrevNlsfQ15, 0,
            state.LastGainIndex, 0,
            state.NlsfDecodeScratch, 0,
            state.NlsfDecodePredScratch, 0,
            state.Nlsf2aScratch, 0,
            state.NlsfIndicesScratch, 0,
            state.GainIndicesScratch, 0,
            scalars.QuantStepSizeQ16,
            scalars.Order, scalars.NbSubfr, scalars.FsKHz, scalars.Conditional,
            intOut, 0,
            shortOut, 0);
    }

    /// <summary>Release.</summary>
    public void Dispose() { /* auto-grouped */ }
}
