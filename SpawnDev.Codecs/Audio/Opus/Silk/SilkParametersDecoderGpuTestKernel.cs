// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Test/integration kernel that drives `SilkParametersDecoderGpu.Decode`.
// The per-stream codebook tables AND the in/out state + scratch regions live in
// one buffer per element type (byte, short, sbyte, int) held by
// SilkParametersInputs; SilkParametersState holds the state offsets into those
// buffers. The kernel binds 7 storage buffers (one view per table needed 20).

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
    /// <summary>Byte tables + byte state (<see cref="PackedDeviceBuffer{T}"/>).</summary>
    public ArrayView<byte> Bytes;
    /// <summary>Short tables + short state.</summary>
    public ArrayView<short> Shorts;
    /// <summary>Sbyte tables + sbyte scratch.</summary>
    public ArrayView<sbyte> SBytes;
    /// <summary>Int tables + int state.</summary>
    public ArrayView<int> Ints;
    /// <summary>NLSF codebook CB1_NLSF_Q8 (in <see cref="Bytes"/>).</summary>
    public int Cb1NlsfQ8At;
    /// <summary>NLSF codebook CB1_Wght_Q9 (in <see cref="Shorts"/>).</summary>
    public int Cb1WghtQ9At;
    /// <summary>NLSF codebook ec_sel (in <see cref="Bytes"/>).</summary>
    public int EcSelAt;
    /// <summary>NLSF codebook pred_Q8 (in <see cref="Bytes"/>).</summary>
    public int PredQ8SourceAt;
    /// <summary>NLSF codebook deltaMin_Q15 (in <see cref="Shorts"/>).</summary>
    public int DeltaMinQ15At;
    /// <summary>silk_LSFCosTab_FIX_Q12 (in <see cref="Shorts"/>).</summary>
    public int LsfCosTabQ12At;
    /// <summary>Pitch contour codebook (in <see cref="SBytes"/>).</summary>
    public int ContourCbAt;
    /// <summary>Flat LTP gain tables (in <see cref="SBytes"/>).</summary>
    public int LtpGainTablesFlatAt;
    /// <summary>LTP gain table offsets per perIndex (in <see cref="Ints"/>).</summary>
    public int LtpGainOffsetsAt;
    /// <summary>silk_LTPScales_table_Q14 (in <see cref="Shorts"/>).</summary>
    public int LtpScaleQ14TableAt;
}

/// <summary>
/// In/out state + scratch regions of one stream: element offsets into the buffers of
/// <see cref="SilkParametersInputs"/>.
/// </summary>
public struct SilkParametersState
{
    /// <summary>Previous frame NLSF Q15 [order] (in Shorts, in/out).</summary>
    public int PrevNlsfQ15At;
    /// <summary>Previous gain index [1] (in Ints, in/out).</summary>
    public int LastGainIndexAt;
    /// <summary>NLSF decode scratch (in Shorts).</summary>
    public int NlsfDecodeScratchAt;
    /// <summary>NLSF decode pred scratch (in Bytes).</summary>
    public int NlsfDecodePredScratchAt;
    /// <summary>NLSF2A scratch (in Ints).</summary>
    public int Nlsf2aScratchAt;
    /// <summary>NLSF indices scratch [order + 1] (in SBytes).</summary>
    public int NlsfIndicesScratchAt;
    /// <summary>Gain indices scratch [nbSubfr] (in SBytes).</summary>
    public int GainIndicesScratchAt;
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
            ArrayView<short>>(SilkParametersTestKernel);
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

    private static void SilkParametersTestKernel(
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
            inputs, scalars.ContourCbSize,
            state,
            scalars.QuantStepSizeQ16,
            scalars.Order, scalars.NbSubfr, scalars.FsKHz, scalars.Conditional,
            intOut, 0,
            shortOut, 0);
    }

    /// <summary>Release.</summary>
    public void Dispose() { /* auto-grouped */ }
}
