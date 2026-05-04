// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Host orchestrator for the SILK per-frame decode pipeline. Mirror of
// CPU SilkDecodeFrame.Decode.
//
// Currently implements **Phase A**: Indices + Pulses (range-coded
// bitstream consumption). The range decoder state crosses the two
// kernel dispatches via a 1-element `ArrayView<OpusRangeDecoderGpuState>`
// buffer, exercising cross-kernel state visibility.
//
// Phase B (Parameters dequant + DecodeCore synthesis + state shift +
// scalar state update) lives in a follow-up commit once Phase A is
// validated bit-exact. Splitting the work this way keeps each
// orchestrator increment testable: Phase A's correctness can be checked
// by comparing the decoded indices + pulses against a CPU oracle that
// drives `SilkIndicesDecoder.Decode` + `SilkPulsesDecoder.Decode` on the
// same bitstream.
//
// Multi-kernel host orchestration matches the SpawnDev codec pattern
// (Vorbis v2, etc.). Each kernel runs as a single thread (sequential
// per-stream entropy decode); cross-stream parallelism happens across
// channels in higher-level orchestration.
//
// Upstream Copyright (c) 2006-2011 Skype Limited. BSD 3-Clause.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// Host orchestrator for SILK per-frame decode (Phase A: indices +
/// pulses; Phase B pending). Owns 3 compiled kernel handles and
/// dispatches them in sequence over GPU-resident buffers.
/// </summary>
public sealed class SilkDecodeFrameGpuOrchestrator : IDisposable
{
    private readonly Action<
        Index1D,
        ArrayView<byte>, int, int,
        ArrayView<OpusRangeDecoderGpuState>> _initStateKernel;

    private readonly Action<
        Index1D,
        ArrayView<byte>, int, int,
        SilkIndicesInputs,
        SilkIndicesScalars,
        ArrayView<OpusRangeDecoderGpuState>,
        ArrayView<int>> _indicesKernel;

    private readonly Action<
        Index1D,
        ArrayView<byte>, int, int,
        SilkPulsesInputs,
        ArrayView<int>,
        int,
        ArrayView<OpusRangeDecoderGpuState>,
        ArrayView<short>> _pulsesKernel;

    private readonly Action<
        Index1D,
        ArrayView<int>,
        SilkParametersInputs,
        SilkParametersState,
        SilkParametersScalars,
        ArrayView<int>,
        ArrayView<short>> _parametersKernel;

    private readonly Action<
        Index1D,
        ArrayView<int>,
        ArrayView<short>,
        ArrayView<short>,
        SilkDecodeCoreInputs,
        SilkDecodeCoreScalars> _decodeCoreKernel;

    /// <summary>Compile all 3 phase kernels for the supplied accelerator.</summary>
    public SilkDecodeFrameGpuOrchestrator(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);

        _initStateKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<byte>, int, int,
            ArrayView<OpusRangeDecoderGpuState>>(InitStateKernel);

        _indicesKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<byte>, int, int,
            SilkIndicesInputs,
            SilkIndicesScalars,
            ArrayView<OpusRangeDecoderGpuState>,
            ArrayView<int>>(IndicesAdapterKernel);

        _pulsesKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<byte>, int, int,
            SilkPulsesInputs,
            ArrayView<int>,
            int,
            ArrayView<OpusRangeDecoderGpuState>,
            ArrayView<short>>(PulsesAdapterKernel);

        _parametersKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<int>,
            SilkParametersInputs,
            SilkParametersState,
            SilkParametersScalars,
            ArrayView<int>,
            ArrayView<short>>(ParametersAdapterKernel);

        _decodeCoreKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<int>,
            ArrayView<short>,
            ArrayView<short>,
            SilkDecodeCoreInputs,
            SilkDecodeCoreScalars>(DecodeCoreAdapterKernel);
    }

    /// <summary>
    /// Phase A: dispatch the indices + pulses decode. Range decoder state
    /// is initialized in <paramref name="stateBuf"/>[0] and threaded through
    /// the 2 dispatches; on return <paramref name="indicesOut"/> holds the
    /// decoded indices (per <see cref="SilkDecodedIndicesLayout"/>) and
    /// <paramref name="pulsesOut"/> holds the decoded pulse train.
    /// </summary>
    public void DecodeIndicesAndPulses(
        ArrayView<byte> packet, int packetStart, int packetStorage,
        SilkIndicesInputs indicesInputs,
        SilkIndicesScalars indicesScalars,
        SilkPulsesInputs pulsesInputs,
        int frameLength,
        ArrayView<OpusRangeDecoderGpuState> stateBuf,
        ArrayView<int> indicesOut,
        ArrayView<short> pulsesOut)
    {
        _initStateKernel(1, packet, packetStart, packetStorage, stateBuf);
        _indicesKernel(1, packet, packetStart, packetStorage,
            indicesInputs, indicesScalars, stateBuf, indicesOut);
        _pulsesKernel(1, packet, packetStart, packetStorage,
            pulsesInputs, indicesOut, frameLength, stateBuf, pulsesOut);
    }

    /// <summary>
    /// Phase A + P: dispatch indices + pulses + parameters dequant. Same as
    /// <see cref="DecodeIndicesAndPulses"/> followed by a parameters kernel
    /// that reads the indices buffer and writes <paramref name="paramsIntOut"/> +
    /// <paramref name="paramsShortOut"/> per <see cref="SilkDecodedParametersLayout"/>.
    /// </summary>
    public void DecodeIndicesPulsesAndParameters(
        ArrayView<byte> packet, int packetStart, int packetStorage,
        SilkIndicesInputs indicesInputs,
        SilkIndicesScalars indicesScalars,
        SilkPulsesInputs pulsesInputs,
        int frameLength,
        SilkParametersInputs parametersInputs,
        SilkParametersState parametersState,
        SilkParametersScalars parametersScalars,
        ArrayView<OpusRangeDecoderGpuState> stateBuf,
        ArrayView<int> indicesOut,
        ArrayView<short> pulsesOut,
        ArrayView<int> paramsIntOut,
        ArrayView<short> paramsShortOut)
    {
        DecodeIndicesAndPulses(
            packet, packetStart, packetStorage,
            indicesInputs, indicesScalars, pulsesInputs,
            frameLength, stateBuf, indicesOut, pulsesOut);
        _parametersKernel(1,
            indicesOut, parametersInputs, parametersState, parametersScalars,
            paramsIntOut, paramsShortOut);
    }

    /// <summary>
    /// Phase A + P + C: full SILK frame decode bitstream-to-PCM. Composes
    /// indices + pulses + parameters + synthesis (decode_core) into one
    /// orchestration. Mirrors the bulk of CPU
    /// <c>SilkDecodeFrame.Decode</c>; the OutBuf shift + scalar state
    /// updates that follow synthesis live in a separate kernel/finalizer
    /// step (caller's responsibility, or future Phase F).
    ///
    /// All buffers + scalars are GPU-resident (cardinal rule). The decode
    /// core kernel reads gainsQ16 + pitchL from <paramref name="paramsIntOut"/>,
    /// reads predCoefQ12 + ltpCoefQ14 + ltpScaleQ14 from <paramref name="paramsShortOut"/>,
    /// reads pulses from <paramref name="pulsesOut"/>, and writes PCM to the
    /// orchestrator-internal XqOut view inside
    /// <paramref name="decodeCoreInputs"/>.
    /// </summary>
    public void DecodeFullFrame(
        ArrayView<byte> packet, int packetStart, int packetStorage,
        SilkIndicesInputs indicesInputs,
        SilkIndicesScalars indicesScalars,
        SilkPulsesInputs pulsesInputs,
        int frameLength,
        SilkParametersInputs parametersInputs,
        SilkParametersState parametersState,
        SilkParametersScalars parametersScalars,
        SilkDecodeCoreInputs decodeCoreInputs,
        int signalType, int quantOffsetType, int seed,
        int lpcOrder, int nbSubfr, int subfrLength, int ltpMemLength,
        int nlsfInterpEnabled,
        ArrayView<OpusRangeDecoderGpuState> stateBuf,
        ArrayView<int> indicesOut,
        ArrayView<short> pulsesOut,
        ArrayView<int> paramsIntOut,
        ArrayView<short> paramsShortOut)
    {
        DecodeIndicesPulsesAndParameters(
            packet, packetStart, packetStorage,
            indicesInputs, indicesScalars,
            pulsesInputs, frameLength,
            parametersInputs, parametersState, parametersScalars,
            stateBuf, indicesOut, pulsesOut, paramsIntOut, paramsShortOut);

        var decodeCoreScalars = new SilkDecodeCoreScalars
        {
            SignalType = signalType,
            QuantOffsetType = quantOffsetType,
            Seed = seed,
            LpcOrder = lpcOrder,
            NbSubfr = nbSubfr,
            SubfrLength = subfrLength,
            FrameLength = frameLength,
            LtpMemLength = ltpMemLength,
            // LtpScaleQ14 is read from paramsShortOut by the kernel adapter.
            LtpScaleQ14 = 0,
            NlsfInterpEnabled = nlsfInterpEnabled,
        };
        _decodeCoreKernel(1,
            paramsIntOut, paramsShortOut, pulsesOut,
            decodeCoreInputs, decodeCoreScalars);
    }

    /// <summary>Release.</summary>
    public void Dispose() { /* auto-grouped kernels owned by accelerator */ }

    // -------- Kernel bodies --------

    /// <summary>Initialize the range-decoder state at <c>stateBuf[0]</c>
    /// from the packet bytes. Run once at the start of each frame.</summary>
    private static void InitStateKernel(
        Index1D _,
        ArrayView<byte> packet, int packetStart, int packetStorage,
        ArrayView<OpusRangeDecoderGpuState> stateBuf)
    {
        stateBuf[0] = OpusRangeDecoderGpu.Init(packet, packetStart, (uint)packetStorage);
    }

    /// <summary>Phase A.1: load state from buffer, call
    /// <see cref="SilkIndicesDecoderGpu.Decode"/>, save state back to
    /// buffer.</summary>
    private static void IndicesAdapterKernel(
        Index1D _,
        ArrayView<byte> packet, int packetStart, int packetStorage,
        SilkIndicesInputs inputs,
        SilkIndicesScalars scalars,
        ArrayView<OpusRangeDecoderGpuState> stateBuf,
        ArrayView<int> indicesOut)
    {
        var state = stateBuf[0];
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
            indicesOut, 0);
        stateBuf[0] = state;
    }

    /// <summary>Phase C: read per-frame parameters from the
    /// SilkDecodedParametersLayout buffers, populate the SilkDecodeCoreInputs
    /// scalar fields that came from those parameter buffers (gainsQ16,
    /// predCoefQ12, etc. are already wired as ArrayView fields on the
    /// inputs struct), and call <see cref="SilkDecodeCoreGpu.Decode"/>.
    /// LtpScaleQ14 is read from the short parameter output and pushed into
    /// the local scalars copy used by the synthesis chain.</summary>
    private static void DecodeCoreAdapterKernel(
        Index1D _,
        ArrayView<int> paramsIntIn,
        ArrayView<short> paramsShortIn,
        ArrayView<short> pulsesIn,
        SilkDecodeCoreInputs inputs,
        SilkDecodeCoreScalars scalars)
    {
        // Pull the scalar LTP scale Q14 out of the short parameter buffer and
        // patch it into the local scalars (host code pre-zeroed it).
        var localScalars = scalars;
        localScalars.LtpScaleQ14 = paramsShortIn[SilkDecodedParametersLayout.ShortLtpScaleQ14Offset];
        SilkDecodeCoreGpu.Decode(inputs, localScalars);
    }

    /// <summary>Phase P: dequantize per-frame parameters from the
    /// <paramref name="indicesIn"/> buffer. No range-decoder state crosses;
    /// pure data-flow kernel that reads indices and writes parameters.</summary>
    private static void ParametersAdapterKernel(
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

    /// <summary>Phase A.2: load state from buffer, read SignalType +
    /// QuantOffsetType from the indices buffer, call
    /// <see cref="SilkPulsesDecoderGpu.Decode"/>, save state back.</summary>
    private static void PulsesAdapterKernel(
        Index1D _,
        ArrayView<byte> packet, int packetStart, int packetStorage,
        SilkPulsesInputs inputs,
        ArrayView<int> indicesIn,
        int frameLength,
        ArrayView<OpusRangeDecoderGpuState> stateBuf,
        ArrayView<short> pulsesOut)
    {
        var state = stateBuf[0];
        int signalType = indicesIn[SilkDecodedIndicesLayout.SignalTypeOffset];
        int quantOffsetType = indicesIn[SilkDecodedIndicesLayout.QuantOffsetTypeOffset];
        SilkPulsesDecoderGpu.Decode(
            ref state, packet, packetStart, (uint)packetStorage,
            inputs,
            signalType, quantOffsetType, frameLength,
            pulsesOut, 0);
        stateBuf[0] = state;
    }
}
