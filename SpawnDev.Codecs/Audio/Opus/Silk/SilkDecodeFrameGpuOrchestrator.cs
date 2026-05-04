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
