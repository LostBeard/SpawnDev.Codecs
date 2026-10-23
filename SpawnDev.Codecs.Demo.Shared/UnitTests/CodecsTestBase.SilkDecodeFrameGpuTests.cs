// Tests for SilkDecodeFrameGpuOrchestrator Phase A: Indices + Pulses
// dispatched as 2 GPU kernels with range-decoder state crossing between
// them via a 1-element ArrayView<OpusRangeDecoderGpuState> buffer.
// Mirrors CPU SilkDecodeFrame.Decode's first 2 phases.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Audio.Opus;
using SpawnDev.Codecs.Audio.Opus.Silk;
using SpawnDev.Codecs.EntropyCoders;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    /// <summary>
    /// Drive the orchestrator's Phase A on the supplied accelerator and return
    /// (decoded indices, decoded pulses) for comparison vs CPU oracle.
    /// </summary>
    private static async Task<(int[] indices, short[] pulses)> SilkDecodeFrameGpuTest_RunPhaseAAsync(
        Accelerator acc,
        byte[] packet,
        SilkNlsfCodebook codebook,
        int fsKHz, int nbSubfr,
        int vadFlag, int decodeLbrr, int conditional,
        int prevLagIndex, int prevSignalTypeWasVoiced,
        int firstFrameAfterReset)
    {
        int frameLength = nbSubfr * 5 * fsKHz;
        int alignedPulsesLen = (frameLength + 15) & ~15;

        // Indices iCDF tables.
        using var dPacket = acc.Allocate1D<byte>(packet.Length);
        var lagLowBits = SilkIcdfTables.SelectPitchLagLowBits(fsKHz);
        var contour = SilkIcdfTables.SelectPitchContour(fsKHz, nbSubfr);

        // LTP gain flat-pack.
        var ltpFlat = new byte[SilkIcdfTables.LtpGain0.Length
                              + SilkIcdfTables.LtpGain1.Length
                              + SilkIcdfTables.LtpGain2.Length];
        Array.Copy(SilkIcdfTables.LtpGain0, 0, ltpFlat, 0, SilkIcdfTables.LtpGain0.Length);
        Array.Copy(SilkIcdfTables.LtpGain1, 0, ltpFlat, SilkIcdfTables.LtpGain0.Length, SilkIcdfTables.LtpGain1.Length);
        Array.Copy(SilkIcdfTables.LtpGain2, 0, ltpFlat,
            SilkIcdfTables.LtpGain0.Length + SilkIcdfTables.LtpGain1.Length,
            SilkIcdfTables.LtpGain2.Length);
        using var dLtpOffsets = acc.Allocate1D<int>(3);

        // Indices scratch.
        using var dEcIxScratch = acc.Allocate1D<short>(codebook.Order);

        // Pulses iCDFs + shell tables + scratches.

        // Outputs + state buffer.
        using var dStateBuf = acc.Allocate1D<OpusRangeDecoderGpuState>(1);
        using var dIndicesOut = acc.Allocate1D<int>(SilkDecodedIndicesLayout.TotalSlots);
        using var dPulsesOut = acc.Allocate1D<short>(alignedPulsesLen);

        // Upload static data.
        dPacket.View.CopyFromCPU(packet);
        dLtpOffsets.View.CopyFromCPU(new int[]
        {
            0,
            SilkIcdfTables.LtpGain0.Length,
            SilkIcdfTables.LtpGain0.Length + SilkIcdfTables.LtpGain1.Length,
        });

        var indicesInputsBytes = new PackedDeviceBuffer<byte>();
        var indicesInputs = new SilkIndicesInputs
        {
            TypeOffsetVadIcdfAt = indicesInputsBytes.Add(SilkIcdfTables.TypeOffsetVad, SilkIcdfTables.TypeOffsetVad.Length),
            TypeOffsetNoVadIcdfAt = indicesInputsBytes.Add(SilkIcdfTables.TypeOffsetNoVad, SilkIcdfTables.TypeOffsetNoVad.Length),
            Uniform4IcdfAt = indicesInputsBytes.Add(SilkIcdfTables.Uniform4, SilkIcdfTables.Uniform4.Length),
            GainIcdfAt = indicesInputsBytes.Add(SilkIcdfTables.Gain, SilkIcdfTables.Gain.Length),
            DeltaGainIcdfAt = indicesInputsBytes.Add(SilkIcdfTables.DeltaGain, SilkIcdfTables.DeltaGain.Length),
            Uniform8IcdfAt = indicesInputsBytes.Add(SilkIcdfTables.Uniform8, SilkIcdfTables.Uniform8.Length),
            Cb1IcdfAt = indicesInputsBytes.Add(codebook.Cb1Icdf, codebook.Cb1Icdf.Length),
            EcIcdfAt = indicesInputsBytes.Add(codebook.EcIcdf, codebook.EcIcdf.Length),
            EcSelAt = indicesInputsBytes.Add(codebook.EcSel, codebook.EcSel.Length),
            PredQ8SourceAt = indicesInputsBytes.Add(codebook.PredQ8, codebook.PredQ8.Length),
            NlsfExtIcdfAt = indicesInputsBytes.Add(SilkIcdfTables.NlsfExt, SilkIcdfTables.NlsfExt.Length),
            NlsfInterpolationFactorIcdfAt = indicesInputsBytes.Add(SilkIcdfTables.NlsfInterpolationFactor, SilkIcdfTables.NlsfInterpolationFactor.Length),
            PitchDeltaIcdfAt = indicesInputsBytes.Add(SilkIcdfTables.PitchDelta, SilkIcdfTables.PitchDelta.Length),
            PitchLagIcdfAt = indicesInputsBytes.Add(SilkIcdfTables.PitchLag, SilkIcdfTables.PitchLag.Length),
            LagLowBitsIcdfAt = indicesInputsBytes.Add(lagLowBits, lagLowBits.Length),
            ContourIcdfAt = indicesInputsBytes.Add(contour, contour.Length),
            LtpPerIndexIcdfAt = indicesInputsBytes.Add(SilkIcdfTables.LtpPerIndex, SilkIcdfTables.LtpPerIndex.Length),
            LtpGainIcdfFlatAt = indicesInputsBytes.Add(ltpFlat, ltpFlat.Length),
            LtpGainOffsets = dLtpOffsets.View,
            LtpScaleIcdfAt = indicesInputsBytes.Add(SilkIcdfTables.LtpScale, SilkIcdfTables.LtpScale.Length),
            EcIxScratch = dEcIxScratch.View,
            PredQ8ScratchAt = indicesInputsBytes.Reserve(codebook.Order),
        };
        using var dIndicesInputsBytes = indicesInputsBytes.Upload(acc);
        indicesInputs.Bytes = dIndicesInputsBytes.View;

        var indicesScalars = new SilkIndicesScalars
        {
            NVectors = codebook.NVectors,
            Order = codebook.Order,
            NbSubfr = nbSubfr,
            FsKHz = fsKHz,
            VadFlag = vadFlag,
            DecodeLbrr = decodeLbrr,
            Conditional = conditional,
            PrevLagIndex = prevLagIndex,
            PrevSignalTypeWasVoiced = prevSignalTypeWasVoiced,
            FirstFrameAfterReset = firstFrameAfterReset,
        };

        var pulsesInputsBytes = new PackedDeviceBuffer<byte>();
        var pulsesInputsInts = new PackedDeviceBuffer<int>();
        var pulsesInputs = SilkPulsesInputs.AddStandard(pulsesInputsBytes, pulsesInputsInts);
        using var dPulsesInputsBytes = pulsesInputsBytes.Upload(acc);
        using var dPulsesInputsInts = pulsesInputsInts.Upload(acc);
        pulsesInputs.Bind(dPulsesInputsBytes.View, dPulsesInputsInts.View);

        using var orchestrator = new SilkDecodeFrameGpuOrchestrator(acc);
        orchestrator.DecodeIndicesAndPulses(
            dPacket.View, 0, packet.Length,
            indicesInputs, indicesScalars, pulsesInputs,
            frameLength,
            dStateBuf.View.BaseView, dIndicesOut.View, dPulsesOut.View);
        await acc.SynchronizeAsync();

        var indices = await dIndicesOut.CopyToHostAsync();
        var pulses = await dPulsesOut.CopyToHostAsync();
        var pulsesSlice = new short[alignedPulsesLen];
        Array.Copy(pulses, pulsesSlice, alignedPulsesLen);
        return (indices, pulsesSlice);
    }

    /// <summary>Drive Phase A+P (indices + pulses + parameters) and return the GPU outputs.</summary>
    private static async Task<(int[] indices, short[] pulses, int[] paramsInt, short[] paramsShort)>
        SilkDecodeFrameGpuTest_RunPhaseAPAsync(
            Accelerator acc,
            byte[] packet,
            SilkNlsfCodebook codebook,
            int fsKHz, int nbSubfr,
            int vadFlag, int decodeLbrr, int conditional,
            int prevLagIndex, int prevSignalTypeWasVoiced,
            int firstFrameAfterReset,
            short[] prevNlsfQ15In, sbyte lastGainIndexIn)
    {
        int frameLength = nbSubfr * 5 * fsKHz;
        int alignedPulsesLen = (frameLength + 15) & ~15;
        int order = codebook.Order;

        // Indices iCDF tables.
        using var dPacket = acc.Allocate1D<byte>(packet.Length);
        var lagLowBits = SilkIcdfTables.SelectPitchLagLowBits(fsKHz);
        var contour = SilkIcdfTables.SelectPitchContour(fsKHz, nbSubfr);

        var ltpFlat = new byte[SilkIcdfTables.LtpGain0.Length
                              + SilkIcdfTables.LtpGain1.Length
                              + SilkIcdfTables.LtpGain2.Length];
        Array.Copy(SilkIcdfTables.LtpGain0, 0, ltpFlat, 0, SilkIcdfTables.LtpGain0.Length);
        Array.Copy(SilkIcdfTables.LtpGain1, 0, ltpFlat, SilkIcdfTables.LtpGain0.Length, SilkIcdfTables.LtpGain1.Length);
        Array.Copy(SilkIcdfTables.LtpGain2, 0, ltpFlat,
            SilkIcdfTables.LtpGain0.Length + SilkIcdfTables.LtpGain1.Length,
            SilkIcdfTables.LtpGain2.Length);
        using var dLtpOffsets = acc.Allocate1D<int>(3);

        using var dEcIxScratch = acc.Allocate1D<short>(order);


        // Parameters codebook + scratches.

        var (ltpGainsFlat, ltpGainsOffsets) = SilkParamsTest_FlatLtpGains();
        var (contourCb, contourCbSize) = SilkParamsTest_SelectContourCb(fsKHz, nbSubfr);


        // Outputs + state buffer.
        using var dStateBuf = acc.Allocate1D<OpusRangeDecoderGpuState>(1);
        using var dIndicesOut = acc.Allocate1D<int>(SilkDecodedIndicesLayout.TotalSlots);
        using var dPulsesOut = acc.Allocate1D<short>(alignedPulsesLen);
        using var dParamsIntOut = acc.Allocate1D<int>(SilkDecodedParametersLayout.IntTotalSlots);
        using var dParamsShortOut = acc.Allocate1D<short>(SilkDecodedParametersLayout.ShortTotalSlots);

        // Upload static data.
        dPacket.View.CopyFromCPU(packet);
        dLtpOffsets.View.CopyFromCPU(new int[]
        {
            0,
            SilkIcdfTables.LtpGain0.Length,
            SilkIcdfTables.LtpGain0.Length + SilkIcdfTables.LtpGain1.Length,
        });


        var indicesInputsBytes = new PackedDeviceBuffer<byte>();
        var indicesInputs = new SilkIndicesInputs
        {
            TypeOffsetVadIcdfAt = indicesInputsBytes.Add(SilkIcdfTables.TypeOffsetVad, SilkIcdfTables.TypeOffsetVad.Length),
            TypeOffsetNoVadIcdfAt = indicesInputsBytes.Add(SilkIcdfTables.TypeOffsetNoVad, SilkIcdfTables.TypeOffsetNoVad.Length),
            Uniform4IcdfAt = indicesInputsBytes.Add(SilkIcdfTables.Uniform4, SilkIcdfTables.Uniform4.Length),
            GainIcdfAt = indicesInputsBytes.Add(SilkIcdfTables.Gain, SilkIcdfTables.Gain.Length),
            DeltaGainIcdfAt = indicesInputsBytes.Add(SilkIcdfTables.DeltaGain, SilkIcdfTables.DeltaGain.Length),
            Uniform8IcdfAt = indicesInputsBytes.Add(SilkIcdfTables.Uniform8, SilkIcdfTables.Uniform8.Length),
            Cb1IcdfAt = indicesInputsBytes.Add(codebook.Cb1Icdf, codebook.Cb1Icdf.Length),
            EcIcdfAt = indicesInputsBytes.Add(codebook.EcIcdf, codebook.EcIcdf.Length),
            EcSelAt = indicesInputsBytes.Add(codebook.EcSel, codebook.EcSel.Length),
            PredQ8SourceAt = indicesInputsBytes.Add(codebook.PredQ8, codebook.PredQ8.Length),
            NlsfExtIcdfAt = indicesInputsBytes.Add(SilkIcdfTables.NlsfExt, SilkIcdfTables.NlsfExt.Length),
            NlsfInterpolationFactorIcdfAt = indicesInputsBytes.Add(SilkIcdfTables.NlsfInterpolationFactor, SilkIcdfTables.NlsfInterpolationFactor.Length),
            PitchDeltaIcdfAt = indicesInputsBytes.Add(SilkIcdfTables.PitchDelta, SilkIcdfTables.PitchDelta.Length),
            PitchLagIcdfAt = indicesInputsBytes.Add(SilkIcdfTables.PitchLag, SilkIcdfTables.PitchLag.Length),
            LagLowBitsIcdfAt = indicesInputsBytes.Add(lagLowBits, lagLowBits.Length),
            ContourIcdfAt = indicesInputsBytes.Add(contour, contour.Length),
            LtpPerIndexIcdfAt = indicesInputsBytes.Add(SilkIcdfTables.LtpPerIndex, SilkIcdfTables.LtpPerIndex.Length),
            LtpGainIcdfFlatAt = indicesInputsBytes.Add(ltpFlat, ltpFlat.Length),
            LtpGainOffsets = dLtpOffsets.View,
            LtpScaleIcdfAt = indicesInputsBytes.Add(SilkIcdfTables.LtpScale, SilkIcdfTables.LtpScale.Length),
            EcIxScratch = dEcIxScratch.View,
            PredQ8ScratchAt = indicesInputsBytes.Reserve(order),
        };
        using var dIndicesInputsBytes = indicesInputsBytes.Upload(acc);
        indicesInputs.Bytes = dIndicesInputsBytes.View;

        var indicesScalars = new SilkIndicesScalars
        {
            NVectors = codebook.NVectors,
            Order = order,
            NbSubfr = nbSubfr,
            FsKHz = fsKHz,
            VadFlag = vadFlag,
            DecodeLbrr = decodeLbrr,
            Conditional = conditional,
            PrevLagIndex = prevLagIndex,
            PrevSignalTypeWasVoiced = prevSignalTypeWasVoiced,
            FirstFrameAfterReset = firstFrameAfterReset,
        };

        var pulsesInputsBytes = new PackedDeviceBuffer<byte>();
        var pulsesInputsInts = new PackedDeviceBuffer<int>();
        var pulsesInputs = SilkPulsesInputs.AddStandard(pulsesInputsBytes, pulsesInputsInts);
        using var dPulsesInputsBytes = pulsesInputsBytes.Upload(acc);
        using var dPulsesInputsInts = pulsesInputsInts.Upload(acc);
        pulsesInputs.Bind(dPulsesInputsBytes.View, dPulsesInputsInts.View);

        var parametersInputsBytes = new PackedDeviceBuffer<byte>();
        var parametersInputsShorts = new PackedDeviceBuffer<short>();
        var parametersInputsSBytes = new PackedDeviceBuffer<sbyte>();
        var parametersInputsInts = new PackedDeviceBuffer<int>();
        var parametersInputs = new SilkParametersInputs
        {
            Cb1NlsfQ8At = parametersInputsBytes.Add(codebook.Cb1NlsfQ8, codebook.Cb1NlsfQ8.Length),
            Cb1WghtQ9At = parametersInputsShorts.Add(codebook.Cb1WghtQ9, codebook.Cb1WghtQ9.Length),
            EcSelAt = parametersInputsBytes.Add(codebook.EcSel, codebook.EcSel.Length),
            PredQ8SourceAt = parametersInputsBytes.Add(codebook.PredQ8, codebook.PredQ8.Length),
            DeltaMinQ15At = parametersInputsShorts.Add(codebook.DeltaMinQ15, codebook.DeltaMinQ15.Length),
            LsfCosTabQ12At = parametersInputsShorts.Add(SilkLsfCosTab.Q12, SilkLsfCosTab.Q12.Length),
            ContourCbAt = parametersInputsSBytes.Add(contourCb, contourCb.Length),
            LtpGainTablesFlatAt = parametersInputsSBytes.Add(ltpGainsFlat, ltpGainsFlat.Length),
            LtpGainOffsetsAt = parametersInputsInts.Add(ltpGainsOffsets, 3),
            LtpScaleQ14TableAt = parametersInputsShorts.Add(SilkParamsTest_LtpScalesQ14, SilkParamsTest_LtpScalesQ14.Length),
        };

        var parametersState = new SilkParametersState
        {
            PrevNlsfQ15At = parametersInputsShorts.Add(prevNlsfQ15In, order),
            LastGainIndexAt = parametersInputsInts.Add(new int[] { lastGainIndexIn }, 1),
            NlsfDecodeScratchAt = parametersInputsShorts.Reserve(3 * 16),
            NlsfDecodePredScratchAt = parametersInputsBytes.Reserve(16),
            Nlsf2aScratchAt = parametersInputsInts.Reserve(66),
            NlsfIndicesScratchAt = parametersInputsSBytes.Reserve(order + 1),
            GainIndicesScratchAt = parametersInputsSBytes.Reserve(nbSubfr),
        };
        using var dParametersInputsBytes = parametersInputsBytes.Upload(acc);
        parametersInputs.Bytes = dParametersInputsBytes.View;
        using var dParametersInputsShorts = parametersInputsShorts.Upload(acc);
        parametersInputs.Shorts = dParametersInputsShorts.View;
        using var dParametersInputsSBytes = parametersInputsSBytes.Upload(acc);
        parametersInputs.SBytes = dParametersInputsSBytes.View;
        using var dParametersInputsInts = parametersInputsInts.Upload(acc);
        parametersInputs.Ints = dParametersInputsInts.View;

        var parametersScalars = new SilkParametersScalars
        {
            QuantStepSizeQ16 = codebook.QuantStepSizeQ16,
            Order = order,
            NbSubfr = nbSubfr,
            FsKHz = fsKHz,
            ContourCbSize = contourCbSize,
            Conditional = conditional,
        };

        using var orchestrator = new SilkDecodeFrameGpuOrchestrator(acc);
        orchestrator.DecodeIndicesPulsesAndParameters(
            dPacket.View, 0, packet.Length,
            indicesInputs, indicesScalars,
            pulsesInputs, frameLength,
            parametersInputs, parametersState, parametersScalars,
            dStateBuf.View.BaseView,
            dIndicesOut.View, dPulsesOut.View,
            dParamsIntOut.View, dParamsShortOut.View);
        await acc.SynchronizeAsync();

        var indices = await dIndicesOut.CopyToHostAsync();
        var pulses = await dPulsesOut.CopyToHostAsync();
        var paramsInt = await dParamsIntOut.CopyToHostAsync();
        var paramsShort = await dParamsShortOut.CopyToHostAsync();
        var pulsesSlice = new short[alignedPulsesLen];
        Array.Copy(pulses, pulsesSlice, alignedPulsesLen);
        return (indices, pulsesSlice, paramsInt, paramsShort);
    }

    [TestMethod(Timeout = 600000)]
    public async Task SilkDecodeFrameGpu_PhaseAP_Voiced_NB_BitExactVsCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {

            const int fsKHz = 8, nbSubfr = 4;
            int frameLength = nbSubfr * 5 * fsKHz;
            var codebook = SilkNlsfCodebookTables.NbMb;

            var indices = new SilkDecodedIndices
            {
                SignalType = SilkSideInfoDecoder.TypeVoiced,
                QuantOffsetType = 0,
                NlsfInterpCoefQ2 = 4,
                LagIndex = 80,
                ContourIndex = 2,
                PerIndex = 1,
                LtpScaleIndex = 0,
                Seed = 1,
            };
            for (int i = 0; i < nbSubfr; i++) indices.GainsIndices[i] = 25;
            indices.NlsfIndices[0] = 7;
            for (int i = 0; i < nbSubfr; i++) indices.LtpIndices[i] = (sbyte)(i + 3);

            short[] pulsesIn = new short[((frameLength + 15) & ~15)];
            byte[] bitstream = EncodeFullSilkFrame(
                codebook, indices, pulsesIn,
                fsKHz: fsKHz, nbSubfr: nbSubfr, conditional: 0, vadFlag: true);

            // CPU oracle: indices + pulses + parameters via the same bitstream.
            var dec = new OpusRangeDecoder(bitstream);
            var cpuIndices = new SilkDecodedIndices();
            SilkIndicesDecoder.Decode(
                cpuIndices, dec, codebook,
                vadFlag: true, decodeLbrr: false,
                fsKHz: fsKHz, nbSubfr: nbSubfr, conditional: 0,
                prevLagIndex: 0, prevSignalTypeWasVoiced: false);
            short[] cpuPulses = new short[((frameLength + 15) & ~15)];
            SilkPulsesDecoder.Decode(
                cpuPulses.AsSpan(), dec,
                signalType: cpuIndices.SignalType,
                quantOffsetType: cpuIndices.QuantOffsetType,
                frameLength: frameLength);
            var cpuParameters = new SilkDecodedParameters();
            sbyte cpuLastGainIdx = 0;
            short[] cpuPrevNlsf = new short[codebook.Order];
            SilkParametersDecoder.Decode(
                cpuParameters, cpuIndices, codebook,
                fsKHz: fsKHz, nbSubfr: nbSubfr,
                lastGainIndex: ref cpuLastGainIdx,
                prevNlsfQ15: cpuPrevNlsf.AsSpan(0, codebook.Order),
                conditional: 0);

            // GPU.
            var (gpuIndices, gpuPulses, gpuParamsInt, gpuParamsShort) =
                await SilkDecodeFrameGpuTest_RunPhaseAPAsync(
                    acc, bitstream, codebook,
                    fsKHz: fsKHz, nbSubfr: nbSubfr,
                    vadFlag: 1, decodeLbrr: 0, conditional: 0,
                    prevLagIndex: 0, prevSignalTypeWasVoiced: 0,
                    firstFrameAfterReset: 1,
                    prevNlsfQ15In: new short[codebook.Order],
                    lastGainIndexIn: 0);

            // Compare gainsQ16 + pitchL.
            for (int i = 0; i < nbSubfr; i++)
            {
                int gpuGain = gpuParamsInt[SilkDecodedParametersLayout.IntGainsQ16Offset + i];
                if (cpuParameters.GainsQ16[i] != gpuGain)
                    throw new Exception($"GainsQ16[{i}] mismatch: cpu={cpuParameters.GainsQ16[i]} gpu={gpuGain}");
                int gpuLag = gpuParamsInt[SilkDecodedParametersLayout.IntPitchLOffset + i];
                if (cpuParameters.PitchL[i] != gpuLag)
                    throw new Exception($"PitchL[{i}] mismatch: cpu={cpuParameters.PitchL[i]} gpu={gpuLag}");
            }
            // Compare nlsfQ15.
            for (int i = 0; i < codebook.Order; i++)
            {
                short gpuNlsf = gpuParamsShort[SilkDecodedParametersLayout.ShortNlsfQ15Offset + i];
                if (cpuParameters.NlsfQ15[i] != gpuNlsf)
                    throw new Exception($"NlsfQ15[{i}] mismatch: cpu={cpuParameters.NlsfQ15[i]} gpu={gpuNlsf}");
            }
            // Compare ltpScaleQ14.
            short gpuLtpScale = gpuParamsShort[SilkDecodedParametersLayout.ShortLtpScaleQ14Offset];
            if (cpuParameters.LtpScaleQ14 != gpuLtpScale)
                throw new Exception($"LtpScaleQ14 mismatch: cpu={cpuParameters.LtpScaleQ14} gpu={gpuLtpScale}");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    /// <summary>Drive Phase A+P+C (full bitstream-to-PCM, optionally with finalize)
    /// and return GPU PCM + updated SLpcQ14Buf + final PrevGainQ16 + OutBuf
    /// snapshot for comparison vs CPU.</summary>
    private static async Task<(short[] pcm, int[] sLpcQ14Buf, int prevGainQ16, short[] outBuf)>
        SilkDecodeFrameGpuTest_RunPhaseAPCAsync(
            Accelerator acc,
            byte[] packet,
            SilkNlsfCodebook codebook,
            int fsKHz, int nbSubfr,
            int vadFlag, int decodeLbrr, int conditional,
            int prevLagIndex, int prevSignalTypeWasVoiced,
            int firstFrameAfterReset,
            short[] prevNlsfQ15In, sbyte lastGainIndexIn,
            int signalType, int quantOffsetType, int seed,
            int lpcOrder, int nlsfInterpEnabled,
            int initialPrevGainQ16,
            int[] initialSLpcQ14Buf,
            short[] initialOutBuf,
            bool withFinalize = false)
    {
        int frameLength = nbSubfr * 5 * fsKHz;
        int alignedPulsesLen = (frameLength + 15) & ~15;
        int order = codebook.Order;
        int subfrLength = 5 * fsKHz;
        int ltpMemLength = 20 * fsKHz; // LTP_MEM_LENGTH_MS * fs_kHz

        const int MaxLpcOrder = 16;
        const int MaxFrameLength = 320;
        const int MaxSubfrLength = 80;
        const int MaxLtpMemLength = 320;

        // Indices iCDF tables (same as Phase A+P).
        using var dPacket = acc.Allocate1D<byte>(packet.Length);
        var lagLowBits = SilkIcdfTables.SelectPitchLagLowBits(fsKHz);
        var contour = SilkIcdfTables.SelectPitchContour(fsKHz, nbSubfr);
        var ltpFlat = new byte[SilkIcdfTables.LtpGain0.Length
                              + SilkIcdfTables.LtpGain1.Length
                              + SilkIcdfTables.LtpGain2.Length];
        Array.Copy(SilkIcdfTables.LtpGain0, 0, ltpFlat, 0, SilkIcdfTables.LtpGain0.Length);
        Array.Copy(SilkIcdfTables.LtpGain1, 0, ltpFlat, SilkIcdfTables.LtpGain0.Length, SilkIcdfTables.LtpGain1.Length);
        Array.Copy(SilkIcdfTables.LtpGain2, 0, ltpFlat,
            SilkIcdfTables.LtpGain0.Length + SilkIcdfTables.LtpGain1.Length,
            SilkIcdfTables.LtpGain2.Length);
        using var dLtpOffsets = acc.Allocate1D<int>(3);
        using var dEcIxScratch = acc.Allocate1D<short>(order);


        var (ltpGainsFlat, ltpGainsOffsets) = SilkParamsTest_FlatLtpGains();
        var (contourCb, contourCbSize) = SilkParamsTest_SelectContourCb(fsKHz, nbSubfr);

        // DecodeCore state + scratches + output.
        using var dStateBuf = acc.Allocate1D<OpusRangeDecoderGpuState>(1);
        using var dIndicesOut = acc.Allocate1D<int>(SilkDecodedIndicesLayout.TotalSlots);
        using var dPulsesOut = acc.Allocate1D<short>(alignedPulsesLen);
        using var dParamsIntOut = acc.Allocate1D<int>(SilkDecodedParametersLayout.IntTotalSlots);
        using var dParamsShortOut = acc.Allocate1D<short>(SilkDecodedParametersLayout.ShortTotalSlots);


        // Upload static data + initial state.
        dPacket.View.CopyFromCPU(packet);
        dLtpOffsets.View.CopyFromCPU(new int[]
        {
            0,
            SilkIcdfTables.LtpGain0.Length,
            SilkIcdfTables.LtpGain0.Length + SilkIcdfTables.LtpGain1.Length,
        });

        var indicesInputsBytes = new PackedDeviceBuffer<byte>();
        var indicesInputs = new SilkIndicesInputs
        {
            TypeOffsetVadIcdfAt = indicesInputsBytes.Add(SilkIcdfTables.TypeOffsetVad, SilkIcdfTables.TypeOffsetVad.Length),
            TypeOffsetNoVadIcdfAt = indicesInputsBytes.Add(SilkIcdfTables.TypeOffsetNoVad, SilkIcdfTables.TypeOffsetNoVad.Length),
            Uniform4IcdfAt = indicesInputsBytes.Add(SilkIcdfTables.Uniform4, SilkIcdfTables.Uniform4.Length),
            GainIcdfAt = indicesInputsBytes.Add(SilkIcdfTables.Gain, SilkIcdfTables.Gain.Length),
            DeltaGainIcdfAt = indicesInputsBytes.Add(SilkIcdfTables.DeltaGain, SilkIcdfTables.DeltaGain.Length),
            Uniform8IcdfAt = indicesInputsBytes.Add(SilkIcdfTables.Uniform8, SilkIcdfTables.Uniform8.Length),
            Cb1IcdfAt = indicesInputsBytes.Add(codebook.Cb1Icdf, codebook.Cb1Icdf.Length),
            EcIcdfAt = indicesInputsBytes.Add(codebook.EcIcdf, codebook.EcIcdf.Length),
            EcSelAt = indicesInputsBytes.Add(codebook.EcSel, codebook.EcSel.Length),
            PredQ8SourceAt = indicesInputsBytes.Add(codebook.PredQ8, codebook.PredQ8.Length),
            NlsfExtIcdfAt = indicesInputsBytes.Add(SilkIcdfTables.NlsfExt, SilkIcdfTables.NlsfExt.Length),
            NlsfInterpolationFactorIcdfAt = indicesInputsBytes.Add(SilkIcdfTables.NlsfInterpolationFactor, SilkIcdfTables.NlsfInterpolationFactor.Length),
            PitchDeltaIcdfAt = indicesInputsBytes.Add(SilkIcdfTables.PitchDelta, SilkIcdfTables.PitchDelta.Length),
            PitchLagIcdfAt = indicesInputsBytes.Add(SilkIcdfTables.PitchLag, SilkIcdfTables.PitchLag.Length),
            LagLowBitsIcdfAt = indicesInputsBytes.Add(lagLowBits, lagLowBits.Length),
            ContourIcdfAt = indicesInputsBytes.Add(contour, contour.Length),
            LtpPerIndexIcdfAt = indicesInputsBytes.Add(SilkIcdfTables.LtpPerIndex, SilkIcdfTables.LtpPerIndex.Length),
            LtpGainIcdfFlatAt = indicesInputsBytes.Add(ltpFlat, ltpFlat.Length),
            LtpGainOffsets = dLtpOffsets.View,
            LtpScaleIcdfAt = indicesInputsBytes.Add(SilkIcdfTables.LtpScale, SilkIcdfTables.LtpScale.Length),
            EcIxScratch = dEcIxScratch.View,
            PredQ8ScratchAt = indicesInputsBytes.Reserve(order),
        };
        using var dIndicesInputsBytes = indicesInputsBytes.Upload(acc);
        indicesInputs.Bytes = dIndicesInputsBytes.View;
        var indicesScalars = new SilkIndicesScalars
        {
            NVectors = codebook.NVectors,
            Order = order,
            NbSubfr = nbSubfr,
            FsKHz = fsKHz,
            VadFlag = vadFlag,
            DecodeLbrr = decodeLbrr,
            Conditional = conditional,
            PrevLagIndex = prevLagIndex,
            PrevSignalTypeWasVoiced = prevSignalTypeWasVoiced,
            FirstFrameAfterReset = firstFrameAfterReset,
        };
        var pulsesInputsBytes = new PackedDeviceBuffer<byte>();
        var pulsesInputsInts = new PackedDeviceBuffer<int>();
        var pulsesInputs = SilkPulsesInputs.AddStandard(pulsesInputsBytes, pulsesInputsInts);
        using var dPulsesInputsBytes = pulsesInputsBytes.Upload(acc);
        using var dPulsesInputsInts = pulsesInputsInts.Upload(acc);
        pulsesInputs.Bind(dPulsesInputsBytes.View, dPulsesInputsInts.View);
        var parametersInputsBytes = new PackedDeviceBuffer<byte>();
        var parametersInputsShorts = new PackedDeviceBuffer<short>();
        var parametersInputsSBytes = new PackedDeviceBuffer<sbyte>();
        var parametersInputsInts = new PackedDeviceBuffer<int>();
        var parametersInputs = new SilkParametersInputs
        {
            Cb1NlsfQ8At = parametersInputsBytes.Add(codebook.Cb1NlsfQ8, codebook.Cb1NlsfQ8.Length),
            Cb1WghtQ9At = parametersInputsShorts.Add(codebook.Cb1WghtQ9, codebook.Cb1WghtQ9.Length),
            EcSelAt = parametersInputsBytes.Add(codebook.EcSel, codebook.EcSel.Length),
            PredQ8SourceAt = parametersInputsBytes.Add(codebook.PredQ8, codebook.PredQ8.Length),
            DeltaMinQ15At = parametersInputsShorts.Add(codebook.DeltaMinQ15, codebook.DeltaMinQ15.Length),
            LsfCosTabQ12At = parametersInputsShorts.Add(SilkLsfCosTab.Q12, SilkLsfCosTab.Q12.Length),
            ContourCbAt = parametersInputsSBytes.Add(contourCb, contourCb.Length),
            LtpGainTablesFlatAt = parametersInputsSBytes.Add(ltpGainsFlat, ltpGainsFlat.Length),
            LtpGainOffsetsAt = parametersInputsInts.Add(ltpGainsOffsets, 3),
            LtpScaleQ14TableAt = parametersInputsShorts.Add(SilkParamsTest_LtpScalesQ14, SilkParamsTest_LtpScalesQ14.Length),
        };
        var parametersState = new SilkParametersState
        {
            PrevNlsfQ15At = parametersInputsShorts.Add(prevNlsfQ15In, order),
            LastGainIndexAt = parametersInputsInts.Add(new int[] { lastGainIndexIn }, 1),
            NlsfDecodeScratchAt = parametersInputsShorts.Reserve(3 * 16),
            NlsfDecodePredScratchAt = parametersInputsBytes.Reserve(16),
            Nlsf2aScratchAt = parametersInputsInts.Reserve(66),
            NlsfIndicesScratchAt = parametersInputsSBytes.Reserve(order + 1),
            GainIndicesScratchAt = parametersInputsSBytes.Reserve(nbSubfr),
        };
        using var dParametersInputsBytes = parametersInputsBytes.Upload(acc);
        parametersInputs.Bytes = dParametersInputsBytes.View;
        using var dParametersInputsShorts = parametersInputsShorts.Upload(acc);
        parametersInputs.Shorts = dParametersInputsShorts.View;
        using var dParametersInputsSBytes = parametersInputsSBytes.Upload(acc);
        parametersInputs.SBytes = dParametersInputsSBytes.View;
        using var dParametersInputsInts = parametersInputsInts.Upload(acc);
        parametersInputs.Ints = dParametersInputsInts.View;
        var parametersScalars = new SilkParametersScalars
        {
            QuantStepSizeQ16 = codebook.QuantStepSizeQ16,
            Order = order,
            NbSubfr = nbSubfr,
            FsKHz = fsKHz,
            ContourCbSize = contourCbSize,
            Conditional = conditional,
        };

        // SilkDecodeCoreInputs: PredCoefQ12 / GainsQ16 / PitchL / LtpCoefQ14
        // are SubViews into the parameter output buffers per
        // SilkDecodedParametersLayout.
        var decodeCoreInputsShorts = new PackedDeviceBuffer<short>();
        var decodeCoreInputsInts = new PackedDeviceBuffer<int>();
        var decodeCoreInputs = new SilkDecodeCoreInputs
        {
            ParamsShort = dParamsShortOut.View,
            ParamsInt = dParamsIntOut.View,
            PredCoefQ12At = SilkDecodedParametersLayout.ShortPredCoefQ12Half1Offset,
            GainsQ16At = SilkDecodedParametersLayout.IntGainsQ16Offset,
            PitchLAt = SilkDecodedParametersLayout.IntPitchLOffset,
            LtpCoefQ14At = SilkDecodedParametersLayout.ShortLtpCoefQ14Offset,
            Pulses = dPulsesOut.View,
            OutBufInOutAt = decodeCoreInputsShorts.Add(initialOutBuf, MaxLtpMemLength + MaxFrameLength),
            SLpcQ14BufInOutAt = decodeCoreInputsInts.Add(initialSLpcQ14Buf, MaxLpcOrder),
            ExcQ14OutAt = decodeCoreInputsInts.Reserve(MaxFrameLength),
            PrevGainQ16InOutAt = decodeCoreInputsInts.Add(new int[] { initialPrevGainQ16 }, 1),
            SLpcScratchAt = decodeCoreInputsInts.Reserve(MaxLpcOrder + MaxSubfrLength),
            SLtpQ15ScratchAt = decodeCoreInputsInts.Reserve(MaxLtpMemLength + MaxFrameLength),
            SLtpScratchAt = decodeCoreInputsShorts.Reserve(MaxLtpMemLength),
            PresQ14ScratchAt = decodeCoreInputsInts.Reserve(MaxSubfrLength),
            GainAdjScratchAt = decodeCoreInputsInts.Reserve(1),
            XqOutAt = decodeCoreInputsShorts.Reserve(MaxFrameLength),
        };
        using var dDecodeCoreInputsShorts = decodeCoreInputsShorts.Upload(acc);
        decodeCoreInputs.Shorts = dDecodeCoreInputsShorts.View;
        using var dDecodeCoreInputsInts = decodeCoreInputsInts.Upload(acc);
        decodeCoreInputs.Ints = dDecodeCoreInputsInts.View;

        using var orchestrator = new SilkDecodeFrameGpuOrchestrator(acc);
        if (withFinalize)
        {
            orchestrator.DecodeFullFrameWithFinalize(
                dPacket.View, 0, packet.Length,
                indicesInputs, indicesScalars,
                pulsesInputs, frameLength,
                parametersInputs, parametersState, parametersScalars,
                decodeCoreInputs,
                signalType, quantOffsetType, seed,
                lpcOrder, nbSubfr, subfrLength, ltpMemLength,
                nlsfInterpEnabled,
                dStateBuf.View.BaseView,
                dIndicesOut.View, dPulsesOut.View,
                dParamsIntOut.View, dParamsShortOut.View);
        }
        else
        {
            orchestrator.DecodeFullFrame(
                dPacket.View, 0, packet.Length,
                indicesInputs, indicesScalars,
                pulsesInputs, frameLength,
                parametersInputs, parametersState, parametersScalars,
                decodeCoreInputs,
                signalType, quantOffsetType, seed,
                lpcOrder, nbSubfr, subfrLength, ltpMemLength,
                nlsfInterpEnabled,
                dStateBuf.View.BaseView,
                dIndicesOut.View, dPulsesOut.View,
                dParamsIntOut.View, dParamsShortOut.View);
        }
        await acc.SynchronizeAsync();

        var pcmFull = await dDecodeCoreInputsShorts.View.SubView(decodeCoreInputs.XqOutAt, MaxFrameLength).CopyToHostAsync();
        var sLpcOut = await dDecodeCoreInputsInts.View.SubView(decodeCoreInputs.SLpcQ14BufInOutAt, MaxLpcOrder).CopyToHostAsync();
        var prevGainOut = await dDecodeCoreInputsInts.View.SubView(decodeCoreInputs.PrevGainQ16InOutAt, 1).CopyToHostAsync();
        var outBufFull = await dDecodeCoreInputsShorts.View.SubView(decodeCoreInputs.OutBufInOutAt, MaxLtpMemLength + MaxFrameLength).CopyToHostAsync();
        var pcm = new short[frameLength];
        Array.Copy(pcmFull, pcm, frameLength);
        return (pcm, sLpcOut, prevGainOut[0], outBufFull);
    }

    [TestMethod(Timeout = 600000)]
    public async Task SilkDecodeFrameGpu_PhaseAPC_Voiced_MB_BitExactVsCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {

            // MB voiced: 12 kHz × 4 subframes × 60 samples = 240 samples/frame.
            // Distinct subfrLength path between NB (40) and WB (80).
            const int fsKHz = 12, nbSubfr = 4, lpcOrder = 10;
            int frameLength = nbSubfr * 5 * fsKHz;
            var codebook = SilkNlsfCodebookTables.NbMb;

            var indices = new SilkDecodedIndices
            {
                SignalType = SilkSideInfoDecoder.TypeVoiced,
                QuantOffsetType = 0,
                NlsfInterpCoefQ2 = 4,
                LagIndex = 80, // safe value; minPitchLag MB ~24, maxPitchLag MB ~216
                ContourIndex = 2,
                PerIndex = 1,
                LtpScaleIndex = 0,
                Seed = 1,
            };
            for (int i = 0; i < nbSubfr; i++) indices.GainsIndices[i] = 25;
            indices.NlsfIndices[0] = 7;
            for (int i = 0; i < nbSubfr; i++) indices.LtpIndices[i] = (sbyte)(i + 3);

            short[] pulsesIn = new short[((frameLength + 15) & ~15)];
            byte[] bitstream = EncodeFullSilkFrame(
                codebook, indices, pulsesIn,
                fsKHz: fsKHz, nbSubfr: nbSubfr, conditional: 0, vadFlag: true);

            // CPU oracle.
            var cpuState = new SilkChannelDecoderState();
            cpuState.Configure(fsKHz: fsKHz, nbSubfr: nbSubfr, lpcOrder: lpcOrder);
            cpuState.Reset();
            var cpuDec = new OpusRangeDecoder(bitstream);
            short[] cpuPcm = new short[frameLength];
            SilkDecodeFrame.Decode(cpuState, cpuDec, cpuPcm, vadFlag: true, conditional: 0);

            // GPU.
            var gpuInitialState = new SilkChannelDecoderState();
            gpuInitialState.Configure(fsKHz: fsKHz, nbSubfr: nbSubfr, lpcOrder: lpcOrder);
            gpuInitialState.Reset();
            var (gpuPcm, _, _, _) = await SilkDecodeFrameGpuTest_RunPhaseAPCAsync(
                acc, bitstream, codebook,
                fsKHz: fsKHz, nbSubfr: nbSubfr,
                vadFlag: 1, decodeLbrr: 0, conditional: 0,
                prevLagIndex: 0, prevSignalTypeWasVoiced: 0,
                firstFrameAfterReset: 1,
                prevNlsfQ15In: new short[codebook.Order],
                lastGainIndexIn: 0,
                signalType: indices.SignalType,
                quantOffsetType: indices.QuantOffsetType,
                seed: indices.Seed,
                lpcOrder: lpcOrder,
                nlsfInterpEnabled: 0,
                initialPrevGainQ16: gpuInitialState.PrevGainQ16,
                initialSLpcQ14Buf: gpuInitialState.SLpcQ14Buf,
                initialOutBuf: gpuInitialState.OutBuf);

            for (int i = 0; i < frameLength; i++)
                if (cpuPcm[i] != gpuPcm[i])
                    throw new Exception($"MB PCM mismatch at sample {i}: cpu={cpuPcm[i]} gpu={gpuPcm[i]}");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod(Timeout = 600000)]
    public async Task SilkDecodeFrameGpu_PhaseAPC_Voiced_WB_BitExactVsCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {

            // WB voiced: 16 kHz × 4 subframes × 80 samples = 320 samples/frame.
            // lpcOrder=16 exercises the WB NLSF codebook + the full 16-tap LPC
            // synthesis filter unroll (vs NB's 10-tap path).
            const int fsKHz = 16, nbSubfr = 4, lpcOrder = 16;
            int frameLength = nbSubfr * 5 * fsKHz;
            var codebook = SilkNlsfCodebookTables.Wb;

            var indices = new SilkDecodedIndices
            {
                SignalType = SilkSideInfoDecoder.TypeVoiced,
                QuantOffsetType = 1,
                NlsfInterpCoefQ2 = 4,
                LagIndex = 200, // within WB max (PE_MAX_LAG_MS * fs_kHz = 18 * 16 = 288)
                ContourIndex = 5,
                PerIndex = 2,
                LtpScaleIndex = 1,
                Seed = 3,
            };
            for (int i = 0; i < nbSubfr; i++) indices.GainsIndices[i] = (sbyte)(30 + i);
            indices.NlsfIndices[0] = 12;
            for (int i = 0; i < nbSubfr; i++) indices.LtpIndices[i] = (sbyte)(i * 2 + 1);

            short[] pulsesIn = new short[((frameLength + 15) & ~15)];
            byte[] bitstream = EncodeFullSilkFrame(
                codebook, indices, pulsesIn,
                fsKHz: fsKHz, nbSubfr: nbSubfr, conditional: 0, vadFlag: true);

            // CPU oracle.
            var cpuState = new SilkChannelDecoderState();
            cpuState.Configure(fsKHz: fsKHz, nbSubfr: nbSubfr, lpcOrder: lpcOrder);
            cpuState.Reset();
            var cpuDec = new OpusRangeDecoder(bitstream);
            short[] cpuPcm = new short[frameLength];
            SilkDecodeFrame.Decode(cpuState, cpuDec, cpuPcm, vadFlag: true, conditional: 0);

            // GPU.
            var gpuInitialState = new SilkChannelDecoderState();
            gpuInitialState.Configure(fsKHz: fsKHz, nbSubfr: nbSubfr, lpcOrder: lpcOrder);
            gpuInitialState.Reset();
            var (gpuPcm, _, _, _) = await SilkDecodeFrameGpuTest_RunPhaseAPCAsync(
                acc, bitstream, codebook,
                fsKHz: fsKHz, nbSubfr: nbSubfr,
                vadFlag: 1, decodeLbrr: 0, conditional: 0,
                prevLagIndex: 0, prevSignalTypeWasVoiced: 0,
                firstFrameAfterReset: 1,
                prevNlsfQ15In: new short[codebook.Order],
                lastGainIndexIn: 0,
                signalType: indices.SignalType,
                quantOffsetType: indices.QuantOffsetType,
                seed: indices.Seed,
                lpcOrder: lpcOrder,
                nlsfInterpEnabled: 0,
                initialPrevGainQ16: gpuInitialState.PrevGainQ16,
                initialSLpcQ14Buf: gpuInitialState.SLpcQ14Buf,
                initialOutBuf: gpuInitialState.OutBuf);

            for (int i = 0; i < frameLength; i++)
                if (cpuPcm[i] != gpuPcm[i])
                    throw new Exception($"WB PCM mismatch at sample {i}: cpu={cpuPcm[i]} gpu={gpuPcm[i]}");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod(Timeout = 600000)]
    public async Task SilkDecodeFrameGpu_TwoFrameStateRolling_NB_BitExactVsCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {

            // Two consecutive voiced NB frames threaded through state. Verifies
            // the orchestrator is multi-frame-ready end-to-end: frame 2's
            // outBuf history (LTP state) comes from frame 1's finalize; the
            // SLpcQ14Buf carry-over comes from frame 1's decode_core; the
            // PrevGainQ16 chains across the gain-adjust math.
            const int fsKHz = 8, nbSubfr = 4, lpcOrder = 10;
            int frameLength = nbSubfr * 5 * fsKHz;
            var codebook = SilkNlsfCodebookTables.NbMb;

            var indices = new SilkDecodedIndices
            {
                SignalType = SilkSideInfoDecoder.TypeVoiced,
                QuantOffsetType = 0,
                NlsfInterpCoefQ2 = 4,
                LagIndex = 80,
                ContourIndex = 2,
                PerIndex = 1,
                LtpScaleIndex = 0,
                Seed = 1,
            };
            for (int i = 0; i < nbSubfr; i++) indices.GainsIndices[i] = 25;
            indices.NlsfIndices[0] = 7;
            for (int i = 0; i < nbSubfr; i++) indices.LtpIndices[i] = (sbyte)(i + 3);

            short[] pulsesIn = new short[((frameLength + 15) & ~15)];
            byte[] bs1 = EncodeFullSilkFrame(
                codebook, indices, pulsesIn,
                fsKHz: fsKHz, nbSubfr: nbSubfr, conditional: 0, vadFlag: true);
            byte[] bs2 = EncodeFullSilkFrame(
                codebook, indices, pulsesIn,
                fsKHz: fsKHz, nbSubfr: nbSubfr, conditional: 0, vadFlag: true);

            // CPU oracle: state-threaded across both frames.
            var cpuState = new SilkChannelDecoderState();
            cpuState.Configure(fsKHz: fsKHz, nbSubfr: nbSubfr, lpcOrder: lpcOrder);
            cpuState.Reset();
            short[] cpu1 = new short[frameLength];
            SilkDecodeFrame.Decode(cpuState, new OpusRangeDecoder(bs1), cpu1, vadFlag: true, conditional: 0);
            // Snapshot persistent state after frame 1 (for GPU frame 2's initial state).
            // Take exactly `order` entries because the GPU PrevNlsfQ15 buffer is sized
            // to the active codebook's order, not the worst-case MAX_LPC_ORDER.
            var prevNlsfAfter1 = new short[codebook.Order];
            Array.Copy(cpuState.PrevNlsfQ15, prevNlsfAfter1, codebook.Order);
            sbyte lastGainIdxAfter1 = cpuState.LastGainIndex;
            int prevLagIdxAfter1 = cpuState.PrevLagIndex;
            int prevSignalTypeWasVoicedAfter1 = cpuState.PrevSignalTypeWasVoiced ? 1 : 0;
            int firstFrameAfterResetAfter1 = cpuState.FirstFrameAfterReset ? 1 : 0;
            short[] cpu2 = new short[frameLength];
            SilkDecodeFrame.Decode(cpuState, new OpusRangeDecoder(bs2), cpu2, vadFlag: true, conditional: 0);

            // GPU: frame 1 with reset state.
            var initial = new SilkChannelDecoderState();
            initial.Configure(fsKHz: fsKHz, nbSubfr: nbSubfr, lpcOrder: lpcOrder);
            initial.Reset();
            var (gpu1, sLpc1, prevGain1, outBuf1) = await SilkDecodeFrameGpuTest_RunPhaseAPCAsync(
                acc, bs1, codebook,
                fsKHz: fsKHz, nbSubfr: nbSubfr,
                vadFlag: 1, decodeLbrr: 0, conditional: 0,
                prevLagIndex: 0, prevSignalTypeWasVoiced: 0,
                firstFrameAfterReset: 1,
                prevNlsfQ15In: new short[codebook.Order],
                lastGainIndexIn: 0,
                signalType: indices.SignalType,
                quantOffsetType: indices.QuantOffsetType,
                seed: indices.Seed,
                lpcOrder: lpcOrder,
                nlsfInterpEnabled: 0,
                initialPrevGainQ16: initial.PrevGainQ16,
                initialSLpcQ14Buf: initial.SLpcQ14Buf,
                initialOutBuf: initial.OutBuf,
                withFinalize: true);

            // Compare frame 1 PCM.
            for (int i = 0; i < frameLength; i++)
                if (cpu1[i] != gpu1[i])
                    throw new Exception($"Frame 1 PCM mismatch at sample {i}: cpu={cpu1[i]} gpu={gpu1[i]}");

            // GPU: frame 2 with state from frame 1 + CPU-oracle scalars.
            var (gpu2, _, _, _) = await SilkDecodeFrameGpuTest_RunPhaseAPCAsync(
                acc, bs2, codebook,
                fsKHz: fsKHz, nbSubfr: nbSubfr,
                vadFlag: 1, decodeLbrr: 0, conditional: 0,
                prevLagIndex: prevLagIdxAfter1,
                prevSignalTypeWasVoiced: prevSignalTypeWasVoicedAfter1,
                firstFrameAfterReset: firstFrameAfterResetAfter1,
                prevNlsfQ15In: prevNlsfAfter1,
                lastGainIndexIn: lastGainIdxAfter1,
                signalType: indices.SignalType,
                quantOffsetType: indices.QuantOffsetType,
                seed: indices.Seed,
                lpcOrder: lpcOrder,
                nlsfInterpEnabled: 0,
                initialPrevGainQ16: prevGain1,
                initialSLpcQ14Buf: sLpc1,
                initialOutBuf: outBuf1,
                withFinalize: true);

            // Compare frame 2 PCM - this is the hard one; if frame 1's persistent state
            // didn't transfer correctly (OutBuf history, SLpcQ14Buf carry, PrevGainQ16),
            // frame 2's LTP synthesis + LPC synthesis would diverge.
            for (int i = 0; i < frameLength; i++)
                if (cpu2[i] != gpu2[i])
                    throw new Exception($"Frame 2 PCM mismatch at sample {i}: cpu={cpu2[i]} gpu={gpu2[i]}");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod(Timeout = 600000)]
    public async Task SilkDecodeFrameGpu_PhaseAPC_Unvoiced_NB_BitExactVsCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {

            // Unvoiced signal type (TYPE_UNVOICED = 1). Exercises the VAD iCDF
            // path in the indices decoder + the unvoiced branch in decode_core
            // (residual = excitation, no LTP), all with vadFlag=true.
            const int fsKHz = 8, nbSubfr = 4, lpcOrder = 10;
            int frameLength = nbSubfr * 5 * fsKHz;
            var codebook = SilkNlsfCodebookTables.NbMb;

            var indices = new SilkDecodedIndices
            {
                SignalType = SilkSideInfoDecoder.TypeUnvoiced,
                QuantOffsetType = 1,
                NlsfInterpCoefQ2 = 4,
                Seed = 2,
            };
            for (int i = 0; i < nbSubfr; i++) indices.GainsIndices[i] = (sbyte)(20 + i);
            indices.NlsfIndices[0] = 10;

            short[] pulsesIn = new short[((frameLength + 15) & ~15)];
            byte[] bitstream = EncodeFullSilkFrame(
                codebook, indices, pulsesIn,
                fsKHz: fsKHz, nbSubfr: nbSubfr, conditional: 0, vadFlag: true);

            // CPU oracle.
            var cpuState = new SilkChannelDecoderState();
            cpuState.Configure(fsKHz: fsKHz, nbSubfr: nbSubfr, lpcOrder: lpcOrder);
            cpuState.Reset();
            var cpuDec = new OpusRangeDecoder(bitstream);
            short[] cpuPcm = new short[frameLength];
            SilkDecodeFrame.Decode(cpuState, cpuDec, cpuPcm, vadFlag: true, conditional: 0);

            // GPU.
            var gpuInitialState = new SilkChannelDecoderState();
            gpuInitialState.Configure(fsKHz: fsKHz, nbSubfr: nbSubfr, lpcOrder: lpcOrder);
            gpuInitialState.Reset();
            var (gpuPcm, _, _, _) = await SilkDecodeFrameGpuTest_RunPhaseAPCAsync(
                acc, bitstream, codebook,
                fsKHz: fsKHz, nbSubfr: nbSubfr,
                vadFlag: 1, decodeLbrr: 0, conditional: 0,
                prevLagIndex: 0, prevSignalTypeWasVoiced: 0,
                firstFrameAfterReset: 1,
                prevNlsfQ15In: new short[codebook.Order],
                lastGainIndexIn: 0,
                signalType: indices.SignalType,
                quantOffsetType: indices.QuantOffsetType,
                seed: indices.Seed,
                lpcOrder: lpcOrder,
                nlsfInterpEnabled: 0,
                initialPrevGainQ16: gpuInitialState.PrevGainQ16,
                initialSLpcQ14Buf: gpuInitialState.SLpcQ14Buf,
                initialOutBuf: gpuInitialState.OutBuf);

            for (int i = 0; i < frameLength; i++)
                if (cpuPcm[i] != gpuPcm[i])
                    throw new Exception($"PCM mismatch at sample {i}: cpu={cpuPcm[i]} gpu={gpuPcm[i]}");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod(Timeout = 600000)]
    public async Task SilkDecodeFrameGpu_PhaseAPC_Inactive_NB_BitExactVsCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {

            // Inactive signal type (TYPE_NO_VOICE_ACTIVITY = 0). Exercises the
            // non-VAD iCDF in the indices decoder + the unvoiced / inactive
            // path in decode_core (no LTP rewhitening, residual = excitation).
            const int fsKHz = 8, nbSubfr = 4, lpcOrder = 10;
            int frameLength = nbSubfr * 5 * fsKHz;
            var codebook = SilkNlsfCodebookTables.NbMb;

            var indices = new SilkDecodedIndices
            {
                SignalType = SilkSideInfoDecoder.TypeInactive,
                QuantOffsetType = 0,
                NlsfInterpCoefQ2 = 4,
                Seed = 0,
            };
            for (int i = 0; i < nbSubfr; i++) indices.GainsIndices[i] = 15;
            indices.NlsfIndices[0] = 5;

            short[] pulsesIn = new short[((frameLength + 15) & ~15)];
            byte[] bitstream = EncodeFullSilkFrame(
                codebook, indices, pulsesIn,
                fsKHz: fsKHz, nbSubfr: nbSubfr, conditional: 0, vadFlag: false);

            // CPU oracle.
            var cpuState = new SilkChannelDecoderState();
            cpuState.Configure(fsKHz: fsKHz, nbSubfr: nbSubfr, lpcOrder: lpcOrder);
            cpuState.Reset();
            var cpuDec = new OpusRangeDecoder(bitstream);
            short[] cpuPcm = new short[frameLength];
            SilkDecodeFrame.Decode(cpuState, cpuDec, cpuPcm, vadFlag: false, conditional: 0);

            // GPU.
            var gpuInitialState = new SilkChannelDecoderState();
            gpuInitialState.Configure(fsKHz: fsKHz, nbSubfr: nbSubfr, lpcOrder: lpcOrder);
            gpuInitialState.Reset();
            var (gpuPcm, _, _, _) = await SilkDecodeFrameGpuTest_RunPhaseAPCAsync(
                acc, bitstream, codebook,
                fsKHz: fsKHz, nbSubfr: nbSubfr,
                vadFlag: 0, decodeLbrr: 0, conditional: 0,
                prevLagIndex: 0, prevSignalTypeWasVoiced: 0,
                firstFrameAfterReset: 1,
                prevNlsfQ15In: new short[codebook.Order],
                lastGainIndexIn: 0,
                signalType: indices.SignalType,
                quantOffsetType: indices.QuantOffsetType,
                seed: indices.Seed,
                lpcOrder: lpcOrder,
                nlsfInterpEnabled: 0,
                initialPrevGainQ16: gpuInitialState.PrevGainQ16,
                initialSLpcQ14Buf: gpuInitialState.SLpcQ14Buf,
                initialOutBuf: gpuInitialState.OutBuf);

            for (int i = 0; i < frameLength; i++)
                if (cpuPcm[i] != gpuPcm[i])
                    throw new Exception($"PCM mismatch at sample {i}: cpu={cpuPcm[i]} gpu={gpuPcm[i]}");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod(Timeout = 600000)]
    public async Task SilkDecodeFrameGpu_PhaseAPCF_Voiced_NB_OutBufShiftBitExactVsCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {

            const int fsKHz = 8, nbSubfr = 4, lpcOrder = 10;
            int frameLength = nbSubfr * 5 * fsKHz;
            int ltpMemLength = 20 * fsKHz;
            var codebook = SilkNlsfCodebookTables.NbMb;

            var indices = new SilkDecodedIndices
            {
                SignalType = SilkSideInfoDecoder.TypeVoiced,
                QuantOffsetType = 0,
                NlsfInterpCoefQ2 = 4,
                LagIndex = 80,
                ContourIndex = 2,
                PerIndex = 1,
                LtpScaleIndex = 0,
                Seed = 1,
            };
            for (int i = 0; i < nbSubfr; i++) indices.GainsIndices[i] = 25;
            indices.NlsfIndices[0] = 7;
            for (int i = 0; i < nbSubfr; i++) indices.LtpIndices[i] = (sbyte)(i + 3);

            short[] pulsesIn = new short[((frameLength + 15) & ~15)];
            byte[] bitstream = EncodeFullSilkFrame(
                codebook, indices, pulsesIn,
                fsKHz: fsKHz, nbSubfr: nbSubfr, conditional: 0, vadFlag: true);

            // CPU oracle.
            var cpuState = new SilkChannelDecoderState();
            cpuState.Configure(fsKHz: fsKHz, nbSubfr: nbSubfr, lpcOrder: lpcOrder);
            cpuState.Reset();
            var cpuDec = new OpusRangeDecoder(bitstream);
            short[] cpuPcm = new short[frameLength];
            SilkDecodeFrame.Decode(cpuState, cpuDec, cpuPcm, vadFlag: true, conditional: 0);
            // After SilkDecodeFrame: cpuState.OutBuf is the post-shift state.

            // GPU with finalize.
            var gpuInitialState = new SilkChannelDecoderState();
            gpuInitialState.Configure(fsKHz: fsKHz, nbSubfr: nbSubfr, lpcOrder: lpcOrder);
            gpuInitialState.Reset();
            var (gpuPcm, _, _, gpuOutBuf) = await SilkDecodeFrameGpuTest_RunPhaseAPCAsync(
                acc, bitstream, codebook,
                fsKHz: fsKHz, nbSubfr: nbSubfr,
                vadFlag: 1, decodeLbrr: 0, conditional: 0,
                prevLagIndex: 0, prevSignalTypeWasVoiced: 0,
                firstFrameAfterReset: 1,
                prevNlsfQ15In: new short[codebook.Order],
                lastGainIndexIn: 0,
                signalType: indices.SignalType,
                quantOffsetType: indices.QuantOffsetType,
                seed: indices.Seed,
                lpcOrder: lpcOrder,
                nlsfInterpEnabled: indices.NlsfInterpCoefQ2 < 4 ? 1 : 0,
                initialPrevGainQ16: gpuInitialState.PrevGainQ16,
                initialSLpcQ14Buf: gpuInitialState.SLpcQ14Buf,
                initialOutBuf: gpuInitialState.OutBuf,
                withFinalize: true);

            for (int i = 0; i < frameLength; i++)
                if (cpuPcm[i] != gpuPcm[i])
                    throw new Exception($"PCM mismatch at sample {i}: cpu={cpuPcm[i]} gpu={gpuPcm[i]}");

            // Verify OutBuf rotation: positions [0..ltpMemLength) on CPU should
            // bit-exactly match GPU. (Both started from zeros; CPU did its
            // internal shift; GPU did finalize; they must converge.)
            for (int i = 0; i < ltpMemLength; i++)
                if (cpuState.OutBuf[i] != gpuOutBuf[i])
                    throw new Exception($"OutBuf[{i}] mismatch: cpu={cpuState.OutBuf[i]} gpu={gpuOutBuf[i]}");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod(Timeout = 600000)]
    public async Task SilkDecodeFrameGpu_PhaseAPC_Voiced_NB_BitExactVsCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {

            const int fsKHz = 8, nbSubfr = 4, lpcOrder = 10;
            int frameLength = nbSubfr * 5 * fsKHz;
            var codebook = SilkNlsfCodebookTables.NbMb;

            var indices = new SilkDecodedIndices
            {
                SignalType = SilkSideInfoDecoder.TypeVoiced,
                QuantOffsetType = 0,
                NlsfInterpCoefQ2 = 4, // 4 = no interp (first frame)
                LagIndex = 80,
                ContourIndex = 2,
                PerIndex = 1,
                LtpScaleIndex = 0,
                Seed = 1,
            };
            for (int i = 0; i < nbSubfr; i++) indices.GainsIndices[i] = 25;
            indices.NlsfIndices[0] = 7;
            for (int i = 0; i < nbSubfr; i++) indices.LtpIndices[i] = (sbyte)(i + 3);

            short[] pulsesIn = new short[((frameLength + 15) & ~15)];
            byte[] bitstream = EncodeFullSilkFrame(
                codebook, indices, pulsesIn,
                fsKHz: fsKHz, nbSubfr: nbSubfr, conditional: 0, vadFlag: true);

            // CPU oracle: drive SilkDecodeFrame.Decode on the bitstream.
            var cpuState = new SilkChannelDecoderState();
            cpuState.Configure(fsKHz: fsKHz, nbSubfr: nbSubfr, lpcOrder: lpcOrder);
            cpuState.Reset();
            var cpuDec = new OpusRangeDecoder(bitstream);
            short[] cpuPcm = new short[frameLength];
            SilkDecodeFrame.Decode(cpuState, cpuDec, cpuPcm, vadFlag: true, conditional: 0);
            // SilkDecodeFrame internally does the OutBuf shift + scalar updates;
            // for this test we compare just PCM (xq) + persistent state buffers
            // BEFORE SilkDecodeFrame's shift step. Replicate the un-shift below
            // so cpuState.SLpcQ14Buf etc. match the orchestrator's view of
            // post-decode-core state. The PCM itself is invariant.

            // GPU.
            var firstFrameState = new SilkChannelDecoderState();
            firstFrameState.Configure(fsKHz: fsKHz, nbSubfr: nbSubfr, lpcOrder: lpcOrder);
            firstFrameState.Reset();
            var (gpuPcm, _, _, _) = await SilkDecodeFrameGpuTest_RunPhaseAPCAsync(
                acc, bitstream, codebook,
                fsKHz: fsKHz, nbSubfr: nbSubfr,
                vadFlag: 1, decodeLbrr: 0, conditional: 0,
                prevLagIndex: 0, prevSignalTypeWasVoiced: 0,
                firstFrameAfterReset: 1,
                prevNlsfQ15In: new short[codebook.Order],
                lastGainIndexIn: 0,
                signalType: indices.SignalType,
                quantOffsetType: indices.QuantOffsetType,
                seed: indices.Seed,
                lpcOrder: lpcOrder,
                nlsfInterpEnabled: indices.NlsfInterpCoefQ2 < 4 ? 1 : 0,
                initialPrevGainQ16: firstFrameState.PrevGainQ16,
                initialSLpcQ14Buf: firstFrameState.SLpcQ14Buf,
                initialOutBuf: firstFrameState.OutBuf);

            for (int i = 0; i < frameLength; i++)
                if (cpuPcm[i] != gpuPcm[i])
                    throw new Exception($"PCM mismatch at sample {i}: cpu={cpuPcm[i]} gpu={gpuPcm[i]}");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod(Timeout = 600000)]
    public async Task SilkDecodeFrameGpu_PhaseA_Unvoiced_NB_BitExactVsCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {

            const int fsKHz = 8, nbSubfr = 4;
            int frameLength = nbSubfr * 5 * fsKHz;
            var codebook = SilkNlsfCodebookTables.NbMb;

            // Build a simple unvoiced NB frame.
            var indices = new SilkDecodedIndices
            {
                SignalType = SilkSideInfoDecoder.TypeUnvoiced,
                QuantOffsetType = 1,
                NlsfInterpCoefQ2 = 4,
                Seed = 2,
            };
            for (int i = 0; i < nbSubfr; i++) indices.GainsIndices[i] = (sbyte)(20 + i);
            indices.NlsfIndices[0] = 10;

            short[] cpuPulses = new short[((frameLength + 15) & ~15)];
            byte[] bitstream = EncodeFullSilkFrame(
                codebook, indices, cpuPulses,
                fsKHz: fsKHz, nbSubfr: nbSubfr, conditional: 0, vadFlag: true);

            // CPU oracle: drive SilkIndicesDecoder + SilkPulsesDecoder on the same bitstream.
            var dec = new OpusRangeDecoder(bitstream);
            var cpuIndices = new SilkDecodedIndices();
            SilkIndicesDecoder.Decode(
                cpuIndices, dec, codebook,
                vadFlag: true, decodeLbrr: false,
                fsKHz: fsKHz, nbSubfr: nbSubfr, conditional: 0,
                prevLagIndex: 0, prevSignalTypeWasVoiced: false);
            short[] cpuPulsesDecoded = new short[((frameLength + 15) & ~15)];
            SilkPulsesDecoder.Decode(
                cpuPulsesDecoded.AsSpan(), dec,
                signalType: cpuIndices.SignalType,
                quantOffsetType: cpuIndices.QuantOffsetType,
                frameLength: frameLength);

            // GPU orchestrator.
            var (gpuIndices, gpuPulses) = await SilkDecodeFrameGpuTest_RunPhaseAAsync(
                acc, bitstream, codebook,
                fsKHz: fsKHz, nbSubfr: nbSubfr,
                vadFlag: 1, decodeLbrr: 0, conditional: 0,
                prevLagIndex: 0, prevSignalTypeWasVoiced: 0,
                firstFrameAfterReset: 1);

            // Compare core indices fields bit-exact.
            if (gpuIndices[SilkDecodedIndicesLayout.SignalTypeOffset] != cpuIndices.SignalType)
                throw new Exception($"SignalType mismatch: cpu={cpuIndices.SignalType} gpu={gpuIndices[SilkDecodedIndicesLayout.SignalTypeOffset]}");
            if (gpuIndices[SilkDecodedIndicesLayout.QuantOffsetTypeOffset] != cpuIndices.QuantOffsetType)
                throw new Exception($"QuantOffsetType mismatch: cpu={cpuIndices.QuantOffsetType} gpu={gpuIndices[SilkDecodedIndicesLayout.QuantOffsetTypeOffset]}");
            if (gpuIndices[SilkDecodedIndicesLayout.SeedOffset] != cpuIndices.Seed)
                throw new Exception($"Seed mismatch: cpu={cpuIndices.Seed} gpu={gpuIndices[SilkDecodedIndicesLayout.SeedOffset]}");

            // Compare pulses bit-exact.
            for (int i = 0; i < frameLength; i++)
                if (cpuPulsesDecoded[i] != gpuPulses[i])
                    throw new Exception($"pulses[{i}] mismatch: cpu={cpuPulsesDecoded[i]} gpu={gpuPulses[i]} - cross-kernel range-decoder state must be wrong");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }
}
