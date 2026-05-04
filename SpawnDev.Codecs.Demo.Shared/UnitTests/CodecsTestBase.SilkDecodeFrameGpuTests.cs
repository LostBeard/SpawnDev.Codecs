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
        using var dTypeOffsetVad = acc.Allocate1D<byte>(SilkIcdfTables.TypeOffsetVad.Length);
        using var dTypeOffsetNoVad = acc.Allocate1D<byte>(SilkIcdfTables.TypeOffsetNoVad.Length);
        using var dUniform4 = acc.Allocate1D<byte>(SilkIcdfTables.Uniform4.Length);
        using var dGain = acc.Allocate1D<byte>(SilkIcdfTables.Gain.Length);
        using var dDeltaGain = acc.Allocate1D<byte>(SilkIcdfTables.DeltaGain.Length);
        using var dUniform8 = acc.Allocate1D<byte>(SilkIcdfTables.Uniform8.Length);
        using var dCb1 = acc.Allocate1D<byte>(codebook.Cb1Icdf.Length);
        using var dEc = acc.Allocate1D<byte>(codebook.EcIcdf.Length);
        using var dEcSel = acc.Allocate1D<byte>(codebook.EcSel.Length);
        using var dPredQ8 = acc.Allocate1D<byte>(codebook.PredQ8.Length);
        using var dNlsfExt = acc.Allocate1D<byte>(SilkIcdfTables.NlsfExt.Length);
        using var dNlsfInterp = acc.Allocate1D<byte>(SilkIcdfTables.NlsfInterpolationFactor.Length);
        using var dPitchDelta = acc.Allocate1D<byte>(SilkIcdfTables.PitchDelta.Length);
        using var dPitchLag = acc.Allocate1D<byte>(SilkIcdfTables.PitchLag.Length);
        var lagLowBits = SilkIcdfTables.SelectPitchLagLowBits(fsKHz);
        using var dLagLowBits = acc.Allocate1D<byte>(lagLowBits.Length);
        var contour = SilkIcdfTables.SelectPitchContour(fsKHz, nbSubfr);
        using var dContour = acc.Allocate1D<byte>(contour.Length);
        using var dLtpPerIndex = acc.Allocate1D<byte>(SilkIcdfTables.LtpPerIndex.Length);

        // LTP gain flat-pack.
        var ltpFlat = new byte[SilkIcdfTables.LtpGain0.Length
                              + SilkIcdfTables.LtpGain1.Length
                              + SilkIcdfTables.LtpGain2.Length];
        Array.Copy(SilkIcdfTables.LtpGain0, 0, ltpFlat, 0, SilkIcdfTables.LtpGain0.Length);
        Array.Copy(SilkIcdfTables.LtpGain1, 0, ltpFlat, SilkIcdfTables.LtpGain0.Length, SilkIcdfTables.LtpGain1.Length);
        Array.Copy(SilkIcdfTables.LtpGain2, 0, ltpFlat,
            SilkIcdfTables.LtpGain0.Length + SilkIcdfTables.LtpGain1.Length,
            SilkIcdfTables.LtpGain2.Length);
        using var dLtpFlat = acc.Allocate1D<byte>(ltpFlat.Length);
        using var dLtpOffsets = acc.Allocate1D<int>(3);
        using var dLtpScale = acc.Allocate1D<byte>(SilkIcdfTables.LtpScale.Length);

        // Indices scratch.
        using var dEcIxScratch = acc.Allocate1D<short>(codebook.Order);
        using var dPredQ8Scratch = acc.Allocate1D<byte>(codebook.Order);

        // Pulses iCDFs + shell tables + scratches.
        using var dRateLevels = acc.Allocate1D<byte>(SilkIcdfTables.RateLevels.Length);
        using var dPulsesPerBlock = acc.Allocate1D<byte>(SilkIcdfTables.PulsesPerBlock.Length);
        using var dLsb = acc.Allocate1D<byte>(SilkIcdfTables.Lsb.Length);
        using var dSign = acc.Allocate1D<byte>(SilkIcdfTables.Sign.Length);
        using var dShellOffsets = acc.Allocate1D<byte>(SilkShellCodeTables.Offsets.Length);
        using var dShellTable0 = acc.Allocate1D<byte>(SilkShellCodeTables.Table0.Length);
        using var dShellTable1 = acc.Allocate1D<byte>(SilkShellCodeTables.Table1.Length);
        using var dShellTable2 = acc.Allocate1D<byte>(SilkShellCodeTables.Table2.Length);
        using var dShellTable3 = acc.Allocate1D<byte>(SilkShellCodeTables.Table3.Length);
        using var dSumPulses = acc.Allocate1D<int>(20);
        using var dNLshifts = acc.Allocate1D<int>(20);

        // Outputs + state buffer.
        using var dStateBuf = acc.Allocate1D<OpusRangeDecoderGpuState>(1);
        using var dIndicesOut = acc.Allocate1D<int>(SilkDecodedIndicesLayout.TotalSlots);
        using var dPulsesOut = acc.Allocate1D<short>(alignedPulsesLen);

        // Upload static data.
        dPacket.View.CopyFromCPU(packet);
        dTypeOffsetVad.View.CopyFromCPU(SilkIcdfTables.TypeOffsetVad);
        dTypeOffsetNoVad.View.CopyFromCPU(SilkIcdfTables.TypeOffsetNoVad);
        dUniform4.View.CopyFromCPU(SilkIcdfTables.Uniform4);
        dGain.View.CopyFromCPU(SilkIcdfTables.Gain);
        dDeltaGain.View.CopyFromCPU(SilkIcdfTables.DeltaGain);
        dUniform8.View.CopyFromCPU(SilkIcdfTables.Uniform8);
        dCb1.View.CopyFromCPU(codebook.Cb1Icdf);
        dEc.View.CopyFromCPU(codebook.EcIcdf);
        dEcSel.View.CopyFromCPU(codebook.EcSel);
        dPredQ8.View.CopyFromCPU(codebook.PredQ8);
        dNlsfExt.View.CopyFromCPU(SilkIcdfTables.NlsfExt);
        dNlsfInterp.View.CopyFromCPU(SilkIcdfTables.NlsfInterpolationFactor);
        dPitchDelta.View.CopyFromCPU(SilkIcdfTables.PitchDelta);
        dPitchLag.View.CopyFromCPU(SilkIcdfTables.PitchLag);
        dLagLowBits.View.CopyFromCPU(lagLowBits);
        dContour.View.CopyFromCPU(contour);
        dLtpPerIndex.View.CopyFromCPU(SilkIcdfTables.LtpPerIndex);
        dLtpFlat.View.CopyFromCPU(ltpFlat);
        dLtpOffsets.View.CopyFromCPU(new int[]
        {
            0,
            SilkIcdfTables.LtpGain0.Length,
            SilkIcdfTables.LtpGain0.Length + SilkIcdfTables.LtpGain1.Length,
        });
        dLtpScale.View.CopyFromCPU(SilkIcdfTables.LtpScale);
        dRateLevels.View.CopyFromCPU(SilkIcdfTables.RateLevels);
        dPulsesPerBlock.View.CopyFromCPU(SilkIcdfTables.PulsesPerBlock);
        dLsb.View.CopyFromCPU(SilkIcdfTables.Lsb);
        dSign.View.CopyFromCPU(SilkIcdfTables.Sign);
        dShellOffsets.View.CopyFromCPU(SilkShellCodeTables.Offsets);
        dShellTable0.View.CopyFromCPU(SilkShellCodeTables.Table0);
        dShellTable1.View.CopyFromCPU(SilkShellCodeTables.Table1);
        dShellTable2.View.CopyFromCPU(SilkShellCodeTables.Table2);
        dShellTable3.View.CopyFromCPU(SilkShellCodeTables.Table3);

        var indicesInputs = new SilkIndicesInputs
        {
            TypeOffsetVadIcdf = dTypeOffsetVad.View,
            TypeOffsetNoVadIcdf = dTypeOffsetNoVad.View,
            Uniform4Icdf = dUniform4.View,
            GainIcdf = dGain.View,
            DeltaGainIcdf = dDeltaGain.View,
            Uniform8Icdf = dUniform8.View,
            Cb1Icdf = dCb1.View,
            EcIcdf = dEc.View,
            EcSel = dEcSel.View,
            PredQ8Source = dPredQ8.View,
            NlsfExtIcdf = dNlsfExt.View,
            NlsfInterpolationFactorIcdf = dNlsfInterp.View,
            PitchDeltaIcdf = dPitchDelta.View,
            PitchLagIcdf = dPitchLag.View,
            LagLowBitsIcdf = dLagLowBits.View,
            ContourIcdf = dContour.View,
            LtpPerIndexIcdf = dLtpPerIndex.View,
            LtpGainIcdfFlat = dLtpFlat.View,
            LtpGainOffsets = dLtpOffsets.View,
            LtpScaleIcdf = dLtpScale.View,
            EcIxScratch = dEcIxScratch.View,
            PredQ8Scratch = dPredQ8Scratch.View,
        };

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

        var pulsesInputs = new SilkPulsesInputs
        {
            RateLevelsIcdf = dRateLevels.View,
            PulsesPerBlockIcdf = dPulsesPerBlock.View,
            LsbIcdf = dLsb.View,
            SignIcdf = dSign.View,
            ShellTables = new SilkShellCoderTables
            {
                Offsets = dShellOffsets.View,
                Table0 = dShellTable0.View,
                Table1 = dShellTable1.View,
                Table2 = dShellTable2.View,
                Table3 = dShellTable3.View,
            },
            SumPulsesScratch = dSumPulses.View,
            NLshiftsScratch = dNLshifts.View,
        };

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

    [TestMethod]
    public async Task SilkDecodeFrameGpu_PhaseA_Unvoiced_NB_BitExactVsCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // CUDA + WebGPU gates inherited from SilkDecodeCoreGpu (same ILGPU
            // backend bugs would apply once Phase B lands).
            // Wasm gated because the orchestrator's 3-kernel constructor
            // compile time on Wasm cold start exceeds PMT's 30s per-test
            // timeout. The orchestrator math is correct (CPU + OpenCL pass
            // bit-exact). When Wasm cold-start kernel-compile speed improves
            // OR PMT's per-test timeout is raised for Codecs, this gate lifts.
            if (acc.AcceleratorType == AcceleratorType.Cuda
                || acc.AcceleratorType == AcceleratorType.WebGPU)
                throw new UnsupportedTestException(
                    "SilkDecodeFrameGpu Phase A gated on backends where SilkDecodeCoreGpu is gated. "
                    + "See _DevComms/SpawnDev.ILGPU/tuvok-to-geordi-silkdecodecoregpu-cuda-webgpu-2026-05-04.md.");
            if (acc.AcceleratorType == AcceleratorType.Wasm)
                throw new UnsupportedTestException(
                    "SilkDecodeFrameGpu Phase A on Wasm exceeds PMT's 30s per-test timeout - 3-kernel "
                    + "orchestrator constructor compile time consumes the budget before the dispatch runs. "
                    + "Math is correct (CPU + OpenCL bit-exact); Wasm cold-start compile pacing is the blocker.");

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
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
