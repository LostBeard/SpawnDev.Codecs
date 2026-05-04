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

        using var dEcIxScratch = acc.Allocate1D<short>(order);
        using var dPredQ8Scratch = acc.Allocate1D<byte>(order);

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

        // Parameters codebook + scratches.
        using var dCb1NlsfQ8 = acc.Allocate1D<byte>(codebook.Cb1NlsfQ8.Length);
        using var dCb1WghtQ9 = acc.Allocate1D<short>(codebook.Cb1WghtQ9.Length);
        using var dDeltaMinQ15 = acc.Allocate1D<short>(codebook.DeltaMinQ15.Length);
        using var dLsfCosTab = acc.Allocate1D<short>(SilkLsfCosTab.Q12.Length);

        var (ltpGainsFlat, ltpGainsOffsets) = SilkParamsTest_FlatLtpGains();
        using var dParLtpGainsFlat = acc.Allocate1D<sbyte>(ltpGainsFlat.Length);
        using var dParLtpGainsOffsets = acc.Allocate1D<int>(3);
        using var dParLtpScaleQ14 = acc.Allocate1D<short>(SilkParamsTest_LtpScalesQ14.Length);
        var (contourCb, contourCbSize) = SilkParamsTest_SelectContourCb(fsKHz, nbSubfr);
        using var dParContourCb = acc.Allocate1D<sbyte>(contourCb.Length);

        using var dPrevNlsfQ15 = acc.Allocate1D<short>(order);
        using var dLastGainIndex = acc.Allocate1D<int>(1);
        using var dNlsfDecodeScratch = acc.Allocate1D<short>(3 * 16);
        using var dNlsfDecodePredScratch = acc.Allocate1D<byte>(16);
        using var dNlsf2aScratch = acc.Allocate1D<int>(66);
        using var dNlsfIndicesScratch = acc.Allocate1D<sbyte>(order + 1);
        using var dGainIndicesScratch = acc.Allocate1D<sbyte>(nbSubfr);

        // Outputs + state buffer.
        using var dStateBuf = acc.Allocate1D<OpusRangeDecoderGpuState>(1);
        using var dIndicesOut = acc.Allocate1D<int>(SilkDecodedIndicesLayout.TotalSlots);
        using var dPulsesOut = acc.Allocate1D<short>(alignedPulsesLen);
        using var dParamsIntOut = acc.Allocate1D<int>(SilkDecodedParametersLayout.IntTotalSlots);
        using var dParamsShortOut = acc.Allocate1D<short>(SilkDecodedParametersLayout.ShortTotalSlots);

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

        dCb1NlsfQ8.View.CopyFromCPU(codebook.Cb1NlsfQ8);
        dCb1WghtQ9.View.CopyFromCPU(codebook.Cb1WghtQ9);
        dDeltaMinQ15.View.CopyFromCPU(codebook.DeltaMinQ15);
        dLsfCosTab.View.CopyFromCPU(SilkLsfCosTab.Q12);
        dParLtpGainsFlat.View.CopyFromCPU(ltpGainsFlat);
        dParLtpGainsOffsets.View.CopyFromCPU(ltpGainsOffsets);
        dParLtpScaleQ14.View.CopyFromCPU(SilkParamsTest_LtpScalesQ14);
        dParContourCb.View.CopyFromCPU(contourCb);
        dPrevNlsfQ15.View.CopyFromCPU(prevNlsfQ15In);
        dLastGainIndex.View.CopyFromCPU(new int[] { lastGainIndexIn });

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

        var parametersInputs = new SilkParametersInputs
        {
            Cb1NlsfQ8 = dCb1NlsfQ8.View,
            Cb1WghtQ9 = dCb1WghtQ9.View,
            EcSel = dEcSel.View,
            PredQ8Source = dPredQ8.View,
            DeltaMinQ15 = dDeltaMinQ15.View,
            LsfCosTabQ12 = dLsfCosTab.View,
            ContourCb = dParContourCb.View,
            LtpGainTablesFlat = dParLtpGainsFlat.View,
            LtpGainOffsets = dParLtpGainsOffsets.View,
            LtpScaleQ14Table = dParLtpScaleQ14.View,
        };

        var parametersState = new SilkParametersState
        {
            PrevNlsfQ15 = dPrevNlsfQ15.View,
            LastGainIndex = dLastGainIndex.View,
            NlsfDecodeScratch = dNlsfDecodeScratch.View,
            NlsfDecodePredScratch = dNlsfDecodePredScratch.View,
            Nlsf2aScratch = dNlsf2aScratch.View,
            NlsfIndicesScratch = dNlsfIndicesScratch.View,
            GainIndicesScratch = dGainIndicesScratch.View,
        };

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

    [TestMethod]
    public async Task SilkDecodeFrameGpu_PhaseAP_Voiced_NB_BitExactVsCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            if (acc.AcceleratorType == AcceleratorType.Cuda
                || acc.AcceleratorType == AcceleratorType.WebGPU)
                throw new UnsupportedTestException(
                    "SilkDecodeFrameGpu Phase A+P inherits SilkDecodeCoreGpu's CUDA + WebGPU gates. "
                    + "See _DevComms/SpawnDev.ILGPU/tuvok-to-geordi-silkdecodecoregpu-cuda-webgpu-2026-05-04.md.");
            if (acc.AcceleratorType == AcceleratorType.Wasm)
                throw new UnsupportedTestException(
                    "SilkDecodeFrameGpu Phase A+P on Wasm exceeds PMT's 30s per-test cold-start timeout.");

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
        finally { acc.Dispose(); ctx.Dispose(); }
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
        using var dEcIxScratch = acc.Allocate1D<short>(order);
        using var dPredQ8Scratch = acc.Allocate1D<byte>(order);

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

        using var dCb1NlsfQ8 = acc.Allocate1D<byte>(codebook.Cb1NlsfQ8.Length);
        using var dCb1WghtQ9 = acc.Allocate1D<short>(codebook.Cb1WghtQ9.Length);
        using var dDeltaMinQ15 = acc.Allocate1D<short>(codebook.DeltaMinQ15.Length);
        using var dLsfCosTab = acc.Allocate1D<short>(SilkLsfCosTab.Q12.Length);
        var (ltpGainsFlat, ltpGainsOffsets) = SilkParamsTest_FlatLtpGains();
        using var dParLtpGainsFlat = acc.Allocate1D<sbyte>(ltpGainsFlat.Length);
        using var dParLtpGainsOffsets = acc.Allocate1D<int>(3);
        using var dParLtpScaleQ14 = acc.Allocate1D<short>(SilkParamsTest_LtpScalesQ14.Length);
        var (contourCb, contourCbSize) = SilkParamsTest_SelectContourCb(fsKHz, nbSubfr);
        using var dParContourCb = acc.Allocate1D<sbyte>(contourCb.Length);
        using var dPrevNlsfQ15 = acc.Allocate1D<short>(order);
        using var dLastGainIndex = acc.Allocate1D<int>(1);
        using var dNlsfDecodeScratch = acc.Allocate1D<short>(3 * 16);
        using var dNlsfDecodePredScratch = acc.Allocate1D<byte>(16);
        using var dNlsf2aScratch = acc.Allocate1D<int>(66);
        using var dNlsfIndicesScratch = acc.Allocate1D<sbyte>(order + 1);
        using var dGainIndicesScratch = acc.Allocate1D<sbyte>(nbSubfr);

        // DecodeCore state + scratches + output.
        using var dStateBuf = acc.Allocate1D<OpusRangeDecoderGpuState>(1);
        using var dIndicesOut = acc.Allocate1D<int>(SilkDecodedIndicesLayout.TotalSlots);
        using var dPulsesOut = acc.Allocate1D<short>(alignedPulsesLen);
        using var dParamsIntOut = acc.Allocate1D<int>(SilkDecodedParametersLayout.IntTotalSlots);
        using var dParamsShortOut = acc.Allocate1D<short>(SilkDecodedParametersLayout.ShortTotalSlots);

        using var dOutBuf = acc.Allocate1D<short>(MaxLtpMemLength + MaxFrameLength);
        using var dSLpcQ14Buf = acc.Allocate1D<int>(MaxLpcOrder);
        using var dExcQ14 = acc.Allocate1D<int>(MaxFrameLength);
        using var dPrevGain = acc.Allocate1D<int>(1);
        using var dSLpcScratch = acc.Allocate1D<int>(MaxLpcOrder + MaxSubfrLength);
        using var dSLtpQ15Scratch = acc.Allocate1D<int>(MaxLtpMemLength + MaxFrameLength);
        using var dSLtpScratch = acc.Allocate1D<short>(MaxLtpMemLength);
        using var dPresQ14Scratch = acc.Allocate1D<int>(MaxSubfrLength);
        using var dGainAdjScratch = acc.Allocate1D<int>(1);
        using var dXqOut = acc.Allocate1D<short>(MaxFrameLength);

        // Upload static data + initial state.
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
        dCb1NlsfQ8.View.CopyFromCPU(codebook.Cb1NlsfQ8);
        dCb1WghtQ9.View.CopyFromCPU(codebook.Cb1WghtQ9);
        dDeltaMinQ15.View.CopyFromCPU(codebook.DeltaMinQ15);
        dLsfCosTab.View.CopyFromCPU(SilkLsfCosTab.Q12);
        dParLtpGainsFlat.View.CopyFromCPU(ltpGainsFlat);
        dParLtpGainsOffsets.View.CopyFromCPU(ltpGainsOffsets);
        dParLtpScaleQ14.View.CopyFromCPU(SilkParamsTest_LtpScalesQ14);
        dParContourCb.View.CopyFromCPU(contourCb);
        dPrevNlsfQ15.View.CopyFromCPU(prevNlsfQ15In);
        dLastGainIndex.View.CopyFromCPU(new int[] { lastGainIndexIn });
        dOutBuf.View.CopyFromCPU(initialOutBuf);
        dSLpcQ14Buf.View.CopyFromCPU(initialSLpcQ14Buf);
        dPrevGain.View.CopyFromCPU(new int[] { initialPrevGainQ16 });

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
        var parametersInputs = new SilkParametersInputs
        {
            Cb1NlsfQ8 = dCb1NlsfQ8.View,
            Cb1WghtQ9 = dCb1WghtQ9.View,
            EcSel = dEcSel.View,
            PredQ8Source = dPredQ8.View,
            DeltaMinQ15 = dDeltaMinQ15.View,
            LsfCosTabQ12 = dLsfCosTab.View,
            ContourCb = dParContourCb.View,
            LtpGainTablesFlat = dParLtpGainsFlat.View,
            LtpGainOffsets = dParLtpGainsOffsets.View,
            LtpScaleQ14Table = dParLtpScaleQ14.View,
        };
        var parametersState = new SilkParametersState
        {
            PrevNlsfQ15 = dPrevNlsfQ15.View,
            LastGainIndex = dLastGainIndex.View,
            NlsfDecodeScratch = dNlsfDecodeScratch.View,
            NlsfDecodePredScratch = dNlsfDecodePredScratch.View,
            Nlsf2aScratch = dNlsf2aScratch.View,
            NlsfIndicesScratch = dNlsfIndicesScratch.View,
            GainIndicesScratch = dGainIndicesScratch.View,
        };
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
        var decodeCoreInputs = new SilkDecodeCoreInputs
        {
            PredCoefQ12 = dParamsShortOut.View.SubView(SilkDecodedParametersLayout.ShortPredCoefQ12Half1Offset, 32),
            GainsQ16 = dParamsIntOut.View.SubView(SilkDecodedParametersLayout.IntGainsQ16Offset, 4),
            PitchL = dParamsIntOut.View.SubView(SilkDecodedParametersLayout.IntPitchLOffset, 4),
            LtpCoefQ14 = dParamsShortOut.View.SubView(SilkDecodedParametersLayout.ShortLtpCoefQ14Offset, 20),
            Pulses = dPulsesOut.View,
            OutBufInOut = dOutBuf.View,
            SLpcQ14BufInOut = dSLpcQ14Buf.View,
            ExcQ14Out = dExcQ14.View,
            PrevGainQ16InOut = dPrevGain.View,
            SLpcScratch = dSLpcScratch.View,
            SLtpQ15Scratch = dSLtpQ15Scratch.View,
            SLtpScratch = dSLtpScratch.View,
            PresQ14Scratch = dPresQ14Scratch.View,
            GainAdjScratch = dGainAdjScratch.View,
            XqOut = dXqOut.View,
        };

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

        var pcmFull = await dXqOut.CopyToHostAsync();
        var sLpcOut = await dSLpcQ14Buf.CopyToHostAsync();
        var prevGainOut = await dPrevGain.CopyToHostAsync();
        var outBufFull = await dOutBuf.CopyToHostAsync();
        var pcm = new short[frameLength];
        Array.Copy(pcmFull, pcm, frameLength);
        return (pcm, sLpcOut, prevGainOut[0], outBufFull);
    }

    [TestMethod]
    public async Task SilkDecodeFrameGpu_PhaseAPC_Voiced_MB_BitExactVsCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            if (acc.AcceleratorType == AcceleratorType.Cuda
                || acc.AcceleratorType == AcceleratorType.WebGPU)
                throw new UnsupportedTestException(
                    "SilkDecodeFrameGpu Phase A+P+C inherits SilkDecodeCoreGpu's CUDA + WebGPU gates.");
            if (acc.AcceleratorType == AcceleratorType.Wasm)
                throw new UnsupportedTestException(
                    "SilkDecodeFrameGpu Phase A+P+C on Wasm exceeds PMT's 30s per-test cold-start timeout.");

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
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkDecodeFrameGpu_PhaseAPC_Voiced_WB_BitExactVsCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            if (acc.AcceleratorType == AcceleratorType.Cuda
                || acc.AcceleratorType == AcceleratorType.WebGPU)
                throw new UnsupportedTestException(
                    "SilkDecodeFrameGpu Phase A+P+C inherits SilkDecodeCoreGpu's CUDA + WebGPU gates.");
            if (acc.AcceleratorType == AcceleratorType.Wasm)
                throw new UnsupportedTestException(
                    "SilkDecodeFrameGpu Phase A+P+C on Wasm exceeds PMT's 30s per-test cold-start timeout.");

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
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkDecodeFrameGpu_PhaseAPC_Inactive_NB_BitExactVsCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            if (acc.AcceleratorType == AcceleratorType.Cuda
                || acc.AcceleratorType == AcceleratorType.WebGPU)
                throw new UnsupportedTestException(
                    "SilkDecodeFrameGpu Phase A+P+C inherits SilkDecodeCoreGpu's CUDA + WebGPU gates.");
            if (acc.AcceleratorType == AcceleratorType.Wasm)
                throw new UnsupportedTestException(
                    "SilkDecodeFrameGpu Phase A+P+C on Wasm exceeds PMT's 30s per-test cold-start timeout.");

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
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkDecodeFrameGpu_PhaseAPCF_Voiced_NB_OutBufShiftBitExactVsCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            if (acc.AcceleratorType == AcceleratorType.Cuda
                || acc.AcceleratorType == AcceleratorType.WebGPU)
                throw new UnsupportedTestException(
                    "SilkDecodeFrameGpu Phase A+P+C+F inherits SilkDecodeCoreGpu's CUDA + WebGPU gates. "
                    + "See _DevComms/SpawnDev.ILGPU/tuvok-to-geordi-silkdecodecoregpu-cuda-webgpu-2026-05-04.md.");
            if (acc.AcceleratorType == AcceleratorType.Wasm)
                throw new UnsupportedTestException(
                    "SilkDecodeFrameGpu Phase A+P+C+F on Wasm exceeds PMT's 30s per-test cold-start timeout.");

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
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkDecodeFrameGpu_PhaseAPC_Voiced_NB_BitExactVsCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            if (acc.AcceleratorType == AcceleratorType.Cuda
                || acc.AcceleratorType == AcceleratorType.WebGPU)
                throw new UnsupportedTestException(
                    "SilkDecodeFrameGpu Phase A+P+C inherits SilkDecodeCoreGpu's CUDA + WebGPU gates. "
                    + "See _DevComms/SpawnDev.ILGPU/tuvok-to-geordi-silkdecodecoregpu-cuda-webgpu-2026-05-04.md.");
            if (acc.AcceleratorType == AcceleratorType.Wasm)
                throw new UnsupportedTestException(
                    "SilkDecodeFrameGpu Phase A+P+C on Wasm exceeds PMT's 30s per-test cold-start timeout.");

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
        finally { acc.Dispose(); ctx.Dispose(); }
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
