// Cross-backend tests for SilkParametersDecoderGpu - the parameter dequantizer
// orchestrator. Builds an indices buffer directly (per SilkDecodedIndicesLayout),
// runs CPU SilkParametersDecoder.Decode as oracle, runs GPU
// SilkParametersDecoderGpu.Decode, compares all output fields.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Audio.Opus.Silk;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    /// <summary>libopus silk_LTPScales_table_Q14 = { 15565, 12288, 8192 }.</summary>
    private static readonly short[] SilkParamsTest_LtpScalesQ14 = { 15565, 12288, 8192 };

    /// <summary>Flat-pack 3 LTP gain Q7 codebooks into one sbyte[] with offsets.</summary>
    private static (sbyte[] flat, int[] offsets) SilkParamsTest_FlatLtpGains()
    {
        var cb0 = SilkLtpGainTables.Select(0); // 8 entries × 5 taps = 40 sbytes
        var cb1 = SilkLtpGainTables.Select(1); // 16 entries × 5 = 80
        var cb2 = SilkLtpGainTables.Select(2); // 32 entries × 5 = 160
        var flat = new sbyte[cb0.Length + cb1.Length + cb2.Length];
        Array.Copy(cb0, 0, flat, 0, cb0.Length);
        Array.Copy(cb1, 0, flat, cb0.Length, cb1.Length);
        Array.Copy(cb2, 0, flat, cb0.Length + cb1.Length, cb2.Length);
        return (flat, new int[] { 0, cb0.Length, cb0.Length + cb1.Length });
    }

    private static (sbyte[] cb, int cbSize) SilkParamsTest_SelectContourCb(int fsKHz, int nbSubfr)
    {
        if (fsKHz == 8)
        {
            if (nbSubfr == 4) return (SilkPitchContourTables.Stage2, 11);
            return (SilkPitchContourTables.Stage210Ms, 3);
        }
        if (nbSubfr == 4) return (SilkPitchContourTables.Stage3, 34);
        return (SilkPitchContourTables.Stage310Ms, 12);
    }

    /// <summary>Build a SilkDecodedIndicesLayout buffer directly from a fixture.</summary>
    private static int[] SilkParamsTest_BuildIndicesBuffer(SilkSideInfoFixture f, int order, int nbSubfr)
    {
        var buf = new int[SilkDecodedIndicesLayout.TotalSlots];
        buf[SilkDecodedIndicesLayout.SignalTypeOffset] = f.SignalType;
        buf[SilkDecodedIndicesLayout.QuantOffsetTypeOffset] = f.QuantOffsetType;
        buf[SilkDecodedIndicesLayout.NlsfInterpCoefQ2Offset] = f.InterpolationFactor;
        buf[SilkDecodedIndicesLayout.LagIndexOffset] = f.PitchCoarseLag * 8 + f.PitchLsb; // simulated lag
        buf[SilkDecodedIndicesLayout.ContourIndexOffset] = f.PitchContour;
        buf[SilkDecodedIndicesLayout.PerIndexOffset] = f.LtpPerIndex;
        buf[SilkDecodedIndicesLayout.LtpScaleIndexOffset] = f.LtpScaleIndex;
        buf[SilkDecodedIndicesLayout.SeedOffset] = f.Seed;
        for (int i = 0; i < nbSubfr; i++)
            buf[SilkDecodedIndicesLayout.GainsIndicesOffset + i] = f.GainIndices[i];
        for (int i = 0; i <= order; i++)
            buf[SilkDecodedIndicesLayout.NlsfIndicesOffset + i] = f.NlsfIndices[i];
        for (int k = 0; k < nbSubfr; k++)
            buf[SilkDecodedIndicesLayout.LtpIndicesOffset + k] = f.LtpGainIndices[k];
        return buf;
    }

    /// <summary>CPU oracle: build a SilkDecodedIndices, run SilkParametersDecoder.Decode.</summary>
    private static (int[] gains, short[] nlsf, short[] lpc, int[] pitch, short[] ltpCoef, int ltpScale, short[] prevNlsf, sbyte lastGainIdx)
        SilkParamsTest_CpuOracle(
            SilkSideInfoFixture f, SilkNlsfCodebook codebook,
            int fsKHz, int nbSubfr, int conditional,
            short[] prevNlsfQ15In, sbyte lastGainIndexIn,
            int simulatedLagIndex)
    {
        int order = codebook.Order;
        var indices = new SilkDecodedIndices
        {
            SignalType = (sbyte)f.SignalType,
            QuantOffsetType = (sbyte)f.QuantOffsetType,
            NlsfInterpCoefQ2 = (sbyte)f.InterpolationFactor,
            LagIndex = (short)simulatedLagIndex,
            ContourIndex = (sbyte)f.PitchContour,
            PerIndex = (sbyte)f.LtpPerIndex,
            LtpScaleIndex = (sbyte)f.LtpScaleIndex,
            Seed = (sbyte)f.Seed,
        };
        for (int i = 0; i < nbSubfr; i++) indices.GainsIndices[i] = (sbyte)f.GainIndices[i];
        for (int i = 0; i <= order; i++) indices.NlsfIndices[i] = f.NlsfIndices[i];
        for (int k = 0; k < nbSubfr; k++) indices.LtpIndices[k] = (sbyte)f.LtpGainIndices[k];

        var output = new SilkDecodedParameters();
        var prevNlsfMutable = new short[order];
        Array.Copy(prevNlsfQ15In, prevNlsfMutable, order);
        sbyte lastGainIdx = lastGainIndexIn;

        SilkParametersDecoder.Decode(
            output, indices, codebook,
            fsKHz: fsKHz, nbSubfr: nbSubfr,
            lastGainIndex: ref lastGainIdx,
            prevNlsfQ15: prevNlsfMutable.AsSpan(0, order),
            conditional: conditional);

        var gains = new int[nbSubfr];
        Array.Copy(output.GainsQ16, gains, nbSubfr);
        var nlsf = new short[order];
        Array.Copy(output.NlsfQ15, nlsf, order);
        var lpc = new short[2 * order];
        // CPU output.PredCoefQ12 layout: first half at [0..order), second half at [MAX_LPC_ORDER..MAX_LPC_ORDER+order).
        Array.Copy(output.PredCoefQ12, 0, lpc, 0, order);
        Array.Copy(output.PredCoefQ12, 16, lpc, order, order);
        var pitch = new int[nbSubfr];
        Array.Copy(output.PitchL, pitch, nbSubfr);
        var ltpCoef = new short[nbSubfr * 5];
        Array.Copy(output.LtpCoefQ14, ltpCoef, nbSubfr * 5);

        return (gains, nlsf, lpc, pitch, ltpCoef, output.LtpScaleQ14, prevNlsfMutable, lastGainIdx);
    }

    private static async Task<(int[] gains, short[] nlsf, short[] lpc, int[] pitch, short[] ltpCoef, int ltpScale, short[] prevNlsf, int lastGainIdx)>
        SilkParamsTest_GpuRun(
            Accelerator acc,
            int[] indicesBuf, SilkNlsfCodebook codebook,
            int fsKHz, int nbSubfr, int conditional,
            short[] prevNlsfQ15In, int lastGainIndexIn,
            int simulatedLagIndex)
    {
        int order = codebook.Order;
        var (ltpFlat, ltpOffsets) = SilkParamsTest_FlatLtpGains();
        var (contourCb, contourCbSize) = SilkParamsTest_SelectContourCb(fsKHz, nbSubfr);

        // Override the lag index in indicesBuf with the simulated value
        // (matches what we'll pass to the CPU oracle).
        indicesBuf[SilkDecodedIndicesLayout.LagIndexOffset] = simulatedLagIndex;

        using var dIndices = acc.Allocate1D<int>(indicesBuf.Length);
        using var dCb1Nlsf = acc.Allocate1D<byte>(codebook.Cb1NlsfQ8.Length);
        using var dCb1Wght = acc.Allocate1D<short>(codebook.Cb1WghtQ9.Length);
        using var dEcSel = acc.Allocate1D<byte>(codebook.EcSel.Length);
        using var dPredQ8 = acc.Allocate1D<byte>(codebook.PredQ8.Length);
        using var dDeltaMin = acc.Allocate1D<short>(codebook.DeltaMinQ15.Length);
        using var dLsfCos = acc.Allocate1D<short>(SilkLsfCosTab.Q12.Length);
        using var dContour = acc.Allocate1D<sbyte>(contourCb.Length);
        using var dLtpFlat = acc.Allocate1D<sbyte>(ltpFlat.Length);
        using var dLtpOffsets = acc.Allocate1D<int>(ltpOffsets.Length);
        using var dLtpScales = acc.Allocate1D<short>(SilkParamsTest_LtpScalesQ14.Length);
        using var dPrevNlsf = acc.Allocate1D<short>(order);
        using var dLastGain = acc.Allocate1D<int>(1);
        using var dNlsfDecScratch = acc.Allocate1D<short>(3 * 16);
        using var dNlsfDecPredScratch = acc.Allocate1D<byte>(16);
        using var dNlsf2aScratch = acc.Allocate1D<int>(66);
        using var dNlsfIdxScratch = acc.Allocate1D<sbyte>(order + 1);
        using var dGainIdxScratch = acc.Allocate1D<sbyte>(nbSubfr);
        using var dIntOut = acc.Allocate1D<int>(SilkDecodedParametersLayout.IntTotalSlots);
        using var dShortOut = acc.Allocate1D<short>(SilkDecodedParametersLayout.ShortTotalSlots);

        dIndices.View.CopyFromCPU(indicesBuf);
        dCb1Nlsf.View.CopyFromCPU(codebook.Cb1NlsfQ8);
        dCb1Wght.View.CopyFromCPU(codebook.Cb1WghtQ9);
        dEcSel.View.CopyFromCPU(codebook.EcSel);
        dPredQ8.View.CopyFromCPU(codebook.PredQ8);
        dDeltaMin.View.CopyFromCPU(codebook.DeltaMinQ15);
        dLsfCos.View.CopyFromCPU(SilkLsfCosTab.Q12);
        dContour.View.CopyFromCPU(contourCb);
        dLtpFlat.View.CopyFromCPU(ltpFlat);
        dLtpOffsets.View.CopyFromCPU(ltpOffsets);
        dLtpScales.View.CopyFromCPU(SilkParamsTest_LtpScalesQ14);
        dPrevNlsf.View.CopyFromCPU(prevNlsfQ15In);
        dLastGain.View.CopyFromCPU(new int[] { lastGainIndexIn });

        var inputs = new SilkParametersInputs
        {
            Cb1NlsfQ8 = dCb1Nlsf.View,
            Cb1WghtQ9 = dCb1Wght.View,
            EcSel = dEcSel.View,
            PredQ8Source = dPredQ8.View,
            DeltaMinQ15 = dDeltaMin.View,
            LsfCosTabQ12 = dLsfCos.View,
            ContourCb = dContour.View,
            LtpGainTablesFlat = dLtpFlat.View,
            LtpGainOffsets = dLtpOffsets.View,
            LtpScaleQ14Table = dLtpScales.View,
        };

        var state = new SilkParametersState
        {
            PrevNlsfQ15 = dPrevNlsf.View,
            LastGainIndex = dLastGain.View,
            NlsfDecodeScratch = dNlsfDecScratch.View,
            NlsfDecodePredScratch = dNlsfDecPredScratch.View,
            Nlsf2aScratch = dNlsf2aScratch.View,
            NlsfIndicesScratch = dNlsfIdxScratch.View,
            GainIndicesScratch = dGainIdxScratch.View,
        };

        var scalars = new SilkParametersScalars
        {
            QuantStepSizeQ16 = codebook.QuantStepSizeQ16,
            Order = order,
            NbSubfr = nbSubfr,
            FsKHz = fsKHz,
            ContourCbSize = contourCbSize,
            Conditional = conditional,
        };

        using var kernel = new SilkParametersDecoderGpuTestKernel(acc);
        kernel.Run(dIndices.View, inputs, state, scalars, dIntOut.View, dShortOut.View);
        await acc.SynchronizeAsync();

        var intOut = await dIntOut.CopyToHostAsync();
        var shortOut = await dShortOut.CopyToHostAsync();
        var prevNlsfOut = await dPrevNlsf.CopyToHostAsync();
        var lastGainOut = await dLastGain.CopyToHostAsync();

        var gains = new int[nbSubfr];
        Array.Copy(intOut, SilkDecodedParametersLayout.IntGainsQ16Offset, gains, 0, nbSubfr);
        var pitch = new int[nbSubfr];
        Array.Copy(intOut, SilkDecodedParametersLayout.IntPitchLOffset, pitch, 0, nbSubfr);
        var nlsf = new short[order];
        Array.Copy(shortOut, SilkDecodedParametersLayout.ShortNlsfQ15Offset, nlsf, 0, order);
        var lpc = new short[2 * order];
        Array.Copy(shortOut, SilkDecodedParametersLayout.ShortPredCoefQ12Half1Offset, lpc, 0, order);
        Array.Copy(shortOut, SilkDecodedParametersLayout.ShortPredCoefQ12Half2Offset, lpc, order, order);
        var ltpCoef = new short[nbSubfr * 5];
        Array.Copy(shortOut, SilkDecodedParametersLayout.ShortLtpCoefQ14Offset, ltpCoef, 0, nbSubfr * 5);
        int ltpScale = shortOut[SilkDecodedParametersLayout.ShortLtpScaleQ14Offset];
        var prevNlsf = new short[order];
        Array.Copy(prevNlsfOut, prevNlsf, order);

        return (gains, nlsf, lpc, pitch, ltpCoef, ltpScale, prevNlsf, lastGainOut[0]);
    }

    private static void SilkParamsTest_AssertEqual<T>(T[] cpu, T[] gpu, string name) where T : IEquatable<T>
    {
        if (cpu.Length != gpu.Length)
            throw new Exception($"{name} length: cpu={cpu.Length} gpu={gpu.Length}");
        for (int i = 0; i < cpu.Length; i++)
            if (!cpu[i].Equals(gpu[i]))
                throw new Exception($"{name}[{i}] mismatch: cpu={cpu[i]} gpu={gpu[i]}");
    }

    [TestMethod]
    public async Task SilkParametersDecoderGpu_VoicedNbWb20msIndependent_BitExactVsCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            var codebook = SilkNlsfCodebookTables.Wb;
            int order = codebook.Order;
            const int fsKHz = 16, nbSubfr = 4, conditional = 0;

            var fixture = new SilkSideInfoFixture
            {
                SignalType = 2,
                QuantOffsetType = 1,
                GainIndices = new int[] { 24, 8, 11, 6 },
                NlsfIndices = new sbyte[order + 1],
                InterpolationFactor = 2,
                PitchCoarseLag = 14, PitchLsb = 5, PitchContour = 11,
                LtpPerIndex = 1,
                LtpGainIndices = new int[] { 7, 12, 3, 9 },
                LtpScaleIndex = 1,
                Seed = 2,
            };
            fixture.NlsfIndices[0] = 9;
            for (int i = 0; i < order; i++)
                fixture.NlsfIndices[i + 1] = (sbyte)((i % 7) - 3);

            // Provide a non-zero prev NLSF so interpolation actually fires.
            var prevNlsfIn = new short[order];
            for (int i = 0; i < order; i++)
                prevNlsfIn[i] = (short)(2000 + i * 1000);
            sbyte lastGainIdxIn = 30;

            // Use an explicit lagIndex (simulated, since we're not chaining
            // through SilkIndicesDecoderGpu).
            int simulatedLag = 80;

            var indicesBuf = SilkParamsTest_BuildIndicesBuffer(fixture, order, nbSubfr);

            var cpu = SilkParamsTest_CpuOracle(
                fixture, codebook, fsKHz, nbSubfr, conditional,
                prevNlsfIn, lastGainIdxIn, simulatedLag);

            var gpu = await SilkParamsTest_GpuRun(
                acc, indicesBuf, codebook, fsKHz, nbSubfr, conditional,
                prevNlsfIn, lastGainIdxIn, simulatedLag);

            SilkParamsTest_AssertEqual(cpu.gains, gpu.gains, "GainsQ16");
            SilkParamsTest_AssertEqual(cpu.nlsf, gpu.nlsf, "NlsfQ15");
            SilkParamsTest_AssertEqual(cpu.lpc, gpu.lpc, "PredCoefQ12");
            SilkParamsTest_AssertEqual(cpu.pitch, gpu.pitch, "PitchL");
            SilkParamsTest_AssertEqual(cpu.ltpCoef, gpu.ltpCoef, "LtpCoefQ14");
            if (cpu.ltpScale != gpu.ltpScale)
                throw new Exception($"LtpScaleQ14: cpu={cpu.ltpScale} gpu={gpu.ltpScale}");
            SilkParamsTest_AssertEqual(cpu.prevNlsf, gpu.prevNlsf, "prevNlsfQ15 (state update)");
            if ((sbyte)gpu.lastGainIdx != cpu.lastGainIdx)
                throw new Exception($"lastGainIndex (state update): cpu={cpu.lastGainIdx} gpu={(sbyte)gpu.lastGainIdx}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkParametersDecoderGpu_UnvoicedNb10msIndependent_BitExactVsCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            var codebook = SilkNlsfCodebookTables.NbMb;
            int order = codebook.Order;
            const int fsKHz = 8, nbSubfr = 2, conditional = 0;

            var fixture = new SilkSideInfoFixture
            {
                SignalType = 1,
                QuantOffsetType = 0,
                GainIndices = new int[] { 10, 9 },
                NlsfIndices = new sbyte[order + 1],
                InterpolationFactor = 0, // unused for nbSubfr=2
                Seed = 1,
                LtpGainIndices = new int[] { 0, 0 },
            };
            fixture.NlsfIndices[0] = 5;
            for (int i = 0; i < order; i++)
                fixture.NlsfIndices[i + 1] = (sbyte)(((i + 1) % 5) - 2);

            var prevNlsfIn = new short[order];
            for (int i = 0; i < order; i++)
                prevNlsfIn[i] = (short)(3000 + i * 500);
            sbyte lastGainIdxIn = 25;

            var indicesBuf = SilkParamsTest_BuildIndicesBuffer(fixture, order, nbSubfr);

            var cpu = SilkParamsTest_CpuOracle(
                fixture, codebook, fsKHz, nbSubfr, conditional,
                prevNlsfIn, lastGainIdxIn, simulatedLagIndex: 0);

            var gpu = await SilkParamsTest_GpuRun(
                acc, indicesBuf, codebook, fsKHz, nbSubfr, conditional,
                prevNlsfIn, lastGainIdxIn, simulatedLagIndex: 0);

            SilkParamsTest_AssertEqual(cpu.gains, gpu.gains, "GainsQ16");
            SilkParamsTest_AssertEqual(cpu.nlsf, gpu.nlsf, "NlsfQ15");
            SilkParamsTest_AssertEqual(cpu.lpc, gpu.lpc, "PredCoefQ12");
            SilkParamsTest_AssertEqual(cpu.pitch, gpu.pitch, "PitchL");
            SilkParamsTest_AssertEqual(cpu.ltpCoef, gpu.ltpCoef, "LtpCoefQ14");
            if (cpu.ltpScale != gpu.ltpScale)
                throw new Exception($"LtpScaleQ14: cpu={cpu.ltpScale} gpu={gpu.ltpScale}");
            SilkParamsTest_AssertEqual(cpu.prevNlsf, gpu.prevNlsf, "prevNlsfQ15 (state update)");
            if ((sbyte)gpu.lastGainIdx != cpu.lastGainIdx)
                throw new Exception($"lastGainIndex (state update): cpu={cpu.lastGainIdx} gpu={(sbyte)gpu.lastGainIdx}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
