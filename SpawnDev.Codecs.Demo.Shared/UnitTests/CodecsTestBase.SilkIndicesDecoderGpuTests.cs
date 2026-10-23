// Cross-backend tests for SilkIndicesDecoderGpu (the full silk_decode_indices
// orchestrator). Encodes a complete known SILK side-info block in the order
// the decoder consumes via OpusRangeEncoder, then verifies GPU bit-exactness.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Audio.Opus.Silk;
using SpawnDev.Codecs.EntropyCoders;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    /// <summary>Inputs we want the encoded SILK side-info block to decode to.</summary>
    private struct SilkSideInfoFixture
    {
        public int SignalType;          // 0/1/2
        public int QuantOffsetType;     // 0/1
        public int[] GainIndices;       // length nbSubfr
        public sbyte[] NlsfIndices;     // length order+1
        public int InterpolationFactor; // 0..4 (only used if nbSubfr==4)
        public int PitchRawDelta;       // 0 = absolute else delta-coded raw
        public int PitchCoarseLag;      // for absolute path
        public int PitchLsb;            // for absolute path
        public int PitchContour;
        public int LtpPerIndex;         // 0/1/2
        public int[] LtpGainIndices;    // length nbSubfr
        public int LtpScaleIndex;       // 0..2 (only used if conditional==0)
        public int Seed;                // 0..3
    }

    /// <summary>
    /// Encode a full SILK side-info block matching the libopus consume order,
    /// using the same iCDF tables as the decoder.
    /// </summary>
    private static byte[] SilkIndicesEncodeFixtureCpu(
        SilkSideInfoFixture f,
        SilkNlsfCodebook codebook,
        int fsKHz, int nbSubfr,
        bool useVadTable,
        bool conditional,
        bool prevSignalTypeWasVoiced)
    {
        var enc = new OpusRangeEncoder(256);

        // 1. Signal type + offset.
        if (useVadTable)
        {
            int combined = (f.SignalType << 1) | f.QuantOffsetType;
            int rawIx = combined - 2; // signalType in {1,2} -> raw in {0,1,2,3}
            enc.EncodeIcdf(rawIx, SilkSideInfoTest_TypeOffsetVad, 8);
        }
        else
        {
            // No-VAD: signalType always 0; raw = quantOffsetType
            enc.EncodeIcdf(f.QuantOffsetType, SilkSideInfoTest_TypeOffsetNoVad, 8);
        }

        // 2. Gain indices.
        if (conditional)
        {
            enc.EncodeIcdf(f.GainIndices[0], SilkGainTest_DeltaGainIcdf, 8);
        }
        else
        {
            int first = f.GainIndices[0];
            int msb = first >> 3;
            int lsb = first & 7;
            int gainIcdfStart = f.SignalType * 8;
            enc.EncodeIcdf(msb, SilkGainTest_GainIcdf.AsSpan(gainIcdfStart, 8), 8);
            enc.EncodeIcdf(lsb, SilkGainTest_Uniform8Icdf, 8);
        }
        for (int i = 1; i < nbSubfr; i++)
            enc.EncodeIcdf(f.GainIndices[i], SilkGainTest_DeltaGainIcdf, 8);

        // 3. NLSF indices.
        int order = codebook.Order;
        int cb1Index = f.NlsfIndices[0];
        int cb1IcdfStart = (f.SignalType >> 1) * codebook.NVectors;
        enc.EncodeIcdf(cb1Index, codebook.Cb1Icdf.AsSpan(cb1IcdfStart, codebook.NVectors), 8);

        Span<short> ecIx = stackalloc short[16];
        Span<byte> predQ8 = stackalloc byte[16];
        SilkNlsfUnpack.Unpack(ecIx, predQ8, codebook, cb1Index);
        for (int i = 0; i < order; i++)
        {
            int rawIx = f.NlsfIndices[i + 1] + 4; // + NLSF_QUANT_MAX_AMPLITUDE
            // Test fixtures avoid rail-extension (rawIx in [1,7]).
            enc.EncodeIcdf(rawIx, codebook.EcIcdf.AsSpan(ecIx[i], 9), 8);
        }
        if (nbSubfr == 4)
        {
            enc.EncodeIcdf(f.InterpolationFactor, SilkNlsfTestHelpers_NlsfInterpFactor, 8);
        }

        // 4. Voiced-only: pitch + LTP.
        if (f.SignalType == 2)
        {
            // Pitch.
            bool useAbsolute = !(conditional && prevSignalTypeWasVoiced && f.PitchRawDelta > 0);
            if (conditional && prevSignalTypeWasVoiced)
            {
                enc.EncodeIcdf(f.PitchRawDelta, SilkPitchTest_PitchDelta, 8);
            }
            if (useAbsolute)
            {
                enc.EncodeIcdf(f.PitchCoarseLag, SilkPitchTest_PitchLag, 8);
                enc.EncodeIcdf(f.PitchLsb, SilkPitchTest_SelectLagLowBits(fsKHz), 8);
            }
            enc.EncodeIcdf(f.PitchContour, SilkPitchTest_SelectContour(fsKHz, nbSubfr), 8);

            // LTP.
            enc.EncodeIcdf(f.LtpPerIndex, SilkLtpTest_PerIndex, 8);
            var gainIcdf = SilkLtpTest_SelectGain(f.LtpPerIndex);
            for (int k = 0; k < nbSubfr; k++)
                enc.EncodeIcdf(f.LtpGainIndices[k], gainIcdf, 8);
            if (!conditional)
                enc.EncodeIcdf(f.LtpScaleIndex, SilkLtpTest_LtpScale, 8);
        }

        // 5. Seed.
        enc.EncodeIcdf(f.Seed, SilkSideInfoTest_Uniform4, 8);

        enc.Done();
        return enc.ToArray();
    }

    private static async Task<int[]> SilkIndicesDecodeFixtureGpuAsync(
        Accelerator acc,
        byte[] packet,
        SilkNlsfCodebook codebook,
        int fsKHz, int nbSubfr, int conditional,
        int vadFlag, int decodeLbrr,
        int prevLagIndex, int prevSignalTypeWasVoiced,
        int firstFrameAfterReset)
    {
        var ltpGainFlat = SilkLtpTest_FlatGains();

        // All buffers needed: 22 ArrayView<byte> + 1 ArrayView<int> + 1 ArrayView<short> + 1 byte scratch.
        using var dPacket = acc.Allocate1D<byte>(packet.Length);
        var lagLowBits = SilkPitchTest_SelectLagLowBits(fsKHz);
        var contour = SilkPitchTest_SelectContour(fsKHz, nbSubfr);
        using var dLtpGainOffsets = acc.Allocate1D<int>(SilkLtpTest_GainOffsets.Length);
        using var dEcIxScratch = acc.Allocate1D<short>(16);
        using var dOutput = acc.Allocate1D<int>(SilkDecodedIndicesLayout.TotalSlots);

        dPacket.View.CopyFromCPU(packet);
        dLtpGainOffsets.View.CopyFromCPU(SilkLtpTest_GainOffsets);

        var inputsBytes = new PackedDeviceBuffer<byte>();
        var inputs = new SilkIndicesInputs
        {
            TypeOffsetVadIcdfAt = inputsBytes.Add(SilkSideInfoTest_TypeOffsetVad, SilkSideInfoTest_TypeOffsetVad.Length),
            TypeOffsetNoVadIcdfAt = inputsBytes.Add(SilkSideInfoTest_TypeOffsetNoVad, SilkSideInfoTest_TypeOffsetNoVad.Length),
            Uniform4IcdfAt = inputsBytes.Add(SilkSideInfoTest_Uniform4, SilkSideInfoTest_Uniform4.Length),
            GainIcdfAt = inputsBytes.Add(SilkGainTest_GainIcdf, SilkGainTest_GainIcdf.Length),
            DeltaGainIcdfAt = inputsBytes.Add(SilkGainTest_DeltaGainIcdf, SilkGainTest_DeltaGainIcdf.Length),
            Uniform8IcdfAt = inputsBytes.Add(SilkGainTest_Uniform8Icdf, SilkGainTest_Uniform8Icdf.Length),
            Cb1IcdfAt = inputsBytes.Add(codebook.Cb1Icdf, codebook.Cb1Icdf.Length),
            EcIcdfAt = inputsBytes.Add(codebook.EcIcdf, codebook.EcIcdf.Length),
            EcSelAt = inputsBytes.Add(codebook.EcSel, codebook.EcSel.Length),
            PredQ8SourceAt = inputsBytes.Add(codebook.PredQ8, codebook.PredQ8.Length),
            NlsfExtIcdfAt = inputsBytes.Add(SilkNlsfTestHelpers_NlsfExt, SilkNlsfTestHelpers_NlsfExt.Length),
            NlsfInterpolationFactorIcdfAt = inputsBytes.Add(SilkNlsfTestHelpers_NlsfInterpFactor, SilkNlsfTestHelpers_NlsfInterpFactor.Length),
            PitchDeltaIcdfAt = inputsBytes.Add(SilkPitchTest_PitchDelta, SilkPitchTest_PitchDelta.Length),
            PitchLagIcdfAt = inputsBytes.Add(SilkPitchTest_PitchLag, SilkPitchTest_PitchLag.Length),
            LagLowBitsIcdfAt = inputsBytes.Add(lagLowBits, lagLowBits.Length),
            ContourIcdfAt = inputsBytes.Add(contour, contour.Length),
            LtpPerIndexIcdfAt = inputsBytes.Add(SilkLtpTest_PerIndex, SilkLtpTest_PerIndex.Length),
            LtpGainIcdfFlatAt = inputsBytes.Add(ltpGainFlat, ltpGainFlat.Length),
            LtpGainOffsets = dLtpGainOffsets.View,
            LtpScaleIcdfAt = inputsBytes.Add(SilkLtpTest_LtpScale, SilkLtpTest_LtpScale.Length),
            EcIxScratch = dEcIxScratch.View,
            PredQ8ScratchAt = inputsBytes.Reserve(16),
        };
        using var dInputsBytes = inputsBytes.Upload(acc);
        inputs.Bytes = dInputsBytes.View;

        var scalars = new SilkIndicesScalars
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

        using var kernel = new SilkIndicesDecoderGpuTestKernel(acc);
        kernel.Run(dPacket.View, 0, packet.Length, inputs, scalars, dOutput.View);
        await acc.SynchronizeAsync();

        var output = await dOutput.CopyToHostAsync();
        var slice = new int[SilkDecodedIndicesLayout.TotalSlots];
        Array.Copy(output, slice, SilkDecodedIndicesLayout.TotalSlots);
        return slice;
    }

    [TestMethod]
    public async Task SilkIndicesDecoderGpu_VoicedNbWb20msIndependent_BitExactVsCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            var codebook = SilkNlsfCodebookTables.Wb;
            int order = codebook.Order;
            const int fsKHz = 16, nbSubfr = 4;

            var fixture = new SilkSideInfoFixture
            {
                SignalType = 2, // voiced
                QuantOffsetType = 1,
                GainIndices = new int[] { (2 << 3) + 4, 8, 11, 6 },
                NlsfIndices = new sbyte[order + 1],
                InterpolationFactor = 2,
                PitchRawDelta = 0,
                PitchCoarseLag = 14,
                PitchLsb = 5,
                PitchContour = 11,
                LtpPerIndex = 1,
                LtpGainIndices = new int[] { 7, 12, 3, 9 },
                LtpScaleIndex = 1,
                Seed = 2,
            };
            fixture.NlsfIndices[0] = 9;
            for (int i = 0; i < order; i++)
                fixture.NlsfIndices[i + 1] = (sbyte)((i % 7) - 3);

            byte[] encoded = SilkIndicesEncodeFixtureCpu(
                fixture, codebook, fsKHz, nbSubfr,
                useVadTable: true, conditional: false, prevSignalTypeWasVoiced: false);

            int[] gpu = await SilkIndicesDecodeFixtureGpuAsync(
                acc, encoded, codebook, fsKHz, nbSubfr, conditional: 0,
                vadFlag: 1, decodeLbrr: 0,
                prevLagIndex: 0, prevSignalTypeWasVoiced: 0,
                firstFrameAfterReset: 0);

            // Verify each named slot.
            int expectedLag = fixture.PitchCoarseLag * (fsKHz >> 1) + fixture.PitchLsb;
            if (gpu[SilkDecodedIndicesLayout.SignalTypeOffset] != fixture.SignalType)
                throw new Exception($"signalType: expected {fixture.SignalType} got {gpu[SilkDecodedIndicesLayout.SignalTypeOffset]}");
            if (gpu[SilkDecodedIndicesLayout.QuantOffsetTypeOffset] != fixture.QuantOffsetType)
                throw new Exception($"quantOffsetType: expected {fixture.QuantOffsetType} got {gpu[SilkDecodedIndicesLayout.QuantOffsetTypeOffset]}");
            if (gpu[SilkDecodedIndicesLayout.NlsfInterpCoefQ2Offset] != fixture.InterpolationFactor)
                throw new Exception($"interpFactor: expected {fixture.InterpolationFactor} got {gpu[SilkDecodedIndicesLayout.NlsfInterpCoefQ2Offset]}");
            if (gpu[SilkDecodedIndicesLayout.LagIndexOffset] != expectedLag)
                throw new Exception($"lagIndex: expected {expectedLag} got {gpu[SilkDecodedIndicesLayout.LagIndexOffset]}");
            if (gpu[SilkDecodedIndicesLayout.ContourIndexOffset] != fixture.PitchContour)
                throw new Exception($"contour: expected {fixture.PitchContour} got {gpu[SilkDecodedIndicesLayout.ContourIndexOffset]}");
            if (gpu[SilkDecodedIndicesLayout.PerIndexOffset] != fixture.LtpPerIndex)
                throw new Exception($"perIndex: expected {fixture.LtpPerIndex} got {gpu[SilkDecodedIndicesLayout.PerIndexOffset]}");
            if (gpu[SilkDecodedIndicesLayout.LtpScaleIndexOffset] != fixture.LtpScaleIndex)
                throw new Exception($"ltpScaleIndex: expected {fixture.LtpScaleIndex} got {gpu[SilkDecodedIndicesLayout.LtpScaleIndexOffset]}");
            if (gpu[SilkDecodedIndicesLayout.SeedOffset] != fixture.Seed)
                throw new Exception($"seed: expected {fixture.Seed} got {gpu[SilkDecodedIndicesLayout.SeedOffset]}");

            for (int i = 0; i < nbSubfr; i++)
                if (gpu[SilkDecodedIndicesLayout.GainsIndicesOffset + i] != fixture.GainIndices[i])
                    throw new Exception($"gainIdx[{i}]: expected {fixture.GainIndices[i]} got {gpu[SilkDecodedIndicesLayout.GainsIndicesOffset + i]}");
            for (int i = 0; i <= order; i++)
                if (gpu[SilkDecodedIndicesLayout.NlsfIndicesOffset + i] != fixture.NlsfIndices[i])
                    throw new Exception($"nlsfIdx[{i}]: expected {fixture.NlsfIndices[i]} got {gpu[SilkDecodedIndicesLayout.NlsfIndicesOffset + i]}");
            for (int k = 0; k < nbSubfr; k++)
                if (gpu[SilkDecodedIndicesLayout.LtpIndicesOffset + k] != fixture.LtpGainIndices[k])
                    throw new Exception($"ltpGain[{k}]: expected {fixture.LtpGainIndices[k]} got {gpu[SilkDecodedIndicesLayout.LtpIndicesOffset + k]}");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod]
    public async Task SilkIndicesDecoderGpu_UnvoicedNb10msIndependent_BitExactVsCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            var codebook = SilkNlsfCodebookTables.NbMb;
            int order = codebook.Order;
            const int fsKHz = 8, nbSubfr = 2;

            var fixture = new SilkSideInfoFixture
            {
                SignalType = 1, // unvoiced
                QuantOffsetType = 0,
                GainIndices = new int[] { (1 << 3) + 2, 9 },
                NlsfIndices = new sbyte[order + 1],
                InterpolationFactor = 0, // unused for nbSubfr=2
                Seed = 1,
                // Voiced-only fields ignored.
                LtpGainIndices = new int[] { 0, 0 },
            };
            fixture.NlsfIndices[0] = 5;
            for (int i = 0; i < order; i++)
                fixture.NlsfIndices[i + 1] = (sbyte)(((i + 1) % 5) - 2);

            byte[] encoded = SilkIndicesEncodeFixtureCpu(
                fixture, codebook, fsKHz, nbSubfr,
                useVadTable: true, conditional: false, prevSignalTypeWasVoiced: false);

            int[] gpu = await SilkIndicesDecodeFixtureGpuAsync(
                acc, encoded, codebook, fsKHz, nbSubfr, conditional: 0,
                vadFlag: 1, decodeLbrr: 0,
                prevLagIndex: 0, prevSignalTypeWasVoiced: 0,
                firstFrameAfterReset: 0);

            if (gpu[SilkDecodedIndicesLayout.SignalTypeOffset] != fixture.SignalType)
                throw new Exception($"signalType mismatch");
            if (gpu[SilkDecodedIndicesLayout.QuantOffsetTypeOffset] != fixture.QuantOffsetType)
                throw new Exception($"quantOffsetType mismatch");
            // Unvoiced -> voiced-only fields must be zero.
            if (gpu[SilkDecodedIndicesLayout.LagIndexOffset] != 0)
                throw new Exception($"unvoiced lagIndex must be 0; got {gpu[SilkDecodedIndicesLayout.LagIndexOffset]}");
            if (gpu[SilkDecodedIndicesLayout.PerIndexOffset] != 0)
                throw new Exception($"unvoiced perIndex must be 0; got {gpu[SilkDecodedIndicesLayout.PerIndexOffset]}");
            // nbSubfr=2 means default interp factor 4.
            if (gpu[SilkDecodedIndicesLayout.NlsfInterpCoefQ2Offset] != 4)
                throw new Exception($"nbSubfr=2 should produce interp=4; got {gpu[SilkDecodedIndicesLayout.NlsfInterpCoefQ2Offset]}");

            for (int i = 0; i < nbSubfr; i++)
                if (gpu[SilkDecodedIndicesLayout.GainsIndicesOffset + i] != fixture.GainIndices[i])
                    throw new Exception($"gainIdx[{i}] mismatch");
            for (int i = 0; i <= order; i++)
                if (gpu[SilkDecodedIndicesLayout.NlsfIndicesOffset + i] != fixture.NlsfIndices[i])
                    throw new Exception($"nlsfIdx[{i}] mismatch");
            if (gpu[SilkDecodedIndicesLayout.SeedOffset] != fixture.Seed)
                throw new Exception($"seed mismatch");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }
}
