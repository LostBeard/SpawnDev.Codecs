// Cross-backend tests for SilkNlsfIndicesDecoderGpu - GPU port of
// SilkNlsfDecoder.DecodeIndices. Round-trip via OpusRangeEncoder using
// the libopus iCDF tables in the same order the decoder consumes them.
// Verify GPU output matches the input indices bit-exact.
//
// Test approach: instead of hand-crafting a packet, we drive the CPU
// SilkNlsfDecoder.DecodeIndices behavior in reverse via OpusRangeEncoder,
// using the SilkNlsfCodebookTables.NbMb codebook (order=10) for
// simplicity. The GPU decoder reads the same iCDF tables in the same
// order; output must match the input indices.

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
    /// <summary>
    /// Encode a sequence of NLSF indices via the libopus reference encoder.
    /// Mirrors the bit-stream layout SilkNlsfDecoder.DecodeIndices consumes:
    ///   1. cb1Index from cb1Icdf at (signalType >> 1) * NVectors offset.
    ///   2. unpack ecIx[] for cb1Index via SilkNlsfUnpack.Unpack.
    ///   3. For each i in [0..order), encode (rawIx + MAX_AMPLITUDE) using
    ///      EcIcdf at offset ecIx[i]; if rawIx hits 0 or 2*MAX_AMPLITUDE,
    ///      emit a NlsfExt extension symbol.
    ///   4. If nbSubfr == MAX_NB_SUBFR, encode the Q2 interpolation factor.
    /// Returns the encoded byte stream.
    /// </summary>
    private static byte[] SilkNlsfEncodeIndicesCpu(
        SilkNlsfCodebook codebook,
        sbyte[] indicesIn,
        int signalType,
        int nbSubfr,
        int interpolationFactorQ2)
    {
        int order = codebook.Order;
        var enc = new OpusRangeEncoder(128);

        // 1. cb1 index from the signal-type-selected half.
        int cb1Index = indicesIn[0];
        int cb1IcdfStart = (signalType >> 1) * codebook.NVectors;
        enc.EncodeIcdf(cb1Index, codebook.Cb1Icdf.AsSpan(cb1IcdfStart, codebook.NVectors), 8);

        // 2. unpack ecIx[] for cb1Index.
        Span<short> ecIx = stackalloc short[16];
        Span<byte> predQ8 = stackalloc byte[16];
        SilkNlsfUnpack.Unpack(ecIx, predQ8, codebook, cb1Index);

        // 3. per-coefficient residuals.
        const int RailTopSymbol = 8; // 2 * NLSF_QUANT_MAX_AMPLITUDE
        for (int i = 0; i < order; i++)
        {
            int rawIx = indicesIn[i + 1] + 4; // + NLSF_QUANT_MAX_AMPLITUDE
            // For test simplicity we produce indices in [1, 7] (no rail-extension cases)
            // so the encoder just emits the inner symbol via EcIcdf.
            // If a caller passes 0 or 8 for rawIx, the encode path here would need to
            // emit the NlsfExt extension; we skip that complexity in this test.
            enc.EncodeIcdf(rawIx, codebook.EcIcdf.AsSpan(ecIx[i], 9), 8);
        }

        // 4. interpolation factor for 20ms frames.
        if (nbSubfr == 4)
        {
            enc.EncodeIcdf(interpolationFactorQ2, SilkNlsfTestHelpers_NlsfInterpFactor, 8);
        }

        enc.Done();
        return enc.ToArray();
    }

    private static readonly byte[] SilkNlsfTestHelpers_NlsfExt =
        { 100, 40, 16, 7, 3, 1, 0 };
    private static readonly byte[] SilkNlsfTestHelpers_NlsfInterpFactor =
        { 243, 221, 192, 181, 0 };

    private static async Task<int[]> SilkNlsfDecodeIndicesGpuAsync(
        Accelerator acc,
        SilkNlsfCodebook codebook,
        byte[] packet,
        int signalType,
        int nbSubfr)
    {
        int order = codebook.Order;
        int nVectors = codebook.NVectors;

        using var dPacket = acc.Allocate1D<byte>(packet.Length);
        using var dCb1Icdf = acc.Allocate1D<byte>(codebook.Cb1Icdf.Length);
        using var dEcIcdf = acc.Allocate1D<byte>(codebook.EcIcdf.Length);
        using var dEcSel = acc.Allocate1D<byte>(codebook.EcSel.Length);
        using var dPredQ8Source = acc.Allocate1D<byte>(codebook.PredQ8.Length);
        using var dNlsfExt = acc.Allocate1D<byte>(SilkNlsfTestHelpers_NlsfExt.Length);
        using var dNlsfInterp = acc.Allocate1D<byte>(SilkNlsfTestHelpers_NlsfInterpFactor.Length);
        using var dEcIxScratch = acc.Allocate1D<short>(16);
        using var dPredQ8Scratch = acc.Allocate1D<byte>(16);
        using var dOutput = acc.Allocate1D<int>(order + 2);

        dPacket.View.CopyFromCPU(packet);
        dCb1Icdf.View.CopyFromCPU(codebook.Cb1Icdf);
        dEcIcdf.View.CopyFromCPU(codebook.EcIcdf);
        dEcSel.View.CopyFromCPU(codebook.EcSel);
        dPredQ8Source.View.CopyFromCPU(codebook.PredQ8);
        dNlsfExt.View.CopyFromCPU(SilkNlsfTestHelpers_NlsfExt);
        dNlsfInterp.View.CopyFromCPU(SilkNlsfTestHelpers_NlsfInterpFactor);

        var inputs = new SilkNlsfIndicesInputs
        {
            Cb1Icdf = dCb1Icdf.View,
            EcIcdf = dEcIcdf.View,
            EcSel = dEcSel.View,
            PredQ8Source = dPredQ8Source.View,
            NlsfExtIcdf = dNlsfExt.View,
            NlsfInterpolationFactorIcdf = dNlsfInterp.View,
            EcIxScratch = dEcIxScratch.View,
            PredQ8Scratch = dPredQ8Scratch.View,
        };

        int cb1IcdfBaseOffset = (signalType >> 1) * nVectors;

        using var kernel = new SilkNlsfIndicesDecoderGpuTestKernel(acc);
        kernel.Run(
            dPacket.View, 0, packet.Length,
            inputs,
            cb1IcdfBaseOffset, order, nbSubfr,
            dOutput.View);
        await acc.SynchronizeAsync();

        var output = await dOutput.CopyToHostAsync();
        var slice = new int[order + 2];
        Array.Copy(output, slice, order + 2);
        return slice;
    }

    [TestMethod]
    public async Task SilkNlsfIndicesDecoderGpu_NbMbVoiced20ms_BitExactVsCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            var codebook = SilkNlsfCodebookTables.NbMb;
            int order = codebook.Order;

            // Voiced (signalType=2), 20ms frame (nbSubfr=4 → interpolation factor).
            // cb1 index in [0, NVectors). Pick a mid-range index.
            // Per-coefficient residuals chosen in [-3, +3] to avoid rail extension
            // (rawIx in [1, 7], avoiding 0 and 8).
            sbyte[] indices = new sbyte[order + 1];
            indices[0] = 13; // cb1 first-stage codebook index
            for (int i = 0; i < order; i++)
                indices[i + 1] = (sbyte)((i % 7) - 3); // -3, -2, -1, 0, 1, 2, 3, -3, -2, -1
            int interpFactorQ2 = 2;

            byte[] encoded = SilkNlsfEncodeIndicesCpu(
                codebook, indices, signalType: 2, nbSubfr: 4, interpolationFactorQ2: interpFactorQ2);

            int[] gpu = await SilkNlsfDecodeIndicesGpuAsync(
                acc, codebook, encoded, signalType: 2, nbSubfr: 4);

            for (int i = 0; i < order + 1; i++)
                if (gpu[i] != indices[i])
                    throw new Exception(
                        $"NLSF index mismatch at i={i}: input={indices[i]} gpu={gpu[i]}");
            if (gpu[order + 1] != interpFactorQ2)
                throw new Exception(
                    $"Interpolation factor mismatch: input={interpFactorQ2} gpu={gpu[order + 1]}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkNlsfIndicesDecoderGpu_NbMbInactive10ms_BitExactVsCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            var codebook = SilkNlsfCodebookTables.NbMb;
            int order = codebook.Order;

            // Inactive (signalType=0), 10ms frame (nbSubfr=2 → no interp factor).
            sbyte[] indices = new sbyte[order + 1];
            indices[0] = 5;
            for (int i = 0; i < order; i++)
                indices[i + 1] = (sbyte)(((i + 2) % 5) - 2); // -2, -1, 0, 1, 2, -2, -1, 0, 1, 2

            byte[] encoded = SilkNlsfEncodeIndicesCpu(
                codebook, indices, signalType: 0, nbSubfr: 2, interpolationFactorQ2: 0);

            int[] gpu = await SilkNlsfDecodeIndicesGpuAsync(
                acc, codebook, encoded, signalType: 0, nbSubfr: 2);

            for (int i = 0; i < order + 1; i++)
                if (gpu[i] != indices[i])
                    throw new Exception(
                        $"NLSF index mismatch at i={i}: input={indices[i]} gpu={gpu[i]}");
            if (gpu[order + 1] != 4)
                throw new Exception(
                    $"Expected interp factor 4 (default for nbSubfr != MAX_NB_SUBFR); got {gpu[order + 1]}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkNlsfIndicesDecoderGpu_WbVoiced20ms_BitExactVsCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // WB codebook (order=16).
            var codebook = SilkNlsfCodebookTables.Wb;
            int order = codebook.Order;

            sbyte[] indices = new sbyte[order + 1];
            indices[0] = 7;
            for (int i = 0; i < order; i++)
                indices[i + 1] = (sbyte)(((i + 1) % 7) - 3);
            int interpFactorQ2 = 3;

            byte[] encoded = SilkNlsfEncodeIndicesCpu(
                codebook, indices, signalType: 2, nbSubfr: 4, interpolationFactorQ2: interpFactorQ2);

            int[] gpu = await SilkNlsfDecodeIndicesGpuAsync(
                acc, codebook, encoded, signalType: 2, nbSubfr: 4);

            for (int i = 0; i < order + 1; i++)
                if (gpu[i] != indices[i])
                    throw new Exception(
                        $"NLSF index mismatch at i={i}: input={indices[i]} gpu={gpu[i]}");
            if (gpu[order + 1] != interpFactorQ2)
                throw new Exception(
                    $"Interpolation factor mismatch: input={interpFactorQ2} gpu={gpu[order + 1]}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
