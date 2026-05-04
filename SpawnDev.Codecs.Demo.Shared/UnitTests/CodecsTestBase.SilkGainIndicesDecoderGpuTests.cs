// Cross-backend tests for SilkGainIndicesDecoderGpu - GPU port of
// SilkGainDecoder.DecodeIndices. Encodes a known sequence of gain
// indices using the libopus reference encoder, decodes via the GPU,
// verifies bit-exact match against the input.

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
    // SilkIcdfTables values mirrored locally so tests don't reach into the
    // codec's internal namespace.
    private static readonly byte[] SilkGainTest_GainIcdf =
    {
        224, 112,  44,  15,   3,   2,   1,   0,  // inactive
        254, 237, 192, 132,  70,  23,   4,   0,  // unvoiced
        255, 252, 226, 155,  61,  11,   2,   0,  // voiced
    };
    private static readonly byte[] SilkGainTest_DeltaGainIcdf =
    {
        250, 245, 234, 203,  71,  50,  42,  38,
         35,  33,  31,  29,  28,  27,  26,  25,
         24,  23,  22,  21,  20,  19,  18,  17,
         16,  15,  14,  13,  12,  11,  10,   9,
          8,   7,   6,   5,   4,   3,   2,   1,
          0,
    };
    private static readonly byte[] SilkGainTest_Uniform8Icdf =
    {
        224, 192, 160, 128, 96, 64, 32, 0,
    };

    /// <summary>
    /// Encode a sequence of gain indices using the CPU reference encoder.
    /// Mirror of SilkGainDecoder.DecodeIndices's bitstream layout in
    /// reverse direction.
    /// </summary>
    private static byte[] SilkGainEncodeIndicesCpu(
        int[] indices, int signalType, int conditional, int nbSubfr)
    {
        var enc = new OpusRangeEncoder(64);
        if (conditional != 0)
        {
            enc.EncodeIcdf(indices[0], SilkGainTest_DeltaGainIcdf, 8);
        }
        else
        {
            int first = indices[0];
            int msb = first >> 3;
            int lsb = first & 7;
            int gainIcdfStart = signalType * 8;
            enc.EncodeIcdf(msb, SilkGainTest_GainIcdf.AsSpan(gainIcdfStart, 8), 8);
            enc.EncodeIcdf(lsb, SilkGainTest_Uniform8Icdf, 8);
        }
        for (int i = 1; i < nbSubfr; i++)
            enc.EncodeIcdf(indices[i], SilkGainTest_DeltaGainIcdf, 8);
        enc.Done();
        return enc.ToArray();
    }

    private static async Task<int[]> SilkGainDecodeIndicesGpuAsync(
        Accelerator acc,
        byte[] packet, int signalType, int conditional, int nbSubfr)
    {
        using var dPacket = acc.Allocate1D<byte>(packet.Length);
        using var dGainIcdf = acc.Allocate1D<byte>(SilkGainTest_GainIcdf.Length);
        using var dDeltaIcdf = acc.Allocate1D<byte>(SilkGainTest_DeltaGainIcdf.Length);
        using var dUniform8 = acc.Allocate1D<byte>(SilkGainTest_Uniform8Icdf.Length);
        using var dOutput = acc.Allocate1D<int>(nbSubfr);

        dPacket.View.CopyFromCPU(packet);
        dGainIcdf.View.CopyFromCPU(SilkGainTest_GainIcdf);
        dDeltaIcdf.View.CopyFromCPU(SilkGainTest_DeltaGainIcdf);
        dUniform8.View.CopyFromCPU(SilkGainTest_Uniform8Icdf);

        using var kernel = new SilkGainIndicesDecoderGpuTestKernel(acc);
        kernel.Run(
            dPacket.View, 0, packet.Length,
            dGainIcdf.View, dDeltaIcdf.View, dUniform8.View,
            signalType, conditional, nbSubfr,
            dOutput.View);
        await acc.SynchronizeAsync();

        var output = await dOutput.CopyToHostAsync();
        var slice = new int[nbSubfr];
        Array.Copy(output, slice, nbSubfr);
        return slice;
    }

    [TestMethod]
    public async Task SilkGainIndicesDecoderGpu_VoicedIndependent4Subfr_BitExactVsCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // Voiced (signalType=2), independent coding, 4 subframes (20ms frame).
            int[] indices = new[] { (2 << 3) + 5, 7, 11, 4 }; // 21, 7, 11, 4
            byte[] encoded = SilkGainEncodeIndicesCpu(
                indices, signalType: 2, conditional: 0, nbSubfr: 4);

            int[] gpu = await SilkGainDecodeIndicesGpuAsync(
                acc, encoded, signalType: 2, conditional: 0, nbSubfr: 4);

            for (int i = 0; i < 4; i++)
                if (gpu[i] != indices[i])
                    throw new Exception($"Gain index mismatch at i={i}: input={indices[i]} gpu={gpu[i]}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkGainIndicesDecoderGpu_UnvoicedConditional2Subfr_BitExactVsCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // Unvoiced (signalType=1), conditional coding (delta), 2 subframes (10ms).
            int[] indices = new[] { 12, 18 };
            byte[] encoded = SilkGainEncodeIndicesCpu(
                indices, signalType: 1, conditional: 1, nbSubfr: 2);

            int[] gpu = await SilkGainDecodeIndicesGpuAsync(
                acc, encoded, signalType: 1, conditional: 1, nbSubfr: 2);

            for (int i = 0; i < 2; i++)
                if (gpu[i] != indices[i])
                    throw new Exception($"Gain index mismatch at i={i}: input={indices[i]} gpu={gpu[i]}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkGainIndicesDecoderGpu_InactiveIndependent4Subfr_BitExactVsCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // Inactive (signalType=0), independent, 4 subframes.
            // first = (0 << 3) + 2 = 2 (low first MSB row).
            int[] indices = new[] { 2, 0, 0, 0 };
            byte[] encoded = SilkGainEncodeIndicesCpu(
                indices, signalType: 0, conditional: 0, nbSubfr: 4);

            int[] gpu = await SilkGainDecodeIndicesGpuAsync(
                acc, encoded, signalType: 0, conditional: 0, nbSubfr: 4);

            for (int i = 0; i < 4; i++)
                if (gpu[i] != indices[i])
                    throw new Exception($"Gain index mismatch at i={i}: input={indices[i]} gpu={gpu[i]}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
