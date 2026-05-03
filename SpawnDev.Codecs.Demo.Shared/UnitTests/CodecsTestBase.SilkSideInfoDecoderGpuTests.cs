// Cross-backend tests for SilkSideInfoDecoderGpu - the GPU port of
// libopus silk/decode_indices.c scalar side-info reads (signal type +
// quantizer offset + PRNG seed). Encodes a known triple on CPU using
// the libopus reference encoder, decodes on GPU, verifies bit-exact.

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
    // Test iCDF tables (mirror SilkIcdfTables values; using public
    // copies here so the tests don't reach into the codec's internal
    // namespace).
    private static readonly byte[] SilkSideInfoTest_TypeOffsetVad =
        new byte[] { 232, 158, 10, 0 };
    private static readonly byte[] SilkSideInfoTest_TypeOffsetNoVad =
        new byte[] { 230, 0 };
    private static readonly byte[] SilkSideInfoTest_Uniform4 =
        new byte[] { 192, 128, 64, 0 };

    /// <summary>
    /// Encode a (signalType, quantOffsetType, seed) triple using the
    /// libopus reference encoder + return the encoded byte buffer.
    /// </summary>
    private static byte[] SilkSideInfoEncodeTripleCpu(
        int signalType, int quantOffsetType, int seed, bool useVadTable)
    {
        var enc = new OpusRangeEncoder(64);

        // Combined index for the signal-type + quantizer-offset symbol.
        // VAD path adds 2 to the raw symbol (CPU
        // SilkSideInfoDecoder.DecodeSignalType reverses with `ix - 2 = raw`,
        // then signalType = ix>>1, quantOffset = ix&1).
        if (useVadTable)
        {
            int combined = (signalType << 1) | quantOffsetType;
            int rawIx = combined - 2; // 0..3 maps signalType {1,2} x offset {0,1}
            enc.EncodeIcdf(rawIx, SilkSideInfoTest_TypeOffsetVad, 8);
        }
        else
        {
            // No-VAD path: signalType is always 0, only the quant-offset
            // bit varies. raw symbol = quantOffsetType (0 or 1).
            enc.EncodeIcdf(quantOffsetType, SilkSideInfoTest_TypeOffsetNoVad, 8);
        }

        // PRNG seed.
        enc.EncodeIcdf(seed, SilkSideInfoTest_Uniform4, 8);

        enc.Done();
        return enc.ToArray();
    }

    private static async Task<int[]> SilkSideInfoDecodeTripleGpuAsync(
        Accelerator acc,
        byte[] packet, bool useVadTable)
    {
        using var dPacket = acc.Allocate1D<byte>(packet.Length);
        using var dVadIcdf = acc.Allocate1D<byte>(SilkSideInfoTest_TypeOffsetVad.Length);
        using var dNoVadIcdf = acc.Allocate1D<byte>(SilkSideInfoTest_TypeOffsetNoVad.Length);
        using var dUniform4Icdf = acc.Allocate1D<byte>(SilkSideInfoTest_Uniform4.Length);
        using var dOutput = acc.Allocate1D<int>(3);

        dPacket.View.CopyFromCPU(packet);
        dVadIcdf.View.CopyFromCPU(SilkSideInfoTest_TypeOffsetVad);
        dNoVadIcdf.View.CopyFromCPU(SilkSideInfoTest_TypeOffsetNoVad);
        dUniform4Icdf.View.CopyFromCPU(SilkSideInfoTest_Uniform4);

        using var kernel = new SilkSideInfoDecoderGpuTestKernel(acc);
        kernel.Run(
            dPacket.View, 0, packet.Length,
            dVadIcdf.View, dNoVadIcdf.View, dUniform4Icdf.View,
            useVadTable ? 1 : 0,
            dOutput.View);
        await acc.SynchronizeAsync();

        var output = await dOutput.CopyToHostAsync();
        var slice = new int[3];
        Array.Copy(output, slice, 3);
        return slice;
    }

    [TestMethod]
    public async Task SilkSideInfoDecoderGpu_VadVoiced_BitExactVsCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // VAD on, signalType=2 (voiced), quantOffsetType=1, seed=2.
            const int signalType = 2;
            const int quantOffsetType = 1;
            const int seed = 2;
            byte[] encoded = SilkSideInfoEncodeTripleCpu(
                signalType, quantOffsetType, seed, useVadTable: true);

            int[] gpu = await SilkSideInfoDecodeTripleGpuAsync(acc, encoded, useVadTable: true);

            if (gpu[0] != signalType)
                throw new Exception($"signalType mismatch: input={signalType} gpu={gpu[0]}");
            if (gpu[1] != quantOffsetType)
                throw new Exception($"quantOffsetType mismatch: input={quantOffsetType} gpu={gpu[1]}");
            if (gpu[2] != seed)
                throw new Exception($"seed mismatch: input={seed} gpu={gpu[2]}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkSideInfoDecoderGpu_VadUnvoiced_BitExactVsCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // VAD on, signalType=1 (unvoiced), quantOffsetType=0, seed=0.
            const int signalType = 1;
            const int quantOffsetType = 0;
            const int seed = 0;
            byte[] encoded = SilkSideInfoEncodeTripleCpu(
                signalType, quantOffsetType, seed, useVadTable: true);

            int[] gpu = await SilkSideInfoDecodeTripleGpuAsync(acc, encoded, useVadTable: true);

            if (gpu[0] != signalType || gpu[1] != quantOffsetType || gpu[2] != seed)
                throw new Exception(
                    $"VAD unvoiced mismatch: input=({signalType},{quantOffsetType},{seed}) " +
                    $"gpu=({gpu[0]},{gpu[1]},{gpu[2]})");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkSideInfoDecoderGpu_NoVadInactive_BitExactVsCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // VAD off, signalType=0 (inactive), quantOffsetType=1, seed=3.
            const int signalType = 0;
            const int quantOffsetType = 1;
            const int seed = 3;
            byte[] encoded = SilkSideInfoEncodeTripleCpu(
                signalType, quantOffsetType, seed, useVadTable: false);

            int[] gpu = await SilkSideInfoDecodeTripleGpuAsync(acc, encoded, useVadTable: false);

            if (gpu[0] != signalType || gpu[1] != quantOffsetType || gpu[2] != seed)
                throw new Exception(
                    $"No-VAD inactive mismatch: input=({signalType},{quantOffsetType},{seed}) " +
                    $"gpu=({gpu[0]},{gpu[1]},{gpu[2]})");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
