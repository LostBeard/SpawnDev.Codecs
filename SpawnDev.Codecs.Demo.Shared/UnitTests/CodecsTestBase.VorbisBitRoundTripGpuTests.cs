// Cross-backend tests for the Vorbis (LSB-first) bit reader/writer
// GPU pair. Verifies VorbisBitWriterGpu + VorbisBitReaderGpu produce
// the same byte format as VorbisBitWriter and that round-trip
// reproduces input values exactly.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Audio.Vorbis;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public async Task VorbisBitGpu_RoundTrip_FixedWidth_8bit()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            const int n = 32;
            var values = new uint[n];
            var bits = new int[n];
            for (int i = 0; i < n; i++) { values[i] = (uint)i; bits[i] = 8; }

            var (gpuBytes, decoded) = await VorbisBitRoundTripGpuAsync(acc, values, bits);

            for (int i = 0; i < n; i++) Equal(values[i], decoded[i]);
            var cpuBytes = VorbisBitEncodeCpu(values, bits);
            Equal(cpuBytes.Length, gpuBytes.Length, "byte length");
            for (int i = 0; i < cpuBytes.Length; i++)
                Equal(cpuBytes[i], gpuBytes[i], $"byte[{i}]");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task VorbisBitGpu_RoundTrip_RandomBatch()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            const int n = 128;
            var rng = new Random(unchecked((int)0xC0DEBAB1u));
            var values = new uint[n];
            var bits = new int[n];
            for (int i = 0; i < n; i++)
            {
                bits[i] = rng.Next(1, 17);
                uint mask = bits[i] == 32 ? 0xFFFFFFFFu : ((1u << bits[i]) - 1);
                values[i] = (uint)rng.Next() & mask;
            }

            var (gpuBytes, decoded) = await VorbisBitRoundTripGpuAsync(acc, values, bits);

            for (int i = 0; i < n; i++)
                if (values[i] != decoded[i])
                    throw new Exception($"sym[{i}] (bits={bits[i]}): input={values[i]} decoded={decoded[i]}");

            var cpuBytes = VorbisBitEncodeCpu(values, bits);
            Equal(cpuBytes.Length, gpuBytes.Length, "byte length");
            for (int i = 0; i < cpuBytes.Length; i++)
                if (cpuBytes[i] != gpuBytes[i])
                    throw new Exception($"byte[{i}]: cpu=0x{cpuBytes[i]:X2} gpu=0x{gpuBytes[i]:X2}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static byte[] VorbisBitEncodeCpu(uint[] values, int[] bits)
    {
        var w = new SpawnDev.Codecs.Audio.Vorbis.VorbisBitWriter();
        for (int i = 0; i < values.Length; i++) w.WriteBits(values[i], bits[i]);
        return w.ToArray();
    }

    private static async Task<(byte[] gpuBytes, uint[] decoded)> VorbisBitRoundTripGpuAsync(
        Accelerator acc, uint[] values, int[] bits)
    {
        int scratchLen = values.Length * 4 + 8;

        using var dValues = acc.Allocate1D<uint>(values.Length);
        using var dBits = acc.Allocate1D<int>(bits.Length);
        using var dDecoded = acc.Allocate1D<uint>(values.Length);
        using var dScratch = acc.Allocate1D<byte>(scratchLen);
        using var dOutLen = acc.Allocate1D<long>(1);

        dValues.View.CopyFromCPU(values);
        dBits.View.CopyFromCPU(bits);
        dScratch.View.CopyFromCPU(new byte[scratchLen]);

        using var kernel = new VorbisBitRoundTripKernel(acc);
        kernel.Run(dValues.View, dBits.View, dDecoded.View, dScratch.View, dOutLen.View, values.Length);
        await acc.SynchronizeAsync();

        long outLen = (await dOutLen.CopyToHostAsync())[0];
        var fullBytes = await dScratch.CopyToHostAsync();
        var bytes = new byte[outLen];
        Array.Copy(fullBytes, bytes, outLen);
        var decoded = await dDecoded.CopyToHostAsync();
        var decodedSlice = new uint[values.Length];
        Array.Copy(decoded, decodedSlice, values.Length);
        return (bytes, decodedSlice);
    }
}
