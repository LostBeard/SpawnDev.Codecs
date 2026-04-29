// Cross-backend tests for the FLAC bit reader/writer GPU pair.
// Verifies FlacBitWriterGpu + FlacBitReaderGpu round-trip with the
// same byte format as the CPU FlacBitWriter/FlacBitReader.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Audio.Flac;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public async Task FlacBitGpu_RoundTrip_FixedWidth_8bit()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            const int n = 32;
            var values = new uint[n];
            var bits = new int[n];
            for (int i = 0; i < n; i++) { values[i] = (uint)i; bits[i] = 8; }

            var (gpuBytes, decoded) = await FlacBitRoundTripGpuAsync(acc, values, bits);

            // Verify decoded matches input.
            for (int i = 0; i < n; i++) Equal(values[i], decoded[i]);
            // Verify bytes match a CPU reference.
            var cpuBytes = FlacBitEncodeCpu(values, bits);
            Equal(cpuBytes.Length, gpuBytes.Length, "byte length");
            for (int i = 0; i < cpuBytes.Length; i++)
                Equal(cpuBytes[i], gpuBytes[i], $"byte[{i}]");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task FlacBitGpu_RoundTrip_VariableWidth_BoundaryAligned()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            // Mix of bit widths that crosses byte boundaries multiple
            // times. Total bit count must align to byte boundary OR
            // be padded with zeros - the writer's AlignToByte zeroes
            // any partial trailing bits.
            var values = new uint[] { 0x5u, 0xAAu, 0x3F0u, 0xFFu, 0xCCCCu, 0x1u, 0xFFFFu, 0x7u };
            var bits = new int[] { 4, 8, 12, 8, 16, 1, 16, 7 }; // total = 72 bits = 9 bytes

            var (gpuBytes, decoded) = await FlacBitRoundTripGpuAsync(acc, values, bits);

            for (int i = 0; i < values.Length; i++) Equal(values[i], decoded[i]);
            var cpuBytes = FlacBitEncodeCpu(values, bits);
            Equal(cpuBytes.Length, gpuBytes.Length, "byte length");
            for (int i = 0; i < cpuBytes.Length; i++)
                Equal(cpuBytes[i], gpuBytes[i], $"byte[{i}]");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task FlacBitGpu_RoundTrip_RandomBatch()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            const int n = 128;
            var rng = new Random(unchecked((int)0xF1ACABCDu));
            var values = new uint[n];
            var bits = new int[n];
            for (int i = 0; i < n; i++)
            {
                bits[i] = rng.Next(1, 17); // 1..16-bit symbols.
                uint mask = bits[i] == 32 ? 0xFFFFFFFFu : ((1u << bits[i]) - 1);
                values[i] = (uint)rng.Next() & mask;
            }

            var (gpuBytes, decoded) = await FlacBitRoundTripGpuAsync(acc, values, bits);

            for (int i = 0; i < n; i++)
                if (values[i] != decoded[i])
                    throw new Exception($"sym[{i}] (bits={bits[i]}): input={values[i]} decoded={decoded[i]}");

            var cpuBytes = FlacBitEncodeCpu(values, bits);
            Equal(cpuBytes.Length, gpuBytes.Length, "byte length");
            for (int i = 0; i < cpuBytes.Length; i++)
                if (cpuBytes[i] != gpuBytes[i])
                    throw new Exception($"byte[{i}]: cpu=0x{cpuBytes[i]:X2} gpu=0x{gpuBytes[i]:X2}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static byte[] FlacBitEncodeCpu(uint[] values, int[] bits)
    {
        // FlacBitWriter is internal; reflect via the SpawnDev.Codecs
        // assembly's InternalsVisibleTo entry that the test project
        // already has (the test project is referenced by the demo
        // shared project).
        var writer = new SpawnDev.Codecs.Audio.Flac.FlacBitWriter();
        for (int i = 0; i < values.Length; i++)
        {
            writer.Write(values[i], bits[i]);
        }
        writer.AlignToByte();
        return writer.ToArray();
    }

    private static async Task<(byte[] gpuBytes, uint[] decoded)> FlacBitRoundTripGpuAsync(
        Accelerator acc, uint[] values, int[] bits)
    {
        // Worst case: every value at 32 bits. Max total bytes = 4N + 8.
        int scratchLen = values.Length * 4 + 8;

        using var dValues = acc.Allocate1D<uint>(values.Length);
        using var dBits = acc.Allocate1D<int>(bits.Length);
        using var dDecoded = acc.Allocate1D<uint>(values.Length);
        using var dScratch = acc.Allocate1D<byte>(scratchLen);
        using var dOutLen = acc.Allocate1D<long>(1);

        dValues.View.CopyFromCPU(values);
        dBits.View.CopyFromCPU(bits);
        dScratch.View.CopyFromCPU(new byte[scratchLen]);

        using var kernel = new FlacBitRoundTripKernel(acc);
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
