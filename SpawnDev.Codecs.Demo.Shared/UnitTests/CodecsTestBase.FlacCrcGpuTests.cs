// Cross-backend tests for FlacCrcGpu. Verifies CRC-8 + CRC-16 byte-
// computation matches the CPU FlacCrc reference for several inputs.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Audio.Flac;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod(Category = "NotOnWebGL")]
    public async Task FlacCrcGpu_EmptyInput_BothZero()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            var (cpu8, cpu16) = ComputeCpu(Array.Empty<byte>());
            var (gpu8, gpu16) = await ComputeGpuAsync(acc, Array.Empty<byte>());
            Equal(cpu8, gpu8, "crc8");
            Equal(cpu16, gpu16, "crc16");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod(Category = "NotOnWebGL")]
    public async Task FlacCrcGpu_SingleByte_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            byte[] data = { 0x42 };
            var (cpu8, cpu16) = ComputeCpu(data);
            var (gpu8, gpu16) = await ComputeGpuAsync(acc, data);
            Equal(cpu8, gpu8, "crc8");
            Equal(cpu16, gpu16, "crc16");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod(Category = "NotOnWebGL")]
    public async Task FlacCrcGpu_FrameHeaderTypical_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            // Typical FLAC frame-header bytes (sync + block-strategy + sample-rate + utf8 sample number).
            byte[] data = { 0xFF, 0xF8, 0x69, 0x18, 0x00 };
            var (cpu8, cpu16) = ComputeCpu(data);
            var (gpu8, gpu16) = await ComputeGpuAsync(acc, data);
            Equal(cpu8, gpu8, "crc8");
            Equal(cpu16, gpu16, "crc16");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod(Category = "NotOnWebGL")]
    public async Task FlacCrcGpu_RandomBatch_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            var rng = new Random(unchecked((int)0xF1ACBABEu));
            byte[] data = new byte[1024];
            rng.NextBytes(data);
            var (cpu8, cpu16) = ComputeCpu(data);
            var (gpu8, gpu16) = await ComputeGpuAsync(acc, data);
            Equal(cpu8, gpu8, "crc8");
            Equal(cpu16, gpu16, "crc16");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    /// <summary>
    /// Ranges around the piece size and past one combine launch (more than FlacCrcGpu.CombinePieces pieces):
    /// the piece-parallel CRCs must equal the serial CPU ones at every cut.
    /// </summary>
    [TestMethod(Category = "NotOnWebGL")]
    public async Task FlacCrcGpu_LongRanges_PieceCombine_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            var rng = new Random(unchecked((int)0xF1AC10C5u));
            int p = FlacCrcGpu.PieceBytes;
            int[] lengths = { p - 1, p, p + 1, 2 * p, 2 * p + 7, 3 * p - 1, 50_000, (FlacCrcGpu.CombinePieces + 2) * p + 123 };
            foreach (int length in lengths)
            {
                byte[] data = new byte[length];
                rng.NextBytes(data);
                var (cpu8, cpu16) = ComputeCpu(data);
                var (gpu8, gpu16) = await ComputeGpuAsync(acc, data);
                Equal(cpu8, gpu8, $"crc8 len={length}");
                Equal(cpu16, gpu16, $"crc16 len={length}");
            }
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    private static (byte crc8, ushort crc16) ComputeCpu(byte[] data)
    {
        return (FlacCrc.Compute8(data), FlacCrc.Compute16(data));
    }

    private static async Task<(byte gpu8, ushort gpu16)> ComputeGpuAsync(Accelerator acc, byte[] data)
    {
        // Allocate1D requires length >= 1; pad if empty.
        int len = Math.Max(1, data.Length);
        using var dData = acc.Allocate1D<byte>(len);
        using var dOut8 = acc.Allocate1D<byte>(1);
        using var dOut16 = acc.Allocate1D<ushort>(1);

        // Build padded input (length 1 byte for empty case).
        var padded = new byte[len];
        if (data.Length > 0) Array.Copy(data, padded, data.Length);
        dData.View.CopyFromCPU(padded);

        using var kernel = new FlacCrcGpuKernel(acc);
        await kernel.RunAsync(dData.View, dOut8.View, dOut16.View, data.Length);

        byte crc8 = (await dOut8.CopyToHostAsync())[0];
        ushort crc16 = (await dOut16.CopyToHostAsync())[0];
        return (crc8, crc16);
    }
}
