// Tests for Vp9ForwardDct16x16Kernel. Validates the ILGPU kernel produces
// bit-for-bit identical output to Vp9ForwardDct16x16.Transform across a
// wide range of inputs.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public async Task Vp9ForwardDct16x16Kernel_ZeroInput_ProducesAllZeroCoefs()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Vp9ForwardDct16x16Kernel(acc);
            var input = new short[256];
            var output = new int[256];
            using var dIn = acc.Allocate1D<short>(256);
            using var dOut = acc.Allocate1D<int>(256);
            dIn.View.CopyFromCPU(input);
            kernel.Run(dIn.View, dOut.View, blockCount: 1);
            await acc.SynchronizeAsync();
            var readback = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dOut);
            readback.AsSpan(0, 256).CopyTo(output);
            for (int i = 0; i < 256; i++) Equal(0, output[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9ForwardDct16x16Kernel_DcOnlyInput_MatchesCpuReference()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Vp9ForwardDct16x16Kernel(acc);
            var input = new short[256];
            for (int i = 0; i < 256; i++) input[i] = 64;
            var cpuOut = new int[256];
            Vp9ForwardDct16x16.Transform(input, rowStrideShorts: 16, cpuOut);

            using var dIn = acc.Allocate1D<short>(256);
            using var dOut = acc.Allocate1D<int>(256);
            dIn.View.CopyFromCPU(input);
            kernel.Run(dIn.View, dOut.View, blockCount: 1);
            await acc.SynchronizeAsync();
            var readback = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dOut);
            var gpuOut = new int[256];
            readback.AsSpan(0, 256).CopyTo(gpuOut);

            for (int i = 0; i < 256; i++) Equal(cpuOut[i], gpuOut[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9ForwardDct16x16Kernel_RandomInput_MatchesCpuReference()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Vp9ForwardDct16x16Kernel(acc);
            const int blockCount = 32;
            var rng = new Random(unchecked((int)0x9F1D1010u));
            var input = new short[blockCount * 256];
            for (int i = 0; i < input.Length; i++) input[i] = (short)rng.Next(-1024, 1024);

            var cpuOut = new int[blockCount * 256];
            for (int b = 0; b < blockCount; b++)
            {
                Vp9ForwardDct16x16.Transform(
                    input.AsSpan(b * 256, 256), rowStrideShorts: 16,
                    cpuOut.AsSpan(b * 256, 256));
            }

            using var dIn = acc.Allocate1D<short>(blockCount * 256);
            using var dOut = acc.Allocate1D<int>(blockCount * 256);
            dIn.View.CopyFromCPU(input);
            kernel.Run(dIn.View, dOut.View, blockCount);
            await acc.SynchronizeAsync();
            var readback = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dOut);
            var gpuOut = new int[blockCount * 256];
            readback.AsSpan(0, gpuOut.Length).CopyTo(gpuOut);

            int mismatches = 0;
            for (int i = 0; i < cpuOut.Length; i++)
                if (cpuOut[i] != gpuOut[i]) mismatches++;
            Equal(0, mismatches);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
