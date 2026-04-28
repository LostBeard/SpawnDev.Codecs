// Tests for Vp9ForwardDct32x32Kernel. Validates the ILGPU kernel produces
// bit-for-bit identical output to Vp9ForwardDct32x32.Transform across a
// wide range of inputs. VP9 32x32 is DCT_DCT only (no ADST).

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
    public async Task Vp9ForwardDct32x32Kernel_ZeroInput_ProducesAllZeroCoefs()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var kernel = new Vp9ForwardDct32x32Kernel(acc);
            var input = new short[1024];
            var output = new int[1024];
            using var dIn = acc.Allocate1D<short>(1024);
            using var dOut = acc.Allocate1D<int>(1024);
            dIn.View.CopyFromCPU(input);
            kernel.Run(dIn.View, dOut.View, blockCount: 1);
            await acc.SynchronizeAsync();
            var readback = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dOut);
            readback.AsSpan(0, 1024).CopyTo(output);
            for (int i = 0; i < 1024; i++) Equal(0, output[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9ForwardDct32x32Kernel_DcOnlyInput_MatchesCpuReference()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var kernel = new Vp9ForwardDct32x32Kernel(acc);
            var input = new short[1024];
            for (int i = 0; i < 1024; i++) input[i] = 64;
            var cpuOut = new int[1024];
            Vp9ForwardDct32x32.Transform(input, rowStrideShorts: 32, cpuOut);

            using var dIn = acc.Allocate1D<short>(1024);
            using var dOut = acc.Allocate1D<int>(1024);
            dIn.View.CopyFromCPU(input);
            kernel.Run(dIn.View, dOut.View, blockCount: 1);
            await acc.SynchronizeAsync();
            var readback = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dOut);
            var gpuOut = new int[1024];
            readback.AsSpan(0, 1024).CopyTo(gpuOut);

            for (int i = 0; i < 1024; i++) Equal(cpuOut[i], gpuOut[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9ForwardDct32x32Kernel_RandomInput_MatchesCpuReference()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var kernel = new Vp9ForwardDct32x32Kernel(acc);
            // 32x32 is heavy: each block is 1024 ints. 32 blocks = 128KB
            // input + 256KB output. Still fits comfortably for unit tests.
            const int blockCount = 32;
            var rng = new Random(unchecked((int)0x9F2D2020u));
            var input = new short[blockCount * 1024];
            for (int i = 0; i < input.Length; i++) input[i] = (short)rng.Next(-1024, 1024);

            var cpuOut = new int[blockCount * 1024];
            for (int b = 0; b < blockCount; b++)
            {
                Vp9ForwardDct32x32.Transform(
                    input.AsSpan(b * 1024, 1024), rowStrideShorts: 32,
                    cpuOut.AsSpan(b * 1024, 1024));
            }

            using var dIn = acc.Allocate1D<short>(blockCount * 1024);
            using var dOut = acc.Allocate1D<int>(blockCount * 1024);
            dIn.View.CopyFromCPU(input);
            kernel.Run(dIn.View, dOut.View, blockCount);
            await acc.SynchronizeAsync();
            var readback = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dOut);
            var gpuOut = new int[blockCount * 1024];
            readback.AsSpan(0, gpuOut.Length).CopyTo(gpuOut);

            int mismatches = 0;
            for (int i = 0; i < cpuOut.Length; i++)
                if (cpuOut[i] != gpuOut[i]) mismatches++;
            Equal(0, mismatches);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
