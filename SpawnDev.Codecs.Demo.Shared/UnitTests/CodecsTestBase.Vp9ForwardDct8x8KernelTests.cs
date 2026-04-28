// Tests for Vp9ForwardDct8x8Kernel. Validates the ILGPU kernel produces
// bit-for-bit identical output to Vp9ForwardDct8x8.Transform across a
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
    public async Task Vp9ForwardDct8x8Kernel_ZeroInput_ProducesAllZeroCoefs()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var kernel = new Vp9ForwardDct8x8Kernel(acc);
            var input = new short[64];
            var output = new int[64];
            using var dIn = acc.Allocate1D<short>(64);
            using var dOut = acc.Allocate1D<int>(64);
            dIn.View.CopyFromCPU(input);
            kernel.Run(dIn.View, dOut.View, blockCount: 1);
            await acc.SynchronizeAsync();
            var readback = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dOut);
            readback.AsSpan(0, 64).CopyTo(output);
            for (int i = 0; i < 64; i++) Equal(0, output[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9ForwardDct8x8Kernel_DcOnlyInput_MatchesCpuReference()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var kernel = new Vp9ForwardDct8x8Kernel(acc);
            var input = new short[64];
            for (int i = 0; i < 64; i++) input[i] = 64;
            var cpuOut = new int[64];
            Vp9ForwardDct8x8.Transform(input, rowStrideShorts: 8, cpuOut);

            using var dIn = acc.Allocate1D<short>(64);
            using var dOut = acc.Allocate1D<int>(64);
            dIn.View.CopyFromCPU(input);
            kernel.Run(dIn.View, dOut.View, blockCount: 1);
            await acc.SynchronizeAsync();
            var readback = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dOut);
            var gpuOut = new int[64];
            readback.AsSpan(0, 64).CopyTo(gpuOut);

            for (int i = 0; i < 64; i++) Equal(cpuOut[i], gpuOut[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9ForwardDct8x8Kernel_RandomInput_MatchesCpuReference()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var kernel = new Vp9ForwardDct8x8Kernel(acc);
            const int blockCount = 32;
            var rng = new Random(unchecked((int)0x9F8D8808u));
            var input = new short[blockCount * 64];
            for (int i = 0; i < input.Length; i++) input[i] = (short)rng.Next(-1024, 1024);

            var cpuOut = new int[blockCount * 64];
            for (int b = 0; b < blockCount; b++)
            {
                Vp9ForwardDct8x8.Transform(
                    input.AsSpan(b * 64, 64), rowStrideShorts: 8,
                    cpuOut.AsSpan(b * 64, 64));
            }

            using var dIn = acc.Allocate1D<short>(blockCount * 64);
            using var dOut = acc.Allocate1D<int>(blockCount * 64);
            dIn.View.CopyFromCPU(input);
            kernel.Run(dIn.View, dOut.View, blockCount);
            await acc.SynchronizeAsync();
            var readback = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dOut);
            var gpuOut = new int[blockCount * 64];
            readback.AsSpan(0, gpuOut.Length).CopyTo(gpuOut);

            int mismatches = 0;
            for (int i = 0; i < cpuOut.Length; i++)
                if (cpuOut[i] != gpuOut[i]) mismatches++;
            Equal(0, mismatches);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
