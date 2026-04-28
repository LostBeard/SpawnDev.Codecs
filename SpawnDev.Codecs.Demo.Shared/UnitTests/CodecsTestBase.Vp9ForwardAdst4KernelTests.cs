// Tests for Vp9ForwardAdst4Kernel. Validates the ILGPU kernel produces
// bit-for-bit identical output to Vp9ForwardAdst4.Transform across a
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
    public async Task Vp9ForwardAdst4Kernel_ZeroInput_ProducesAllZeroCoefs()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var kernel = new Vp9ForwardAdst4Kernel(acc);
            var input = new int[4];
            var output = new int[4];
            using var dIn = acc.Allocate1D<int>(4);
            using var dOut = acc.Allocate1D<int>(4);
            dIn.View.CopyFromCPU(input);
            kernel.Run(dIn.View, dOut.View, blockCount: 1);
            await acc.SynchronizeAsync();
            var readback = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dOut);
            readback.AsSpan(0, 4).CopyTo(output);
            for (int i = 0; i < 4; i++) Equal(0, output[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9ForwardAdst4Kernel_DcOnlyInput_MatchesCpuReference()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var kernel = new Vp9ForwardAdst4Kernel(acc);
            // DC-only on a 1D primitive: only x0 is nonzero.
            var input = new int[] { 1024, 0, 0, 0 };
            var cpuOut = new int[4];
            Vp9ForwardAdst4.Transform(input, cpuOut);

            using var dIn = acc.Allocate1D<int>(4);
            using var dOut = acc.Allocate1D<int>(4);
            dIn.View.CopyFromCPU(input);
            kernel.Run(dIn.View, dOut.View, blockCount: 1);
            await acc.SynchronizeAsync();
            var readback = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dOut);
            var gpuOut = new int[4];
            readback.AsSpan(0, 4).CopyTo(gpuOut);

            for (int i = 0; i < 4; i++) Equal(cpuOut[i], gpuOut[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9ForwardAdst4Kernel_RandomInput_MatchesCpuReference()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var kernel = new Vp9ForwardAdst4Kernel(acc);
            const int blockCount = 32;
            var rng = new Random(unchecked((int)0x9FA50404u));
            var input = new int[blockCount * 4];
            for (int i = 0; i < input.Length; i++) input[i] = rng.Next(-2048, 2048);

            var cpuOut = new int[blockCount * 4];
            for (int b = 0; b < blockCount; b++)
            {
                Vp9ForwardAdst4.Transform(
                    input.AsSpan(b * 4, 4),
                    cpuOut.AsSpan(b * 4, 4));
            }

            using var dIn = acc.Allocate1D<int>(blockCount * 4);
            using var dOut = acc.Allocate1D<int>(blockCount * 4);
            dIn.View.CopyFromCPU(input);
            kernel.Run(dIn.View, dOut.View, blockCount);
            await acc.SynchronizeAsync();
            var readback = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dOut);
            var gpuOut = new int[blockCount * 4];
            readback.AsSpan(0, gpuOut.Length).CopyTo(gpuOut);

            int mismatches = 0;
            for (int i = 0; i < cpuOut.Length; i++)
                if (cpuOut[i] != gpuOut[i]) mismatches++;
            Equal(0, mismatches);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
