// Tests for Vp9ForwardWht4x4Kernel - bit-exact mirror of
// Vp9ForwardWht4x4.Transform (lossless-mode 4x4 Walsh-Hadamard).

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
    public async Task Vp9ForwardWht4x4Kernel_ZeroInput_ProducesAllZero()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var kernel = new Vp9ForwardWht4x4Kernel(acc);
            const int blockCount = 4;
            int total = blockCount * 16;
            using var dIn = acc.Allocate1D<short>(total);
            using var dOut = acc.Allocate1D<int>(total);
            dIn.View.MemSetToZero();
            kernel.Run(dIn.View, dOut.View, blockCount);
            await acc.SynchronizeAsync();
            var readback = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dOut);
            for (int i = 0; i < total; i++) Equal(0, readback[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9ForwardWht4x4Kernel_RandomInput_MatchesCpuReference()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var kernel = new Vp9ForwardWht4x4Kernel(acc);
            const int blockCount = 32;
            int total = blockCount * 16;
            var rng = new Random(0xFA110);
            var input = new short[total];
            for (int i = 0; i < input.Length; i++) input[i] = (short)rng.Next(-1024, 1024);

            // CPU reference output.
            var cpuOut = new int[total];
            for (int b = 0; b < blockCount; b++)
            {
                Vp9ForwardWht4x4.Transform(
                    input.AsSpan(b * 16, 16), rowStrideShorts: 4,
                    cpuOut.AsSpan(b * 16, 16));
            }

            using var dIn = acc.Allocate1D<short>(total);
            using var dOut = acc.Allocate1D<int>(total);
            dIn.View.CopyFromCPU(input);
            kernel.Run(dIn.View, dOut.View, blockCount);
            await acc.SynchronizeAsync();
            var readback = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dOut);
            var gpuOut = new int[total];
            readback.AsSpan(0, total).CopyTo(gpuOut);

            int mismatches = 0;
            for (int i = 0; i < total; i++)
                if (cpuOut[i] != gpuOut[i]) mismatches++;
            Equal(0, mismatches);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
