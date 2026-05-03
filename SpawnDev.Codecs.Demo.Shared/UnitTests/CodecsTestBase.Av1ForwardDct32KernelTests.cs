// Tests for Av1ForwardDct32Kernel. Validates bit-exact match with
// Av1ForwardDct32.Transform across (a) zero, (b) DC-only / structured,
// (c) random batches, (d) all cos_bit values.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Av1;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public async Task Av1ForwardDct32Kernel_ZeroInput_MatchesCpuReference()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Av1ForwardDct32Kernel(acc);
            var input = new int[32];
            var cpuOut = new int[32];
            Av1ForwardDct32.Transform(input, cpuOut);

            using var dIn = acc.Allocate1D<int>(32);
            using var dOut = acc.Allocate1D<int>(32);
            dIn.View.CopyFromCPU(input);
            kernel.Run(dIn.View, dOut.View, transformCount: 1);
            await acc.SynchronizeAsync();
            var readback = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dOut);
            var gpuOut = new int[32];
            readback.AsSpan(0, 32).CopyTo(gpuOut);
            for (int i = 0; i < 32; i++) Equal(cpuOut[i], gpuOut[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Av1ForwardDct32Kernel_DcOnlyInput_MatchesCpuReference()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Av1ForwardDct32Kernel(acc);
            var input = new int[32];
            for (int i = 0; i < 32; i++) input[i] = 256;
            var cpuOut = new int[32];
            Av1ForwardDct32.Transform(input, cpuOut);

            using var dIn = acc.Allocate1D<int>(32);
            using var dOut = acc.Allocate1D<int>(32);
            dIn.View.CopyFromCPU(input);
            kernel.Run(dIn.View, dOut.View, transformCount: 1);
            await acc.SynchronizeAsync();
            var readback = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dOut);
            var gpuOut = new int[32];
            readback.AsSpan(0, 32).CopyTo(gpuOut);
            for (int i = 0; i < 32; i++) Equal(cpuOut[i], gpuOut[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Av1ForwardDct32Kernel_RandomBatch_MatchesCpuReference()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Av1ForwardDct32Kernel(acc);
            const int transformCount = 16;
            var rng = new Random(0xA1F2);
            var input = new int[transformCount * 32];
            for (int i = 0; i < input.Length; i++) input[i] = rng.Next(-2048, 2048);

            var cpuOut = new int[transformCount * 32];
            for (int t = 0; t < transformCount; t++)
                Av1ForwardDct32.Transform(input.AsSpan(t * 32, 32), cpuOut.AsSpan(t * 32, 32));

            using var dIn = acc.Allocate1D<int>(transformCount * 32);
            using var dOut = acc.Allocate1D<int>(transformCount * 32);
            dIn.View.CopyFromCPU(input);
            kernel.Run(dIn.View, dOut.View, transformCount);
            await acc.SynchronizeAsync();
            var readback = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dOut);
            var gpuOut = new int[transformCount * 32];
            readback.AsSpan(0, gpuOut.Length).CopyTo(gpuOut);

            int mismatches = 0;
            for (int i = 0; i < cpuOut.Length; i++)
                if (cpuOut[i] != gpuOut[i]) mismatches++;
            Equal(0, mismatches);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Av1ForwardDct32Kernel_AllCosBits_MatchCpuReference()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Av1ForwardDct32Kernel(acc);
            const int transformCount = 4;
            var rng = new Random(0xFD32);
            var input = new int[transformCount * 32];
            for (int i = 0; i < input.Length; i++) input[i] = rng.Next(-1024, 1024);

            for (int cosBit = 10; cosBit <= 13; cosBit++)
            {
                var cpuOut = new int[transformCount * 32];
                for (int t = 0; t < transformCount; t++)
                    Av1ForwardDct32.Transform(input.AsSpan(t * 32, 32), cpuOut.AsSpan(t * 32, 32), cosBit);

                using var dIn = acc.Allocate1D<int>(transformCount * 32);
                using var dOut = acc.Allocate1D<int>(transformCount * 32);
                dIn.View.CopyFromCPU(input);
                kernel.Run(dIn.View, dOut.View, transformCount, cosBit);
                await acc.SynchronizeAsync();
                var readback = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dOut);
                var gpuOut = new int[transformCount * 32];
                readback.AsSpan(0, gpuOut.Length).CopyTo(gpuOut);

                int mismatches = 0;
                for (int i = 0; i < cpuOut.Length; i++)
                    if (cpuOut[i] != gpuOut[i]) mismatches++;
                Equal(0, mismatches, $"cosBit={cosBit}");
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
