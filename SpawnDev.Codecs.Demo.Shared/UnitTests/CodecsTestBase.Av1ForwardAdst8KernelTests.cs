// Tests for Av1ForwardAdst8Kernel. Validates bit-exact match with
// Av1ForwardAdst8.Transform across (a) zero, (b) DC-only / structured,
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
    public async Task Av1ForwardAdst8Kernel_ZeroInput_MatchesCpuReference()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var kernel = new Av1ForwardAdst8Kernel(acc);
            var input = new int[8];
            var cpuOut = new int[8];
            Av1ForwardAdst8.Transform(input, cpuOut);

            using var dIn = acc.Allocate1D<int>(8);
            using var dOut = acc.Allocate1D<int>(8);
            dIn.View.CopyFromCPU(input);
            kernel.Run(dIn.View, dOut.View, transformCount: 1);
            await acc.SynchronizeAsync();
            var readback = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dOut);
            var gpuOut = new int[8];
            readback.AsSpan(0, 8).CopyTo(gpuOut);
            for (int i = 0; i < 8; i++) Equal(cpuOut[i], gpuOut[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Av1ForwardAdst8Kernel_DcOnlyInput_MatchesCpuReference()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var kernel = new Av1ForwardAdst8Kernel(acc);
            var input = new int[8];
            for (int i = 0; i < 8; i++) input[i] = 256;
            var cpuOut = new int[8];
            Av1ForwardAdst8.Transform(input, cpuOut);

            using var dIn = acc.Allocate1D<int>(8);
            using var dOut = acc.Allocate1D<int>(8);
            dIn.View.CopyFromCPU(input);
            kernel.Run(dIn.View, dOut.View, transformCount: 1);
            await acc.SynchronizeAsync();
            var readback = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dOut);
            var gpuOut = new int[8];
            readback.AsSpan(0, 8).CopyTo(gpuOut);
            for (int i = 0; i < 8; i++) Equal(cpuOut[i], gpuOut[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Av1ForwardAdst8Kernel_RandomBatch_MatchesCpuReference()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var kernel = new Av1ForwardAdst8Kernel(acc);
            const int transformCount = 64;
            var rng = new Random(0xAD58);
            var input = new int[transformCount * 8];
            for (int i = 0; i < input.Length; i++) input[i] = rng.Next(-2048, 2048);

            var cpuOut = new int[transformCount * 8];
            for (int t = 0; t < transformCount; t++)
                Av1ForwardAdst8.Transform(input.AsSpan(t * 8, 8), cpuOut.AsSpan(t * 8, 8));

            using var dIn = acc.Allocate1D<int>(transformCount * 8);
            using var dOut = acc.Allocate1D<int>(transformCount * 8);
            dIn.View.CopyFromCPU(input);
            kernel.Run(dIn.View, dOut.View, transformCount);
            await acc.SynchronizeAsync();
            var readback = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dOut);
            var gpuOut = new int[transformCount * 8];
            readback.AsSpan(0, gpuOut.Length).CopyTo(gpuOut);

            int mismatches = 0;
            for (int i = 0; i < cpuOut.Length; i++)
                if (cpuOut[i] != gpuOut[i]) mismatches++;
            Equal(0, mismatches);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Av1ForwardAdst8Kernel_AllCosBits_MatchCpuReference()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var kernel = new Av1ForwardAdst8Kernel(acc);
            const int transformCount = 16;
            var rng = new Random(0xFD8A);
            var input = new int[transformCount * 8];
            for (int i = 0; i < input.Length; i++) input[i] = rng.Next(-1024, 1024);

            for (int cosBit = 10; cosBit <= 13; cosBit++)
            {
                var cpuOut = new int[transformCount * 8];
                for (int t = 0; t < transformCount; t++)
                    Av1ForwardAdst8.Transform(input.AsSpan(t * 8, 8), cpuOut.AsSpan(t * 8, 8), cosBit);

                using var dIn = acc.Allocate1D<int>(transformCount * 8);
                using var dOut = acc.Allocate1D<int>(transformCount * 8);
                dIn.View.CopyFromCPU(input);
                kernel.Run(dIn.View, dOut.View, transformCount, cosBit);
                await acc.SynchronizeAsync();
                var readback = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dOut);
                var gpuOut = new int[transformCount * 8];
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
