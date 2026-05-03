// Tests for Av1InverseDct8Gpu + Av1InverseDct16Gpu helpers driven
// through the wrapper kernels. Verifies bit-exact match vs the CPU
// reference Av1InverseDct{8,16}.Transform across zero, dc, random
// and all-cosBits scenarios.

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
    public async Task Av1InverseDct8Gpu_ZeroInput_MatchesCpuReference()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Av1InverseDct8GpuKernel(acc);
            var input = new int[8];
            var cpuOut = new int[8];
            Av1InverseDct8.Transform(input, cpuOut);

            using var dIn = acc.Allocate1D<int>(8);
            using var dOut = acc.Allocate1D<int>(8);
            dIn.View.CopyFromCPU(input);
            kernel.Run(dIn.View, dOut.View, transformCount: 1);
            await acc.SynchronizeAsync();
            var readback = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dOut);
            for (int i = 0; i < 8; i++) Equal(cpuOut[i], readback[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Av1InverseDct8Gpu_RandomBatch_MatchesCpuReference()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Av1InverseDct8GpuKernel(acc);
            const int transformCount = 64;
            var rng = new Random(unchecked((int)0x1A8D0C8Au));
            var input = new int[transformCount * 8];
            for (int i = 0; i < input.Length; i++) input[i] = rng.Next(-2048, 2048);

            var cpuOut = new int[transformCount * 8];
            for (int t = 0; t < transformCount; t++)
                Av1InverseDct8.Transform(input.AsSpan(t * 8, 8), cpuOut.AsSpan(t * 8, 8));

            using var dIn = acc.Allocate1D<int>(transformCount * 8);
            using var dOut = acc.Allocate1D<int>(transformCount * 8);
            dIn.View.CopyFromCPU(input);
            kernel.Run(dIn.View, dOut.View, transformCount);
            await acc.SynchronizeAsync();
            var readback = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dOut);

            int mismatches = 0;
            for (int i = 0; i < cpuOut.Length; i++)
                if (cpuOut[i] != readback[i]) mismatches++;
            Equal(0, mismatches);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Av1InverseDct8Gpu_AllCosBits_MatchCpuReference()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Av1InverseDct8GpuKernel(acc);
            const int transformCount = 16;
            var rng = new Random(unchecked((int)0x1A8DCAFEu));
            var input = new int[transformCount * 8];
            for (int i = 0; i < input.Length; i++) input[i] = rng.Next(-1024, 1024);

            for (int cosBit = 10; cosBit <= 13; cosBit++)
            {
                var cpuOut = new int[transformCount * 8];
                for (int t = 0; t < transformCount; t++)
                    Av1InverseDct8.Transform(input.AsSpan(t * 8, 8), cpuOut.AsSpan(t * 8, 8), cosBit);

                using var dIn = acc.Allocate1D<int>(transformCount * 8);
                using var dOut = acc.Allocate1D<int>(transformCount * 8);
                dIn.View.CopyFromCPU(input);
                kernel.Run(dIn.View, dOut.View, transformCount, cosBit);
                await acc.SynchronizeAsync();
                var readback = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dOut);

                int mismatches = 0;
                for (int i = 0; i < cpuOut.Length; i++)
                    if (cpuOut[i] != readback[i]) mismatches++;
                Equal(0, mismatches, $"cosBit={cosBit}");
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Av1InverseDct16Gpu_ZeroInput_MatchesCpuReference()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Av1InverseDct16GpuKernel(acc);
            var input = new int[16];
            var cpuOut = new int[16];
            Av1InverseDct16.Transform(input, cpuOut);

            using var dIn = acc.Allocate1D<int>(16);
            using var dOut = acc.Allocate1D<int>(16);
            dIn.View.CopyFromCPU(input);
            kernel.Run(dIn.View, dOut.View, transformCount: 1);
            await acc.SynchronizeAsync();
            var readback = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dOut);
            for (int i = 0; i < 16; i++) Equal(cpuOut[i], readback[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Av1InverseDct16Gpu_RandomBatch_MatchesCpuReference()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Av1InverseDct16GpuKernel(acc);
            const int transformCount = 32;
            var rng = new Random(unchecked((int)0x1A16D0DEu));
            var input = new int[transformCount * 16];
            for (int i = 0; i < input.Length; i++) input[i] = rng.Next(-2048, 2048);

            var cpuOut = new int[transformCount * 16];
            for (int t = 0; t < transformCount; t++)
                Av1InverseDct16.Transform(input.AsSpan(t * 16, 16), cpuOut.AsSpan(t * 16, 16));

            using var dIn = acc.Allocate1D<int>(transformCount * 16);
            using var dOut = acc.Allocate1D<int>(transformCount * 16);
            dIn.View.CopyFromCPU(input);
            kernel.Run(dIn.View, dOut.View, transformCount);
            await acc.SynchronizeAsync();
            var readback = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dOut);

            int mismatches = 0;
            for (int i = 0; i < cpuOut.Length; i++)
                if (cpuOut[i] != readback[i]) mismatches++;
            Equal(0, mismatches);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Av1InverseDct16Gpu_AllCosBits_MatchCpuReference()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Av1InverseDct16GpuKernel(acc);
            const int transformCount = 16;
            var rng = new Random(unchecked((int)0x1A16BABEu));
            var input = new int[transformCount * 16];
            for (int i = 0; i < input.Length; i++) input[i] = rng.Next(-1024, 1024);

            for (int cosBit = 10; cosBit <= 13; cosBit++)
            {
                var cpuOut = new int[transformCount * 16];
                for (int t = 0; t < transformCount; t++)
                    Av1InverseDct16.Transform(input.AsSpan(t * 16, 16), cpuOut.AsSpan(t * 16, 16), cosBit);

                using var dIn = acc.Allocate1D<int>(transformCount * 16);
                using var dOut = acc.Allocate1D<int>(transformCount * 16);
                dIn.View.CopyFromCPU(input);
                kernel.Run(dIn.View, dOut.View, transformCount, cosBit);
                await acc.SynchronizeAsync();
                var readback = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dOut);

                int mismatches = 0;
                for (int i = 0; i < cpuOut.Length; i++)
                    if (cpuOut[i] != readback[i]) mismatches++;
                Equal(0, mismatches, $"cosBit={cosBit}");
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
