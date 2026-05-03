// Tests for Vp9ForwardAdst8Kernel - bit-exact mirror of
// Vp9ForwardAdst8.Transform (1D 8-point ADST). One thread per
// 8-coef block.

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
    public async Task Vp9ForwardAdst8Kernel_ZeroInput_ProducesAllZeroCoefs()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Vp9ForwardAdst8Kernel(acc);
            const int blockCount = 4;
            int total = blockCount * 8;
            var input = new int[total];
            var cpuOut = new int[total];
            for (int b = 0; b < blockCount; b++)
                Vp9ForwardAdst8.Transform(input.AsSpan(b * 8, 8), cpuOut.AsSpan(b * 8, 8));

            using var dIn = acc.Allocate1D<int>(total);
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

    [TestMethod]
    public async Task Vp9ForwardAdst8Kernel_DcOnlyInput_MatchesCpuReference()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Vp9ForwardAdst8Kernel(acc);
            // Constant 8-point input - typical for a flat residual.
            var input = new int[8];
            for (int i = 0; i < 8; i++) input[i] = 256;
            var cpuOut = new int[8];
            Vp9ForwardAdst8.Transform(input, cpuOut);

            using var dIn = acc.Allocate1D<int>(8);
            using var dOut = acc.Allocate1D<int>(8);
            dIn.View.CopyFromCPU(input);
            kernel.Run(dIn.View, dOut.View, blockCount: 1);
            await acc.SynchronizeAsync();
            var readback = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dOut);
            var gpuOut = new int[8];
            readback.AsSpan(0, 8).CopyTo(gpuOut);

            int mismatches = 0;
            for (int i = 0; i < 8; i++)
                if (cpuOut[i] != gpuOut[i]) mismatches++;
            Equal(0, mismatches);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9ForwardAdst8Kernel_RandomInput_MatchesCpuReference()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Vp9ForwardAdst8Kernel(acc);
            const int blockCount = 32;
            int total = blockCount * 8;
            var rng = new Random(0xAD58A);
            var input = new int[total];
            for (int i = 0; i < total; i++) input[i] = rng.Next(-2048, 2048);

            var cpuOut = new int[total];
            for (int b = 0; b < blockCount; b++)
                Vp9ForwardAdst8.Transform(input.AsSpan(b * 8, 8), cpuOut.AsSpan(b * 8, 8));

            using var dIn = acc.Allocate1D<int>(total);
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
