// Cross-backend tests for Vp9ForwardDct8x8Gpu (the single-block
// in-kernel helper). Verifies element-by-element agreement with
// Vp9ForwardDct8x8.Transform (CPU oracle).

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private static int[] Vp9Fdct8x8Cpu(short[] input)
    {
        var output = new int[64];
        Vp9ForwardDct8x8.Transform(input, rowStrideShorts: 8, output);
        return output;
    }

    private static async Task<int[]> Vp9Fdct8x8GpuAsync(Accelerator acc, short[] input)
    {
        using var kernel = new Vp9ForwardDct8x8GpuTestKernel(acc);
        using var dIn = acc.Allocate1D<short>(64);
        using var dOut = acc.Allocate1D<int>(64);
        using var dScratch = acc.Allocate1D<int>(64);
        dIn.View.CopyFromCPU(input);
        kernel.Run(dIn.View, dOut.View, dScratch.View, inStride: 8);
        await acc.SynchronizeAsync();
        return await dOut.CopyToHostAsync();
    }

    private static async Task AssertVp9Fdct8x8GpuMatchesCpuAsync(Accelerator acc, short[] input)
    {
        var cpu = Vp9Fdct8x8Cpu(input);
        var gpu = await Vp9Fdct8x8GpuAsync(acc, input);
        for (int i = 0; i < 64; i++)
        {
            if (cpu[i] != gpu[i])
                throw new Exception(
                    $"FDCT 8x8 mismatch at output[{i}]: cpu={cpu[i]} gpu={gpu[i]}");
        }
    }

    [TestMethod]
    public async Task Vp9ForwardDct8x8Gpu_AllZero_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            await AssertVp9Fdct8x8GpuMatchesCpuAsync(acc, new short[64]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9ForwardDct8x8Gpu_Impulses_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // DC impulse, mid impulse, negative impulse - exercises every
            // butterfly node because each impulse spreads to all bins.
            for (int pos = 0; pos < 64; pos++)
            {
                var input = new short[64];
                input[pos] = (short)((pos & 1) == 0 ? 100 : -100);
                await AssertVp9Fdct8x8GpuMatchesCpuAsync(acc, input);
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9ForwardDct8x8Gpu_RandomResiduals_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            var rng = new Random(unchecked((int)0xFD088DC7u));
            for (int trial = 0; trial < 16; trial++)
            {
                var input = new short[64];
                for (int i = 0; i < 64; i++)
                    input[i] = (short)rng.Next(-255, 256);
                await AssertVp9Fdct8x8GpuMatchesCpuAsync(acc, input);
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9ForwardDct8x8Gpu_SaturationLimits_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            var positive = new short[64];
            var negative = new short[64];
            var alternating = new short[64];
            for (int i = 0; i < 64; i++)
            {
                positive[i] = 255;
                negative[i] = -255;
                alternating[i] = (short)((i & 1) == 0 ? 255 : -255);
            }
            await AssertVp9Fdct8x8GpuMatchesCpuAsync(acc, positive);
            await AssertVp9Fdct8x8GpuMatchesCpuAsync(acc, negative);
            await AssertVp9Fdct8x8GpuMatchesCpuAsync(acc, alternating);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
