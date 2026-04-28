// Cross-backend tests for Vp9ForwardDct16x16Gpu (the single-block
// in-kernel helper). Verifies element-by-element agreement with
// Vp9ForwardDct16x16.Transform (CPU oracle) across a sweep of
// realistic encoder residual inputs.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private static int[] Vp9Fdct16x16Cpu(short[] input)
    {
        var output = new int[256];
        Vp9ForwardDct16x16.Transform(input, rowStrideShorts: 16, output);
        return output;
    }

    private static async Task<int[]> Vp9Fdct16x16GpuAsync(Accelerator acc, short[] input)
    {
        using var kernel = new Vp9ForwardDct16x16GpuTestKernel(acc);
        using var dIn = acc.Allocate1D<short>(256);
        using var dOut = acc.Allocate1D<int>(256);
        using var dScratch = acc.Allocate1D<int>(256);
        dIn.View.CopyFromCPU(input);
        kernel.Run(dIn.View, dOut.View, dScratch.View, inStride: 16);
        await acc.SynchronizeAsync();
        return await dOut.CopyToHostAsync();
    }

    private static async Task AssertVp9Fdct16x16GpuMatchesCpuAsync(Accelerator acc, short[] input)
    {
        var cpu = Vp9Fdct16x16Cpu(input);
        var gpu = await Vp9Fdct16x16GpuAsync(acc, input);
        for (int i = 0; i < 256; i++)
        {
            if (cpu[i] != gpu[i])
                throw new Exception(
                    $"FDCT 16x16 mismatch at output[{i}]: cpu={cpu[i]} gpu={gpu[i]}");
        }
    }

    [TestMethod]
    public async Task Vp9ForwardDct16x16Gpu_AllZeroInput_ProducesZeroOutput()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            var input = new short[256];
            await AssertVp9Fdct16x16GpuMatchesCpuAsync(acc, input);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9ForwardDct16x16Gpu_DcImpulse_MatchesCpu()
    {
        // Single non-zero in spatial domain - exercises every butterfly
        // node because the impulse spreads to every frequency bin.
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            var input = new short[256];
            input[0] = 100;
            await AssertVp9Fdct16x16GpuMatchesCpuAsync(acc, input);

            // Mid-block impulse.
            input = new short[256];
            input[8 * 16 + 8] = 100;
            await AssertVp9Fdct16x16GpuMatchesCpuAsync(acc, input);

            // Negative impulse.
            input = new short[256];
            input[5 * 16 + 11] = -100;
            await AssertVp9Fdct16x16GpuMatchesCpuAsync(acc, input);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9ForwardDct16x16Gpu_RandomResiduals_MatchesCpu()
    {
        // Sweep 16 random 16x16 blocks across the typical residual range.
        // VP9 residuals after subtract are in [-255, 255]; we test the
        // full int8/int9 range.
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            var rng = new Random(unchecked((int)0xFD09CDC7u));
            for (int trial = 0; trial < 16; trial++)
            {
                var input = new short[256];
                for (int i = 0; i < 256; i++)
                    input[i] = (short)rng.Next(-255, 256);
                await AssertVp9Fdct16x16GpuMatchesCpuAsync(acc, input);
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9ForwardDct16x16Gpu_SaturationLimits_MatchesCpu()
    {
        // Worst-case residuals at int16 saturation - extreme inputs flag
        // any overflow / sign-extension drift between CPU and GPU.
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            var positive = new short[256];
            var negative = new short[256];
            var alternating = new short[256];
            for (int i = 0; i < 256; i++)
            {
                positive[i] = 255;
                negative[i] = -255;
                alternating[i] = (short)((i & 1) == 0 ? 255 : -255);
            }
            await AssertVp9Fdct16x16GpuMatchesCpuAsync(acc, positive);
            await AssertVp9Fdct16x16GpuMatchesCpuAsync(acc, negative);
            await AssertVp9Fdct16x16GpuMatchesCpuAsync(acc, alternating);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
