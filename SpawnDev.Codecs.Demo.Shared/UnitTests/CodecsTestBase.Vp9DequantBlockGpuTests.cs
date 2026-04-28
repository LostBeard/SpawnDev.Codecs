// Cross-backend tests for Vp9DequantBlockGpu. Verifies element-by-
// element agreement with Vp9Dequantizer.DequantizeInPlace across
// every block size + a sweep of saturating coefficient magnitudes.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private static short[] Vp9DequantCpu(short[] input, int dcQ, int acQ)
    {
        var output = (short[])input.Clone();
        Vp9Dequantizer.DequantizeInPlace(output, new Vp9PlaneQuantizer((short)dcQ, (short)acQ));
        return output;
    }

    private static async Task<short[]> Vp9DequantGpuAsync(Accelerator acc, short[] input, int dcQ, int acQ)
    {
        using var kernel = new Vp9DequantBlockGpuTestKernel(acc);
        using var dCoefs = acc.Allocate1D<short>(input.Length);
        dCoefs.View.CopyFromCPU(input);
        kernel.Run(dCoefs.View, input.Length, dcQ, acQ);
        await acc.SynchronizeAsync();
        return await dCoefs.CopyToHostAsync();
    }

    private static async Task AssertVp9DequantGpuMatchesCpuAsync(
        Accelerator acc, short[] input, int dcQ, int acQ)
    {
        var cpu = Vp9DequantCpu(input, dcQ, acQ);
        var gpu = await Vp9DequantGpuAsync(acc, input, dcQ, acQ);
        for (int i = 0; i < input.Length; i++)
        {
            if (cpu[i] != gpu[i])
                throw new Exception(
                    $"Dequant mismatch at coef[{i}]: input={input[i]} dcQ={dcQ} acQ={acQ} " +
                    $"cpu={cpu[i]} gpu={gpu[i]}");
        }
    }

    [TestMethod]
    public async Task Vp9DequantBlockGpu_AllSizes_RandomCoefs_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            int[] sizes = { 16, 64, 256, 1024 };
            (int dc, int ac)[] qPairs = { (4, 4), (16, 23), (100, 200), (1336, 1828) };

            var rng = new Random(unchecked((int)0xDE9CDE91u));
            foreach (var n in sizes)
            {
                var input = new short[n];
                for (int i = 0; i < n; i++)
                    input[i] = (short)rng.Next(-2048, 2048);

                foreach (var (dcQ, acQ) in qPairs)
                    await AssertVp9DequantGpuMatchesCpuAsync(acc, input, dcQ, acQ);
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9DequantBlockGpu_SaturatesAtInt16Bounds()
    {
        // Push coef * dequant past short.MaxValue / short.MinValue and
        // verify GPU saturates identically to CPU.
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            // 4096 * 16 = 65536 -> saturates to 32767.
            // -4096 * 16 = -65536 -> saturates to -32768.
            var input = new short[16];
            input[0] = 4096;
            input[1] = -4096;
            input[2] = 100;
            input[3] = -100;
            await AssertVp9DequantGpuMatchesCpuAsync(acc, input, dcQ: 16, acQ: 16);

            // Push DC slot to extremes too.
            input = new short[16];
            input[0] = -4096;
            input[1] = 4096;
            await AssertVp9DequantGpuMatchesCpuAsync(acc, input, dcQ: 16, acQ: 16);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9DequantBlockGpu_AllZeroInput_StaysZero()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            var input = new short[256];
            await AssertVp9DequantGpuMatchesCpuAsync(acc, input, dcQ: 64, acQ: 64);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
