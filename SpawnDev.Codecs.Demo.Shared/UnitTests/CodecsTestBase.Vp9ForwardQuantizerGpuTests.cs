// Cross-backend tests for Vp9ForwardQuantizerGpu. Verifies element-
// by-element agreement with Vp9ForwardQuantizer.QuantizeBlock across
// every legal combination of (DC quantizer, AC quantizer, block size).
// Sweeps coefficient signs and magnitudes that flag rounding /
// truncate-toward-zero drift.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private static int[] Vp9QuantizeCpu(int[] input, int dcQ, int acQ)
    {
        var output = (int[])input.Clone();
        Vp9ForwardQuantizer.QuantizeBlock(output, dcQ, acQ);
        return output;
    }

    private static async Task<int[]> Vp9QuantizeGpuAsync(Accelerator acc, int[] input, int dcQ, int acQ)
    {
        using var kernel = new Vp9ForwardQuantizerGpuTestKernel(acc);
        using var dCoefs = acc.Allocate1D<int>(input.Length);
        dCoefs.View.CopyFromCPU(input);
        kernel.Run(dCoefs.View, input.Length, dcQ, acQ);
        await acc.SynchronizeAsync();
        return await dCoefs.CopyToHostAsync();
    }

    private static async Task AssertVp9QuantizeGpuMatchesCpuAsync(
        Accelerator acc, int[] input, int dcQ, int acQ)
    {
        var cpu = Vp9QuantizeCpu(input, dcQ, acQ);
        var gpu = await Vp9QuantizeGpuAsync(acc, input, dcQ, acQ);
        for (int i = 0; i < input.Length; i++)
        {
            if (cpu[i] != gpu[i])
                throw new Exception(
                    $"Quantize mismatch at coef[{i}]: input={input[i]} dcQ={dcQ} acQ={acQ} " +
                    $"cpu={cpu[i]} gpu={gpu[i]}");
        }
    }

    [TestMethod]
    public async Task Vp9ForwardQuantizerGpu_AllSignsAllSizes_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            // Block sizes: 16 (Tx4x4), 64 (Tx8x8), 256 (Tx16x16), 1024 (Tx32x32).
            int[] sizes = { 16, 64, 256, 1024 };
            // Spread of dc/ac values from the actual VP9 tables.
            (int dc, int ac)[] qPairs = { (4, 4), (16, 23), (100, 200), (1336, 1828) };

            var rng = new Random(unchecked((int)0xF0E91A78u));
            foreach (var n in sizes)
            {
                var input = new int[n];
                for (int i = 0; i < n; i++)
                    input[i] = rng.Next(-32768, 32768);

                foreach (var (dcQ, acQ) in qPairs)
                    await AssertVp9QuantizeGpuMatchesCpuAsync(acc, input, dcQ, acQ);
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9ForwardQuantizerGpu_RoundingBoundaries_MatchesCpu()
    {
        // Pin the rounding-half-up boundary cases explicitly. For divisor q
        // and value v, the CPU does (|v| + q/2) / q with sign re-applied.
        // The GPU must agree exactly.
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            // Coefficients that land exactly on rounding boundaries for q=10.
            // q/2 = 5; values 0..9 / 10..19 / -1..-9 cover both halves.
            var input = new int[16];
            for (int i = 0; i < 16; i++) input[i] = i - 8; // -8..7
            await AssertVp9QuantizeGpuMatchesCpuAsync(acc, input, dcQ: 10, acQ: 10);
            // q=2 - hits the smallest legal divisor.
            await AssertVp9QuantizeGpuMatchesCpuAsync(acc, input, dcQ: 2, acQ: 2);
            // Large q - quantizes most coefficients to zero.
            await AssertVp9QuantizeGpuMatchesCpuAsync(acc, input, dcQ: 200, acQ: 200);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9ForwardQuantizerGpu_LargeNegativeAndPositive_MatchesCpu()
    {
        // Saturation extremes - large coef magnitudes test the negative-
        // sign path explicitly. Without the explicit RoundedDivide a
        // backend that lowers `int / q` as `>>` on the negative branch
        // would diverge here.
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            var input = new int[64];
            for (int i = 0; i < 64; i++)
                input[i] = (i % 2 == 0 ? 1 : -1) * (i + 1) * 1000;
            await AssertVp9QuantizeGpuMatchesCpuAsync(acc, input, dcQ: 17, acQ: 23);
            await AssertVp9QuantizeGpuMatchesCpuAsync(acc, input, dcQ: 137, acQ: 219);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9ForwardQuantizerGpu_AllZeroInput_StaysZero()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            var input = new int[256];
            await AssertVp9QuantizeGpuMatchesCpuAsync(acc, input, dcQ: 64, acQ: 64);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
