// Cross-backend test for SilkNlsfInterpolateGpu.InterpolateAt. Verifies the
// GPU per-coefficient NLSF interpolation matches a direct CPU implementation
// of the SilkParametersDecoder.Decode interpolation block bit-exactly across
// representative NLSF pairs and interpCoefQ2 values.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Audio.Opus.Silk;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public async Task SilkNlsfInterpolateGpu_QuarterStep_Order10_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            short[] prev = { 2000, 5000, 8000, 11000, 14500, 17500, 21000, 24500, 28000, 31000 };
            short[] cur  = { 2200, 5200, 7800, 11400, 14200, 17800, 20800, 24800, 27800, 31200 };
            await InterpolateAndVerify(acc, prev, cur, interpCoefQ2: 1);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkNlsfInterpolateGpu_HalfStep_Order10_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            short[] prev = { 2000, 5000, 8000, 11000, 14500, 17500, 21000, 24500, 28000, 31000 };
            short[] cur  = { 2400, 4800, 8200, 11500, 14000, 18200, 20500, 25000, 27500, 31500 };
            await InterpolateAndVerify(acc, prev, cur, interpCoefQ2: 2);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkNlsfInterpolateGpu_ThreeQuarterStep_Order16_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            short[] prev = { 1500, 3500, 5500, 7500, 9500, 11500, 13500, 15500,
                             17500, 19500, 21500, 23500, 25500, 27500, 29500, 31500 };
            short[] cur  = { 1700, 3300, 5700, 7300, 9700, 11300, 13700, 15300,
                             17700, 19300, 21700, 23300, 25700, 27300, 29700, 31300 };
            await InterpolateAndVerify(acc, prev, cur, interpCoefQ2: 3);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkNlsfInterpolateGpu_ZeroStep_Identity_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            // interpCoefQ2 = 0 -> nlsf0 == prev (no interpolation).
            short[] prev = { 1000, 4000, 7000, 10000, 13000, 16000, 19000, 22000, 25000, 28000 };
            short[] cur  = { 5000, 6000, 9000, 12000, 15000, 18000, 21000, 24000, 27000, 30000 };
            await InterpolateAndVerify(acc, prev, cur, interpCoefQ2: 0);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static async Task InterpolateAndVerify(
        Accelerator acc, short[] prev, short[] cur, int interpCoefQ2)
    {
        int order = prev.Length;

        // CPU reference (matches SilkParametersDecoder.Decode interpolation block).
        short[] cpuOut = new short[order];
        for (int i = 0; i < order; i++)
        {
            int delta = cur[i] - prev[i];
            cpuOut[i] = (short)(prev[i] + ((interpCoefQ2 * delta) >> 2));
        }

        // GPU dispatch: per-coefficient parallel.
        using var dPrev = acc.Allocate1D<short>(order);
        using var dCur = acc.Allocate1D<short>(order);
        using var dOut = acc.Allocate1D<short>(order);
        dPrev.View.CopyFromCPU(prev);
        dCur.View.CopyFromCPU(cur);
        dOut.MemSetToZero();

        var kernel = acc.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<short>, ArrayView<short>, ArrayView<short>, int>(InterpKernel);
        kernel(new Index1D(order), dOut.View, dPrev.View, dCur.View, interpCoefQ2);
        await acc.SynchronizeAsync();

        var gpuOut = await dOut.CopyToHostAsync();

        for (int i = 0; i < order; i++)
        {
            if (cpuOut[i] != gpuOut[i])
                throw new Exception($"nlsf0[{i}]: cpu={cpuOut[i]} gpu={gpuOut[i]} (interpCoefQ2={interpCoefQ2})");
        }
    }

    private static void InterpKernel(
        Index1D index,
        ArrayView<short> nlsf0Q15, ArrayView<short> prevNlsfQ15, ArrayView<short> curNlsfQ15,
        int interpCoefQ2)
    {
        SilkNlsfInterpolateGpu.InterpolateAt(nlsf0Q15, 0, prevNlsfQ15, 0, curNlsfQ15, 0,
            interpCoefQ2, index.X);
    }
}
