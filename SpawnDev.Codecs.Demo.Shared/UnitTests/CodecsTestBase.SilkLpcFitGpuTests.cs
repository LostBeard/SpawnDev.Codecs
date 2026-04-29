// Cross-backend test for SilkLpcFitGpu.FitAt. Verifies the GPU LPC
// coefficient fit (with iterative bandwidth expansion on overflow)
// matches the CPU reference SilkLpcFit.Fit bit-exactly.

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
    public async Task SilkLpcFitGpu_NoOverflow_Order10_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            // Q24 -> Q12 fit, all coefs fit in int16 after rshift_round so no bwexpand needed.
            int[] aQ24 = { 1 << 23, -(1 << 22), 1 << 21, -(1 << 20), 1 << 19, -(1 << 18), 1 << 17, -(1 << 16), 1 << 15, -(1 << 14) };
            await FitAndVerify(acc, aQ24, qIn: 24, qOut: 12);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkLpcFitGpu_SingleTapOverflow_Order10_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            // Q24 -> Q12: one tap is huge, forces bwexpand iteration(s).
            int[] aQ24 = { 1 << 28, -(1 << 22), 1 << 21, -(1 << 20), 1 << 19, -(1 << 18), 1 << 17, -(1 << 16), 1 << 15, -(1 << 14) };
            await FitAndVerify(acc, aQ24, qIn: 24, qOut: 12);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkLpcFitGpu_MultiBwexpand_Order16_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            // Q24 -> Q12, multiple coefs over int16 limit, requires multiple bwexpand passes.
            int[] aQ24 = new int[16];
            var rng = new Random(unchecked((int)0xCAFEFEEDu));
            for (int i = 0; i < 16; i++)
                aQ24[i] = rng.Next(-(1 << 28), 1 << 28);
            await FitAndVerify(acc, aQ24, qIn: 24, qOut: 12);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkLpcFitGpu_ExtremeOverflow_TenIterations_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            // Force the 10-iteration clip path: coefs near int.MaxValue.
            int[] aQ24 = new int[10];
            for (int i = 0; i < 10; i++)
                aQ24[i] = (i & 1) == 0 ? int.MaxValue / 2 : -int.MaxValue / 2;
            await FitAndVerify(acc, aQ24, qIn: 24, qOut: 12);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static async Task FitAndVerify(Accelerator acc, int[] aQIn, int qIn, int qOut)
    {
        int d = aQIn.Length;

        // CPU reference (mutates aQIn).
        int[] cpuAIn = (int[])aQIn.Clone();
        short[] cpuAOut = new short[d];
        SilkLpcFit.Fit(cpuAOut, cpuAIn, qOut, qIn, d);

        // GPU reference (mutates aQIn).
        int[] gpuAIn = (int[])aQIn.Clone();
        using var dAIn = acc.Allocate1D<int>(d);
        using var dAOut = acc.Allocate1D<short>(d);
        dAIn.View.CopyFromCPU(gpuAIn);
        dAOut.MemSetToZero();

        var kernel = acc.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<short>, ArrayView<int>, int, int, int>(FitKernel);
        kernel(new Index1D(1), dAOut.View, dAIn.View, qOut, qIn, d);
        await acc.SynchronizeAsync();

        var gpuAOut = await dAOut.CopyToHostAsync();
        var gpuAInOut = await dAIn.CopyToHostAsync();

        for (int i = 0; i < d; i++)
        {
            if (cpuAOut[i] != gpuAOut[i])
                throw new Exception($"aQOut[{i}]: cpu={cpuAOut[i]} gpu={gpuAOut[i]} (d={d})");
            if (cpuAIn[i] != gpuAInOut[i])
                throw new Exception($"aQIn[{i}]: cpu={cpuAIn[i]} gpu={gpuAInOut[i]} (d={d})");
        }
    }

    private static void FitKernel(
        Index1D _,
        ArrayView<short> aQOut, ArrayView<int> aQIn,
        int qOut, int qIn, int d)
    {
        SilkLpcFitGpu.FitAt(aQOut, 0, aQIn, 0, qOut, qIn, d);
    }
}
