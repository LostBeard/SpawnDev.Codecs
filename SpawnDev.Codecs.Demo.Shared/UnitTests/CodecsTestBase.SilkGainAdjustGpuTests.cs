// Cross-backend test for SilkGainAdjustGpu.ApplyAt (and indirectly
// SilkDivVarQGpu.Compute). Verifies the GPU gain-adjust step matches the
// CPU reference SilkGainAdjust.Apply bit-exactly for representative gain
// transitions.

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
    public async Task SilkGainAdjustGpu_EqualGains_NoChange_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            await GainAdjustVerify(acc, prevGainQ16: 1 << 16, curGainQ16: 1 << 16, seed: 0xCAFE0001u);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkGainAdjustGpu_GainIncrease_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // 2x gain step: prev=1.0, cur=2.0 in Q16.
            await GainAdjustVerify(acc, prevGainQ16: 1 << 16, curGainQ16: 2 << 16, seed: 0xCAFE0002u);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkGainAdjustGpu_GainDecrease_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // 0.25x gain step: prev=4.0, cur=1.0 in Q16. Stresses the LSHIFT_SAT32 path
            // when ratio > 1 (gainAdj = 4 << 16 = 262144).
            await GainAdjustVerify(acc, prevGainQ16: 4 << 16, curGainQ16: 1 << 16, seed: 0xCAFE0003u);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkGainAdjustGpu_OddRatio_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // Non-power-of-two ratio - exercises the silk_DIV32_varQ approximation.
            await GainAdjustVerify(acc, prevGainQ16: 73219, curGainQ16: 91737, seed: 0xCAFE0004u);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static async Task GainAdjustVerify(
        Accelerator acc, int prevGainQ16, int curGainQ16, uint seed)
    {
        const int maxLpcOrder = 16;
        var rng = new Random(unchecked((int)seed));

        // Random Q14 LPC state.
        int[] cpuState = new int[maxLpcOrder + 80]; // include some tail samples to verify they're untouched
        int[] gpuState = new int[cpuState.Length];
        for (int i = 0; i < cpuState.Length; i++)
        {
            cpuState[i] = rng.Next(-(1 << 18), 1 << 18);
            gpuState[i] = cpuState[i];
        }

        // CPU reference.
        int cpuGainAdj = SilkGainAdjust.Apply(cpuState, prevGainQ16, curGainQ16);

        // GPU dispatch: single-thread (the work is tiny).
        using var dState = acc.Allocate1D<int>(gpuState.Length);
        using var dGainAdj = acc.Allocate1D<int>(1);
        dState.View.CopyFromCPU(gpuState);
        dGainAdj.MemSetToZero();

        var kernel = acc.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, int, int, ArrayView<int>>(GainAdjustKernel);
        kernel(new Index1D(1), dState.View, prevGainQ16, curGainQ16, dGainAdj.View);
        await acc.SynchronizeAsync();

        var gpuOut = await dState.CopyToHostAsync();
        int gpuGainAdj = (await dGainAdj.CopyToHostAsync())[0];

        if (cpuGainAdj != gpuGainAdj)
            throw new Exception($"gainAdj: cpu={cpuGainAdj} gpu={gpuGainAdj}");
        for (int i = 0; i < cpuState.Length; i++)
        {
            if (cpuState[i] != gpuOut[i])
                throw new Exception($"state[{i}]: cpu={cpuState[i]} gpu={gpuOut[i]}");
        }
    }

    private static void GainAdjustKernel(
        Index1D _,
        ArrayView<int> stateQ14, int prevGainQ16, int curGainQ16, ArrayView<int> gainAdjOut)
    {
        SilkGainAdjustGpu.ApplyAt(stateQ14, 0, prevGainQ16, curGainQ16, gainAdjOut, 0);
    }
}
