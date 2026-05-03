// Cross-backend test for SilkLpcSynthesisFilterGpu.ApplyAt. Verifies the
// GPU LPC synthesis filter (decoder reconstruction inner loop) matches the
// CPU reference SilkLpcSynthesisFilter.Apply bit-exactly across orders 10
// and 16 with representative gains and residual signals.

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
    public async Task SilkLpcSynthesisFilterGpu_Order10_NB_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            short[] aQ12 = { 600, -300, 100, -50, 25, -12, 6, -3, 1, -1 };
            int gainQ10 = 1024; // unity Q10 gain
            int subfrLen = 80;  // typical 5 ms NB subframe at 16 kHz
            int order = 10;
            await SynthAndVerify(acc, aQ12, order, gainQ10, subfrLen, 0xC0DEC0DEu);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkLpcSynthesisFilterGpu_Order16_WB_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            short[] aQ12 = { 800, -400, 200, -100, 50, -25, 12, -6,
                             3, -1, 1, -1, 1, -1, 1, -1 };
            int gainQ10 = 2048; // 2x Q10 gain
            int subfrLen = 160; // typical 10 ms WB subframe at 16 kHz
            int order = 16;
            await SynthAndVerify(acc, aQ12, order, gainQ10, subfrLen, 0xFEEDF00Du);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkLpcSynthesisFilterGpu_LargeGain_Saturation_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // Stress the LSHIFT_SAT32 + SAT16 paths with high gain + large residual.
            short[] aQ12 = { 2000, -1000, 500, -250, 125, -62, 31, -15, 7, -3 };
            int gainQ10 = 16384; // 16x gain
            int subfrLen = 120;
            int order = 10;
            await SynthAndVerify(acc, aQ12, order, gainQ10, subfrLen, 0xBAADF00Du, residualScale: 1 << 20);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static async Task SynthAndVerify(
        Accelerator acc, short[] aQ12, int order, int gainQ10, int subfrLen, uint seed,
        int residualScale = 1 << 14)
    {
        const int maxLpcOrder = 16;
        var rng = new Random(unchecked((int)seed));

        // Random Q14 residual + random Q14 history.
        int[] presQ14 = new int[subfrLen];
        for (int i = 0; i < subfrLen; i++) presQ14[i] = rng.Next(-residualScale, residualScale);

        int[] cpuState = new int[maxLpcOrder + subfrLen];
        for (int i = 0; i < maxLpcOrder; i++)
            cpuState[i] = rng.Next(-(1 << 18), 1 << 18);
        int[] gpuState = (int[])cpuState.Clone();

        // CPU reference.
        short[] cpuOut = new short[subfrLen];
        SilkLpcSynthesisFilter.Apply(cpuState, presQ14, aQ12, gainQ10, order, subfrLen, cpuOut);

        // GPU dispatch: single-thread per stream (sequential filter state).
        using var dState = acc.Allocate1D<int>(maxLpcOrder + subfrLen);
        using var dPres = acc.Allocate1D<int>(subfrLen);
        using var dA = acc.Allocate1D<short>(aQ12.Length);
        using var dPcm = acc.Allocate1D<short>(subfrLen);
        dState.View.CopyFromCPU(gpuState);
        dPres.View.CopyFromCPU(presQ14);
        dA.View.CopyFromCPU(aQ12);
        dPcm.MemSetToZero();

        var kernel = acc.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, ArrayView<int>, ArrayView<short>,
            int, int, int, ArrayView<short>>(SynthKernel);
        kernel(new Index1D(1), dState.View, dPres.View, dA.View,
            gainQ10, order, subfrLen, dPcm.View);
        await acc.SynchronizeAsync();

        var gpuOut = await dPcm.CopyToHostAsync();
        var gpuStateOut = await dState.CopyToHostAsync();

        for (int i = 0; i < subfrLen; i++)
        {
            if (cpuOut[i] != gpuOut[i])
                throw new Exception($"pcm[{i}]: cpu={cpuOut[i]} gpu={gpuOut[i]} (order={order})");
        }
        for (int i = 0; i < maxLpcOrder + subfrLen; i++)
        {
            if (cpuState[i] != gpuStateOut[i])
                throw new Exception($"state[{i}]: cpu={cpuState[i]} gpu={gpuStateOut[i]} (order={order})");
        }
    }

    private static void SynthKernel(
        Index1D _,
        ArrayView<int> stateQ14, ArrayView<int> presQ14, ArrayView<short> aQ12,
        int gainQ10, int order, int subfrLen, ArrayView<short> pcmOut)
    {
        SilkLpcSynthesisFilterGpu.ApplyAt(stateQ14, 0, presQ14, 0, aQ12, 0,
            gainQ10, order, subfrLen, pcmOut, 0);
    }
}
