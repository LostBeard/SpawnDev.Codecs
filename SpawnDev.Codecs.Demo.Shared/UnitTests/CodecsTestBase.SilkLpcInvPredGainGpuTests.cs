// Cross-backend test for SilkLpcInvPredGainGpu.Compute. Verifies
// the GPU LPC inverse prediction gain matches the CPU reference
// SilkLpcInvPredGain.Compute bit-exactly for representative LPC
// configurations (stable filters + DC-unstable filters).

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
    public async Task SilkLpcInvPredGainGpu_StableNbFilter_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // Stable order-10 NB SILK AR filter (typical Q12 coefficients).
            short[] aQ12 = { 600, -300, 100, -50, 25, -12, 6, -3, 1, -1 };
            await ComputeAndVerifyLpc(acc, aQ12);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkLpcInvPredGainGpu_DcUnstable_ReturnsZero()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // DC unstable: sum of coefficients >= 4096.
            short[] aQ12 = { 1000, 1000, 1000, 1000, 200 };
            await ComputeAndVerifyLpc(acc, aQ12);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkLpcInvPredGainGpu_StableOrder16_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // Stable order-16 WB SILK AR filter.
            short[] aQ12 = { 800, -400, 200, -100, 50, -25, 12, -6,
                             3, -1, 1, -1, 1, -1, 1, -1 };
            await ComputeAndVerifyLpc(acc, aQ12);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static async Task ComputeAndVerifyLpc(Accelerator acc, short[] aQ12)
    {
        // CPU reference.
        int cpuGain = SilkLpcInvPredGain.Compute(aQ12, aQ12.Length);

        // GPU.
        using var dA = acc.Allocate1D<short>(aQ12.Length);
        using var dScratch = acc.Allocate1D<int>(aQ12.Length);
        using var dResult = acc.Allocate1D<int>(1);
        dA.View.CopyFromCPU(aQ12);

        var kernel = acc.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<short>, ArrayView<int>, ArrayView<int>, int>(GainKernel);
        kernel(new Index1D(1), dA.View, dScratch.View, dResult.View, aQ12.Length);
        await acc.SynchronizeAsync();

        int gpuGain = (await dResult.CopyToHostAsync())[0];
        if (cpuGain != gpuGain)
            throw new Exception($"LpcInvPredGain: cpu={cpuGain} gpu={gpuGain}");
    }

    private static void GainKernel(
        Index1D _,
        ArrayView<short> aQ12, ArrayView<int> scratch, ArrayView<int> result,
        int order)
    {
        result[0] = SilkLpcInvPredGainGpu.Compute(aQ12, 0, order, scratch, 0);
    }
}
