// Cross-backend tests for SilkSumSqrShiftGpu.
// Verifies the GPU silk_sum_sqr_shift produces bit-exact (energy, shift)
// pairs vs the CPU SilkSumSqrShift.Compute reference.

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
    public async Task SilkSumSqrShiftGpu_SmallSignal_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            short[] x = { 100, -200, 50, -75, 300, -150, 25, 0 };
            await ComputeAndVerify(acc, x);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkSumSqrShiftGpu_FullScale_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // Full-scale int16 values that force the dynamic shift to engage.
            short[] x = new short[80];
            for (int i = 0; i < x.Length; i++) x[i] = (short)((i * 1729) & 0x7FFF);
            await ComputeAndVerify(acc, x);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkSumSqrShiftGpu_NbWideband_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // Typical SILK NB-WB analysis frame size: 320 samples (40 ms WB).
            var rng = new Random(unchecked((int)0x511CC319u));
            short[] x = new short[320];
            for (int i = 0; i < x.Length; i++) x[i] = (short)rng.Next(-32768, 32768);
            await ComputeAndVerify(acc, x);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static async Task ComputeAndVerify(Accelerator acc, short[] x)
    {
        SilkSumSqrShift.Compute(x, out int cpuEnergy, out int cpuShift);

        using var dX = acc.Allocate1D<short>(x.Length);
        using var dEnergy = acc.Allocate1D<int>(1);
        using var dShift = acc.Allocate1D<int>(1);
        dX.View.CopyFromCPU(x);

        var kernel = acc.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<short>, ArrayView<int>, ArrayView<int>, int>(
            SumSqrShiftKernel);
        kernel(new Index1D(1), dX.View, dEnergy.View, dShift.View, x.Length);
        await acc.SynchronizeAsync();

        int gpuEnergy = (await dEnergy.CopyToHostAsync())[0];
        int gpuShift = (await dShift.CopyToHostAsync())[0];

        if (cpuShift != gpuShift)
            throw new Exception($"shift mismatch: cpu={cpuShift} gpu={gpuShift}");
        if (cpuEnergy != gpuEnergy)
            throw new Exception($"energy mismatch: cpu={cpuEnergy} gpu={gpuEnergy}");
    }

    private static void SumSqrShiftKernel(
        Index1D _, ArrayView<short> x, ArrayView<int> energy, ArrayView<int> shift, int len)
    {
        SilkSumSqrShiftGpu.Compute(x, 0, len, out int e, out int s);
        energy[0] = e;
        shift[0] = s;
    }
}
