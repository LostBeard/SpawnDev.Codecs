// Cross-backend test for SilkNlsfStabilizeGpu.Stabilize. Verifies the
// GPU NLSF stabilizer matches CPU SilkNlsfStabilize.Stabilize bit-
// exactly across CUDA + OpenCL + CPU.

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
    public async Task SilkNlsfStabilizeGpu_AlreadyStable_NoChange()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // 10-tap NLSF that's already well-spaced (each adjacent pair > 100 apart).
            short[] nlsf = { 100, 1100, 2200, 3300, 4400, 5500, 6600, 7700, 8800, 9900 };
            short[] nDelta = new short[11];
            for (int i = 0; i < 11; i++) nDelta[i] = 100;

            await StabilizeAndVerify(acc, nlsf, nDelta);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkNlsfStabilizeGpu_NeedsAdjustment_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // NLSF with violated spacing - some adjacent pairs are too close.
            short[] nlsf = { 50, 80, 200, 400, 1000, 5000, 5050, 8000, 9000, 31000 };
            short[] nDelta = new short[11];
            for (int i = 0; i < 11; i++) nDelta[i] = 200;

            await StabilizeAndVerify(acc, nlsf, nDelta);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkNlsfStabilizeGpu_HeavyDisorder_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // Severely disordered NLSF that forces the fallback insertion-sort + clamp path.
            short[] nlsf = { 5000, 1000, 30000, 2000, 8000, 25000, 4000, 100, 28000, 12000 };
            short[] nDelta = new short[11];
            for (int i = 0; i < 11; i++) nDelta[i] = 250;

            await StabilizeAndVerify(acc, nlsf, nDelta);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static async Task StabilizeAndVerify(Accelerator acc, short[] nlsf, short[] nDelta)
    {
        // CPU reference.
        var cpuNlsf = (short[])nlsf.Clone();
        SilkNlsfStabilize.Stabilize(cpuNlsf, nDelta);

        // GPU.
        using var dNlsf = acc.Allocate1D<short>(nlsf.Length);
        using var dDelta = acc.Allocate1D<short>(nDelta.Length);
        dNlsf.View.CopyFromCPU(nlsf);
        dDelta.View.CopyFromCPU(nDelta);

        var kernel = acc.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<short>, ArrayView<short>, int>(StabilizeKernel);
        kernel(new Index1D(1), dNlsf.View, dDelta.View, nlsf.Length);
        await acc.SynchronizeAsync();

        var gpuNlsf = await dNlsf.CopyToHostAsync();
        for (int i = 0; i < nlsf.Length; i++)
            if (cpuNlsf[i] != gpuNlsf[i])
                throw new Exception($"nlsf[{i}]: cpu={cpuNlsf[i]} gpu={gpuNlsf[i]}");
    }

    private static void StabilizeKernel(
        Index1D _, ArrayView<short> nlsf, ArrayView<short> nDelta, int L)
    {
        SilkNlsfStabilizeGpu.Stabilize(nlsf, 0, L, nDelta, 0);
    }
}
