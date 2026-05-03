// Cross-backend test for SilkNlsf2AGpu.ComputeAt. Verifies the full
// NLSF -> Q12 LPC pipeline on the GPU matches the CPU reference
// SilkNlsf2A.Compute bit-exactly. Composes SilkLpcFitGpu +
// SilkLpcInvPredGainGpu + SilkBwexpanderGpu + LSF cosine table lookup.

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
    public async Task SilkNlsf2AGpu_Order10_NB_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // Order-10 NB-style NLSFs: monotonically increasing in [0, 32768).
            short[] nlsf = { 2000, 5000, 8000, 11000, 14500, 17500, 21000, 24500, 28000, 31000 };
            await Nlsf2AAndVerify(acc, nlsf);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkNlsf2AGpu_Order16_WB_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // Order-16 WB-style NLSFs.
            short[] nlsf = { 1500, 3500, 5500, 7500, 9500, 11500, 13500, 15500,
                             17500, 19500, 21500, 23500, 25500, 27500, 29500, 31500 };
            await Nlsf2AAndVerify(acc, nlsf);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkNlsf2AGpu_Order10_RandomMonotonic_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // Random monotonic NLSFs.
            var rng = new Random(unchecked((int)0xCAFE0010u));
            short[] nlsf = new short[10];
            int sum = 0;
            for (int i = 0; i < 10; i++)
            {
                sum += rng.Next(800, 4000);
                nlsf[i] = (short)Math.Min(sum, 32767);
            }
            await Nlsf2AAndVerify(acc, nlsf);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static async Task Nlsf2AAndVerify(Accelerator acc, short[] nlsf)
    {
        int d = nlsf.Length;

        // CPU reference.
        short[] cpuAQ12 = new short[d];
        SilkNlsf2A.Compute(cpuAQ12, nlsf, d);

        // GPU dispatch: single-thread per stream (sequential pipeline).
        using var dNlsf = acc.Allocate1D<short>(d);
        using var dCosTab = acc.Allocate1D<short>(SilkLsfCosTab.Q12.Length);
        using var dScratch = acc.Allocate1D<int>(72); // safe headroom over the 65 needed
        using var dAQ12 = acc.Allocate1D<short>(d);
        dNlsf.View.CopyFromCPU(nlsf);
        dCosTab.View.CopyFromCPU(SilkLsfCosTab.Q12);
        dScratch.MemSetToZero();
        dAQ12.MemSetToZero();

        var kernel = acc.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<short>, ArrayView<short>, ArrayView<short>,
            ArrayView<int>, int>(Nlsf2AKernel);
        kernel(new Index1D(1), dAQ12.View, dNlsf.View, dCosTab.View, dScratch.View, d);
        await acc.SynchronizeAsync();

        var gpuAQ12 = await dAQ12.CopyToHostAsync();
        for (int i = 0; i < d; i++)
        {
            if (cpuAQ12[i] != gpuAQ12[i])
                throw new Exception($"aQ12[{i}]: cpu={cpuAQ12[i]} gpu={gpuAQ12[i]} (d={d})");
        }
    }

    private static void Nlsf2AKernel(
        Index1D _,
        ArrayView<short> aQ12, ArrayView<short> nlsf, ArrayView<short> lsfCosTab,
        ArrayView<int> scratch, int d)
    {
        SilkNlsf2AGpu.ComputeAt(aQ12, 0, nlsf, 0, lsfCosTab, 0, scratch, 0, d);
    }
}
