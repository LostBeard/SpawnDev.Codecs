// Cross-backend test for SilkLsfToCosGpu.Convert. Verifies the GPU
// LSF->cos lookup matches the per-k SilkNlsf2A inner loop bit-exactly.

using System.Reflection;
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
    public async Task SilkLsfToCosGpu_FullLsfRange_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // Sweep across the LSF Q15 range with an arbitrary stride.
            const int n = 128;
            int[] nlsfValues = new int[n];
            var rng = new Random(unchecked((int)0x511C0F50u));
            for (int i = 0; i < n; i++)
                nlsfValues[i] = rng.Next(0, 32768); // Q15 range

            // Get the 129-entry cosine table (internal access via reflection).
            var cosTabField = typeof(SilkLsfCosTab).GetField("Q12",
                BindingFlags.NonPublic | BindingFlags.Static)
                ?? throw new InvalidOperationException("Q12 field not found");
            var cosTab = (short[])cosTabField.GetValue(null)!;

            // CPU reference (replicate the SilkNlsf2A inner loop math).
            int[] cpuResults = new int[n];
            const int qa = 16;
            for (int i = 0; i < n; i++)
            {
                int nlsf = nlsfValues[i];
                int fInt = nlsf >> 8;
                int fFrac = nlsf - (fInt << 8);
                int cosVal = cosTab[fInt];
                int delta = cosTab[fInt + 1] - cosVal;
                long sum = ((long)cosVal << 8) + (long)delta * fFrac;
                int shift = 20 - qa;
                cpuResults[i] = shift > 0
                    ? (int)((sum + (1L << (shift - 1))) >> shift)
                    : (int)sum;
            }

            // GPU.
            using var dNlsf = acc.Allocate1D<int>(n);
            using var dCosTab = acc.Allocate1D<short>(cosTab.Length);
            using var dResults = acc.Allocate1D<int>(n);
            dNlsf.View.CopyFromCPU(nlsfValues);
            dCosTab.View.CopyFromCPU(cosTab);

            var kernel = acc.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<int>, ArrayView<short>, ArrayView<int>, int, int>(
                LsfToCosKernel);
            kernel(new Index1D(n), dNlsf.View, dCosTab.View, dResults.View, n, qa);
            await acc.SynchronizeAsync();

            int[] gpuResults = await dResults.CopyToHostAsync();
            for (int i = 0; i < n; i++)
                if (cpuResults[i] != gpuResults[i])
                    throw new Exception($"LsfToCos[{i}] (nlsf={nlsfValues[i]}): cpu={cpuResults[i]} gpu={gpuResults[i]}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static void LsfToCosKernel(
        Index1D idx,
        ArrayView<int> nlsf, ArrayView<short> cosTab, ArrayView<int> output,
        int count, int qa)
    {
        if (idx >= count) return;
        output[idx] = SilkLsfToCosGpu.Convert(nlsf[idx], cosTab, 0, qa);
    }
}
