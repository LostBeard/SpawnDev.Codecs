// Cross-backend test for SilkInverseQ32Gpu.Compute. Verifies the GPU
// silk_INVERSE32_varQ matches the CPU SilkMacros.silk_INVERSE32_varQ
// bit-exactly for typical Q-format inputs.

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
    public async Task SilkInverseQ32Gpu_Compute_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // Mix of small / medium / large inputs spanning the typical
            // SILK Q-format range. Skip 0 (caller responsibility per the API contract).
            (int b32, int Qres)[] cases =
            {
                (1 << 30, 60),       // simplest: 1.0 in Q30 -> result = 1 in Q60? actually 2^60 / 2^30 = 2^30
                (1 << 24, 30),
                (123456, 30),
                (-987654, 32),
                (int.MaxValue, 30),
                (-int.MaxValue, 30),
                (1, 30),
                (-1, 30),
                (1 << 15, 60),
                (-(1 << 25), 50),
            };

            // CPU reference via reflection (silk_INVERSE32_varQ is internal).
            var cpuMethod = typeof(SilkMacros).GetMethod(
                "silk_INVERSE32_varQ",
                BindingFlags.NonPublic | BindingFlags.Static)
                ?? throw new InvalidOperationException("silk_INVERSE32_varQ not found");

            int[] cpuResults = new int[cases.Length];
            for (int i = 0; i < cases.Length; i++)
                cpuResults[i] = (int)cpuMethod.Invoke(null, new object[] { cases[i].b32, cases[i].Qres })!;

            // GPU.
            int n = cases.Length;
            int[] inputs = new int[n * 2];
            for (int i = 0; i < n; i++) { inputs[i * 2] = cases[i].b32; inputs[i * 2 + 1] = cases[i].Qres; }
            using var dInputs = acc.Allocate1D<int>(inputs.Length);
            using var dResults = acc.Allocate1D<int>(n);
            dInputs.View.CopyFromCPU(inputs);

            var kernel = acc.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<int>, ArrayView<int>, int>(InverseKernel);
            kernel(new Index1D(n), dInputs.View, dResults.View, n);
            await acc.SynchronizeAsync();

            var gpuResults = await dResults.CopyToHostAsync();
            for (int i = 0; i < n; i++)
                if (cpuResults[i] != gpuResults[i])
                    throw new Exception(
                        $"INVERSE32[{i}] (b32={cases[i].b32}, Qres={cases[i].Qres}): cpu={cpuResults[i]} gpu={gpuResults[i]}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static void InverseKernel(
        Index1D idx, ArrayView<int> inputs, ArrayView<int> results, int count)
    {
        if (idx >= count) return;
        int b32 = inputs[idx * 2];
        int Qres = inputs[idx * 2 + 1];
        results[idx] = SilkInverseQ32Gpu.Compute(b32, Qres);
    }
}
