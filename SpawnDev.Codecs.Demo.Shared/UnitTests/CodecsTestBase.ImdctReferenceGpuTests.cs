// Cross-backend tests for ImdctReferenceGpu. Verifies the parallel
// O(N^2) IMDCT produces output matching ImdctReference (CPU) within
// floating-point tolerance.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Audio.Transforms;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public async Task ImdctReferenceGpu_DcCoef_N16_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            int n = 16;
            var input = new float[n];
            input[0] = 1.0f;
            await ImdctRoundTripAndVerify(acc, input, n, blockCount: 1);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task ImdctReferenceGpu_RandomCoefs_N32_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            int n = 32;
            var rng = new Random(unchecked((int)0xAD323BAEu));
            var input = new float[n];
            for (int i = 0; i < n; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);
            await ImdctRoundTripAndVerify(acc, input, n, blockCount: 1);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task ImdctReferenceGpu_RandomBatch_N64_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            int n = 64;
            int blockCount = 8;
            var rng = new Random(unchecked((int)0xAD64BABEu));
            var input = new float[blockCount * n];
            for (int i = 0; i < input.Length; i++)
                input[i] = (float)(rng.NextDouble() * 2 - 1);
            await ImdctRoundTripAndVerify(acc, input, n, blockCount);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static async Task ImdctRoundTripAndVerify(Accelerator acc, float[] input, int n, int blockCount)
    {
        var cpuOut = new float[blockCount * 2 * n];
        for (int b = 0; b < blockCount; b++)
        {
            var inSpan = ((ReadOnlySpan<float>)input.AsSpan(b * n, n));
            ImdctReference.Transform(inSpan, cpuOut.AsSpan(b * 2 * n, 2 * n));
        }

        using var dInput = acc.Allocate1D<float>(input.Length);
        using var dOutput = acc.Allocate1D<float>(blockCount * 2 * n);
        dInput.View.CopyFromCPU(input);

        using var kernel = new ImdctReferenceGpuKernel(acc);
        kernel.Run(dInput.View, dOutput.View, blockCount, n);
        await acc.SynchronizeAsync();

        var gpuOut = await dOutput.CopyToHostAsync();

        const float tol = 1e-3f;
        int mismatches = 0;
        int firstMismatch = -1;
        float worstDelta = 0;
        for (int i = 0; i < cpuOut.Length; i++)
        {
            float delta = MathF.Abs(cpuOut[i] - gpuOut[i]);
            if (delta > tol)
            {
                if (firstMismatch < 0) firstMismatch = i;
                if (delta > worstDelta) worstDelta = delta;
                mismatches++;
            }
        }
        if (mismatches > 0)
            throw new Exception(
                $"{mismatches} mismatches > tol={tol}; worst delta={worstDelta}; first at i={firstMismatch} cpu={cpuOut[firstMismatch]} gpu={gpuOut[firstMismatch]}");
    }
}
