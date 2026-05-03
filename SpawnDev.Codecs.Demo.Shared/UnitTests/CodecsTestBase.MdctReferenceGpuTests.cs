// Cross-backend tests for MdctReferenceGpu. Verifies the parallel
// O(N^2) MDCT produces output matching MdctReference (CPU) within
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
    public async Task MdctReferenceGpu_Impulse_N16_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            int n = 16;
            var input = new float[2 * n];
            input[0] = 1.0f; // unit impulse at start
            await MdctRoundTripAndVerify(acc, input, n, blockCount: 1);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task MdctReferenceGpu_Sinusoid_N32_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            int n = 32;
            var input = new float[2 * n];
            for (int i = 0; i < input.Length; i++)
                input[i] = MathF.Sin(2 * MathF.PI * i / 32f);
            await MdctRoundTripAndVerify(acc, input, n, blockCount: 1);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task MdctReferenceGpu_RandomBatch_N64_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            int n = 64;
            int blockCount = 8;
            var rng = new Random(unchecked((int)0xAD64BADu));
            var input = new float[blockCount * 2 * n];
            for (int i = 0; i < input.Length; i++)
                input[i] = (float)(rng.NextDouble() * 2 - 1);
            await MdctRoundTripAndVerify(acc, input, n, blockCount);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static async Task MdctRoundTripAndVerify(Accelerator acc, float[] input, int n, int blockCount)
    {
        // CPU reference.
        var cpuOut = new float[blockCount * n];
        for (int b = 0; b < blockCount; b++)
        {
            var inSpan = ((ReadOnlySpan<float>)input.AsSpan(b * 2 * n, 2 * n));
            MdctReference.Transform(inSpan, cpuOut.AsSpan(b * n, n));
        }

        using var dInput = acc.Allocate1D<float>(input.Length);
        using var dOutput = acc.Allocate1D<float>(blockCount * n);
        dInput.View.CopyFromCPU(input);

        using var kernel = new MdctReferenceGpuKernel(acc);
        kernel.Run(dInput.View, dOutput.View, blockCount, n);
        await acc.SynchronizeAsync();

        var gpuOut = await dOutput.CopyToHostAsync();

        // Floating-point tolerance: O(N^2) sum accumulates ~N rounding
        // errors per output. For N=64 that's ~6e-5 relative; allow
        // 1e-3 absolute as a generous bound.
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
