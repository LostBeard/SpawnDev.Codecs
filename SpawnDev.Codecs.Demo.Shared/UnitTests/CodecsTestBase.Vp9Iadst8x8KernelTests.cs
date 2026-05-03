// Cross-backend tests for Vp9Iadst8x8Kernel. Same structure as iDCT
// 8x8 kernel tests. N=64 LocalMemory scratch now works on WebGPU
// per rc.10; WebGL guarded as before.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    /// <summary>
    /// iADST 8x8 kernel shares the iDCT 8x8 kernel's WebGL constraint:
    /// 64 `flat out` varyings per thread exceeds GL_MAX_VARYING_VECTORS.
    /// WebGPU is green after rc.10.
    /// </summary>
    private static bool IsIadst8x8KernelSupported(Accelerator acc)
    {
        var name = acc.AcceleratorType.ToString();
        return !name.Equals("WebGL", StringComparison.OrdinalIgnoreCase);
    }

    [TestMethod]
    public async Task Vp9Iadst8x8Kernel_ZeroCoefficients_LeavesPredictorUnchanged()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            if (!IsIadst8x8KernelSupported(acc)) return;
            using var kernel = new Vp9Iadst8x8Kernel(acc);
            var coeffs = new short[64];
            var dest = new byte[64];
            for (int i = 0; i < 64; i++) dest[i] = 128;
            await kernel.RunAsync(coeffs.AsMemory(), dest.AsMemory(), blockCount: 1);
            for (int i = 0; i < 64; i++) Equal((byte)128, dest[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9Iadst8x8Kernel_RandomInputs_BitExactMatchReference()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            if (!IsIadst8x8KernelSupported(acc)) return;
            using var kernel = new Vp9Iadst8x8Kernel(acc);
            var rng = new Random(unchecked((int)0xADA58080u));
            for (int trial = 0; trial < 10; trial++)
            {
                var coeffs = new short[64];
                for (int i = 0; i < 64; i++)
                    coeffs[i] = (short)rng.Next(-4096, 4096);
                var cpuDest = new byte[64];
                var gpuDest = new byte[64];
                for (int i = 0; i < 64; i++)
                {
                    byte p = (byte)rng.Next(0, 256);
                    cpuDest[i] = p;
                    gpuDest[i] = p;
                }

                Vp9Iadst8x8Reference.IadstAdst8x8_64_Add(coeffs, cpuDest, 8);
                await kernel.RunAsync(coeffs.AsMemory(), gpuDest.AsMemory(), blockCount: 1);

                for (int i = 0; i < 64; i++)
                    Equal(cpuDest[i], gpuDest[i]);
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9Iadst8x8Kernel_TwoBlockDispatch_MatchesReference()
    {
        // Diagnostic: isolates whether "any batching" or ">= N blocks"
        // is the failure boundary. If 2 blocks also fails but 1 passed,
        // the batch dispatch path itself is broken - not a size issue.
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            if (!IsIadst8x8KernelSupported(acc)) return;
            using var kernel = new Vp9Iadst8x8Kernel(acc);
            const int n = 2;
            var rng = new Random(0xBC0);
            var coeffsFlat = new short[n * 64];
            var predFlat = new byte[n * 64];
            for (int i = 0; i < coeffsFlat.Length; i++)
                coeffsFlat[i] = (short)rng.Next(-4096, 4096);
            for (int i = 0; i < predFlat.Length; i++)
                predFlat[i] = (byte)rng.Next(0, 256);

            var cpuResults = (byte[])predFlat.Clone();
            for (int b = 0; b < n; b++)
                Vp9Iadst8x8Reference.IadstAdst8x8_64_Add(
                    coeffsFlat.AsSpan(b * 64, 64),
                    cpuResults.AsSpan(b * 64, 64),
                    8);

            var gpuResults = (byte[])predFlat.Clone();
            await kernel.RunAsync(coeffsFlat.AsMemory(), gpuResults.AsMemory(), blockCount: n);

            for (int i = 0; i < n * 64; i++)
                Equal(cpuResults[i], gpuResults[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9Iadst8x8Kernel_BatchedDispatch_AllBlocksMatchReference()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            if (!IsIadst8x8KernelSupported(acc)) return;
            using var kernel = new Vp9Iadst8x8Kernel(acc);
            const int n = 8;
            var rng = new Random(unchecked((int)0xADA5B47Cu));
            var coeffsFlat = new short[n * 64];
            var predFlat = new byte[n * 64];
            for (int b = 0; b < n; b++)
            {
                for (int i = 0; i < 64; i++)
                    coeffsFlat[b * 64 + i] = (short)rng.Next(-4096, 4096);
                for (int i = 0; i < 64; i++)
                    predFlat[b * 64 + i] = (byte)rng.Next(0, 256);
            }

            var cpuResults = (byte[])predFlat.Clone();
            for (int b = 0; b < n; b++)
            {
                Vp9Iadst8x8Reference.IadstAdst8x8_64_Add(
                    coeffsFlat.AsSpan(b * 64, 64),
                    cpuResults.AsSpan(b * 64, 64),
                    8);
            }

            var gpuResults = (byte[])predFlat.Clone();
            await kernel.RunAsync(coeffsFlat.AsMemory(), gpuResults.AsMemory(), blockCount: n);

            for (int i = 0; i < n * 64; i++)
                Equal(cpuResults[i], gpuResults[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
