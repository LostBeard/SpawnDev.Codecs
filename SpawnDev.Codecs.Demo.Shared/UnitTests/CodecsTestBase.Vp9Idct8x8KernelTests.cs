// Cross-backend tests for Vp9Idct8x8Kernel. Each runner dispatches
// via CreateKernelAcceleratorAsync; byte-for-byte parity vs reference
// is enforced on every backend that can compile the kernel.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    /// <summary>
    /// WebGL can't execute the 8x8 kernel bit-exactly - not for the usual
    /// "no atomics" reason the 4x4 kernel hits, but because each thread
    /// emits 64 `flat out` varyings which exceeds GL_MAX_VARYING_VECTORS
    /// on most WebGL implementations. Per Geordi's 2026-04-24 analysis
    /// (geordi-to-tuvok-vp9-idct8x8-fixed), getting this kernel green on
    /// WebGL requires the kernel topology to change from one-thread-per-
    /// block to one-thread-per-output-element. That's a future slice if
    /// WebGL coverage becomes a requirement.
    ///
    /// WebGPU was previously blocked by a LocalMemory<int>(N>=32) codegen
    /// bug; that was fixed in SpawnDev.ILGPU 4.9.2-rc.10 (commit 9bc8ec2)
    /// and WebGPU now runs the kernel bit-exact. The guard below no longer
    /// filters it.
    /// </summary>
    private static bool Is8x8KernelSupported(Accelerator acc)
    {
        var name = acc.AcceleratorType.ToString();
        return !name.Equals("WebGL", StringComparison.OrdinalIgnoreCase);
    }

    [TestMethod]
    public async Task Vp9Idct8x8Kernel_ZeroCoefficients_LeavesPredictorUnchanged()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            if (!Is8x8KernelSupported(acc)) return; // tracked upstream
            using var kernel = new Vp9Idct8x8Kernel(acc);
            var coeffs = new short[64];
            var dest = new byte[64];
            for (int i = 0; i < 64; i++) dest[i] = 128;
            await kernel.RunAsync(coeffs.AsMemory(), dest.AsMemory(), blockCount: 1);
            for (int i = 0; i < 64; i++) Equal((byte)128, dest[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9Idct8x8Kernel_DcOnly_MatchesReference()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            if (!Is8x8KernelSupported(acc)) return; // tracked upstream
            using var kernel = new Vp9Idct8x8Kernel(acc);
            var coeffs = new short[64];
            coeffs[0] = 1024;
            var cpuDest = new byte[64];
            var gpuDest = new byte[64];
            for (int i = 0; i < 64; i++) { cpuDest[i] = 100; gpuDest[i] = 100; }

            Vp9Idct8x8Reference.Idct8x8_64_Add(coeffs, cpuDest, 8);
            await kernel.RunAsync(coeffs.AsMemory(), gpuDest.AsMemory(), blockCount: 1);

            True(cpuDest.AsSpan().SequenceEqual(gpuDest),
                $"8x8 kernel DC-only must match reference on {acc.AcceleratorType}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9Idct8x8Kernel_RandomInputs_BitExactMatchReference()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            if (!Is8x8KernelSupported(acc)) return; // tracked upstream
            using var kernel = new Vp9Idct8x8Kernel(acc);
            var rng = new Random(unchecked((int)0xDEADBEEFu));
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

                Vp9Idct8x8Reference.Idct8x8_64_Add(coeffs, cpuDest, 8);
                await kernel.RunAsync(coeffs.AsMemory(), gpuDest.AsMemory(), blockCount: 1);

                for (int i = 0; i < 64; i++)
                    Equal(cpuDest[i], gpuDest[i]);
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9Idct8x8Kernel_BatchedDispatch_AllBlocksMatchReference()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            if (!Is8x8KernelSupported(acc)) return; // tracked upstream
            using var kernel = new Vp9Idct8x8Kernel(acc);
            const int n = 16;
            var rng = new Random(unchecked((int)0xFEEDFACEu));
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
                Vp9Idct8x8Reference.Idct8x8_64_Add(
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
