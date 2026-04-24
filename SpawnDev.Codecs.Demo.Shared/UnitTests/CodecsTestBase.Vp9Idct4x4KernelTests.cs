// Tests for Vp9Idct4x4Kernel. Validates that the ILGPU kernel produces
// byte-for-byte identical output to Vp9Idct4x4Reference across a wide
// range of random coefficient inputs. VP9 is a normative bitstream -
// any divergence between the reference and the GPU kernel would make
// the decoder produce visibly wrong pixels on that backend.
//
// Each runner (CpuCodecsTests, CudaCodecsTests, OpenCLCodecsTests,
// WebGPUCodecsTests, WebGLCodecsTests, WasmCodecsTests) overrides
// CreateKernelAcceleratorAsync() to return its native accelerator.
// The same kernel code therefore runs on all 6 backends through
// PlaywrightMultiTest, and bit-exact agreement is enforced on each.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public async Task Vp9Idct4x4Kernel_ZeroCoefficients_LeavesPredictorUnchanged()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var kernel = new Vp9Idct4x4Kernel(acc);
            var coeffs = new short[16];
            var dest = new byte[16];
            for (int i = 0; i < 16; i++) dest[i] = 128;
            await kernel.RunAsync(coeffs, dest, blockCount: 1);
            for (int i = 0; i < 16; i++) Equal((byte)128, dest[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9Idct4x4Kernel_DcOnly_MatchesReference()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var kernel = new Vp9Idct4x4Kernel(acc);
            var coeffs = new short[16];
            coeffs[0] = 1024;
            var cpuDest = new byte[16];
            var gpuDest = new byte[16];
            for (int i = 0; i < 16; i++) { cpuDest[i] = 100; gpuDest[i] = 100; }

            Vp9Idct4x4Reference.Idct4x4_16_Add(coeffs, cpuDest, 4);
            await kernel.RunAsync(coeffs.AsMemory(), gpuDest.AsMemory(), blockCount: 1);

            True(cpuDest.AsSpan().SequenceEqual(gpuDest),
                $"kernel DC-only output must match CPU reference on {acc.AcceleratorType}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9Idct4x4Kernel_RandomInputs_BitExactMatchReference()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var kernel = new Vp9Idct4x4Kernel(acc);
            var rng = new Random(unchecked((int)0xBADDC0DEu));
            for (int trial = 0; trial < 25; trial++)
            {
                var coeffs = new short[16];
                for (int i = 0; i < 16; i++)
                    coeffs[i] = (short)rng.Next(-2048, 2048);
                var cpuDest = new byte[16];
                var gpuDest = new byte[16];
                for (int i = 0; i < 16; i++)
                {
                    byte p = (byte)rng.Next(0, 256);
                    cpuDest[i] = p;
                    gpuDest[i] = p;
                }

                Vp9Idct4x4Reference.Idct4x4_16_Add(coeffs, cpuDest, 4);
                await kernel.RunAsync(coeffs.AsMemory(), gpuDest.AsMemory(), blockCount: 1);

                for (int i = 0; i < 16; i++)
                {
                    Equal(cpuDest[i], gpuDest[i]);
                }
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9Idct4x4Kernel_BatchedDispatch_AllBlocksMatchReference()
    {
        // THE flex: N=64 independent blocks through one kernel dispatch
        // on this runner's native accelerator. Every output byte must
        // match the single-block CPU reference.
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var kernel = new Vp9Idct4x4Kernel(acc);
            const int n = 64;
            var rng = new Random(unchecked((int)0xBEEFCAFEu));
            var coeffsFlat = new short[n * 16];
            var predFlat = new byte[n * 16];
            for (int b = 0; b < n; b++)
            {
                for (int i = 0; i < 16; i++)
                    coeffsFlat[b * 16 + i] = (short)rng.Next(-2048, 2048);
                for (int i = 0; i < 16; i++)
                    predFlat[b * 16 + i] = (byte)rng.Next(0, 256);
            }

            // CPU reference: process each block.
            var cpuResults = (byte[])predFlat.Clone();
            for (int b = 0; b < n; b++)
            {
                Vp9Idct4x4Reference.Idct4x4_16_Add(
                    coeffsFlat.AsSpan(b * 16, 16),
                    cpuResults.AsSpan(b * 16, 16),
                    4);
            }

            // GPU batched: one dispatch.
            var gpuResults = (byte[])predFlat.Clone();
            await kernel.RunAsync(coeffsFlat.AsMemory(), gpuResults.AsMemory(), blockCount: n);

            // Compare byte-for-byte.
            for (int i = 0; i < n * 16; i++)
            {
                Equal(cpuResults[i], gpuResults[i]);
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
