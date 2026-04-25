// Tests for Vp9DcPredict4x4Kernel (slice 176). Validates that the
// ILGPU kernel produces byte-for-byte identical output to
// Vp9DcPredictor across all four DC variants (full / top-only /
// left-only / 128) and on every backend.
//
// Each runner overrides CreateKernelAcceleratorAsync() so the same
// kernel runs on CPU emulator, CUDA, OpenCL, WebGPU, WebGL, and
// Wasm. Bit-exact agreement is enforced on each backend.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public async Task Vp9DcPredict4x4Kernel_BothEdges_MatchesCpuReference()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var kernel = new Vp9DcPredict4x4Kernel(acc);
            byte[] above = { 10, 20, 30, 40 };
            byte[] left = { 50, 60, 70, 80 };

            var cpuDst = new byte[16];
            Vp9DcPredictor.DcPredict(above, left, cpuDst, 4, 4);

            var gpuDst = new byte[16];
            await kernel.RunAsync(above, left, gpuDst, Vp9DcVariant.Both, blockCount: 1);

            for (int i = 0; i < 16; i++) Equal(cpuDst[i], gpuDst[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9DcPredict4x4Kernel_TopOnly_MatchesCpuReference()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var kernel = new Vp9DcPredict4x4Kernel(acc);
            byte[] above = { 100, 100, 100, 100 };
            byte[] left = { 0, 0, 0, 0 };

            var cpuDst = new byte[16];
            Vp9DcPredictor.DcPredictTop(above, cpuDst, 4, 4);

            var gpuDst = new byte[16];
            await kernel.RunAsync(above, left, gpuDst, Vp9DcVariant.TopOnly, blockCount: 1);

            for (int i = 0; i < 16; i++) Equal(cpuDst[i], gpuDst[i]);
            // DC for flat above=100 is 100.
            for (int i = 0; i < 16; i++) Equal((byte)100, gpuDst[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9DcPredict4x4Kernel_LeftOnly_MatchesCpuReference()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var kernel = new Vp9DcPredict4x4Kernel(acc);
            byte[] above = { 0, 0, 0, 0 };
            byte[] left = { 50, 60, 70, 80 };

            var cpuDst = new byte[16];
            Vp9DcPredictor.DcPredictLeft(left, cpuDst, 4, 4);

            var gpuDst = new byte[16];
            await kernel.RunAsync(above, left, gpuDst, Vp9DcVariant.LeftOnly, blockCount: 1);

            for (int i = 0; i < 16; i++) Equal(cpuDst[i], gpuDst[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9DcPredict4x4Kernel_None_FillsWith128()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var kernel = new Vp9DcPredict4x4Kernel(acc);
            byte[] above = new byte[4];
            byte[] left = new byte[4];

            var gpuDst = new byte[16];
            await kernel.RunAsync(above, left, gpuDst, Vp9DcVariant.None, blockCount: 1);

            for (int i = 0; i < 16; i++) Equal((byte)128, gpuDst[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9DcPredict4x4Kernel_RandomInputs_BitExact()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var kernel = new Vp9DcPredict4x4Kernel(acc);
            var rng = new Random(unchecked((int)0xCAFE1234u));
            for (int trial = 0; trial < 20; trial++)
            {
                var above = new byte[4];
                var left = new byte[4];
                for (int i = 0; i < 4; i++)
                {
                    above[i] = (byte)rng.Next(0, 256);
                    left[i] = (byte)rng.Next(0, 256);
                }

                var cpuDst = new byte[16];
                Vp9DcPredictor.DcPredict(above, left, cpuDst, 4, 4);

                var gpuDst = new byte[16];
                await kernel.RunAsync(above, left, gpuDst, Vp9DcVariant.Both, blockCount: 1);

                for (int i = 0; i < 16; i++) Equal(cpuDst[i], gpuDst[i]);
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9DcPredict4x4Kernel_BatchedDispatch_AllBlocksBitExact()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var kernel = new Vp9DcPredict4x4Kernel(acc);
            const int n = 64;
            var rng = new Random(unchecked((int)0xBEEFFEEDu));
            var aboveFlat = new byte[n * 4];
            var leftFlat = new byte[n * 4];
            for (int i = 0; i < aboveFlat.Length; i++) aboveFlat[i] = (byte)rng.Next(0, 256);
            for (int i = 0; i < leftFlat.Length; i++) leftFlat[i] = (byte)rng.Next(0, 256);

            // CPU reference per block.
            var cpuFlat = new byte[n * 16];
            for (int b = 0; b < n; b++)
            {
                Vp9DcPredictor.DcPredict(
                    aboveFlat.AsSpan(b * 4, 4),
                    leftFlat.AsSpan(b * 4, 4),
                    cpuFlat.AsSpan(b * 16, 16),
                    4, 4);
            }

            // GPU batched.
            var gpuFlat = new byte[n * 16];
            await kernel.RunAsync(aboveFlat, leftFlat, gpuFlat, Vp9DcVariant.Both, blockCount: n);

            for (int i = 0; i < cpuFlat.Length; i++) Equal(cpuFlat[i], gpuFlat[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
