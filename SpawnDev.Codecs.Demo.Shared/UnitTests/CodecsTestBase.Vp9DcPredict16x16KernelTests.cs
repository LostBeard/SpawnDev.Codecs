// Tests for Vp9DcPredict16x16Kernel (slice 178). 16x16 sibling -
// same structure as the 4x4 / 8x8 kernel tests.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public async Task Vp9DcPredict16x16Kernel_BothEdges_MatchesCpuReference()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var kernel = new Vp9DcPredict16x16Kernel(acc);
            var above = new byte[16];
            var left = new byte[16];
            for (int i = 0; i < 16; i++) { above[i] = (byte)(i * 5); left[i] = (byte)(i * 7); }

            var cpuDst = new byte[256];
            Vp9DcPredictor.DcPredict(above, left, cpuDst, 16, 16);

            var gpuDst = new byte[256];
            await kernel.RunAsync(above, left, gpuDst, Vp9DcVariant.Both, blockCount: 1);

            for (int i = 0; i < 256; i++) Equal(cpuDst[i], gpuDst[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9DcPredict16x16Kernel_TopOnly_MatchesCpuReference()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var kernel = new Vp9DcPredict16x16Kernel(acc);
            var above = new byte[16];
            for (int i = 0; i < 16; i++) above[i] = 64;
            var left = new byte[16];

            var cpuDst = new byte[256];
            Vp9DcPredictor.DcPredictTop(above, cpuDst, 16, 16);

            var gpuDst = new byte[256];
            await kernel.RunAsync(above, left, gpuDst, Vp9DcVariant.TopOnly, blockCount: 1);

            for (int i = 0; i < 256; i++) Equal(cpuDst[i], gpuDst[i]);
            for (int i = 0; i < 256; i++) Equal((byte)64, gpuDst[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9DcPredict16x16Kernel_LeftOnly_MatchesCpuReference()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var kernel = new Vp9DcPredict16x16Kernel(acc);
            var above = new byte[16];
            var left = new byte[16];
            for (int i = 0; i < 16; i++) left[i] = (byte)(20 + i * 3);

            var cpuDst = new byte[256];
            Vp9DcPredictor.DcPredictLeft(left, cpuDst, 16, 16);

            var gpuDst = new byte[256];
            await kernel.RunAsync(above, left, gpuDst, Vp9DcVariant.LeftOnly, blockCount: 1);

            for (int i = 0; i < 256; i++) Equal(cpuDst[i], gpuDst[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9DcPredict16x16Kernel_None_FillsWith128()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var kernel = new Vp9DcPredict16x16Kernel(acc);
            var above = new byte[16];
            var left = new byte[16];
            var gpuDst = new byte[256];
            await kernel.RunAsync(above, left, gpuDst, Vp9DcVariant.None, blockCount: 1);
            for (int i = 0; i < 256; i++) Equal((byte)128, gpuDst[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9DcPredict16x16Kernel_RandomInputs_BitExact()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var kernel = new Vp9DcPredict16x16Kernel(acc);
            var rng = new Random(unchecked((int)0xCAFEBEEFu));
            for (int trial = 0; trial < 12; trial++)
            {
                var above = new byte[16];
                var left = new byte[16];
                for (int i = 0; i < 16; i++)
                {
                    above[i] = (byte)rng.Next(0, 256);
                    left[i] = (byte)rng.Next(0, 256);
                }

                var cpuDst = new byte[256];
                Vp9DcPredictor.DcPredict(above, left, cpuDst, 16, 16);

                var gpuDst = new byte[256];
                await kernel.RunAsync(above, left, gpuDst, Vp9DcVariant.Both, blockCount: 1);

                for (int i = 0; i < 256; i++) Equal(cpuDst[i], gpuDst[i]);
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9DcPredict16x16Kernel_BatchedDispatch_AllBlocksBitExact()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var kernel = new Vp9DcPredict16x16Kernel(acc);
            const int n = 16;
            var rng = new Random(unchecked((int)0xCAFEFEEDu));
            var aboveFlat = new byte[n * 16];
            var leftFlat = new byte[n * 16];
            for (int i = 0; i < aboveFlat.Length; i++) aboveFlat[i] = (byte)rng.Next(0, 256);
            for (int i = 0; i < leftFlat.Length; i++) leftFlat[i] = (byte)rng.Next(0, 256);

            var cpuFlat = new byte[n * 256];
            for (int b = 0; b < n; b++)
            {
                Vp9DcPredictor.DcPredict(
                    aboveFlat.AsSpan(b * 16, 16),
                    leftFlat.AsSpan(b * 16, 16),
                    cpuFlat.AsSpan(b * 256, 256),
                    16, 16);
            }

            var gpuFlat = new byte[n * 256];
            await kernel.RunAsync(aboveFlat, leftFlat, gpuFlat, Vp9DcVariant.Both, blockCount: n);

            for (int i = 0; i < cpuFlat.Length; i++) Equal(cpuFlat[i], gpuFlat[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
