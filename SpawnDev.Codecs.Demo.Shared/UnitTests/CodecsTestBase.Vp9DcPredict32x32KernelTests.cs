// Tests for Vp9DcPredict32x32Kernel (slice 179). 32x32 sibling of
// the 4x4 / 8x8 / 16x16 kernel tests.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public async Task Vp9DcPredict32x32Kernel_BothEdges_MatchesCpuReference()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Vp9DcPredict32x32Kernel(acc);
            var above = new byte[32];
            var left = new byte[32];
            for (int i = 0; i < 32; i++) { above[i] = (byte)i; left[i] = (byte)(i * 2); }

            var cpuDst = new byte[1024];
            Vp9DcPredictor.DcPredict(above, left, cpuDst, 32, 32);

            var gpuDst = new byte[1024];
            await kernel.RunAsync(above, left, gpuDst, Vp9DcVariant.Both, blockCount: 1);

            for (int i = 0; i < 1024; i++) Equal(cpuDst[i], gpuDst[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9DcPredict32x32Kernel_TopOnly_MatchesCpuReference()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Vp9DcPredict32x32Kernel(acc);
            var above = new byte[32];
            for (int i = 0; i < 32; i++) above[i] = 50;
            var left = new byte[32];

            var cpuDst = new byte[1024];
            Vp9DcPredictor.DcPredictTop(above, cpuDst, 32, 32);

            var gpuDst = new byte[1024];
            await kernel.RunAsync(above, left, gpuDst, Vp9DcVariant.TopOnly, blockCount: 1);

            for (int i = 0; i < 1024; i++) Equal(cpuDst[i], gpuDst[i]);
            for (int i = 0; i < 1024; i++) Equal((byte)50, gpuDst[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9DcPredict32x32Kernel_LeftOnly_MatchesCpuReference()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Vp9DcPredict32x32Kernel(acc);
            var above = new byte[32];
            var left = new byte[32];
            for (int i = 0; i < 32; i++) left[i] = (byte)(100 + i);

            var cpuDst = new byte[1024];
            Vp9DcPredictor.DcPredictLeft(left, cpuDst, 32, 32);

            var gpuDst = new byte[1024];
            await kernel.RunAsync(above, left, gpuDst, Vp9DcVariant.LeftOnly, blockCount: 1);

            for (int i = 0; i < 1024; i++) Equal(cpuDst[i], gpuDst[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9DcPredict32x32Kernel_None_FillsWith128()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Vp9DcPredict32x32Kernel(acc);
            var above = new byte[32];
            var left = new byte[32];
            var gpuDst = new byte[1024];
            await kernel.RunAsync(above, left, gpuDst, Vp9DcVariant.None, blockCount: 1);
            for (int i = 0; i < 1024; i++) Equal((byte)128, gpuDst[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9DcPredict32x32Kernel_RandomInputs_BitExact()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Vp9DcPredict32x32Kernel(acc);
            var rng = new Random(unchecked((int)0xCAFEABCDu));
            for (int trial = 0; trial < 8; trial++)
            {
                var above = new byte[32];
                var left = new byte[32];
                for (int i = 0; i < 32; i++)
                {
                    above[i] = (byte)rng.Next(0, 256);
                    left[i] = (byte)rng.Next(0, 256);
                }

                var cpuDst = new byte[1024];
                Vp9DcPredictor.DcPredict(above, left, cpuDst, 32, 32);

                var gpuDst = new byte[1024];
                await kernel.RunAsync(above, left, gpuDst, Vp9DcVariant.Both, blockCount: 1);

                for (int i = 0; i < 1024; i++) Equal(cpuDst[i], gpuDst[i]);
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9DcPredict32x32Kernel_BatchedDispatch_AllBlocksBitExact()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Vp9DcPredict32x32Kernel(acc);
            const int n = 8;
            var rng = new Random(unchecked((int)0xCAFE9999u));
            var aboveFlat = new byte[n * 32];
            var leftFlat = new byte[n * 32];
            for (int i = 0; i < aboveFlat.Length; i++) aboveFlat[i] = (byte)rng.Next(0, 256);
            for (int i = 0; i < leftFlat.Length; i++) leftFlat[i] = (byte)rng.Next(0, 256);

            var cpuFlat = new byte[n * 1024];
            for (int b = 0; b < n; b++)
            {
                Vp9DcPredictor.DcPredict(
                    aboveFlat.AsSpan(b * 32, 32),
                    leftFlat.AsSpan(b * 32, 32),
                    cpuFlat.AsSpan(b * 1024, 1024),
                    32, 32);
            }

            var gpuFlat = new byte[n * 1024];
            await kernel.RunAsync(aboveFlat, leftFlat, gpuFlat, Vp9DcVariant.Both, blockCount: n);

            for (int i = 0; i < cpuFlat.Length; i++) Equal(cpuFlat[i], gpuFlat[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
