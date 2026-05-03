// Tests for Vp9D63Predict16x16Kernel (slice 202).

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public async Task Vp9D63Predict16x16Kernel_KnownPattern_MatchesCpuReference()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Vp9D63Predict16x16Kernel(acc);
            var above = new byte[32];
            for (int i = 0; i < 32; i++) above[i] = (byte)(8 + i * 4);

            var cpuDst = new byte[256];
            Vp9DirectionalPredictor.D63Predict(above, cpuDst, 16, 16);

            var gpuDst = new byte[256];
            await kernel.RunAsync(above, gpuDst, blockCount: 1);

            for (int i = 0; i < 256; i++) Equal(cpuDst[i], gpuDst[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9D63Predict16x16Kernel_RandomInputs_BitExact()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Vp9D63Predict16x16Kernel(acc);
            var rng = new Random(unchecked((int)0xCAFE0202u));
            for (int trial = 0; trial < 8; trial++)
            {
                var above = new byte[32];
                for (int i = 0; i < 32; i++) above[i] = (byte)rng.Next(0, 256);

                var cpuDst = new byte[256];
                Vp9DirectionalPredictor.D63Predict(above, cpuDst, 16, 16);

                var gpuDst = new byte[256];
                await kernel.RunAsync(above, gpuDst, blockCount: 1);

                for (int i = 0; i < 256; i++) Equal(cpuDst[i], gpuDst[i]);
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9D63Predict16x16Kernel_BatchedDispatch_AllBlocksBitExact()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Vp9D63Predict16x16Kernel(acc);
            const int n = 4;
            var rng = new Random(unchecked((int)0xCAFE2020u));
            var aboveFlat = new byte[n * 32];
            for (int i = 0; i < aboveFlat.Length; i++) aboveFlat[i] = (byte)rng.Next(0, 256);

            var cpuFlat = new byte[n * 256];
            for (int b = 0; b < n; b++)
            {
                Vp9DirectionalPredictor.D63Predict(
                    aboveFlat.AsSpan(b * 32, 32),
                    cpuFlat.AsSpan(b * 256, 256),
                    16, 16);
            }

            var gpuFlat = new byte[n * 256];
            await kernel.RunAsync(aboveFlat, gpuFlat, blockCount: n);

            for (int i = 0; i < cpuFlat.Length; i++) Equal(cpuFlat[i], gpuFlat[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
