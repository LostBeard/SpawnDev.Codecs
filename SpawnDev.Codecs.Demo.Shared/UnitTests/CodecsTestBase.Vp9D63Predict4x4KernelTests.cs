// Tests for Vp9D63Predict4x4Kernel (slice 190).

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public async Task Vp9D63Predict4x4Kernel_KnownPattern_MatchesCpuReference()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Vp9D63Predict4x4Kernel(acc);
            byte[] above = { 10, 20, 30, 40, 50, 60, 70, 80 };

            var cpuDst = new byte[16];
            Vp9DirectionalPredictor.D63Predict(above, cpuDst, 4, 4);

            var gpuDst = new byte[16];
            await kernel.RunAsync(above, gpuDst, blockCount: 1);

            for (int i = 0; i < 16; i++) Equal(cpuDst[i], gpuDst[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9D63Predict4x4Kernel_FlatInput_MatchesCpuReference()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Vp9D63Predict4x4Kernel(acc);
            var above = new byte[8];
            for (int i = 0; i < 8; i++) above[i] = 200;

            var gpuDst = new byte[16];
            await kernel.RunAsync(above, gpuDst, blockCount: 1);

            for (int i = 0; i < 16; i++) Equal((byte)200, gpuDst[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9D63Predict4x4Kernel_RandomInputs_BitExact()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Vp9D63Predict4x4Kernel(acc);
            var rng = new Random(unchecked((int)0xCAFE0190u));
            for (int trial = 0; trial < 16; trial++)
            {
                var above = new byte[8];
                for (int i = 0; i < 8; i++) above[i] = (byte)rng.Next(0, 256);

                var cpuDst = new byte[16];
                Vp9DirectionalPredictor.D63Predict(above, cpuDst, 4, 4);

                var gpuDst = new byte[16];
                await kernel.RunAsync(above, gpuDst, blockCount: 1);

                for (int i = 0; i < 16; i++) Equal(cpuDst[i], gpuDst[i]);
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9D63Predict4x4Kernel_BatchedDispatch_AllBlocksBitExact()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Vp9D63Predict4x4Kernel(acc);
            const int n = 16;
            var rng = new Random(unchecked((int)0xCAFE1900u));
            var aboveFlat = new byte[n * 8];
            for (int i = 0; i < aboveFlat.Length; i++) aboveFlat[i] = (byte)rng.Next(0, 256);

            var cpuFlat = new byte[n * 16];
            for (int b = 0; b < n; b++)
            {
                Vp9DirectionalPredictor.D63Predict(
                    aboveFlat.AsSpan(b * 8, 8),
                    cpuFlat.AsSpan(b * 16, 16),
                    4, 4);
            }

            var gpuFlat = new byte[n * 16];
            await kernel.RunAsync(aboveFlat, gpuFlat, blockCount: n);

            for (int i = 0; i < cpuFlat.Length; i++) Equal(cpuFlat[i], gpuFlat[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
