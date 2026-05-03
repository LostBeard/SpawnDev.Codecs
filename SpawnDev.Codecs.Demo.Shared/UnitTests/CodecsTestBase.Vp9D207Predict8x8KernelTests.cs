// Tests for Vp9D207Predict8x8Kernel (slice 197).

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public async Task Vp9D207Predict8x8Kernel_KnownPattern_MatchesCpuReference()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Vp9D207Predict8x8Kernel(acc);
            var left = new byte[8];
            for (int i = 0; i < 8; i++) left[i] = (byte)(40 + i * 9);

            var cpuDst = new byte[64];
            Vp9DirectionalPredictor.D207Predict(left, cpuDst, 8, 8);

            var gpuDst = new byte[64];
            await kernel.RunAsync(left, gpuDst, blockCount: 1);

            for (int i = 0; i < 64; i++) Equal(cpuDst[i], gpuDst[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9D207Predict8x8Kernel_RandomInputs_BitExact()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Vp9D207Predict8x8Kernel(acc);
            var rng = new Random(unchecked((int)0xCAFE0197u));
            for (int trial = 0; trial < 12; trial++)
            {
                var left = new byte[8];
                for (int i = 0; i < 8; i++) left[i] = (byte)rng.Next(0, 256);

                var cpuDst = new byte[64];
                Vp9DirectionalPredictor.D207Predict(left, cpuDst, 8, 8);

                var gpuDst = new byte[64];
                await kernel.RunAsync(left, gpuDst, blockCount: 1);

                for (int i = 0; i < 64; i++) Equal(cpuDst[i], gpuDst[i]);
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9D207Predict8x8Kernel_BatchedDispatch_AllBlocksBitExact()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Vp9D207Predict8x8Kernel(acc);
            const int n = 8;
            var rng = new Random(unchecked((int)0xCAFE1970u));
            var leftFlat = new byte[n * 8];
            for (int i = 0; i < leftFlat.Length; i++) leftFlat[i] = (byte)rng.Next(0, 256);

            var cpuFlat = new byte[n * 64];
            for (int b = 0; b < n; b++)
            {
                Vp9DirectionalPredictor.D207Predict(
                    leftFlat.AsSpan(b * 8, 8),
                    cpuFlat.AsSpan(b * 64, 64),
                    8, 8);
            }

            var gpuFlat = new byte[n * 64];
            await kernel.RunAsync(leftFlat, gpuFlat, blockCount: n);

            for (int i = 0; i < cpuFlat.Length; i++) Equal(cpuFlat[i], gpuFlat[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
