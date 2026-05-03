// Tests for Vp9D153Predict8x8Kernel (slice 200). Closes the 10-mode
// 8x8 GPU intra prediction set.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public async Task Vp9D153Predict8x8Kernel_KnownPattern_MatchesCpuReference()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Vp9D153Predict8x8Kernel(acc);
            var above = new byte[8];
            var left = new byte[8];
            for (int i = 0; i < 8; i++) { above[i] = (byte)(20 + i * 5); left[i] = (byte)(60 + i * 7); }
            byte topLeft = 30;

            var cpuDst = new byte[64];
            Vp9DirectionalPredictor.D153Predict(topLeft, above, left, cpuDst, 8, 8);

            var gpuDst = new byte[64];
            await kernel.RunAsync(new[] { topLeft }, above, left, gpuDst, blockCount: 1);

            for (int i = 0; i < 64; i++) Equal(cpuDst[i], gpuDst[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9D153Predict8x8Kernel_RandomInputs_BitExact()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Vp9D153Predict8x8Kernel(acc);
            var rng = new Random(unchecked((int)0xCAFE0200u));
            for (int trial = 0; trial < 12; trial++)
            {
                var above = new byte[8];
                var left = new byte[8];
                for (int i = 0; i < 8; i++)
                {
                    above[i] = (byte)rng.Next(0, 256);
                    left[i] = (byte)rng.Next(0, 256);
                }
                byte topLeft = (byte)rng.Next(0, 256);

                var cpuDst = new byte[64];
                Vp9DirectionalPredictor.D153Predict(topLeft, above, left, cpuDst, 8, 8);

                var gpuDst = new byte[64];
                await kernel.RunAsync(new[] { topLeft }, above, left, gpuDst, blockCount: 1);

                for (int i = 0; i < 64; i++) Equal(cpuDst[i], gpuDst[i]);
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9D153Predict8x8Kernel_BatchedDispatch_AllBlocksBitExact()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Vp9D153Predict8x8Kernel(acc);
            const int n = 8;
            var rng = new Random(unchecked((int)0xCAFE2000u));
            var topLeftFlat = new byte[n];
            var aboveFlat = new byte[n * 8];
            var leftFlat = new byte[n * 8];
            for (int i = 0; i < n; i++) topLeftFlat[i] = (byte)rng.Next(0, 256);
            for (int i = 0; i < aboveFlat.Length; i++) aboveFlat[i] = (byte)rng.Next(0, 256);
            for (int i = 0; i < leftFlat.Length; i++) leftFlat[i] = (byte)rng.Next(0, 256);

            var cpuFlat = new byte[n * 64];
            for (int b = 0; b < n; b++)
            {
                Vp9DirectionalPredictor.D153Predict(
                    topLeftFlat[b],
                    aboveFlat.AsSpan(b * 8, 8),
                    leftFlat.AsSpan(b * 8, 8),
                    cpuFlat.AsSpan(b * 64, 64),
                    8, 8);
            }

            var gpuFlat = new byte[n * 64];
            await kernel.RunAsync(topLeftFlat, aboveFlat, leftFlat, gpuFlat, blockCount: n);

            for (int i = 0; i < cpuFlat.Length; i++) Equal(cpuFlat[i], gpuFlat[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
