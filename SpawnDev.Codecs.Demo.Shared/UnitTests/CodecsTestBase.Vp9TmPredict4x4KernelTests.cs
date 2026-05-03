// Tests for Vp9TmPredict4x4Kernel (slice 183).

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public async Task Vp9TmPredict4x4Kernel_KnownPattern_MatchesCpuReference()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Vp9TmPredict4x4Kernel(acc);
            byte[] above = { 10, 20, 30, 40 };
            byte[] left = { 50, 60, 70, 80 };
            byte topLeft = 5;

            var cpuDst = new byte[16];
            Vp9TmPredictor.TmPredict(topLeft, above, left, cpuDst, 4, 4);

            var gpuDst = new byte[16];
            await kernel.RunAsync(new[] { topLeft }, above, left, gpuDst, blockCount: 1);

            for (int i = 0; i < 16; i++) Equal(cpuDst[i], gpuDst[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9TmPredict4x4Kernel_ClipsAtPixelRange()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Vp9TmPredict4x4Kernel(acc);
            // Force overflow: above = 255, left = 255, topLeft = 0 -> 510 -> clip to 255.
            byte[] above = { 255, 255, 255, 255 };
            byte[] left = { 255, 255, 255, 255 };
            byte topLeft = 0;

            var gpuDst = new byte[16];
            await kernel.RunAsync(new[] { topLeft }, above, left, gpuDst, blockCount: 1);
            for (int i = 0; i < 16; i++) Equal((byte)255, gpuDst[i]);

            // Force underflow: above = 0, left = 0, topLeft = 200 -> -200 -> clip to 0.
            byte[] aboveZero = { 0, 0, 0, 0 };
            byte[] leftZero = { 0, 0, 0, 0 };
            byte topLeftLarge = 200;

            var gpuDst2 = new byte[16];
            await kernel.RunAsync(new[] { topLeftLarge }, aboveZero, leftZero, gpuDst2, blockCount: 1);
            for (int i = 0; i < 16; i++) Equal((byte)0, gpuDst2[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9TmPredict4x4Kernel_RandomInputs_BitExact()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Vp9TmPredict4x4Kernel(acc);
            var rng = new Random(unchecked((int)0xCAFE0183u));
            for (int trial = 0; trial < 16; trial++)
            {
                var above = new byte[4];
                var left = new byte[4];
                for (int i = 0; i < 4; i++)
                {
                    above[i] = (byte)rng.Next(0, 256);
                    left[i] = (byte)rng.Next(0, 256);
                }
                byte topLeft = (byte)rng.Next(0, 256);

                var cpuDst = new byte[16];
                Vp9TmPredictor.TmPredict(topLeft, above, left, cpuDst, 4, 4);

                var gpuDst = new byte[16];
                await kernel.RunAsync(new[] { topLeft }, above, left, gpuDst, blockCount: 1);

                for (int i = 0; i < 16; i++) Equal(cpuDst[i], gpuDst[i]);
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9TmPredict4x4Kernel_BatchedDispatch_AllBlocksBitExact()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Vp9TmPredict4x4Kernel(acc);
            const int n = 32;
            var rng = new Random(unchecked((int)0xCAFE1830u));
            var topLeftFlat = new byte[n];
            var aboveFlat = new byte[n * 4];
            var leftFlat = new byte[n * 4];
            for (int i = 0; i < n; i++) topLeftFlat[i] = (byte)rng.Next(0, 256);
            for (int i = 0; i < aboveFlat.Length; i++) aboveFlat[i] = (byte)rng.Next(0, 256);
            for (int i = 0; i < leftFlat.Length; i++) leftFlat[i] = (byte)rng.Next(0, 256);

            var cpuFlat = new byte[n * 16];
            for (int b = 0; b < n; b++)
            {
                Vp9TmPredictor.TmPredict(
                    topLeftFlat[b],
                    aboveFlat.AsSpan(b * 4, 4),
                    leftFlat.AsSpan(b * 4, 4),
                    cpuFlat.AsSpan(b * 16, 16),
                    4, 4);
            }

            var gpuFlat = new byte[n * 16];
            await kernel.RunAsync(topLeftFlat, aboveFlat, leftFlat, gpuFlat, blockCount: n);

            for (int i = 0; i < cpuFlat.Length; i++) Equal(cpuFlat[i], gpuFlat[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
