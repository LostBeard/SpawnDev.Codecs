// Tests for Vp9TmPredict16x16Kernel (slice 185).

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public async Task Vp9TmPredict16x16Kernel_KnownPattern_MatchesCpuReference()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var kernel = new Vp9TmPredict16x16Kernel(acc);
            var above = new byte[16];
            var left = new byte[16];
            for (int i = 0; i < 16; i++) { above[i] = (byte)(i * 11); left[i] = (byte)(i * 13); }
            byte topLeft = 50;

            var cpuDst = new byte[256];
            Vp9TmPredictor.TmPredict(topLeft, above, left, cpuDst, 16, 16);

            var gpuDst = new byte[256];
            await kernel.RunAsync(new[] { topLeft }, above, left, gpuDst, blockCount: 1);

            for (int i = 0; i < 256; i++) Equal(cpuDst[i], gpuDst[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9TmPredict16x16Kernel_RandomInputs_BitExact()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var kernel = new Vp9TmPredict16x16Kernel(acc);
            var rng = new Random(unchecked((int)0xCAFE0185u));
            for (int trial = 0; trial < 8; trial++)
            {
                var above = new byte[16];
                var left = new byte[16];
                for (int i = 0; i < 16; i++)
                {
                    above[i] = (byte)rng.Next(0, 256);
                    left[i] = (byte)rng.Next(0, 256);
                }
                byte topLeft = (byte)rng.Next(0, 256);

                var cpuDst = new byte[256];
                Vp9TmPredictor.TmPredict(topLeft, above, left, cpuDst, 16, 16);

                var gpuDst = new byte[256];
                await kernel.RunAsync(new[] { topLeft }, above, left, gpuDst, blockCount: 1);

                for (int i = 0; i < 256; i++) Equal(cpuDst[i], gpuDst[i]);
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9TmPredict16x16Kernel_BatchedDispatch_AllBlocksBitExact()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var kernel = new Vp9TmPredict16x16Kernel(acc);
            const int n = 8;
            var rng = new Random(unchecked((int)0xCAFE1850u));
            var topLeftFlat = new byte[n];
            var aboveFlat = new byte[n * 16];
            var leftFlat = new byte[n * 16];
            for (int i = 0; i < n; i++) topLeftFlat[i] = (byte)rng.Next(0, 256);
            for (int i = 0; i < aboveFlat.Length; i++) aboveFlat[i] = (byte)rng.Next(0, 256);
            for (int i = 0; i < leftFlat.Length; i++) leftFlat[i] = (byte)rng.Next(0, 256);

            var cpuFlat = new byte[n * 256];
            for (int b = 0; b < n; b++)
            {
                Vp9TmPredictor.TmPredict(
                    topLeftFlat[b],
                    aboveFlat.AsSpan(b * 16, 16),
                    leftFlat.AsSpan(b * 16, 16),
                    cpuFlat.AsSpan(b * 256, 256),
                    16, 16);
            }

            var gpuFlat = new byte[n * 256];
            await kernel.RunAsync(topLeftFlat, aboveFlat, leftFlat, gpuFlat, blockCount: n);

            for (int i = 0; i < cpuFlat.Length; i++) Equal(cpuFlat[i], gpuFlat[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
