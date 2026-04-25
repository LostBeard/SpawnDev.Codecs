// Tests for Vp9DcPredict8x8Kernel (slice 177). 8x8 sibling of the
// 4x4 kernel tests - same structure, wider edge buffers.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public async Task Vp9DcPredict8x8Kernel_BothEdges_MatchesCpuReference()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var kernel = new Vp9DcPredict8x8Kernel(acc);
            var above = new byte[8];
            var left = new byte[8];
            for (int i = 0; i < 8; i++) { above[i] = (byte)(10 + i * 7); left[i] = (byte)(20 + i * 11); }

            var cpuDst = new byte[64];
            Vp9DcPredictor.DcPredict(above, left, cpuDst, 8, 8);

            var gpuDst = new byte[64];
            await kernel.RunAsync(above, left, gpuDst, Vp9DcVariant.Both, blockCount: 1);

            for (int i = 0; i < 64; i++) Equal(cpuDst[i], gpuDst[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9DcPredict8x8Kernel_TopOnly_MatchesCpuReference()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var kernel = new Vp9DcPredict8x8Kernel(acc);
            var above = new byte[8];
            for (int i = 0; i < 8; i++) above[i] = 200;
            var left = new byte[8];

            var cpuDst = new byte[64];
            Vp9DcPredictor.DcPredictTop(above, cpuDst, 8, 8);

            var gpuDst = new byte[64];
            await kernel.RunAsync(above, left, gpuDst, Vp9DcVariant.TopOnly, blockCount: 1);

            for (int i = 0; i < 64; i++) Equal(cpuDst[i], gpuDst[i]);
            for (int i = 0; i < 64; i++) Equal((byte)200, gpuDst[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9DcPredict8x8Kernel_LeftOnly_MatchesCpuReference()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var kernel = new Vp9DcPredict8x8Kernel(acc);
            var above = new byte[8];
            var left = new byte[8];
            for (int i = 0; i < 8; i++) left[i] = (byte)(40 + i * 5);

            var cpuDst = new byte[64];
            Vp9DcPredictor.DcPredictLeft(left, cpuDst, 8, 8);

            var gpuDst = new byte[64];
            await kernel.RunAsync(above, left, gpuDst, Vp9DcVariant.LeftOnly, blockCount: 1);

            for (int i = 0; i < 64; i++) Equal(cpuDst[i], gpuDst[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9DcPredict8x8Kernel_None_FillsWith128()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var kernel = new Vp9DcPredict8x8Kernel(acc);
            var above = new byte[8];
            var left = new byte[8];
            var gpuDst = new byte[64];
            await kernel.RunAsync(above, left, gpuDst, Vp9DcVariant.None, blockCount: 1);
            for (int i = 0; i < 64; i++) Equal((byte)128, gpuDst[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9DcPredict8x8Kernel_RandomInputs_BitExact()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var kernel = new Vp9DcPredict8x8Kernel(acc);
            var rng = new Random(unchecked((int)0xCAFE5678u));
            for (int trial = 0; trial < 16; trial++)
            {
                var above = new byte[8];
                var left = new byte[8];
                for (int i = 0; i < 8; i++)
                {
                    above[i] = (byte)rng.Next(0, 256);
                    left[i] = (byte)rng.Next(0, 256);
                }

                var cpuDst = new byte[64];
                Vp9DcPredictor.DcPredict(above, left, cpuDst, 8, 8);

                var gpuDst = new byte[64];
                await kernel.RunAsync(above, left, gpuDst, Vp9DcVariant.Both, blockCount: 1);

                for (int i = 0; i < 64; i++) Equal(cpuDst[i], gpuDst[i]);
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9DcPredict8x8Kernel_BatchedDispatch_AllBlocksBitExact()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var kernel = new Vp9DcPredict8x8Kernel(acc);
            const int n = 32;
            var rng = new Random(unchecked((int)0xBEEFCAFEu));
            var aboveFlat = new byte[n * 8];
            var leftFlat = new byte[n * 8];
            for (int i = 0; i < aboveFlat.Length; i++) aboveFlat[i] = (byte)rng.Next(0, 256);
            for (int i = 0; i < leftFlat.Length; i++) leftFlat[i] = (byte)rng.Next(0, 256);

            var cpuFlat = new byte[n * 64];
            for (int b = 0; b < n; b++)
            {
                Vp9DcPredictor.DcPredict(
                    aboveFlat.AsSpan(b * 8, 8),
                    leftFlat.AsSpan(b * 8, 8),
                    cpuFlat.AsSpan(b * 64, 64),
                    8, 8);
            }

            var gpuFlat = new byte[n * 64];
            await kernel.RunAsync(aboveFlat, leftFlat, gpuFlat, Vp9DcVariant.Both, blockCount: n);

            for (int i = 0; i < cpuFlat.Length; i++) Equal(cpuFlat[i], gpuFlat[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
