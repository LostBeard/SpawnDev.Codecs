// Tests for Vp9D207Predict4x4Kernel (slice 191).

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public async Task Vp9D207Predict4x4Kernel_KnownPattern_MatchesCpuReference()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var kernel = new Vp9D207Predict4x4Kernel(acc);
            byte[] left = { 50, 60, 70, 80 };

            var cpuDst = new byte[16];
            Vp9DirectionalPredictor.D207Predict(left, cpuDst, 4, 4);

            var gpuDst = new byte[16];
            await kernel.RunAsync(left, gpuDst, blockCount: 1);

            for (int i = 0; i < 16; i++) Equal(cpuDst[i], gpuDst[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9D207Predict4x4Kernel_FlatInput_MatchesCpuReference()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var kernel = new Vp9D207Predict4x4Kernel(acc);
            var left = new byte[4];
            for (int i = 0; i < 4; i++) left[i] = 150;

            var gpuDst = new byte[16];
            await kernel.RunAsync(left, gpuDst, blockCount: 1);

            for (int i = 0; i < 16; i++) Equal((byte)150, gpuDst[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9D207Predict4x4Kernel_RandomInputs_BitExact()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var kernel = new Vp9D207Predict4x4Kernel(acc);
            var rng = new Random(unchecked((int)0xCAFE0191u));
            for (int trial = 0; trial < 16; trial++)
            {
                var left = new byte[4];
                for (int i = 0; i < 4; i++) left[i] = (byte)rng.Next(0, 256);

                var cpuDst = new byte[16];
                Vp9DirectionalPredictor.D207Predict(left, cpuDst, 4, 4);

                var gpuDst = new byte[16];
                await kernel.RunAsync(left, gpuDst, blockCount: 1);

                for (int i = 0; i < 16; i++) Equal(cpuDst[i], gpuDst[i]);
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9D207Predict4x4Kernel_BatchedDispatch_AllBlocksBitExact()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var kernel = new Vp9D207Predict4x4Kernel(acc);
            const int n = 16;
            var rng = new Random(unchecked((int)0xCAFE1910u));
            var leftFlat = new byte[n * 4];
            for (int i = 0; i < leftFlat.Length; i++) leftFlat[i] = (byte)rng.Next(0, 256);

            var cpuFlat = new byte[n * 16];
            for (int b = 0; b < n; b++)
            {
                Vp9DirectionalPredictor.D207Predict(
                    leftFlat.AsSpan(b * 4, 4),
                    cpuFlat.AsSpan(b * 16, 16),
                    4, 4);
            }

            var gpuFlat = new byte[n * 16];
            await kernel.RunAsync(leftFlat, gpuFlat, blockCount: n);

            for (int i = 0; i < cpuFlat.Length; i++) Equal(cpuFlat[i], gpuFlat[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
