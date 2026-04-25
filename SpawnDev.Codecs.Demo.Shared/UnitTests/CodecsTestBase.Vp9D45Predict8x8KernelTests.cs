// Tests for Vp9D45Predict8x8Kernel (slice 195).

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public async Task Vp9D45Predict8x8Kernel_KnownPattern_MatchesCpuReference()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var kernel = new Vp9D45Predict8x8Kernel(acc);
            var above = new byte[16];
            for (int i = 0; i < 16; i++) above[i] = (byte)(10 + i * 5);

            var cpuDst = new byte[64];
            Vp9DirectionalPredictor.D45Predict(above, cpuDst, 8, 8);

            var gpuDst = new byte[64];
            await kernel.RunAsync(above, gpuDst, blockCount: 1);

            for (int i = 0; i < 64; i++) Equal(cpuDst[i], gpuDst[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9D45Predict8x8Kernel_FlatInput_MatchesCpuReference()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var kernel = new Vp9D45Predict8x8Kernel(acc);
            var above = new byte[16];
            for (int i = 0; i < 16; i++) above[i] = 50;

            var gpuDst = new byte[64];
            await kernel.RunAsync(above, gpuDst, blockCount: 1);

            for (int i = 0; i < 64; i++) Equal((byte)50, gpuDst[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9D45Predict8x8Kernel_RandomInputs_BitExact()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var kernel = new Vp9D45Predict8x8Kernel(acc);
            var rng = new Random(unchecked((int)0xCAFE0195u));
            for (int trial = 0; trial < 12; trial++)
            {
                var above = new byte[16];
                for (int i = 0; i < 16; i++) above[i] = (byte)rng.Next(0, 256);

                var cpuDst = new byte[64];
                Vp9DirectionalPredictor.D45Predict(above, cpuDst, 8, 8);

                var gpuDst = new byte[64];
                await kernel.RunAsync(above, gpuDst, blockCount: 1);

                for (int i = 0; i < 64; i++) Equal(cpuDst[i], gpuDst[i]);
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9D45Predict8x8Kernel_BatchedDispatch_AllBlocksBitExact()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var kernel = new Vp9D45Predict8x8Kernel(acc);
            const int n = 8;
            var rng = new Random(unchecked((int)0xCAFE1950u));
            var aboveFlat = new byte[n * 16];  // 2N=16 per block
            for (int i = 0; i < aboveFlat.Length; i++) aboveFlat[i] = (byte)rng.Next(0, 256);

            var cpuFlat = new byte[n * 64];
            for (int b = 0; b < n; b++)
            {
                Vp9DirectionalPredictor.D45Predict(
                    aboveFlat.AsSpan(b * 16, 16),
                    cpuFlat.AsSpan(b * 64, 64),
                    8, 8);
            }

            var gpuFlat = new byte[n * 64];
            await kernel.RunAsync(aboveFlat, gpuFlat, blockCount: n);

            for (int i = 0; i < cpuFlat.Length; i++) Equal(cpuFlat[i], gpuFlat[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
