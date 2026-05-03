// Tests for Vp9VhPredict16x16Kernel (slice 182). 16x16 sibling.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public async Task Vp9VhPredict16x16Kernel_V_MatchesCpuReference()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Vp9VhPredict16x16Kernel(acc);
            var above = new byte[16];
            for (int i = 0; i < 16; i++) above[i] = (byte)(i * 17);  // 0, 17, 34, ..., 255
            var left = new byte[16];

            var cpuDst = new byte[256];
            Vp9VHPredictor.VPredict(above, cpuDst, 16, 16);

            var gpuDst = new byte[256];
            await kernel.RunAsync(above, left, gpuDst, Vp9VhMode.V, blockCount: 1);

            for (int i = 0; i < 256; i++) Equal(cpuDst[i], gpuDst[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9VhPredict16x16Kernel_H_MatchesCpuReference()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Vp9VhPredict16x16Kernel(acc);
            var above = new byte[16];
            var left = new byte[16];
            for (int i = 0; i < 16; i++) left[i] = (byte)(255 - i * 17);

            var cpuDst = new byte[256];
            Vp9VHPredictor.HPredict(left, cpuDst, 16, 16);

            var gpuDst = new byte[256];
            await kernel.RunAsync(above, left, gpuDst, Vp9VhMode.H, blockCount: 1);

            for (int i = 0; i < 256; i++) Equal(cpuDst[i], gpuDst[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9VhPredict16x16Kernel_V_RandomInputs_BitExact()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Vp9VhPredict16x16Kernel(acc);
            var rng = new Random(unchecked((int)0xCAFE0182u));
            for (int trial = 0; trial < 8; trial++)
            {
                var above = new byte[16];
                var left = new byte[16];
                for (int i = 0; i < 16; i++) above[i] = (byte)rng.Next(0, 256);

                var cpuDst = new byte[256];
                Vp9VHPredictor.VPredict(above, cpuDst, 16, 16);

                var gpuDst = new byte[256];
                await kernel.RunAsync(above, left, gpuDst, Vp9VhMode.V, blockCount: 1);

                for (int i = 0; i < 256; i++) Equal(cpuDst[i], gpuDst[i]);
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9VhPredict16x16Kernel_H_RandomInputs_BitExact()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Vp9VhPredict16x16Kernel(acc);
            var rng = new Random(unchecked((int)0xCAFE0183u));
            for (int trial = 0; trial < 8; trial++)
            {
                var above = new byte[16];
                var left = new byte[16];
                for (int i = 0; i < 16; i++) left[i] = (byte)rng.Next(0, 256);

                var cpuDst = new byte[256];
                Vp9VHPredictor.HPredict(left, cpuDst, 16, 16);

                var gpuDst = new byte[256];
                await kernel.RunAsync(above, left, gpuDst, Vp9VhMode.H, blockCount: 1);

                for (int i = 0; i < 256; i++) Equal(cpuDst[i], gpuDst[i]);
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9VhPredict16x16Kernel_BatchedDispatch_AllBlocksBitExact()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Vp9VhPredict16x16Kernel(acc);
            const int n = 8;
            var rng = new Random(unchecked((int)0xCAFE0184u));
            var aboveFlat = new byte[n * 16];
            var leftFlat = new byte[n * 16];
            for (int i = 0; i < aboveFlat.Length; i++) aboveFlat[i] = (byte)rng.Next(0, 256);
            for (int i = 0; i < leftFlat.Length; i++) leftFlat[i] = (byte)rng.Next(0, 256);

            var cpuV = new byte[n * 256];
            for (int b = 0; b < n; b++)
                Vp9VHPredictor.VPredict(aboveFlat.AsSpan(b * 16, 16), cpuV.AsSpan(b * 256, 256), 16, 16);
            var gpuV = new byte[n * 256];
            await kernel.RunAsync(aboveFlat, leftFlat, gpuV, Vp9VhMode.V, blockCount: n);
            for (int i = 0; i < cpuV.Length; i++) Equal(cpuV[i], gpuV[i]);

            var cpuH = new byte[n * 256];
            for (int b = 0; b < n; b++)
                Vp9VHPredictor.HPredict(leftFlat.AsSpan(b * 16, 16), cpuH.AsSpan(b * 256, 256), 16, 16);
            var gpuH = new byte[n * 256];
            await kernel.RunAsync(aboveFlat, leftFlat, gpuH, Vp9VhMode.H, blockCount: n);
            for (int i = 0; i < cpuH.Length; i++) Equal(cpuH[i], gpuH[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
