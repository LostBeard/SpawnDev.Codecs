// Tests for Vp9VhPredict8x8Kernel (slice 181). 8x8 sibling.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public async Task Vp9VhPredict8x8Kernel_V_MatchesCpuReference()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Vp9VhPredict8x8Kernel(acc);
            var above = new byte[8];
            for (int i = 0; i < 8; i++) above[i] = (byte)(10 * (i + 1));
            var left = new byte[8];

            var cpuDst = new byte[64];
            Vp9VHPredictor.VPredict(above, cpuDst, 8, 8);

            var gpuDst = new byte[64];
            await kernel.RunAsync(above, left, gpuDst, Vp9VhMode.V, blockCount: 1);

            for (int i = 0; i < 64; i++) Equal(cpuDst[i], gpuDst[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9VhPredict8x8Kernel_H_MatchesCpuReference()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Vp9VhPredict8x8Kernel(acc);
            var above = new byte[8];
            var left = new byte[8];
            for (int i = 0; i < 8; i++) left[i] = (byte)(15 * (i + 1));

            var cpuDst = new byte[64];
            Vp9VHPredictor.HPredict(left, cpuDst, 8, 8);

            var gpuDst = new byte[64];
            await kernel.RunAsync(above, left, gpuDst, Vp9VhMode.H, blockCount: 1);

            for (int i = 0; i < 64; i++) Equal(cpuDst[i], gpuDst[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9VhPredict8x8Kernel_V_RandomInputs_BitExact()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Vp9VhPredict8x8Kernel(acc);
            var rng = new Random(unchecked((int)0xCAFE0181u));
            for (int trial = 0; trial < 12; trial++)
            {
                var above = new byte[8];
                var left = new byte[8];
                for (int i = 0; i < 8; i++) above[i] = (byte)rng.Next(0, 256);

                var cpuDst = new byte[64];
                Vp9VHPredictor.VPredict(above, cpuDst, 8, 8);

                var gpuDst = new byte[64];
                await kernel.RunAsync(above, left, gpuDst, Vp9VhMode.V, blockCount: 1);

                for (int i = 0; i < 64; i++) Equal(cpuDst[i], gpuDst[i]);
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9VhPredict8x8Kernel_H_RandomInputs_BitExact()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Vp9VhPredict8x8Kernel(acc);
            var rng = new Random(unchecked((int)0xCAFE0182u));
            for (int trial = 0; trial < 12; trial++)
            {
                var above = new byte[8];
                var left = new byte[8];
                for (int i = 0; i < 8; i++) left[i] = (byte)rng.Next(0, 256);

                var cpuDst = new byte[64];
                Vp9VHPredictor.HPredict(left, cpuDst, 8, 8);

                var gpuDst = new byte[64];
                await kernel.RunAsync(above, left, gpuDst, Vp9VhMode.H, blockCount: 1);

                for (int i = 0; i < 64; i++) Equal(cpuDst[i], gpuDst[i]);
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9VhPredict8x8Kernel_BatchedDispatch_AllBlocksBitExact()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Vp9VhPredict8x8Kernel(acc);
            const int n = 16;
            var rng = new Random(unchecked((int)0xCAFE0183u));
            var aboveFlat = new byte[n * 8];
            var leftFlat = new byte[n * 8];
            for (int i = 0; i < aboveFlat.Length; i++) aboveFlat[i] = (byte)rng.Next(0, 256);
            for (int i = 0; i < leftFlat.Length; i++) leftFlat[i] = (byte)rng.Next(0, 256);

            var cpuV = new byte[n * 64];
            for (int b = 0; b < n; b++)
                Vp9VHPredictor.VPredict(aboveFlat.AsSpan(b * 8, 8), cpuV.AsSpan(b * 64, 64), 8, 8);
            var gpuV = new byte[n * 64];
            await kernel.RunAsync(aboveFlat, leftFlat, gpuV, Vp9VhMode.V, blockCount: n);
            for (int i = 0; i < cpuV.Length; i++) Equal(cpuV[i], gpuV[i]);

            var cpuH = new byte[n * 64];
            for (int b = 0; b < n; b++)
                Vp9VHPredictor.HPredict(leftFlat.AsSpan(b * 8, 8), cpuH.AsSpan(b * 64, 64), 8, 8);
            var gpuH = new byte[n * 64];
            await kernel.RunAsync(aboveFlat, leftFlat, gpuH, Vp9VhMode.H, blockCount: n);
            for (int i = 0; i < cpuH.Length; i++) Equal(cpuH[i], gpuH[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
