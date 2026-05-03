// Tests for Vp9VhPredict4x4Kernel (slice 180). Verifies V_PRED and
// H_PRED kernel outputs match the CPU oracle bit-exactly across
// every supported backend.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public async Task Vp9VhPredict4x4Kernel_V_MatchesCpuReference()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Vp9VhPredict4x4Kernel(acc);
            byte[] above = { 10, 20, 30, 40 };
            byte[] left = { 0, 0, 0, 0 };

            var cpuDst = new byte[16];
            Vp9VHPredictor.VPredict(above, cpuDst, 4, 4);

            var gpuDst = new byte[16];
            await kernel.RunAsync(above, left, gpuDst, Vp9VhMode.V, blockCount: 1);

            for (int i = 0; i < 16; i++) Equal(cpuDst[i], gpuDst[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9VhPredict4x4Kernel_H_MatchesCpuReference()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Vp9VhPredict4x4Kernel(acc);
            byte[] above = { 0, 0, 0, 0 };
            byte[] left = { 11, 22, 33, 44 };

            var cpuDst = new byte[16];
            Vp9VHPredictor.HPredict(left, cpuDst, 4, 4);

            var gpuDst = new byte[16];
            await kernel.RunAsync(above, left, gpuDst, Vp9VhMode.H, blockCount: 1);

            for (int i = 0; i < 16; i++) Equal(cpuDst[i], gpuDst[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9VhPredict4x4Kernel_V_RandomInputs_BitExact()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Vp9VhPredict4x4Kernel(acc);
            var rng = new Random(unchecked((int)0xCAFEFACEu));
            for (int trial = 0; trial < 16; trial++)
            {
                var above = new byte[4];
                var left = new byte[4];
                for (int i = 0; i < 4; i++) above[i] = (byte)rng.Next(0, 256);

                var cpuDst = new byte[16];
                Vp9VHPredictor.VPredict(above, cpuDst, 4, 4);

                var gpuDst = new byte[16];
                await kernel.RunAsync(above, left, gpuDst, Vp9VhMode.V, blockCount: 1);

                for (int i = 0; i < 16; i++) Equal(cpuDst[i], gpuDst[i]);
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9VhPredict4x4Kernel_H_RandomInputs_BitExact()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Vp9VhPredict4x4Kernel(acc);
            var rng = new Random(unchecked((int)0xCAFEFEEDu));
            for (int trial = 0; trial < 16; trial++)
            {
                var above = new byte[4];
                var left = new byte[4];
                for (int i = 0; i < 4; i++) left[i] = (byte)rng.Next(0, 256);

                var cpuDst = new byte[16];
                Vp9VHPredictor.HPredict(left, cpuDst, 4, 4);

                var gpuDst = new byte[16];
                await kernel.RunAsync(above, left, gpuDst, Vp9VhMode.H, blockCount: 1);

                for (int i = 0; i < 16; i++) Equal(cpuDst[i], gpuDst[i]);
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9VhPredict4x4Kernel_BatchedDispatch_AllBlocksBitExact()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Vp9VhPredict4x4Kernel(acc);
            const int n = 32;
            var rng = new Random(unchecked((int)0xCAFE0042u));
            var aboveFlat = new byte[n * 4];
            var leftFlat = new byte[n * 4];
            for (int i = 0; i < aboveFlat.Length; i++) aboveFlat[i] = (byte)rng.Next(0, 256);
            for (int i = 0; i < leftFlat.Length; i++) leftFlat[i] = (byte)rng.Next(0, 256);

            // Reference: per-block VPredict.
            var cpuV = new byte[n * 16];
            for (int b = 0; b < n; b++)
            {
                Vp9VHPredictor.VPredict(
                    aboveFlat.AsSpan(b * 4, 4),
                    cpuV.AsSpan(b * 16, 16),
                    4, 4);
            }
            var gpuV = new byte[n * 16];
            await kernel.RunAsync(aboveFlat, leftFlat, gpuV, Vp9VhMode.V, blockCount: n);
            for (int i = 0; i < cpuV.Length; i++) Equal(cpuV[i], gpuV[i]);

            // Reference: per-block HPredict.
            var cpuH = new byte[n * 16];
            for (int b = 0; b < n; b++)
            {
                Vp9VHPredictor.HPredict(
                    leftFlat.AsSpan(b * 4, 4),
                    cpuH.AsSpan(b * 16, 16),
                    4, 4);
            }
            var gpuH = new byte[n * 16];
            await kernel.RunAsync(aboveFlat, leftFlat, gpuH, Vp9VhMode.H, blockCount: n);
            for (int i = 0; i < cpuH.Length; i++) Equal(cpuH[i], gpuH[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
