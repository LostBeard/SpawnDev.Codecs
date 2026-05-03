// Cross-backend tests for Vp9Idct16x16Gpu (the single-block in-kernel
// helper). Verifies element-by-element agreement with
// Vp9Idct16x16Reference.Idct16x16_256_Add (CPU oracle).

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private static byte[] Vp9Idct16x16Cpu(short[] coefs, byte[] predictor)
    {
        var dest = (byte[])predictor.Clone();
        Vp9Idct16x16Reference.Idct16x16_256_Add(coefs, dest, 16);
        return dest;
    }

    private static async Task<byte[]> Vp9Idct16x16GpuAsync(Accelerator acc, short[] coefs, byte[] predictor)
    {
        using var kernel = new Vp9Idct16x16GpuTestKernel(acc);
        using var dCoefs = acc.Allocate1D<short>(256);
        using var dDest = acc.Allocate1D<byte>(256);
        using var dScratch = acc.Allocate1D<int>(256);
        dCoefs.View.CopyFromCPU(coefs);
        dDest.View.CopyFromCPU(predictor);
        kernel.Run(dCoefs.View, dDest.View, dScratch.View, destStride: 16);
        await acc.SynchronizeAsync();
        return await dDest.CopyToHostAsync();
    }

    private static async Task AssertVp9Idct16x16GpuMatchesCpuAsync(
        Accelerator acc, short[] coefs, byte[] predictor)
    {
        var cpu = Vp9Idct16x16Cpu(coefs, predictor);
        var gpu = await Vp9Idct16x16GpuAsync(acc, coefs, predictor);
        for (int i = 0; i < 256; i++)
        {
            if (cpu[i] != gpu[i])
                throw new Exception(
                    $"iDCT 16x16 mismatch at pixel[{i}]: cpu={cpu[i]} gpu={gpu[i]}");
        }
    }

    [TestMethod]
    public async Task Vp9Idct16x16Gpu_AllZeroCoefs_LeavesPredictorUnchanged()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            var coefs = new short[256];
            var predictor = new byte[256];
            for (int i = 0; i < 256; i++) predictor[i] = 128;
            await AssertVp9Idct16x16GpuMatchesCpuAsync(acc, coefs, predictor);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9Idct16x16Gpu_DcOnly_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            int[] dcValues = { 32, 96, -32, -96, 256, -256, 1, -1 };
            var predictor = new byte[256];
            for (int i = 0; i < 256; i++) predictor[i] = 100;
            foreach (var dc in dcValues)
            {
                var coefs = new short[256];
                coefs[0] = (short)dc;
                await AssertVp9Idct16x16GpuMatchesCpuAsync(acc, coefs, predictor);
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9Idct16x16Gpu_RandomCoefs_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            var rng = new Random(unchecked((int)0xDC161616u));
            for (int trial = 0; trial < 8; trial++)
            {
                var coefs = new short[256];
                var predictor = new byte[256];
                for (int i = 0; i < 256; i++)
                {
                    coefs[i] = (short)rng.Next(-512, 512);
                    predictor[i] = (byte)rng.Next(0, 256);
                }
                await AssertVp9Idct16x16GpuMatchesCpuAsync(acc, coefs, predictor);
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9Idct16x16Gpu_ClampsAtPredictorBoundaries_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            var coefs = new short[256];
            for (int i = 0; i < 256; i++) coefs[i] = 1024;

            var lowPred = new byte[256];
            var highPred = new byte[256];
            for (int i = 0; i < 256; i++) highPred[i] = 255;

            await AssertVp9Idct16x16GpuMatchesCpuAsync(acc, coefs, lowPred);
            await AssertVp9Idct16x16GpuMatchesCpuAsync(acc, coefs, highPred);

            for (int i = 0; i < 256; i++) coefs[i] = -1024;
            await AssertVp9Idct16x16GpuMatchesCpuAsync(acc, coefs, lowPred);
            await AssertVp9Idct16x16GpuMatchesCpuAsync(acc, coefs, highPred);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
