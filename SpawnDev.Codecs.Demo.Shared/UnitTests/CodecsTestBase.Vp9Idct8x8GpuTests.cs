// Cross-backend tests for Vp9Idct8x8Gpu (the single-block in-kernel
// helper). Verifies element-by-element agreement with
// Vp9Idct8x8Reference.Idct8x8_64_Add (CPU oracle) across realistic
// coefficient + predictor combinations.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private static byte[] Vp9Idct8x8Cpu(short[] coefs, byte[] predictor)
    {
        var dest = (byte[])predictor.Clone();
        Vp9Idct8x8Reference.Idct8x8_64_Add(coefs, dest, 8);
        return dest;
    }

    private static async Task<byte[]> Vp9Idct8x8GpuAsync(Accelerator acc, short[] coefs, byte[] predictor)
    {
        using var kernel = new Vp9Idct8x8GpuTestKernel(acc);
        using var dCoefs = acc.Allocate1D<short>(64);
        using var dDest = acc.Allocate1D<byte>(64);
        using var dScratch = acc.Allocate1D<int>(64);
        dCoefs.View.CopyFromCPU(coefs);
        dDest.View.CopyFromCPU(predictor);
        kernel.Run(dCoefs.View, dDest.View, dScratch.View, destStride: 8);
        await acc.SynchronizeAsync();
        return await dDest.CopyToHostAsync();
    }

    private static async Task AssertVp9Idct8x8GpuMatchesCpuAsync(
        Accelerator acc, short[] coefs, byte[] predictor)
    {
        var cpu = Vp9Idct8x8Cpu(coefs, predictor);
        var gpu = await Vp9Idct8x8GpuAsync(acc, coefs, predictor);
        for (int i = 0; i < 64; i++)
        {
            if (cpu[i] != gpu[i])
                throw new Exception(
                    $"iDCT 8x8 mismatch at pixel[{i}]: cpu={cpu[i]} gpu={gpu[i]}");
        }
    }

    [TestMethod]
    public async Task Vp9Idct8x8Gpu_AllZeroCoefs_LeavesPredictorUnchanged()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            var coefs = new short[64];
            var predictor = new byte[64];
            for (int i = 0; i < 64; i++) predictor[i] = 128;
            await AssertVp9Idct8x8GpuMatchesCpuAsync(acc, coefs, predictor);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9Idct8x8Gpu_DcOnly_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // DC-only coefficient produces a flat residual. A range of
            // DC values exercises positive + negative residual paths.
            int[] dcValues = { 32, 96, -32, -96, 128, -128, 1, -1 };
            var predictor = new byte[64];
            for (int i = 0; i < 64; i++) predictor[i] = 100;
            foreach (var dc in dcValues)
            {
                var coefs = new short[64];
                coefs[0] = (short)dc;
                await AssertVp9Idct8x8GpuMatchesCpuAsync(acc, coefs, predictor);
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9Idct8x8Gpu_RandomCoefs_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            var rng = new Random(unchecked((int)0xDC0E8C18u));
            for (int trial = 0; trial < 16; trial++)
            {
                var coefs = new short[64];
                var predictor = new byte[64];
                for (int i = 0; i < 64; i++)
                {
                    coefs[i] = (short)rng.Next(-256, 256);
                    predictor[i] = (byte)rng.Next(0, 256);
                }
                await AssertVp9Idct8x8GpuMatchesCpuAsync(acc, coefs, predictor);
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9Idct8x8Gpu_ClampsAtPredictorBoundaries_MatchesCpu()
    {
        // Push the residual past 0 / 255 boundary to verify clip3
        // matches the CPU oracle exactly.
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            var coefs = new short[64];
            for (int i = 0; i < 64; i++) coefs[i] = 1024; // large positive

            var lowPred = new byte[64];      // all 0 - clamps top
            var highPred = new byte[64];
            for (int i = 0; i < 64; i++) highPred[i] = 255; // all 255 - clamps bottom

            await AssertVp9Idct8x8GpuMatchesCpuAsync(acc, coefs, lowPred);
            await AssertVp9Idct8x8GpuMatchesCpuAsync(acc, coefs, highPred);

            for (int i = 0; i < 64; i++) coefs[i] = -1024;
            await AssertVp9Idct8x8GpuMatchesCpuAsync(acc, coefs, lowPred);
            await AssertVp9Idct8x8GpuMatchesCpuAsync(acc, coefs, highPred);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
