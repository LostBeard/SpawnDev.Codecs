// Cross-backend test for Vp9Iadst8x8Gpu.Iadst8x8. Verifies the GPU 8x8
// inverse ADST helper matches the CPU reference Vp9Iadst8x8Reference
// bit-exactly across representative coefficient patterns.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public async Task Vp9Iadst8x8Gpu_DcOnly_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            short[] coefs = new short[64];
            coefs[0] = 200;
            await Iadst8AndVerify(acc, coefs, predictor: 50);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9Iadst8x8Gpu_RandomCoefs_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            var rng = new Random(unchecked((int)0xC088_AD88u));
            short[] coefs = new short[64];
            for (int i = 0; i < 64; i++) coefs[i] = (short)rng.Next(-200, 200);
            await Iadst8AndVerify(acc, coefs, predictor: 128);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9Iadst8x8Gpu_LargeCoefs_ClipsCorrect_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            short[] coefs = new short[64];
            coefs[0] = 4000;
            coefs[1] = -2000;
            coefs[8] = -2000;
            await Iadst8AndVerify(acc, coefs, predictor: 200);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9Iadst8x8Gpu_ZeroCoefs_Identity_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            short[] coefs = new short[64];
            await Iadst8AndVerify(acc, coefs, predictor: 100);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static async Task Iadst8AndVerify(Accelerator acc, short[] coefs, byte predictor)
    {
        const int stride = 16;

        // CPU reference.
        byte[] cpuDest = new byte[stride * 8];
        for (int j = 0; j < 8; j++)
            for (int i = 0; i < 8; i++)
                cpuDest[j * stride + i] = predictor;
        Vp9Iadst8x8Reference.IadstAdst8x8_64_Add(coefs, cpuDest, stride);

        // GPU dispatch: single-thread per block.
        using var dCoefs = acc.Allocate1D<short>(64);
        using var dDest = acc.Allocate1D<byte>(stride * 8);
        using var dScratch = acc.Allocate1D<short>(64);
        dCoefs.View.CopyFromCPU(coefs);

        byte[] initDest = new byte[stride * 8];
        for (int j = 0; j < 8; j++)
            for (int i = 0; i < 8; i++)
                initDest[j * stride + i] = predictor;
        dDest.View.CopyFromCPU(initDest);
        dScratch.MemSetToZero();

        var kernel = acc.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<short>, ArrayView<byte>, ArrayView<short>, int>(Iadst8Kernel);
        kernel(new Index1D(1), dCoefs.View, dDest.View, dScratch.View, stride);
        await acc.SynchronizeAsync();

        var gpuDest = await dDest.CopyToHostAsync();

        for (int j = 0; j < 8; j++)
        {
            for (int i = 0; i < 8; i++)
            {
                if (cpuDest[j * stride + i] != gpuDest[j * stride + i])
                    throw new Exception($"pixel[{j},{i}]: cpu={cpuDest[j * stride + i]} gpu={gpuDest[j * stride + i]}");
            }
        }
    }

    private static void Iadst8Kernel(
        Index1D _,
        ArrayView<short> coefs, ArrayView<byte> dest, ArrayView<short> scratch, int stride)
    {
        Vp9Iadst8x8Gpu.Iadst8x8(coefs, 0, dest, 0, stride, scratch);
    }
}
