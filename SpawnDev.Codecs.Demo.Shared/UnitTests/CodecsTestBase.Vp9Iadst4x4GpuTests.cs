// Cross-backend test for Vp9Iadst4x4Gpu.Iadst4x4. Verifies the GPU 4x4
// inverse ADST helper matches the CPU reference Vp9Iadst4x4Reference
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
    public async Task Vp9Iadst4x4Gpu_DcOnly_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            short[] coefs = new short[16];
            coefs[0] = 100;
            await IadstAndVerify(acc, coefs, predictor: 50);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9Iadst4x4Gpu_RandomCoefs_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            var rng = new Random(unchecked((int)0xC044_AD44u));
            short[] coefs = new short[16];
            for (int i = 0; i < 16; i++) coefs[i] = (short)rng.Next(-200, 200);
            await IadstAndVerify(acc, coefs, predictor: 128);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9Iadst4x4Gpu_LargeCoefs_ClipsCorrect_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            short[] coefs = new short[16];
            coefs[0] = 4000;
            coefs[1] = -2000;
            coefs[4] = -2000;
            await IadstAndVerify(acc, coefs, predictor: 200);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9Iadst4x4Gpu_ZeroCoefs_Identity_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            short[] coefs = new short[16];
            await IadstAndVerify(acc, coefs, predictor: 100);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static async Task IadstAndVerify(Accelerator acc, short[] coefs, byte predictor)
    {
        const int stride = 8;

        // CPU reference.
        byte[] cpuDest = new byte[stride * 4];
        for (int j = 0; j < 4; j++)
            for (int i = 0; i < 4; i++)
                cpuDest[j * stride + i] = predictor;
        Vp9Iadst4x4Reference.IadstAdst4x4_16_Add(coefs, cpuDest, stride);

        // GPU dispatch: single-thread per block.
        using var dCoefs = acc.Allocate1D<short>(16);
        using var dDest = acc.Allocate1D<byte>(stride * 4);
        using var dScratch = acc.Allocate1D<short>(16);
        dCoefs.View.CopyFromCPU(coefs);

        byte[] initDest = new byte[stride * 4];
        for (int j = 0; j < 4; j++)
            for (int i = 0; i < 4; i++)
                initDest[j * stride + i] = predictor;
        dDest.View.CopyFromCPU(initDest);
        dScratch.MemSetToZero();

        var kernel = acc.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<short>, ArrayView<byte>, ArrayView<short>, int>(IadstKernel);
        kernel(new Index1D(1), dCoefs.View, dDest.View, dScratch.View, stride);
        await acc.SynchronizeAsync();

        var gpuDest = await dDest.CopyToHostAsync();

        for (int j = 0; j < 4; j++)
        {
            for (int i = 0; i < 4; i++)
            {
                if (cpuDest[j * stride + i] != gpuDest[j * stride + i])
                    throw new Exception($"pixel[{j},{i}]: cpu={cpuDest[j * stride + i]} gpu={gpuDest[j * stride + i]}");
            }
        }
    }

    private static void IadstKernel(
        Index1D _,
        ArrayView<short> coefs, ArrayView<byte> dest, ArrayView<short> scratch, int stride)
    {
        Vp9Iadst4x4Gpu.Iadst4x4(coefs, 0, dest, 0, stride, scratch);
    }
}
