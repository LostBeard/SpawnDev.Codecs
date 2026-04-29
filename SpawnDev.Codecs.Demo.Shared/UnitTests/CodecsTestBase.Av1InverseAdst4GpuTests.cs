// Cross-backend test for Av1InverseAdst4Gpu.Inverse4. Verifies the GPU
// 4-point inverse ADST helper matches the CPU reference Av1InverseAdst4
// bit-exactly across representative coefficient patterns + cosBit values.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Av1;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public async Task Av1InverseAdst4Gpu_DefaultCosBit_RandomCoefs_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            var rng = new Random(unchecked((int)0xAA01_AD04u));
            int[] input = new int[4];
            for (int i = 0; i < 4; i++) input[i] = rng.Next(-2000, 2000);
            await Iadst4AndVerify(acc, input, cosBit: 12);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Av1InverseAdst4Gpu_AllCosBits_MatchCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            int[] input = { 100, -200, 300, -400 };
            await Iadst4AndVerify(acc, input, cosBit: 10);
            await Iadst4AndVerify(acc, input, cosBit: 11);
            await Iadst4AndVerify(acc, input, cosBit: 12);
            await Iadst4AndVerify(acc, input, cosBit: 13);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Av1InverseAdst4Gpu_AllZero_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            // Exercises the early-out zero path.
            int[] input = { 0, 0, 0, 0 };
            await Iadst4AndVerify(acc, input, cosBit: 12);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Av1InverseAdst4Gpu_LargeCoefs_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            // Stress the 64-bit intermediate path.
            int[] input = { 100000, -50000, 25000, -10000 };
            await Iadst4AndVerify(acc, input, cosBit: 12);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static async Task Iadst4AndVerify(Accelerator acc, int[] input, int cosBit)
    {
        // CPU reference.
        int[] cpuOut = new int[4];
        Av1InverseAdst4.Transform(input, cpuOut, cosBit);

        // GPU dispatch: single-thread per row.
        using var dIn = acc.Allocate1D<int>(4);
        using var dOut = acc.Allocate1D<int>(4);
        dIn.View.CopyFromCPU(input);
        dOut.MemSetToZero();

        var kernel = acc.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, ArrayView<int>, int>(Iadst4Kernel);
        kernel(new Index1D(1), dIn.View, dOut.View, cosBit);
        await acc.SynchronizeAsync();

        var gpuOut = await dOut.CopyToHostAsync();

        for (int i = 0; i < 4; i++)
        {
            if (cpuOut[i] != gpuOut[i])
                throw new Exception($"output[{i}]: cpu={cpuOut[i]} gpu={gpuOut[i]} (cosBit={cosBit})");
        }
    }

    private static void Iadst4Kernel(
        Index1D _,
        ArrayView<int> input, ArrayView<int> output, int cosBit)
    {
        Av1InverseAdst4Gpu.Inverse4(input, 0, output, 0, cosBit);
    }
}
