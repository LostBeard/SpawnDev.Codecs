// Cross-backend test for Av1InverseAdst16Gpu.Inverse16. Verifies the GPU
// 16-point inverse ADST helper matches the CPU reference Av1InverseAdst16
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
    public async Task Av1InverseAdst16Gpu_DefaultCosBit_RandomCoefs_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            var rng = new Random(unchecked((int)0xAA01_AD16u));
            int[] input = new int[16];
            for (int i = 0; i < 16; i++) input[i] = rng.Next(-2000, 2000);
            await Iadst16AndVerify(acc, input, cosBit: 12);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Av1InverseAdst16Gpu_CosBit13_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            int[] input = new int[16];
            for (int i = 0; i < 16; i++) input[i] = (i % 2 == 0 ? 1 : -1) * (i + 1) * 100;
            await Iadst16AndVerify(acc, input, cosBit: 13);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Av1InverseAdst16Gpu_DcOnly_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            int[] input = new int[16];
            input[0] = 5000;
            await Iadst16AndVerify(acc, input, cosBit: 12);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Av1InverseAdst16Gpu_CosBit10_LargeCoefs_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            // Large coefs to stress the long-mul + shift path.
            int[] input = { 5000, -3000, 4000, -2000, 1000, -500, 250, -100,
                              50,   -25,   12,    -6,    3,   -1,   1,    0 };
            await Iadst16AndVerify(acc, input, cosBit: 10);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static async Task Iadst16AndVerify(Accelerator acc, int[] input, int cosBit)
    {
        // CPU reference.
        int[] cpuOut = new int[16];
        Av1InverseAdst16.Transform(input, cpuOut, cosBit);

        // GPU dispatch: single-thread per row.
        using var dIn = acc.Allocate1D<int>(16);
        using var dOut = acc.Allocate1D<int>(16);
        dIn.View.CopyFromCPU(input);
        dOut.MemSetToZero();

        var kernel = acc.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, ArrayView<int>, int>(Iadst16Kernel);
        kernel(new Index1D(1), dIn.View, dOut.View, cosBit);
        await acc.SynchronizeAsync();

        var gpuOut = await dOut.CopyToHostAsync();

        for (int i = 0; i < 16; i++)
        {
            if (cpuOut[i] != gpuOut[i])
                throw new Exception($"output[{i}]: cpu={cpuOut[i]} gpu={gpuOut[i]} (cosBit={cosBit})");
        }
    }

    private static void Iadst16Kernel(
        Index1D _,
        ArrayView<int> input, ArrayView<int> output, int cosBit)
    {
        Av1InverseAdst16Gpu.Inverse16(input, 0, output, 0, cosBit);
    }
}
