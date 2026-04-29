// Cross-backend test for Av1InverseAdst8Gpu.Inverse8. Verifies the GPU
// 8-point inverse ADST helper matches the CPU reference Av1InverseAdst8
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
    public async Task Av1InverseAdst8Gpu_DefaultCosBit_RandomCoefs_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            var rng = new Random(unchecked((int)0xAA01ADC8u));
            int[] input = new int[8];
            for (int i = 0; i < 8; i++) input[i] = rng.Next(-2000, 2000);
            await IadstAndVerify(acc, input, cosBit: 12);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Av1InverseAdst8Gpu_CosBit13_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            int[] input = { 100, -200, 300, -400, 500, -600, 700, -800 };
            await IadstAndVerify(acc, input, cosBit: 13);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Av1InverseAdst8Gpu_CosBit10_LargeCoefs_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            // Large coefficients to stress the long-mul + shift path.
            int[] input = { 5000, -3000, 4000, -2000, 1000, -500, 250, -100 };
            await IadstAndVerify(acc, input, cosBit: 10);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Av1InverseAdst8Gpu_DcOnly_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            int[] input = new int[8];
            input[0] = 1000; // DC-only
            await IadstAndVerify(acc, input, cosBit: 12);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static async Task IadstAndVerify(Accelerator acc, int[] input, int cosBit)
    {
        // CPU reference.
        int[] cpuOut = new int[8];
        Av1InverseAdst8.Transform(input, cpuOut, cosBit);

        // GPU dispatch: single-thread per row.
        using var dIn = acc.Allocate1D<int>(8);
        using var dOut = acc.Allocate1D<int>(8);
        dIn.View.CopyFromCPU(input);
        dOut.MemSetToZero();

        var kernel = acc.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, ArrayView<int>, int>(IadstKernel);
        kernel(new Index1D(1), dIn.View, dOut.View, cosBit);
        await acc.SynchronizeAsync();

        var gpuOut = await dOut.CopyToHostAsync();

        for (int i = 0; i < 8; i++)
        {
            if (cpuOut[i] != gpuOut[i])
                throw new Exception($"output[{i}]: cpu={cpuOut[i]} gpu={gpuOut[i]} (cosBit={cosBit})");
        }
    }

    private static void IadstKernel(
        Index1D _,
        ArrayView<int> input, ArrayView<int> output, int cosBit)
    {
        Av1InverseAdst8Gpu.Inverse8(input, 0, output, 0, cosBit);
    }
}
