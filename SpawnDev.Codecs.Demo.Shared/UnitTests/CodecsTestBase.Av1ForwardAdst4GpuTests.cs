// Cross-backend test for Av1ForwardAdst4Gpu.Forward4. Verifies the GPU
// 4-point forward ADST helper matches the CPU reference Av1ForwardAdst4
// bit-exactly across representative input patterns + cosBit values.

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
    public async Task Av1ForwardAdst4Gpu_DefaultCosBit_RandomInput_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            var rng = new Random(unchecked((int)0xAA01_FA04u));
            int[] input = new int[4];
            for (int i = 0; i < 4; i++) input[i] = rng.Next(-256, 256);
            await Fadst4AndVerify(acc, input, cosBit: 13);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Av1ForwardAdst4Gpu_AllCosBits_MatchCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            int[] input = { 100, -50, 200, -150 };
            await Fadst4AndVerify(acc, input, cosBit: 10);
            await Fadst4AndVerify(acc, input, cosBit: 11);
            await Fadst4AndVerify(acc, input, cosBit: 12);
            await Fadst4AndVerify(acc, input, cosBit: 13);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Av1ForwardAdst4Gpu_AllZero_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // Exercises the early-out zero path.
            int[] input = { 0, 0, 0, 0 };
            await Fadst4AndVerify(acc, input, cosBit: 13);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Av1ForwardAdst4Gpu_LargeInput_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // Stress the 64-bit intermediate path.
            int[] input = { 16000, -16000, 16000, -16000 };
            await Fadst4AndVerify(acc, input, cosBit: 13);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static async Task Fadst4AndVerify(Accelerator acc, int[] input, int cosBit)
    {
        // CPU reference.
        int[] cpuOut = new int[4];
        Av1ForwardAdst4.Transform(input, cpuOut, cosBit);

        // GPU dispatch: single-thread per row.
        using var dIn = acc.Allocate1D<int>(4);
        using var dOut = acc.Allocate1D<int>(4);
        dIn.View.CopyFromCPU(input);
        dOut.MemSetToZero();

        var kernel = acc.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, ArrayView<int>, int>(Fadst4Kernel);
        kernel(new Index1D(1), dIn.View, dOut.View, cosBit);
        await acc.SynchronizeAsync();

        var gpuOut = await dOut.CopyToHostAsync();

        for (int i = 0; i < 4; i++)
        {
            if (cpuOut[i] != gpuOut[i])
                throw new Exception($"output[{i}]: cpu={cpuOut[i]} gpu={gpuOut[i]} (cosBit={cosBit})");
        }
    }

    private static void Fadst4Kernel(
        Index1D _,
        ArrayView<int> input, ArrayView<int> output, int cosBit)
    {
        Av1ForwardAdst4Gpu.Forward4(input, 0, output, 0, cosBit);
    }
}
