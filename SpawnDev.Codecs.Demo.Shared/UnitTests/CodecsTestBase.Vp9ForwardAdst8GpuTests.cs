// Cross-backend test for Vp9ForwardAdst8Gpu.Forward8. Verifies the GPU
// 8-point forward ADST helper matches the CPU reference Vp9ForwardAdst8
// bit-exactly across representative input patterns.

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
    public async Task Vp9ForwardAdst8Gpu_RandomInput_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            var rng = new Random(unchecked((int)0xC088_FA88u));
            int[] input = new int[8];
            for (int i = 0; i < 8; i++) input[i] = rng.Next(-1000, 1000);
            await Fadst8AndVerify(acc, input);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9ForwardAdst8Gpu_LargeInput_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            int[] input = { 16000, -16000, 16000, -16000, 16000, -16000, 16000, -16000 };
            await Fadst8AndVerify(acc, input);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9ForwardAdst8Gpu_AllZero_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            int[] input = new int[8];
            await Fadst8AndVerify(acc, input);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9ForwardAdst8Gpu_MonotonicInput_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            int[] input = { 100, 200, 300, 400, 500, 600, 700, 800 };
            await Fadst8AndVerify(acc, input);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static async Task Fadst8AndVerify(Accelerator acc, int[] input)
    {
        // CPU reference.
        int[] cpuOut = new int[8];
        Vp9ForwardAdst8.Transform(input, cpuOut);

        // GPU dispatch: single-thread per row.
        using var dIn = acc.Allocate1D<int>(8);
        using var dOut = acc.Allocate1D<int>(8);
        dIn.View.CopyFromCPU(input);
        dOut.MemSetToZero();

        var kernel = acc.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, ArrayView<int>>(Fadst8Kernel);
        kernel(new Index1D(1), dIn.View, dOut.View);
        await acc.SynchronizeAsync();

        var gpuOut = await dOut.CopyToHostAsync();

        for (int i = 0; i < 8; i++)
        {
            if (cpuOut[i] != gpuOut[i])
                throw new Exception($"output[{i}]: cpu={cpuOut[i]} gpu={gpuOut[i]}");
        }
    }

    private static void Fadst8Kernel(
        Index1D _,
        ArrayView<int> input, ArrayView<int> output)
    {
        Vp9ForwardAdst8Gpu.Forward8(input, 0, output, 0);
    }
}
