// Cross-backend test for Vp9ForwardAdst16Gpu.Forward16. Verifies the GPU
// 16-point forward ADST helper matches the CPU reference Vp9ForwardAdst16
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
    public async Task Vp9ForwardAdst16Gpu_RandomInput_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            var rng = new Random(unchecked((int)0xC161_FA61u));
            int[] input = new int[16];
            for (int i = 0; i < 16; i++) input[i] = rng.Next(-1000, 1000);
            await Fadst16AndVerify(acc, input);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9ForwardAdst16Gpu_LargeInput_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            int[] input = new int[16];
            for (int i = 0; i < 16; i++) input[i] = (i & 1) == 0 ? 16000 : -16000;
            await Fadst16AndVerify(acc, input);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9ForwardAdst16Gpu_AllZero_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            int[] input = new int[16];
            await Fadst16AndVerify(acc, input);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9ForwardAdst16Gpu_MonotonicInput_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            int[] input = new int[16];
            for (int i = 0; i < 16; i++) input[i] = (i + 1) * 100;
            await Fadst16AndVerify(acc, input);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static async Task Fadst16AndVerify(Accelerator acc, int[] input)
    {
        // CPU reference.
        int[] cpuOut = new int[16];
        Vp9ForwardAdst16.Transform(input, cpuOut);

        // GPU dispatch: single-thread per row.
        using var dIn = acc.Allocate1D<int>(16);
        using var dOut = acc.Allocate1D<int>(16);
        dIn.View.CopyFromCPU(input);
        dOut.MemSetToZero();

        var kernel = acc.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, ArrayView<int>>(Fadst16Kernel);
        kernel(new Index1D(1), dIn.View, dOut.View);
        await acc.SynchronizeAsync();

        var gpuOut = await dOut.CopyToHostAsync();

        for (int i = 0; i < 16; i++)
        {
            if (cpuOut[i] != gpuOut[i])
                throw new Exception($"output[{i}]: cpu={cpuOut[i]} gpu={gpuOut[i]}");
        }
    }

    private static void Fadst16Kernel(
        Index1D _,
        ArrayView<int> input, ArrayView<int> output)
    {
        Vp9ForwardAdst16Gpu.Forward16(input, 0, output, 0);
    }
}
