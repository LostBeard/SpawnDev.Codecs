// Cross-backend test for Vp9ForwardDct4x4Gpu.Transform. Verifies the GPU
// 4x4 forward DCT helper matches the CPU reference Vp9ForwardDct4x4
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
    public async Task Vp9ForwardDct4x4Gpu_RandomInput_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            var rng = new Random(unchecked((int)0xC044_FD44u));
            short[] input = new short[16];
            for (int i = 0; i < 16; i++) input[i] = (short)rng.Next(-256, 256);
            await FdctAndVerify(acc, input, rowStride: 4);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9ForwardDct4x4Gpu_FlatInput_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            // Flat input - exercises the DC bias path.
            short[] input = new short[16];
            for (int i = 0; i < 16; i++) input[i] = 100;
            await FdctAndVerify(acc, input, rowStride: 4);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9ForwardDct4x4Gpu_ZeroInput_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            short[] input = new short[16];
            await FdctAndVerify(acc, input, rowStride: 4);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9ForwardDct4x4Gpu_LargeStride_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            // Source 4x4 block at the start of an 8-wide row stride.
            short[] input = new short[8 * 4];
            var rng = new Random(unchecked((int)0xC044_F388u));
            for (int j = 0; j < 4; j++)
                for (int i = 0; i < 4; i++)
                    input[j * 8 + i] = (short)rng.Next(-256, 256);
            await FdctAndVerify(acc, input, rowStride: 8);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static async Task FdctAndVerify(Accelerator acc, short[] input, int rowStride)
    {
        // CPU reference.
        int[] cpuOut = new int[16];
        Vp9ForwardDct4x4.Transform(input, rowStride, cpuOut);

        // GPU dispatch: single-thread per block.
        using var dIn = acc.Allocate1D<short>(input.Length);
        using var dOut = acc.Allocate1D<int>(16);
        using var dScratch = acc.Allocate1D<int>(16);
        dIn.View.CopyFromCPU(input);
        dOut.MemSetToZero();
        dScratch.MemSetToZero();

        var kernel = acc.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<short>, int, ArrayView<int>, ArrayView<int>>(FdctKernel);
        kernel(new Index1D(1), dIn.View, rowStride, dOut.View, dScratch.View);
        await acc.SynchronizeAsync();

        var gpuOut = await dOut.CopyToHostAsync();

        for (int i = 0; i < 16; i++)
        {
            if (cpuOut[i] != gpuOut[i])
                throw new Exception($"output[{i}]: cpu={cpuOut[i]} gpu={gpuOut[i]}");
        }
    }

    private static void FdctKernel(
        Index1D _,
        ArrayView<short> input, int rowStride, ArrayView<int> output, ArrayView<int> scratch)
    {
        Vp9ForwardDct4x4Gpu.Transform(input, 0, rowStride, output, 0, scratch, 0);
    }
}
