// Cross-backend test for Av1InverseQuantizerGpu.DequantizeAt. Verifies the
// GPU AV1 inverse quantizer matches the symmetric round-trip of the
// CPU forward quantizer + a manual dequant step bit-exactly across
// representative block sizes and Q values.

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
    public async Task Av1InverseQuantizerGpu_4x4Block_LowQ_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            await DequantAndVerify(acc, blockSize: 16, dcQ: 8, acQ: 12, seed: 0xAA01_4404u);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Av1InverseQuantizerGpu_8x8Block_MidQ_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            await DequantAndVerify(acc, blockSize: 64, dcQ: 50, acQ: 100, seed: 0xAA01_8808u);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Av1InverseQuantizerGpu_16x16Block_HighQ_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            await DequantAndVerify(acc, blockSize: 256, dcQ: 1200, acQ: 1800, seed: 0xAA01_1616u);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Av1InverseQuantizerGpu_32x32Block_LargeQ_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            await DequantAndVerify(acc, blockSize: 1024, dcQ: 5000, acQ: 7500, seed: 0xAA01_3232u);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static async Task DequantAndVerify(
        Accelerator acc, int blockSize, int dcQ, int acQ, uint seed)
    {
        var rng = new Random(unchecked((int)seed));

        // Random quantized coefficients (range bounded so int*Q stays safe).
        int[] quantized = new int[blockSize];
        for (int i = 0; i < blockSize; i++) quantized[i] = rng.Next(-1000, 1001);

        // CPU reference - direct dequant.
        int[] cpuOut = new int[blockSize];
        cpuOut[0] = quantized[0] * dcQ;
        for (int i = 1; i < blockSize; i++) cpuOut[i] = quantized[i] * acQ;

        // GPU dispatch: per-coefficient parallel.
        using var dQ = acc.Allocate1D<int>(blockSize);
        using var dOut = acc.Allocate1D<int>(blockSize);
        dQ.View.CopyFromCPU(quantized);
        dOut.MemSetToZero();

        var kernel = acc.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, ArrayView<int>, int, int>(DequantKernel);
        kernel(new Index1D(blockSize), dQ.View, dOut.View, dcQ, acQ);
        await acc.SynchronizeAsync();

        var gpuOut = await dOut.CopyToHostAsync();

        for (int i = 0; i < blockSize; i++)
        {
            if (cpuOut[i] != gpuOut[i])
                throw new Exception($"output[{i}]: cpu={cpuOut[i]} gpu={gpuOut[i]} (block={blockSize}, dcQ={dcQ}, acQ={acQ})");
        }
    }

    private static void DequantKernel(
        Index1D index,
        ArrayView<int> quantized, ArrayView<int> output, int dcQ, int acQ)
    {
        Av1InverseQuantizerGpu.DequantizeAt(quantized, 0, output, 0, dcQ, acQ, index.X);
    }
}
