// Cross-backend test for SilkGainDecoderGpu.DequantizeAt. Verifies the GPU
// gain dequantizer matches the CPU reference SilkGainDecoder.Dequantize
// bit-exactly across conditional / non-conditional paths and nbSubfr 2/4.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Audio.Opus.Silk;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public async Task SilkGainDecoderGpu_Independent_NbSubfr4_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            // Independent first index, 4-subframe (20 ms) frame.
            sbyte[] indices = { 25, 5, -2, 3 };
            await DequantAndVerify(acc, indices, prevInd: 0, conditional: 0);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkGainDecoderGpu_Conditional_NbSubfr4_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            // Conditional (delta-coded) first index, with non-zero prevInd.
            sbyte[] indices = { 0, 5, 8, -2 };
            await DequantAndVerify(acc, indices, prevInd: 30, conditional: 1);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkGainDecoderGpu_NbSubfr2_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            // 10 ms frame: 2 subframes only.
            sbyte[] indices = { 35, 2 };
            await DequantAndVerify(acc, indices, prevInd: 15, conditional: 1);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkGainDecoderGpu_DoubleStepBranch_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            // Force the double-step branch: indTmp > doubleStepThreshold for low prevInd.
            sbyte[] indices = { 40, 39, 1, 0 };
            await DequantAndVerify(acc, indices, prevInd: 5, conditional: 1);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static async Task DequantAndVerify(
        Accelerator acc, sbyte[] indices, sbyte prevInd, int conditional)
    {
        int nbSubfr = indices.Length;

        // CPU reference (mutates prevInd).
        sbyte cpuPrev = prevInd;
        int[] cpuGain = new int[nbSubfr];
        SilkGainDecoder.Dequantize(cpuGain, indices, ref cpuPrev, conditional, nbSubfr);

        // GPU dispatch.
        using var dInd = acc.Allocate1D<sbyte>(nbSubfr);
        using var dGain = acc.Allocate1D<int>(nbSubfr);
        using var dPrev = acc.Allocate1D<int>(1);
        dInd.View.CopyFromCPU(indices);
        dGain.MemSetToZero();
        dPrev.View.CopyFromCPU(new[] { (int)prevInd });

        var kernel = acc.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, ArrayView<sbyte>, ArrayView<int>, int, int>(GainDecodeKernel);
        kernel(new Index1D(1), dGain.View, dInd.View, dPrev.View, conditional, nbSubfr);
        await acc.SynchronizeAsync();

        var gpuGain = await dGain.CopyToHostAsync();
        int gpuPrev = (await dPrev.CopyToHostAsync())[0];

        for (int i = 0; i < nbSubfr; i++)
        {
            if (cpuGain[i] != gpuGain[i])
                throw new Exception($"gainQ16[{i}]: cpu={cpuGain[i]} gpu={gpuGain[i]}");
        }
        if ((int)cpuPrev != gpuPrev)
            throw new Exception($"prevInd: cpu={cpuPrev} gpu={gpuPrev}");
    }

    private static void GainDecodeKernel(
        Index1D _,
        ArrayView<int> gainQ16, ArrayView<sbyte> ind, ArrayView<int> prevIndOut,
        int conditional, int nbSubfr)
    {
        SilkGainDecoderGpu.DequantizeAt(gainQ16, 0, ind, 0, prevIndOut, 0,
            conditional, nbSubfr);
    }
}
