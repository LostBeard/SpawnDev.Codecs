// Cross-backend test for SilkStereoDecodePredGpu.ApplyAt. Verifies the
// GPU stereo predictor dequantizer matches the CPU reference
// SilkStereoDecodePred.DequantizePredictors bit-exactly across
// representative index triples.

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
    public async Task SilkStereoDecodePredGpu_ZeroIndices_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            await DequantAndVerify(acc, new[] { 0, 0, 0 }, new[] { 0, 0, 0 });
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkStereoDecodePredGpu_TypicalIndices_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // Mid-table index triples from a typical stereo voiced frame.
            await DequantAndVerify(acc, new[] { 1, 2, 3 }, new[] { 2, 3, 2 });
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkStereoDecodePredGpu_BoundaryIndices_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // Near-edge index triples (idx[0]+3*idx[2] = 14 hits the last cell).
            await DequantAndVerify(acc, new[] { 2, 4, 4 }, new[] { 0, 0, 4 });
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkStereoDecodePredGpu_AllSweep_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // Sweep several combinations.
            await DequantAndVerify(acc, new[] { 0, 1, 2 }, new[] { 1, 4, 1 });
            await DequantAndVerify(acc, new[] { 1, 0, 4 }, new[] { 2, 2, 0 });
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static async Task DequantAndVerify(Accelerator acc, int[] ix0, int[] ix1)
    {
        // CPU reference (mutates ix0/ix1).
        int[] cpuIx0 = (int[])ix0.Clone();
        int[] cpuIx1 = (int[])ix1.Clone();
        int[] cpuPred = new int[2];
        SilkStereoDecodePred.DequantizePredictors(cpuIx0, cpuIx1, cpuPred);

        // GPU dispatch: single-thread.
        using var dPred = acc.Allocate1D<int>(2);
        using var dTab = acc.Allocate1D<short>(SilkStereoDecodePred.StereoPredQuantQ13.Length);
        dPred.MemSetToZero();
        dTab.View.CopyFromCPU(SilkStereoDecodePred.StereoPredQuantQ13);

        var kernel = acc.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, ArrayView<short>,
            int, int, int, int, int, int>(StereoPredKernel);
        kernel(new Index1D(1), dPred.View, dTab.View,
            ix0[0], ix0[1], ix0[2], ix1[0], ix1[1], ix1[2]);
        await acc.SynchronizeAsync();

        var gpuPred = await dPred.CopyToHostAsync();

        for (int i = 0; i < 2; i++)
        {
            if (cpuPred[i] != gpuPred[i])
                throw new Exception($"predQ13[{i}]: cpu={cpuPred[i]} gpu={gpuPred[i]} (ix0=[{ix0[0]},{ix0[1]},{ix0[2]}], ix1=[{ix1[0]},{ix1[1]},{ix1[2]}])");
        }
    }

    private static void StereoPredKernel(
        Index1D _,
        ArrayView<int> predQ13, ArrayView<short> stereoPredQuantQ13,
        int ix0_0, int ix0_1, int ix0_2,
        int ix1_0, int ix1_1, int ix1_2)
    {
        SilkStereoDecodePredGpu.ApplyAt(predQ13, 0, stereoPredQuantQ13, 0,
            ix0_0, ix0_1, ix0_2, ix1_0, ix1_1, ix1_2);
    }
}
