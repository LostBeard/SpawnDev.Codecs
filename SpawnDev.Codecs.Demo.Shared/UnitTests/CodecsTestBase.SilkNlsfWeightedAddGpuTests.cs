// Cross-backend test for SilkNlsfWeightedAddGpu.ApplyAt. Verifies the GPU
// per-coefficient NLSF weighted-add stage matches a direct CPU
// implementation of the SilkNlsfDecoder.Decode weighted-add block bit-
// exactly across both NB/MB and WB codebooks.

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
    public async Task SilkNlsfWeightedAddGpu_NbMb_FirstStageZero_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            var codebook = SilkNlsfCodebookTables.NbMb;
            short[] resQ10 = { 5, -3, 2, -1, 0, 4, -2, 1, -4, 3 };
            await WeightedAddAndVerify(acc, codebook, cb1Index: 0, resQ10);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkNlsfWeightedAddGpu_NbMb_MiddleEntry_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            var codebook = SilkNlsfCodebookTables.NbMb;
            int cb1Index = codebook.NVectors / 2;
            short[] resQ10 = { 0, 1, -1, 2, -2, 3, -3, 4, -4, 0 };
            await WeightedAddAndVerify(acc, codebook, cb1Index, resQ10);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkNlsfWeightedAddGpu_Wb_FullScaleResidual_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            var codebook = SilkNlsfCodebookTables.Wb;
            // Larger residuals stress the divide + clamp paths.
            short[] resQ10 = { 100, -100, 80, -80, 60, -60, 40, -40,
                                20,  -20, 10, -10,  5,  -5,  2,  -2 };
            await WeightedAddAndVerify(acc, codebook, cb1Index: 7, resQ10);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkNlsfWeightedAddGpu_Wb_LastEntry_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            var codebook = SilkNlsfCodebookTables.Wb;
            int cb1Index = codebook.NVectors - 1;
            short[] resQ10 = new short[16];
            var rng = new Random(unchecked((int)0xDEAD0028u));
            for (int i = 0; i < 16; i++) resQ10[i] = (short)rng.Next(-200, 200);
            await WeightedAddAndVerify(acc, codebook, cb1Index, resQ10);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static async Task WeightedAddAndVerify(
        Accelerator acc, SilkNlsfCodebook codebook, int cb1Index, short[] resQ10)
    {
        int order = codebook.Order;
        int cbBase = cb1Index * order;

        // CPU reference (matches the SilkNlsfDecoder.Decode weighted-add block).
        short[] cpuOut = new short[order];
        for (int i = 0; i < order; i++)
        {
            int residual = (int)resQ10[i] << 14;
            int weightedResidual = residual / codebook.Cb1WghtQ9[cbBase + i];
            int nlsfQ15Tmp = weightedResidual + ((int)codebook.Cb1NlsfQ8[cbBase + i] << 7);
            if (nlsfQ15Tmp < 0) nlsfQ15Tmp = 0;
            else if (nlsfQ15Tmp > 32767) nlsfQ15Tmp = 32767;
            cpuOut[i] = (short)nlsfQ15Tmp;
        }

        // GPU dispatch: per-coefficient parallel.
        using var dRes = acc.Allocate1D<short>(order);
        using var dCb1 = acc.Allocate1D<byte>(codebook.Cb1NlsfQ8.Length);
        using var dCbWght = acc.Allocate1D<short>(codebook.Cb1WghtQ9.Length);
        using var dOut = acc.Allocate1D<short>(order);
        dRes.View.CopyFromCPU(resQ10);
        dCb1.View.CopyFromCPU(codebook.Cb1NlsfQ8);
        dCbWght.View.CopyFromCPU(codebook.Cb1WghtQ9);
        dOut.MemSetToZero();

        var kernel = acc.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<short>, ArrayView<short>, ArrayView<byte>, ArrayView<short>, int>(
            WeightedAddKernel);
        kernel(new Index1D(order), dOut.View, dRes.View, dCb1.View, dCbWght.View, cbBase);
        await acc.SynchronizeAsync();

        var gpuOut = await dOut.CopyToHostAsync();

        for (int i = 0; i < order; i++)
        {
            if (cpuOut[i] != gpuOut[i])
                throw new Exception($"pNlsfQ15[{i}]: cpu={cpuOut[i]} gpu={gpuOut[i]} (cb1Index={cb1Index})");
        }
    }

    private static void WeightedAddKernel(
        Index1D index,
        ArrayView<short> pNlsfQ15, ArrayView<short> resQ10,
        ArrayView<byte> cb1, ArrayView<short> cbWght,
        int cbBase)
    {
        SilkNlsfWeightedAddGpu.ApplyAt(pNlsfQ15, 0, resQ10, 0, cb1, cbBase, cbWght, cbBase, index.X);
    }
}
