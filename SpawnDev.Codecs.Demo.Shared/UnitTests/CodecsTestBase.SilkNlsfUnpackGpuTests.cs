// Cross-backend test for SilkNlsfUnpackGpu.UnpackPairAt. Verifies the GPU
// per-coefficient-pair unpacker matches the CPU reference SilkNlsfUnpack.Unpack
// bit-exactly for both NB/MB (order 10) and WB (order 16) codebooks.

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
    public async Task SilkNlsfUnpackGpu_NbMb_Order10_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            await UnpackAndVerify(acc, SilkNlsfCodebookTables.NbMb, cb1Index: 0);
            await UnpackAndVerify(acc, SilkNlsfCodebookTables.NbMb, cb1Index: 5);
            await UnpackAndVerify(acc, SilkNlsfCodebookTables.NbMb,
                cb1Index: SilkNlsfCodebookTables.NbMb.NVectors - 1);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkNlsfUnpackGpu_Wb_Order16_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            await UnpackAndVerify(acc, SilkNlsfCodebookTables.Wb, cb1Index: 0);
            await UnpackAndVerify(acc, SilkNlsfCodebookTables.Wb, cb1Index: 7);
            await UnpackAndVerify(acc, SilkNlsfCodebookTables.Wb,
                cb1Index: SilkNlsfCodebookTables.Wb.NVectors - 1);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static async Task UnpackAndVerify(
        Accelerator acc, SilkNlsfCodebook codebook, int cb1Index)
    {
        int order = codebook.Order;

        // CPU reference.
        short[] cpuEcIx = new short[order];
        byte[] cpuPredQ8 = new byte[order];
        SilkNlsfUnpack.Unpack(cpuEcIx, cpuPredQ8, codebook, cb1Index);

        // GPU dispatch: per-coefficient-pair parallel (order/2 threads).
        int ecSelOffset = cb1Index * order / 2;
        using var dEcSel = acc.Allocate1D<byte>(codebook.EcSel.Length);
        using var dPredSrc = acc.Allocate1D<byte>(codebook.PredQ8.Length);
        using var dEcIx = acc.Allocate1D<short>(order);
        using var dPredOut = acc.Allocate1D<byte>(order);
        dEcSel.View.CopyFromCPU(codebook.EcSel);
        dPredSrc.View.CopyFromCPU(codebook.PredQ8);
        dEcIx.MemSetToZero();
        dPredOut.MemSetToZero();

        var kernel = acc.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<short>, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>,
            int, int>(UnpackKernel);
        kernel(new Index1D(order / 2), dEcIx.View, dPredOut.View, dEcSel.View, dPredSrc.View,
            ecSelOffset, order);
        await acc.SynchronizeAsync();

        var gpuEcIx = await dEcIx.CopyToHostAsync();
        var gpuPredOut = await dPredOut.CopyToHostAsync();

        for (int i = 0; i < order; i++)
        {
            if (cpuEcIx[i] != gpuEcIx[i])
                throw new Exception($"ecIx[{i}]: cpu={cpuEcIx[i]} gpu={gpuEcIx[i]} (order={order}, cb1Index={cb1Index})");
            if (cpuPredQ8[i] != gpuPredOut[i])
                throw new Exception($"predQ8[{i}]: cpu={cpuPredQ8[i]} gpu={gpuPredOut[i]} (order={order}, cb1Index={cb1Index})");
        }
    }

    private static void UnpackKernel(
        Index1D index,
        ArrayView<short> ecIx, ArrayView<byte> predQ8Out,
        ArrayView<byte> ecSel, ArrayView<byte> predQ8Source,
        int ecSelOffset, int order)
    {
        SilkNlsfUnpackGpu.UnpackPairAt(ecIx, 0, predQ8Out, 0,
            ecSel, ecSelOffset, predQ8Source, 0, order, index.X);
    }
}
