// Cross-backend test for SilkNlsfDecodeGpu.DecodeAt. Verifies the full
// GPU NLSF decode pipeline (Unpack -> ResidualDequant -> WeightedAdd ->
// Stabilize) matches the CPU reference SilkNlsfDecoder.Decode bit-exactly
// across both NB/MB and WB codebooks with representative codebook paths.

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
    public async Task SilkNlsfDecodeGpu_NbMb_FirstStageZero_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            var codebook = SilkNlsfCodebookTables.NbMb;
            // Codebook path: cb1Index=0, then 10 residual indices.
            sbyte[] indices = { 0, 1, -1, 2, 0, -2, 1, 3, -1, 0, 2 };
            await NlsfDecodeAndVerify(acc, codebook, indices);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkNlsfDecodeGpu_NbMb_MiddleEntry_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            var codebook = SilkNlsfCodebookTables.NbMb;
            sbyte cb1Index = (sbyte)(codebook.NVectors / 2);
            sbyte[] indices = { cb1Index, 0, 1, -1, 2, -2, 0, 3, 1, -3, 2 };
            await NlsfDecodeAndVerify(acc, codebook, indices);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkNlsfDecodeGpu_Wb_TypicalFrame_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            var codebook = SilkNlsfCodebookTables.Wb;
            sbyte[] indices = { 7, 1, -1, 2, 0, -2, 3, -1, 1, 0, 2, -3, 1, -1, 2, 0, -1 };
            await NlsfDecodeAndVerify(acc, codebook, indices);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkNlsfDecodeGpu_Wb_LastEntry_RandomResiduals_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            var codebook = SilkNlsfCodebookTables.Wb;
            var rng = new Random(unchecked((int)0xDEAD0030u));
            sbyte[] indices = new sbyte[17];
            indices[0] = (sbyte)(codebook.NVectors - 1);
            for (int i = 1; i < 17; i++) indices[i] = (sbyte)rng.Next(-4, 5);
            await NlsfDecodeAndVerify(acc, codebook, indices);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static async Task NlsfDecodeAndVerify(
        Accelerator acc, SilkNlsfCodebook codebook, sbyte[] nlsfIndices)
    {
        int order = codebook.Order;

        // CPU reference.
        short[] cpuOut = new short[order];
        SilkNlsfDecoder.Decode(cpuOut, nlsfIndices, codebook);

        // GPU dispatch: single-thread.
        using var dIndices = acc.Allocate1D<sbyte>(nlsfIndices.Length);
        using var dCb1 = acc.Allocate1D<byte>(codebook.Cb1NlsfQ8.Length);
        using var dCbWght = acc.Allocate1D<short>(codebook.Cb1WghtQ9.Length);
        using var dEcSel = acc.Allocate1D<byte>(codebook.EcSel.Length);
        using var dPredSrc = acc.Allocate1D<byte>(codebook.PredQ8.Length);
        using var dDeltaMin = acc.Allocate1D<short>(codebook.DeltaMinQ15.Length);
        using var dOut = acc.Allocate1D<short>(order);
        using var dScratch = acc.Allocate1D<short>(2 * order);
        using var dPredScratch = acc.Allocate1D<byte>(order);

        dIndices.View.CopyFromCPU(nlsfIndices);
        dCb1.View.CopyFromCPU(codebook.Cb1NlsfQ8);
        dCbWght.View.CopyFromCPU(codebook.Cb1WghtQ9);
        dEcSel.View.CopyFromCPU(codebook.EcSel);
        dPredSrc.View.CopyFromCPU(codebook.PredQ8);
        dDeltaMin.View.CopyFromCPU(codebook.DeltaMinQ15);
        dOut.MemSetToZero();
        dScratch.MemSetToZero();
        dPredScratch.MemSetToZero();

        var kernel = acc.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<short>, ArrayView<sbyte>,
            ArrayView<byte>, ArrayView<short>, ArrayView<byte>, ArrayView<byte>, ArrayView<short>,
            int, int,
            ArrayView<short>, ArrayView<byte>>(NlsfDecodeKernel);
        kernel(new Index1D(1), dOut.View, dIndices.View,
            dCb1.View, dCbWght.View, dEcSel.View, dPredSrc.View, dDeltaMin.View,
            codebook.QuantStepSizeQ16, order,
            dScratch.View, dPredScratch.View);
        await acc.SynchronizeAsync();

        var gpuOut = await dOut.CopyToHostAsync();

        for (int i = 0; i < order; i++)
        {
            if (cpuOut[i] != gpuOut[i])
                throw new Exception($"pNlsfQ15[{i}]: cpu={cpuOut[i]} gpu={gpuOut[i]} (cb1Index={nlsfIndices[0]}, order={order})");
        }
    }

    private static void NlsfDecodeKernel(
        Index1D _,
        ArrayView<short> pNlsfQ15, ArrayView<sbyte> nlsfIndices,
        ArrayView<byte> cb1NlsfQ8, ArrayView<short> cb1WghtQ9,
        ArrayView<byte> ecSel, ArrayView<byte> predQ8Source, ArrayView<short> deltaMinQ15,
        int quantStepSizeQ16, int order,
        ArrayView<short> scratch, ArrayView<byte> predScratch)
    {
        SilkNlsfDecodeGpu.DecodeAt(pNlsfQ15, 0, nlsfIndices, 0,
            cb1NlsfQ8, 0, cb1WghtQ9, 0, ecSel, 0, predQ8Source, 0, deltaMinQ15, 0,
            quantStepSizeQ16, order, scratch, 0, predScratch, 0);
    }
}
