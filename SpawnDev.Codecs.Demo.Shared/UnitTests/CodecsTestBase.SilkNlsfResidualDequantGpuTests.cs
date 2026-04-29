// Cross-backend test for SilkNlsfResidualDequantGpu.DequantizeAt. Verifies
// the GPU NLSF residual dequantizer matches the CPU reference
// SilkNlsfDecoder.ResidualDequant bit-exactly across both NB/MB (order 10)
// and WB (order 16) codebooks.

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
    public async Task SilkNlsfResidualDequantGpu_NbMb_Order10_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            var codebook = SilkNlsfCodebookTables.NbMb;
            // Realistic predictor variant 0 (lower half of PredQ8).
            byte[] predQ8 = new byte[codebook.Order];
            Array.Copy(codebook.PredQ8, 0, predQ8, 0, codebook.Order - 1);
            // Last entry is unused by ResidualDequant per libopus, set to 0.
            sbyte[] indices = { 0, 1, -1, 2, 0, -2, 1, 3, -1, 0 };
            await DequantAndVerify(acc, indices, predQ8, codebook.QuantStepSizeQ16, codebook.Order);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkNlsfResidualDequantGpu_Wb_Order16_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            var codebook = SilkNlsfCodebookTables.Wb;
            byte[] predQ8 = new byte[codebook.Order];
            Array.Copy(codebook.PredQ8, 0, predQ8, 0, codebook.Order - 1);
            sbyte[] indices = { 1, -1, 2, 0, -2, 3, -1, 1, 0, 2, -3, 1, -1, 2, 0, -1 };
            await DequantAndVerify(acc, indices, predQ8, codebook.QuantStepSizeQ16, codebook.Order);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkNlsfResidualDequantGpu_AllZeroIndices_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            // All-zero indices stress the predictor-only path (both branches skip).
            var codebook = SilkNlsfCodebookTables.NbMb;
            byte[] predQ8 = new byte[codebook.Order];
            Array.Copy(codebook.PredQ8, codebook.Order - 1, predQ8, 0, codebook.Order - 1);
            sbyte[] indices = new sbyte[10];
            await DequantAndVerify(acc, indices, predQ8, codebook.QuantStepSizeQ16, codebook.Order);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkNlsfResidualDequantGpu_MaxIndices_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            // Large-magnitude indices stress both quant-adjust branches.
            var codebook = SilkNlsfCodebookTables.Wb;
            byte[] predQ8 = new byte[codebook.Order];
            Array.Copy(codebook.PredQ8, 0, predQ8, 0, codebook.Order - 1);
            sbyte[] indices = { 8, -8, 7, -7, 6, -6, 5, -5, 4, -4, 3, -3, 2, -2, 1, -1 };
            await DequantAndVerify(acc, indices, predQ8, codebook.QuantStepSizeQ16, codebook.Order);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static async Task DequantAndVerify(
        Accelerator acc, sbyte[] indices, byte[] predCoefQ8, int quantStepSizeQ16, int order)
    {
        // CPU reference.
        short[] cpuOut = new short[order];
        SilkNlsfDecoder.ResidualDequant(cpuOut, indices, predCoefQ8, quantStepSizeQ16, order);

        // GPU dispatch: single-thread per stream (sequential).
        using var dOut = acc.Allocate1D<short>(order);
        using var dIndices = acc.Allocate1D<sbyte>(order);
        using var dPred = acc.Allocate1D<byte>(order);
        dIndices.View.CopyFromCPU(indices);
        dPred.View.CopyFromCPU(predCoefQ8);
        dOut.MemSetToZero();

        var kernel = acc.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<short>, ArrayView<sbyte>, ArrayView<byte>,
            int, int>(NlsfResidualKernel);
        kernel(new Index1D(1), dOut.View, dIndices.View, dPred.View, quantStepSizeQ16, order);
        await acc.SynchronizeAsync();

        var gpuOut = await dOut.CopyToHostAsync();

        for (int i = 0; i < order; i++)
        {
            if (cpuOut[i] != gpuOut[i])
                throw new Exception($"xQ10[{i}]: cpu={cpuOut[i]} gpu={gpuOut[i]} (order={order})");
        }
    }

    private static void NlsfResidualKernel(
        Index1D _,
        ArrayView<short> xQ10, ArrayView<sbyte> indices, ArrayView<byte> predCoefQ8,
        int quantStepSizeQ16, int order)
    {
        SilkNlsfResidualDequantGpu.DequantizeAt(xQ10, 0, indices, 0, predCoefQ8, 0,
            quantStepSizeQ16, order);
    }
}
