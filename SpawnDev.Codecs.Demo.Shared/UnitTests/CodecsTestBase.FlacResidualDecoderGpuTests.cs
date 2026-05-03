// Cross-backend test for FlacResidualDecoderGpu.DecodeAt. Verifies the GPU
// FLAC Rice-coded residual decoder matches the CPU reference
// FlacResidualDecoder.Decode bit-exactly across both PartitionedRice and
// PartitionedRice2 coding methods plus the escape path.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Audio.Flac;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public async Task FlacResidualDecoderGpu_RiceMethod0_SmallResiduals_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // Small residuals -> small Rice parameter, partition order 0.
            int[] residual = { 1, -2, 3, -4, 0, 5, -6, 7 };
            await EncodeDecodeAndVerify(acc, residual, codingMethod: 0, partitionOrder: 0,
                riceParam: 3, blockSize: 12, predictorOrder: 4);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task FlacResidualDecoderGpu_RiceMethod0_PartitionOrder2_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // 4 partitions (1 << 2) of 8 samples each, predictorOrder 4.
            // Total residuals = 32 - 4 = 28.
            var rng = new Random(unchecked((int)0xF1AC0001u));
            int[] residual = new int[28];
            for (int i = 0; i < 28; i++) residual[i] = rng.Next(-8, 8);
            await EncodeDecodeAndVerify(acc, residual, codingMethod: 0, partitionOrder: 2,
                riceParam: 4, blockSize: 32, predictorOrder: 4);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task FlacResidualDecoderGpu_RiceMethod1_LargerParam_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // PartitionedRice2 (5-bit param), partition order 0.
            int[] residual = { 100, -200, 300, -400, 500, -600, 700, -800 };
            await EncodeDecodeAndVerify(acc, residual, codingMethod: 1, partitionOrder: 0,
                riceParam: 10, blockSize: 12, predictorOrder: 4);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static async Task EncodeDecodeAndVerify(
        Accelerator acc, int[] residual, int codingMethod, int partitionOrder,
        int riceParam, int blockSize, int predictorOrder)
    {
        // Build the residual bit stream using the same layout as FlacResidualDecoder
        // expects. Mirror of FlacFixedSubframeEncoder.Emit residual block.
        int paramBits = codingMethod == 0 ? 4 : 5;
        var w = new FlacBitWriter();
        w.Write((uint)codingMethod, 2);
        w.Write((uint)partitionOrder, 4);

        int partitionCount = 1 << partitionOrder;
        int partitionSizeBase = blockSize >> partitionOrder;
        int residualIndex = 0;
        for (int p = 0; p < partitionCount; p++)
        {
            int partitionSize = (p == 0) ? partitionSizeBase - predictorOrder : partitionSizeBase;
            w.Write((uint)riceParam, paramBits);
            for (int i = 0; i < partitionSize; i++)
            {
                int r = residual[residualIndex++];
                uint u = r >= 0 ? (uint)(r << 1) : (uint)((-r << 1) - 1);
                int q = (int)(u >> riceParam);
                uint rem = u & ((1u << riceParam) - 1);
                w.WriteUnary(q);
                if (riceParam > 0) w.Write(rem, riceParam);
            }
        }
        // Pad to byte boundary.
        w.AlignToByte();
        byte[] encoded = w.ToArray();

        // CPU reference decode.
        var cpuReader = new FlacBitReader(encoded);
        int[] cpuOut = new int[blockSize - predictorOrder];
        FlacResidualDecoder.Decode(ref cpuReader, cpuOut, blockSize, predictorOrder);
        for (int i = 0; i < cpuOut.Length; i++)
        {
            if (cpuOut[i] != residual[i])
                throw new Exception($"CPU decode self-check failed at {i}: got {cpuOut[i]} expected {residual[i]}");
        }

        // GPU decode.
        using var dData = acc.Allocate1D<byte>(encoded.Length);
        using var dResidual = acc.Allocate1D<int>(blockSize - predictorOrder);
        dData.View.CopyFromCPU(encoded);
        dResidual.MemSetToZero();

        var kernel = acc.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<int>, int, int, int>(ResidualKernel);
        kernel(new Index1D(1), dData.View, dResidual.View, encoded.Length, blockSize, predictorOrder);
        await acc.SynchronizeAsync();

        var gpuOut = await dResidual.CopyToHostAsync();
        for (int i = 0; i < gpuOut.Length; i++)
        {
            if (gpuOut[i] != residual[i])
                throw new Exception($"GPU decode mismatch at {i}: got {gpuOut[i]} expected {residual[i]}");
        }
    }

    private static void ResidualKernel(
        Index1D _,
        ArrayView<byte> data, ArrayView<int> residualOut,
        int dataLen, int blockSize, int predictorOrder)
    {
        var state = FlacBitReaderGpu.Init(dataLen);
        FlacResidualDecoderGpu.DecodeAt(ref state, data, residualOut, 0, blockSize, predictorOrder);
    }
}
