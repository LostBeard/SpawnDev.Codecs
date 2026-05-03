// Cross-backend tests for VorbisFloor1RenderCurveGpu. Verifies the
// full Floor 1 curve render (control point synthesis + sort + line
// rendering + tail) matches the CPU VorbisFloor1Curve.Render reference
// bit-exactly across CUDA + OpenCL + CPU.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Audio.Vorbis;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public async Task VorbisFloor1RenderCurveGpu_SmallFloor_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // Minimal floor: 4 control points across the half-block.
            var config = new VorbisFloor1Config
            {
                Partitions = 1,
                PartitionClassList = new[] { 0 },
                ClassDimensions = new[] { 2 },
                ClassSubclasses = new[] { 0 },
                ClassMasterbooks = new[] { -1 },
                ClassSubclassBooks = new[] { new[] { -1 } },
                Multiplier = 2,
                RangeBits = 8,
                XList = new[] { 0, 256, 80, 200 },
            };
            int[] decodedY = { 50, 100, 75, 90 };
            const int halfBlock = 256;
            await VerifyRender(acc, config, decodedY, halfBlock);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task VorbisFloor1RenderCurveGpu_LargerFloor_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // 8-control-point floor with multiplier 4 (steepest quantization).
            var config = new VorbisFloor1Config
            {
                Partitions = 1,
                PartitionClassList = new[] { 0 },
                ClassDimensions = new[] { 6 },
                ClassSubclasses = new[] { 0 },
                ClassMasterbooks = new[] { -1 },
                ClassSubclassBooks = new[] { new[] { -1 } },
                Multiplier = 4,
                RangeBits = 9,
                XList = new[] { 0, 512, 50, 150, 250, 320, 400, 460 },
            };
            int[] decodedY = { 30, 50, 40, 35, 45, 38, 42, 47 };
            const int halfBlock = 512;
            await VerifyRender(acc, config, decodedY, halfBlock);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static async Task VerifyRender(
        Accelerator acc, VorbisFloor1Config config, int[] decodedY, int halfBlock)
    {
        // CPU reference.
        var cpuOut = new float[halfBlock];
        VorbisFloor1Curve.Render(config, decodedY, halfBlock, cpuOut);

        // GPU.
        int values = config.XList.Length;
        var inverseDb = VorbisFloor1InverseDbGpu.BuildInverseDbTable();

        using var dXList = acc.Allocate1D<int>(values);
        using var dDecodedY = acc.Allocate1D<int>(values);
        using var dCurveOut = acc.Allocate1D<float>(halfBlock);
        using var dInverseDb = acc.Allocate1D<float>(inverseDb.Length);
        using var dScratchInt = acc.Allocate1D<int>(2 * values);
        using var dScratchByte = acc.Allocate1D<byte>(values);

        dXList.View.CopyFromCPU(config.XList);
        dDecodedY.View.CopyFromCPU(decodedY);
        dCurveOut.View.CopyFromCPU(new float[halfBlock]);
        dInverseDb.View.CopyFromCPU(inverseDb);
        dScratchInt.View.CopyFromCPU(new int[2 * values]);
        dScratchByte.View.CopyFromCPU(new byte[values]);

        var kernel = acc.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, ArrayView<int>,
            ArrayView<float>, ArrayView<float>,
            ArrayView<int>, ArrayView<byte>,
            int, int, int>(RenderCurveKernel);
        kernel(new Index1D(1),
            dXList.View, dDecodedY.View,
            dCurveOut.View, dInverseDb.View,
            dScratchInt.View, dScratchByte.View,
            values, config.Multiplier, halfBlock);
        await acc.SynchronizeAsync();

        var gpuOut = await dCurveOut.CopyToHostAsync();
        for (int i = 0; i < halfBlock; i++)
            if (cpuOut[i] != gpuOut[i])
                throw new Exception($"Curve[{i}]: cpu={cpuOut[i]} gpu={gpuOut[i]}");
    }

    private static void RenderCurveKernel(
        Index1D _,
        ArrayView<int> xList, ArrayView<int> decodedY,
        ArrayView<float> curveOut, ArrayView<float> inverseDb,
        ArrayView<int> scratchInt, ArrayView<byte> scratchByte,
        int values, int multiplier, int halfBlock)
    {
        VorbisFloor1RenderCurveGpu.Render(
            xList, 0, values,
            decodedY, 0,
            multiplier, halfBlock,
            curveOut, 0,
            inverseDb, 0,
            scratchInt, 0,
            scratchByte, 0);
    }
}
