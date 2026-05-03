// Cross-backend test for FlacChannelDecorrelationGpu.DecorrelateAt. Verifies
// the GPU per-sample stereo decorrelation matches a direct CPU implementation
// of the FlacFrameDecoder.Decode decorrelation loop bit-exactly across all
// 3 stereo encoding modes (LeftSide / RightSide / MidSide).

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
    public async Task FlacChannelDecorrelationGpu_LeftSide_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            await DecorrelateAndVerify(acc,
                mode: FlacChannelDecorrelationGpu.MODE_LEFT_SIDE,
                blockSize: 1024, seed: 0xDEC008u);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task FlacChannelDecorrelationGpu_RightSide_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            await DecorrelateAndVerify(acc,
                mode: FlacChannelDecorrelationGpu.MODE_RIGHT_SIDE,
                blockSize: 1024, seed: 0xDEC009u);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task FlacChannelDecorrelationGpu_MidSide_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            await DecorrelateAndVerify(acc,
                mode: FlacChannelDecorrelationGpu.MODE_MID_SIDE,
                blockSize: 1024, seed: 0xDEC00Au);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task FlacChannelDecorrelationGpu_MidSide_FullBlock_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // Full 4096-sample FLAC block - exercises the parallel pattern at scale.
            await DecorrelateAndVerify(acc,
                mode: FlacChannelDecorrelationGpu.MODE_MID_SIDE,
                blockSize: 4096, seed: 0xDEC04096u);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static async Task DecorrelateAndVerify(
        Accelerator acc, int mode, int blockSize, uint seed)
    {
        var rng = new Random(unchecked((int)seed));

        // Two-channel signal in a single buffer: ch0 at [0..blockSize), ch1 at [blockSize..2*blockSize).
        // For the side channel (ch1 in LeftSide / MidSide, ch0 in RightSide), libFLAC stores it at
        // bps + 1 - so use a slightly wider int range to exercise that.
        int[] inBuffer = new int[2 * blockSize];
        for (int i = 0; i < 2 * blockSize; i++) inBuffer[i] = rng.Next(-65536, 65536);

        // CPU reference - direct port of FlacFrameDecoder.Decode decorrelation block.
        int[] cpuOut = (int[])inBuffer.Clone();
        for (int n = 0; n < blockSize; n++)
        {
            int a = cpuOut[n];
            int b = cpuOut[blockSize + n];
            switch (mode)
            {
                case FlacChannelDecorrelationGpu.MODE_LEFT_SIDE:
                    cpuOut[blockSize + n] = a - b;
                    break;
                case FlacChannelDecorrelationGpu.MODE_RIGHT_SIDE:
                    cpuOut[n] = a + b;
                    break;
                case FlacChannelDecorrelationGpu.MODE_MID_SIDE:
                    int midScaled = (a << 1) | (b & 1);
                    cpuOut[n] = (midScaled + b) >> 1;
                    cpuOut[blockSize + n] = (midScaled - b) >> 1;
                    break;
            }
        }

        // GPU dispatch: per-sample parallel.
        using var dSamples = acc.Allocate1D<int>(2 * blockSize);
        dSamples.View.CopyFromCPU(inBuffer);

        var kernel = acc.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, int, int>(DecorrelateKernel);
        kernel(new Index1D(blockSize), dSamples.View, blockSize, mode);
        await acc.SynchronizeAsync();

        var gpuOut = await dSamples.CopyToHostAsync();

        for (int i = 0; i < 2 * blockSize; i++)
        {
            if (cpuOut[i] != gpuOut[i])
                throw new Exception($"samples[{i}]: cpu={cpuOut[i]} gpu={gpuOut[i]} (mode={mode}, block={blockSize})");
        }
    }

    private static void DecorrelateKernel(
        Index1D index,
        ArrayView<int> samples, int blockSize, int mode)
    {
        FlacChannelDecorrelationGpu.DecorrelateAt(samples, 0, blockSize, mode, index.X);
    }
}
