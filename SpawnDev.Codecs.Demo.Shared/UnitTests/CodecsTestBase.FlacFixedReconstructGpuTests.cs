// Cross-backend test for FlacFixedReconstructGpu.ReconstructAt. Verifies
// the GPU FLAC FIXED-subframe reconstruction matches a direct CPU
// implementation of the FlacSubframeDecoder.DecodeFixed reconstruction
// loop bit-exactly across all 5 valid orders (0..4).

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
    public async Task FlacFixedReconstructGpu_Order0_NoOp_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // Order 0 -> no prediction, samples == residuals.
            await ReconstructAndVerify(acc, order: 0, length: 64, seed: 0xCAFE0000u);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task FlacFixedReconstructGpu_Order1_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            await ReconstructAndVerify(acc, order: 1, length: 64, seed: 0xCAFE0001u);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task FlacFixedReconstructGpu_Order2_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            await ReconstructAndVerify(acc, order: 2, length: 256, seed: 0xCAFE0002u);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task FlacFixedReconstructGpu_Order3_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            await ReconstructAndVerify(acc, order: 3, length: 512, seed: 0xCAFE0003u);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task FlacFixedReconstructGpu_Order4_FullBlock_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // Order 4 with 4096-block (typical FLAC default) - exercises the long-running
            // sequential reconstruction path.
            await ReconstructAndVerify(acc, order: 4, length: 4096, seed: 0xCAFE0004u);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static async Task ReconstructAndVerify(
        Accelerator acc, int order, int length, uint seed)
    {
        var rng = new Random(unchecked((int)seed));

        // Build initial buffer: warm-up samples [0..order) + residuals [order..length).
        // Use small random int values so the int+long math stays comfortable.
        int[] inBuffer = new int[length];
        for (int i = 0; i < length; i++) inBuffer[i] = rng.Next(-1000, 1001);

        // CPU reference - replicate FlacSubframeDecoder.DecodeFixed reconstruction loop.
        int[] cpuOut = (int[])inBuffer.Clone();
        for (int n = order; n < length; n++)
        {
            long pred = 0;
            switch (order)
            {
                case 1: pred = cpuOut[n - 1]; break;
                case 2: pred = 2L * cpuOut[n - 1] - cpuOut[n - 2]; break;
                case 3: pred = 3L * cpuOut[n - 1] - 3L * cpuOut[n - 2] + cpuOut[n - 3]; break;
                case 4: pred = 4L * cpuOut[n - 1] - 6L * cpuOut[n - 2] + 4L * cpuOut[n - 3] - cpuOut[n - 4]; break;
            }
            cpuOut[n] = (int)(cpuOut[n] + pred);
        }

        // GPU dispatch: single-thread per stream.
        using var dSamples = acc.Allocate1D<int>(length);
        dSamples.View.CopyFromCPU(inBuffer);

        var kernel = acc.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, int, int>(ReconstructKernel);
        kernel(new Index1D(1), dSamples.View, length, order);
        await acc.SynchronizeAsync();

        var gpuOut = await dSamples.CopyToHostAsync();

        for (int i = 0; i < length; i++)
        {
            if (cpuOut[i] != gpuOut[i])
                throw new Exception($"samples[{i}]: cpu={cpuOut[i]} gpu={gpuOut[i]} (order={order}, length={length})");
        }
    }

    private static void ReconstructKernel(
        Index1D _,
        ArrayView<int> samples, int length, int order)
    {
        FlacFixedReconstructGpu.ReconstructAt(samples, 0, length, order);
    }
}
