// Cross-backend test for FlacLpcResidualGpu.ComputeAt. Verifies the GPU
// FLAC LPC encoder-side residual computation matches a direct CPU
// implementation bit-exactly across representative LPC orders and
// quantization levels.

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
    public async Task FlacLpcResidualGpu_Order8_Q12_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            int[] coefs = { 3654, -3210, 2876, -2543, 2210, -1876, 1543, -1210 };
            await ResidualAndVerify(acc, coefs, length: 256, quantLevel: 12, seed: 0xC0FFEE08u);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task FlacLpcResidualGpu_Order16_Q14_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            int[] coefs = { 14000, -12000, 10000, -8000, 6000, -4000, 2000, -1000,
                              500,   -250,   100,   -50,   25,   -12,    6,    -3 };
            await ResidualAndVerify(acc, coefs, length: 1024, quantLevel: 14, seed: 0xC0FFEE16u);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task FlacLpcResidualGpu_Order32_FullBlock_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // Max FLAC order 32, full 4096-sample block.
            var rng = new Random(unchecked((int)0xC0FFEE32u));
            int[] coefs = new int[32];
            for (int i = 0; i < 32; i++) coefs[i] = rng.Next(-16384, 16384);
            await ResidualAndVerify(acc, coefs, length: 4096, quantLevel: 14, seed: 0xC0FFEEABu);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static async Task ResidualAndVerify(
        Accelerator acc, int[] coefs, int length, int quantLevel, uint seed)
    {
        int order = coefs.Length;
        var rng = new Random(unchecked((int)seed));

        // Build random PCM samples.
        int[] samples = new int[length];
        for (int i = 0; i < length; i++) samples[i] = rng.Next(-1000, 1001);

        // CPU reference - direct port of ComputeResidualWithQuantizedCoefs.
        int residualLen = length - order;
        int[] cpuResidual = new int[residualLen];
        for (int n = 0; n < residualLen; n++)
        {
            long pred = 0;
            for (int i = 0; i < order; i++)
                pred += (long)coefs[i] * samples[order + n - 1 - i];
            cpuResidual[n] = samples[order + n] - (int)(pred >> quantLevel);
        }

        // GPU dispatch: per-output-sample parallel.
        using var dSamples = acc.Allocate1D<int>(length);
        using var dCoefs = acc.Allocate1D<int>(order);
        using var dResidual = acc.Allocate1D<int>(residualLen);
        dSamples.View.CopyFromCPU(samples);
        dCoefs.View.CopyFromCPU(coefs);
        dResidual.MemSetToZero();

        var kernel = acc.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, int, int>(LpcResidualKernel);
        kernel(new Index1D(residualLen), dSamples.View, dCoefs.View, dResidual.View, order, quantLevel);
        await acc.SynchronizeAsync();

        var gpuResidual = await dResidual.CopyToHostAsync();

        for (int i = 0; i < residualLen; i++)
        {
            if (cpuResidual[i] != gpuResidual[i])
                throw new Exception($"residual[{i}]: cpu={cpuResidual[i]} gpu={gpuResidual[i]} (order={order}, q={quantLevel})");
        }
    }

    private static void LpcResidualKernel(
        Index1D index,
        ArrayView<int> samples, ArrayView<int> coefs, ArrayView<int> residual,
        int order, int quantLevel)
    {
        FlacLpcResidualGpu.ComputeAt(samples, 0, coefs, 0, residual, 0,
            order, quantLevel, index.X);
    }
}
