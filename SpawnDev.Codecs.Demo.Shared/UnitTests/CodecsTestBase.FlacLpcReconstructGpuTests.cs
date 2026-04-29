// Cross-backend test for FlacLpcReconstructGpu.ReconstructAt. Verifies the
// GPU FLAC LPC-subframe reconstruction matches a direct CPU implementation
// of the FlacSubframeDecoder.DecodeLpc reconstruction loop bit-exactly
// across representative LPC orders and quantization levels.

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
    public async Task FlacLpcReconstructGpu_Order8_Q12_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            // Typical FLAC encoder choice: order 8, quant level 12.
            int[] coefs = { 3654, -3210, 2876, -2543, 2210, -1876, 1543, -1210 };
            await ReconstructAndVerify(acc, coefs, length: 256, quantLevel: 12, seed: 0xC0FFEE08u);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task FlacLpcReconstructGpu_Order16_Q14_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            // Order 16, max-typical quant level 14.
            int[] coefs = { 14000, -12000, 10000, -8000, 6000, -4000, 2000, -1000,
                              500,   -250,   100,   -50,   25,   -12,    6,    -3 };
            await ReconstructAndVerify(acc, coefs, length: 1024, quantLevel: 14, seed: 0xC0FFEE16u);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task FlacLpcReconstructGpu_Order32_Q14_FullBlock_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            // Max FLAC order 32, full 4096-sample block.
            var rng = new Random(unchecked((int)0xC0FFEE32u));
            int[] coefs = new int[32];
            for (int i = 0; i < 32; i++) coefs[i] = rng.Next(-16384, 16384);
            await ReconstructAndVerify(acc, coefs, length: 4096, quantLevel: 14, seed: 0xC0FFEEABu);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task FlacLpcReconstructGpu_Order4_LowQuant_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            // Low quant level (5) - exercises the right-shift path with smaller divisor.
            int[] coefs = { 100, -75, 50, -25 };
            await ReconstructAndVerify(acc, coefs, length: 128, quantLevel: 5, seed: 0xC0FFEE04u);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static async Task ReconstructAndVerify(
        Accelerator acc, int[] coefs, int length, int quantLevel, uint seed)
    {
        int order = coefs.Length;
        var rng = new Random(unchecked((int)seed));

        // Build initial buffer: warm-up samples + residuals (use modest int values
        // so the int+long math doesn't overflow).
        int[] inBuffer = new int[length];
        for (int i = 0; i < length; i++) inBuffer[i] = rng.Next(-1000, 1001);

        // CPU reference - direct port of FlacSubframeDecoder.DecodeLpc reconstruction loop.
        int[] cpuOut = (int[])inBuffer.Clone();
        for (int n = order; n < length; n++)
        {
            long pred = 0;
            for (int i = 0; i < order; i++)
                pred += (long)coefs[i] * cpuOut[n - 1 - i];
            cpuOut[n] = (int)(cpuOut[n] + (pred >> quantLevel));
        }

        // GPU dispatch: single-thread per stream.
        using var dSamples = acc.Allocate1D<int>(length);
        using var dCoefs = acc.Allocate1D<int>(order);
        dSamples.View.CopyFromCPU(inBuffer);
        dCoefs.View.CopyFromCPU(coefs);

        var kernel = acc.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, ArrayView<int>, int, int, int>(LpcReconstructKernel);
        kernel(new Index1D(1), dSamples.View, dCoefs.View, length, order, quantLevel);
        await acc.SynchronizeAsync();

        var gpuOut = await dSamples.CopyToHostAsync();

        for (int i = 0; i < length; i++)
        {
            if (cpuOut[i] != gpuOut[i])
                throw new Exception($"samples[{i}]: cpu={cpuOut[i]} gpu={gpuOut[i]} (order={order}, quant={quantLevel})");
        }
    }

    private static void LpcReconstructKernel(
        Index1D _,
        ArrayView<int> samples, ArrayView<int> coefs, int length, int order, int quantLevel)
    {
        FlacLpcReconstructGpu.ReconstructAt(samples, 0, coefs, 0, length, order, quantLevel);
    }
}
