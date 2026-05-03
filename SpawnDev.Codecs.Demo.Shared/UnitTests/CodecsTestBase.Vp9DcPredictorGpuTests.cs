// Cross-backend tests for Vp9DcPredictorGpu (the single-block
// in-kernel helper). Verifies byte-exact agreement with
// Vp9DcPredictor (CPU oracle) across every (size, variant)
// combination the future sequential encode kernel will exercise.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private static byte[] DcPredictCpu(byte[] above, byte[] left, int n, Vp9DcVariant variant)
    {
        var dst = new byte[n * n];
        switch (variant)
        {
            case Vp9DcVariant.Both:
                Vp9DcPredictor.DcPredict(above, left, dst, n, n);
                break;
            case Vp9DcVariant.TopOnly:
                Vp9DcPredictor.DcPredictTop(above, dst, n, n);
                break;
            case Vp9DcVariant.LeftOnly:
                Vp9DcPredictor.DcPredictLeft(left, dst, n, n);
                break;
            case Vp9DcVariant.None:
                Vp9DcPredictor.DcPredict128(dst, n, n);
                break;
        }
        return dst;
    }

    private static async Task<byte[]> DcPredictGpuAsync(
        Accelerator acc, byte[] above, byte[] left, int n, Vp9DcVariant variant)
    {
        using var kernel = new Vp9DcPredictorGpuTestKernel(acc);
        using var dAbove = acc.Allocate1D<byte>(n);
        using var dLeft = acc.Allocate1D<byte>(n);
        using var dDst = acc.Allocate1D<byte>(n * n);
        dAbove.View.CopyFromCPU(above);
        dLeft.View.CopyFromCPU(left);
        kernel.Run(dAbove.View, dLeft.View, dDst.View, n, variant);
        await acc.SynchronizeAsync();
        return await dDst.CopyToHostAsync();
    }

    private static async Task AssertDcPredictGpuMatchesCpuAsync(
        Accelerator acc, byte[] above, byte[] left, int n, Vp9DcVariant variant)
    {
        var cpu = DcPredictCpu(above, left, n, variant);
        var gpu = await DcPredictGpuAsync(acc, above, left, n, variant);
        for (int i = 0; i < n * n; i++)
        {
            if (cpu[i] != gpu[i])
                throw new Exception(
                    $"DC predict mismatch at offset {i}: cpu={cpu[i]} gpu={gpu[i]} " +
                    $"(n={n}, variant={variant})");
        }
    }

    [TestMethod]
    public async Task Vp9DcPredictorGpu_AllVariants_AllSizes_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            int[] sizes = { 4, 8, 16, 32 };
            Vp9DcVariant[] variants =
            {
                Vp9DcVariant.Both,
                Vp9DcVariant.TopOnly,
                Vp9DcVariant.LeftOnly,
                Vp9DcVariant.None,
            };

            var rng = new Random(unchecked((int)0xDC9C0DE7u));
            foreach (var n in sizes)
            {
                var above = new byte[n];
                var left = new byte[n];
                rng.NextBytes(above);
                rng.NextBytes(left);

                foreach (var v in variants)
                    await AssertDcPredictGpuMatchesCpuAsync(acc, above, left, n, v);
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9DcPredictorGpu_RandomSweep_MatchesCpu()
    {
        // Stress sweep: 32 random (above, left) pairs across all
        // sizes / variants. Catches any sum-overflow / shift / rounding
        // drift that hand-picked cases could miss.
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            var rng = new Random(unchecked((int)0xA1DCBA75u));
            int[] sizes = { 4, 8, 16, 32 };
            Vp9DcVariant[] variants =
            {
                Vp9DcVariant.Both,
                Vp9DcVariant.TopOnly,
                Vp9DcVariant.LeftOnly,
                Vp9DcVariant.None,
            };

            for (int trial = 0; trial < 32; trial++)
            {
                int n = sizes[rng.Next(sizes.Length)];
                var v = variants[rng.Next(variants.Length)];
                var above = new byte[n];
                var left = new byte[n];
                rng.NextBytes(above);
                rng.NextBytes(left);
                await AssertDcPredictGpuMatchesCpuAsync(acc, above, left, n, v);
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9DcPredictorGpu_Boundary_AllOnesAllZeros_MatchesCpu()
    {
        // Pin the rounding boundary cases: all-zeros must give DC=0,
        // all-255 must give DC=255 (no overflow above byte range).
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            foreach (var n in new[] { 4, 8, 16, 32 })
            {
                var zeros = new byte[n];
                var ones = new byte[n];
                for (int i = 0; i < n; i++) ones[i] = 255;

                await AssertDcPredictGpuMatchesCpuAsync(acc, zeros, zeros, n, Vp9DcVariant.Both);
                await AssertDcPredictGpuMatchesCpuAsync(acc, ones, ones, n, Vp9DcVariant.Both);
                await AssertDcPredictGpuMatchesCpuAsync(acc, ones, zeros, n, Vp9DcVariant.Both);
                await AssertDcPredictGpuMatchesCpuAsync(acc, zeros, ones, n, Vp9DcVariant.Both);
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
