// Tests for Vp8SubtractKernel - residual = (short)(src - pred).

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Vp8;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public async Task Vp8SubtractKernel_RandomInput_MatchesCpuReference()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Vp8SubtractKernel(acc);
            const int pixelCount = 1024;
            var rng = new Random(0x5AB);
            var src = new byte[pixelCount];
            var pred = new byte[pixelCount];
            for (int i = 0; i < pixelCount; i++) { src[i] = (byte)rng.Next(0, 256); pred[i] = (byte)rng.Next(0, 256); }

            var cpuOut = new short[pixelCount];
            for (int i = 0; i < pixelCount; i++) cpuOut[i] = (short)(src[i] - pred[i]);

            using var dSrc = acc.Allocate1D<byte>(pixelCount);
            using var dPred = acc.Allocate1D<byte>(pixelCount);
            using var dRes = acc.Allocate1D<short>(pixelCount);
            dSrc.View.CopyFromCPU(src);
            dPred.View.CopyFromCPU(pred);
            kernel.Run(dSrc.View, dPred.View, dRes.View, pixelCount);
            await acc.SynchronizeAsync();

            int mismatches = await GpuTestVerifyCodecs.CountShortMismatches(
                acc, dRes.View, cpuOut, pixelCount);
            Equal(0, mismatches);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
