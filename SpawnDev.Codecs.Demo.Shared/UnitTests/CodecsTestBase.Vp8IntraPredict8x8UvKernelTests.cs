// Tests for Vp8IntraPredict8x8UvKernel - bit-exact vs Vp8IntraPredictor8x8.Predict.
// Each block in the batch picks a different mode + neighbor-availability
// corner to exercise all 16 (mode, haveAbove, haveLeft) combinations.

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
    public async Task Vp8IntraPredict8x8UvKernel_AllModesAndAvailabilities_MatchesCpuReference()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var kernel = new Vp8IntraPredict8x8UvKernel(acc);
            // 16 blocks: 4 modes x 4 (haveAbove, haveLeft) corners.
            const int blockCount = 16;
            var rng = new Random(2031);
            var above = new byte[blockCount * 8];
            var left = new byte[blockCount * 8];
            var topLeft = new byte[blockCount];
            var modeAndFlags = new byte[blockCount];
            for (int i = 0; i < above.Length; i++) above[i] = (byte)rng.Next(0, 256);
            for (int i = 0; i < left.Length; i++) left[i] = (byte)rng.Next(0, 256);
            for (int b = 0; b < blockCount; b++)
            {
                topLeft[b] = (byte)rng.Next(0, 256);
                int mode = b & 0x3;
                int corner = (b >> 2) & 0x3;
                bool haveAbove = (corner & 1) != 0;
                bool haveLeft = (corner & 2) != 0;
                modeAndFlags[b] = (byte)(mode | (haveAbove ? 0x10 : 0) | (haveLeft ? 0x20 : 0));
            }

            // CPU reference output.
            var cpuDst = new byte[blockCount * 64];
            for (int b = 0; b < blockCount; b++)
            {
                int mode = modeAndFlags[b] & 0x0F;
                bool haveAbove = (modeAndFlags[b] & 0x10) != 0;
                bool haveLeft = (modeAndFlags[b] & 0x20) != 0;
                Vp8IntraPredictor8x8.Predict(
                    (Vp8IntraMode16x16)mode,
                    above.AsSpan(b * 8, 8),
                    left.AsSpan(b * 8, 8),
                    topLeft[b],
                    haveAbove, haveLeft,
                    cpuDst.AsSpan(b * 64, 64), stride: 8);
            }

            // GPU kernel output - stays on device, verified GPU-side.
            using var dAbove = acc.Allocate1D<byte>(above.Length);
            using var dLeft = acc.Allocate1D<byte>(left.Length);
            using var dTopLeft = acc.Allocate1D<byte>(topLeft.Length);
            using var dModeFlags = acc.Allocate1D<byte>(modeAndFlags.Length);
            using var dDst = acc.Allocate1D<byte>(blockCount * 64);
            dAbove.View.CopyFromCPU(above);
            dLeft.View.CopyFromCPU(left);
            dTopLeft.View.CopyFromCPU(topLeft);
            dModeFlags.View.CopyFromCPU(modeAndFlags);
            kernel.Run(dAbove.View, dLeft.View, dTopLeft.View, dModeFlags.View, dDst.View, blockCount);
            await acc.SynchronizeAsync();

            int mismatches = await GpuTestVerifyCodecs.CountByteMismatches(
                acc, dDst.View, cpuDst, cpuDst.Length);
            Equal(0, mismatches);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
