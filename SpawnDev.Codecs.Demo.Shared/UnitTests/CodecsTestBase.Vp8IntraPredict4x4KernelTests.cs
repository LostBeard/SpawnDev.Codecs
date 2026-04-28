// Tests for Vp8IntraPredict4x4Kernel - bit-exact vs Vp8IntraPredictor4x4.Predict.
// Each block in the batch uses a different mode to exercise all 10 cases.

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
    public async Task Vp8IntraPredict4x4Kernel_AllModes_MatchesCpuReference()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var kernel = new Vp8IntraPredict4x4Kernel(acc);
            // 30 blocks: 3 of each of 10 modes with different random neighbours.
            const int blockCount = 30;
            var rng = new Random(2029);
            var above = new byte[blockCount * 9];
            var left = new byte[blockCount * 4];
            var modes = new byte[blockCount];
            for (int i = 0; i < above.Length; i++) above[i] = (byte)rng.Next(0, 256);
            for (int i = 0; i < left.Length; i++) left[i] = (byte)rng.Next(0, 256);
            for (int b = 0; b < blockCount; b++) modes[b] = (byte)(b % 10);

            // CPU reference output. Vp8IntraPredictor4x4 uses an above buffer
            // where index 'aboveOffset' = above[0] and aboveOffset-1 = above[-1].
            // Our kernel layout is: aBase+0 = above[-1], aBase+1..aBase+8 = above[0..7].
            // So when calling the CPU reference we pass aboveOffset = 1 into the
            // 9-byte slice.
            var cpuDst = new byte[blockCount * 16];
            for (int b = 0; b < blockCount; b++)
            {
                var aboveSlice = above.AsSpan(b * 9, 9);
                var leftSlice = left.AsSpan(b * 4, 4);
                var dstSlice = cpuDst.AsSpan(b * 16, 16);
                Vp8IntraPredictor4x4.Predict(
                    (Vp8IntraMode4x4)modes[b],
                    aboveSlice, aboveOffset: 1,
                    leftSlice,
                    dstSlice, stride: 4);
            }

            // GPU kernel output.
            using var dAbove = acc.Allocate1D<byte>(above.Length);
            using var dLeft = acc.Allocate1D<byte>(left.Length);
            using var dModes = acc.Allocate1D<byte>(modes.Length);
            using var dDst = acc.Allocate1D<byte>(blockCount * 16);
            dAbove.View.CopyFromCPU(above);
            dLeft.View.CopyFromCPU(left);
            dModes.View.CopyFromCPU(modes);
            kernel.Run(dAbove.View, dLeft.View, dModes.View, dDst.View, blockCount);
            await acc.SynchronizeAsync();
            var readback = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dDst);
            var gpuDst = new byte[blockCount * 16];
            readback.AsSpan(0, gpuDst.Length).CopyTo(gpuDst);

            int mismatches = 0;
            int firstMismatchBlock = -1;
            for (int b = 0; b < blockCount && firstMismatchBlock < 0; b++)
                for (int i = 0; i < 16; i++)
                    if (cpuDst[b * 16 + i] != gpuDst[b * 16 + i])
                    {
                        firstMismatchBlock = b;
                        mismatches++;
                        break;
                    }
            Equal(0, mismatches);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
