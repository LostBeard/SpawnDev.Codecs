// Tests for Vp8DcOnlyIdctAddKernel - bit-exact vs Vp8InverseTransform.DcOnlyIdctAdd.

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
    public async Task Vp8DcOnlyIdctAddKernel_RandomInput_MatchesCpuReference()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var kernel = new Vp8DcOnlyIdctAddKernel(acc);
            const int blockCount = 32;
            var rng = new Random(2028);
            var inputDc = new short[blockCount];
            var pred = new byte[blockCount * 16];
            for (int i = 0; i < blockCount; i++) inputDc[i] = (short)rng.Next(-2048, 2048);
            for (int i = 0; i < pred.Length; i++) pred[i] = (byte)rng.Next(0, 256);

            // CPU reference output.
            var cpuDst = new byte[blockCount * 16];
            for (int b = 0; b < blockCount; b++)
            {
                Vp8InverseTransform.DcOnlyIdctAdd(
                    inputDc[b],
                    pred.AsSpan(b * 16, 16), predStride: 4,
                    cpuDst.AsSpan(b * 16, 16), dstStride: 4);
            }

            // GPU kernel output.
            using var dDc = acc.Allocate1D<short>(blockCount);
            using var dPred = acc.Allocate1D<byte>(blockCount * 16);
            using var dDst = acc.Allocate1D<byte>(blockCount * 16);
            dDc.View.CopyFromCPU(inputDc);
            dPred.View.CopyFromCPU(pred);
            kernel.Run(dDc.View, dPred.View, dDst.View, blockCount);
            await acc.SynchronizeAsync();
            var readback = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dDst);
            var gpuDst = new byte[blockCount * 16];
            readback.AsSpan(0, gpuDst.Length).CopyTo(gpuDst);

            int mismatches = 0;
            for (int i = 0; i < cpuDst.Length; i++)
                if (cpuDst[i] != gpuDst[i]) mismatches++;
            Equal(0, mismatches);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
