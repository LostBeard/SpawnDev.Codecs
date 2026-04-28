// Tests for Vp8DequantizerKernel - bit-exact match to coef[0] *= dcQ;
// coef[i] *= acQ for i in 1..15.

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
    public async Task Vp8DequantizerKernel_RandomInput_MatchesCpuReference()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var kernel = new Vp8DequantizerKernel(acc);
            const int blockCount = 24;
            var rng = new Random(43);
            var coefs = new short[blockCount * 16];
            var dcQ = new short[blockCount];
            var acQ = new short[blockCount];
            for (int i = 0; i < coefs.Length; i++) coefs[i] = (short)rng.Next(-128, 128);
            for (int b = 0; b < blockCount; b++)
            {
                dcQ[b] = (short)(8 + rng.Next(120));
                acQ[b] = (short)(8 + rng.Next(120));
            }

            // CPU reference output (mirrors Vp8KeyframeWalker dequant pattern).
            var cpuOut = new short[blockCount * 16];
            for (int b = 0; b < blockCount; b++)
            {
                cpuOut[b * 16 + 0] = (short)(coefs[b * 16 + 0] * dcQ[b]);
                for (int j = 1; j < 16; j++)
                    cpuOut[b * 16 + j] = (short)(coefs[b * 16 + j] * acQ[b]);
            }

            // GPU kernel output (in-place).
            using var dCoefs = acc.Allocate1D<short>(blockCount * 16);
            using var dDc = acc.Allocate1D<short>(blockCount);
            using var dAc = acc.Allocate1D<short>(blockCount);
            dCoefs.View.CopyFromCPU(coefs);
            dDc.View.CopyFromCPU(dcQ);
            dAc.View.CopyFromCPU(acQ);
            kernel.Run(dCoefs.View, dDc.View, dAc.View, blockCount);
            await acc.SynchronizeAsync();
            var readback = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dCoefs);
            var gpuOut = new short[blockCount * 16];
            readback.AsSpan(0, gpuOut.Length).CopyTo(gpuOut);

            int mismatches = 0;
            for (int i = 0; i < cpuOut.Length; i++)
                if (cpuOut[i] != gpuOut[i]) mismatches++;
            Equal(0, mismatches);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
