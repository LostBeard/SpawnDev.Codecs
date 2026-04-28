// Tests for Av1ForwardQuantizerKernel - bit-exact match to
// Av1ForwardQuantizer.QuantizeBlock across multiple block sizes.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Av1;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public async Task Av1ForwardQuantizerKernel_4x4Blocks_MatchesCpuReference()
    {
        await RunAv1QuantizerMatchesCpu(coefsPerBlock: 16, blockCount: 16, seed: 0xA1FE);
    }

    [TestMethod]
    public async Task Av1ForwardQuantizerKernel_8x8Blocks_MatchesCpuReference()
    {
        await RunAv1QuantizerMatchesCpu(coefsPerBlock: 64, blockCount: 8, seed: 0xA1ED);
    }

    [TestMethod]
    public async Task Av1ForwardQuantizerKernel_16x16Blocks_MatchesCpuReference()
    {
        await RunAv1QuantizerMatchesCpu(coefsPerBlock: 256, blockCount: 4, seed: 0xA1DC);
    }

    private async Task RunAv1QuantizerMatchesCpu(int coefsPerBlock, int blockCount, int seed)
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var kernel = new Av1ForwardQuantizerKernel(acc);
            var rng = new Random(seed);
            int total = blockCount * coefsPerBlock;
            var coefs = new int[total];
            var dcQ = new int[blockCount];
            var acQ = new int[blockCount];
            for (int i = 0; i < total; i++) coefs[i] = rng.Next(-32768, 32768);
            for (int b = 0; b < blockCount; b++)
            {
                dcQ[b] = 8 + rng.Next(120);
                acQ[b] = 8 + rng.Next(120);
            }

            // CPU reference: QuantizeBlock per block.
            var cpuOut = new int[total];
            coefs.CopyTo(cpuOut, 0);
            for (int b = 0; b < blockCount; b++)
                Av1ForwardQuantizer.QuantizeBlock(
                    cpuOut.AsSpan(b * coefsPerBlock, coefsPerBlock),
                    dcQ[b], acQ[b]);

            // GPU kernel (in-place).
            using var dCoefs = acc.Allocate1D<int>(total);
            using var dDc = acc.Allocate1D<int>(blockCount);
            using var dAc = acc.Allocate1D<int>(blockCount);
            dCoefs.View.CopyFromCPU(coefs);
            dDc.View.CopyFromCPU(dcQ);
            dAc.View.CopyFromCPU(acQ);
            kernel.Run(dCoefs.View, dDc.View, dAc.View, blockCount, coefsPerBlock);
            await acc.SynchronizeAsync();
            var readback = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dCoefs);
            var gpuOut = new int[total];
            readback.AsSpan(0, total).CopyTo(gpuOut);

            int mismatches = 0;
            for (int i = 0; i < total; i++)
                if (cpuOut[i] != gpuOut[i]) mismatches++;
            Equal(0, mismatches, $"coefsPerBlock={coefsPerBlock}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
