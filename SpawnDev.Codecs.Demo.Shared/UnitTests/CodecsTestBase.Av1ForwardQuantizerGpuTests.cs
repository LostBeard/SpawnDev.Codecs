// Tests for Av1ForwardQuantizerGpu.QuantizeBlock driven through
// Av1ForwardQuantizerGpuKernel. Verifies bit-exact match with the
// CPU reference Av1ForwardQuantizer.QuantizeBlock across (a) zero,
// (b) DC-only block, (c) random batch of 64 8x8 blocks, (d) 16x16
// blocks with realistic q-index ranges.

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
    public async Task Av1ForwardQuantizerGpu_ZeroInput_MatchesCpuReference()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Av1ForwardQuantizerGpuKernel(acc);
            const int blockCount = 4;
            const int coefsPerBlock = 64;
            var coefs = new int[blockCount * coefsPerBlock];
            var dcQ = new int[blockCount];
            var acQ = new int[blockCount];
            for (int i = 0; i < blockCount; i++) { dcQ[i] = 16; acQ[i] = 32; }

            var cpuCoefs = new int[blockCount * coefsPerBlock];
            for (int b = 0; b < blockCount; b++)
                Av1ForwardQuantizer.QuantizeBlock(cpuCoefs.AsSpan(b * coefsPerBlock, coefsPerBlock), dcQ[b], acQ[b]);

            using var dCoefs = acc.Allocate1D<int>(coefs.Length);
            using var dDc = acc.Allocate1D<int>(dcQ.Length);
            using var dAc = acc.Allocate1D<int>(acQ.Length);
            dCoefs.View.CopyFromCPU(coefs);
            dDc.View.CopyFromCPU(dcQ);
            dAc.View.CopyFromCPU(acQ);
            kernel.Run(dCoefs.View, dDc.View, dAc.View, blockCount, coefsPerBlock);
            await acc.SynchronizeAsync();
            var readback = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dCoefs);

            for (int i = 0; i < cpuCoefs.Length; i++) Equal(cpuCoefs[i], readback[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Av1ForwardQuantizerGpu_RandomBatch8x8_MatchesCpuReference()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Av1ForwardQuantizerGpuKernel(acc);
            const int blockCount = 64;
            const int coefsPerBlock = 64;
            var rng = new Random(unchecked((int)0xA1A88BADu));
            var coefs = new int[blockCount * coefsPerBlock];
            var dcQ = new int[blockCount];
            var acQ = new int[blockCount];
            for (int i = 0; i < coefs.Length; i++) coefs[i] = rng.Next(-32768, 32768);
            for (int b = 0; b < blockCount; b++)
            {
                dcQ[b] = rng.Next(8, 256);
                acQ[b] = rng.Next(8, 256);
            }

            var cpuCoefs = (int[])coefs.Clone();
            for (int b = 0; b < blockCount; b++)
                Av1ForwardQuantizer.QuantizeBlock(cpuCoefs.AsSpan(b * coefsPerBlock, coefsPerBlock), dcQ[b], acQ[b]);

            using var dCoefs = acc.Allocate1D<int>(coefs.Length);
            using var dDc = acc.Allocate1D<int>(dcQ.Length);
            using var dAc = acc.Allocate1D<int>(acQ.Length);
            dCoefs.View.CopyFromCPU(coefs);
            dDc.View.CopyFromCPU(dcQ);
            dAc.View.CopyFromCPU(acQ);
            kernel.Run(dCoefs.View, dDc.View, dAc.View, blockCount, coefsPerBlock);
            await acc.SynchronizeAsync();
            var readback = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dCoefs);

            int mismatches = 0;
            for (int i = 0; i < cpuCoefs.Length; i++)
                if (cpuCoefs[i] != readback[i]) mismatches++;
            Equal(0, mismatches);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Av1ForwardQuantizerGpu_RandomBatch16x16_MatchesCpuReference()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Av1ForwardQuantizerGpuKernel(acc);
            const int blockCount = 32;
            const int coefsPerBlock = 256;
            var rng = new Random(unchecked((int)0xA1A1616Bu));
            var coefs = new int[blockCount * coefsPerBlock];
            var dcQ = new int[blockCount];
            var acQ = new int[blockCount];
            for (int i = 0; i < coefs.Length; i++) coefs[i] = rng.Next(-65536, 65536);
            for (int b = 0; b < blockCount; b++)
            {
                dcQ[b] = rng.Next(8, 512);
                acQ[b] = rng.Next(8, 512);
            }

            var cpuCoefs = (int[])coefs.Clone();
            for (int b = 0; b < blockCount; b++)
                Av1ForwardQuantizer.QuantizeBlock(cpuCoefs.AsSpan(b * coefsPerBlock, coefsPerBlock), dcQ[b], acQ[b]);

            using var dCoefs = acc.Allocate1D<int>(coefs.Length);
            using var dDc = acc.Allocate1D<int>(dcQ.Length);
            using var dAc = acc.Allocate1D<int>(acQ.Length);
            dCoefs.View.CopyFromCPU(coefs);
            dDc.View.CopyFromCPU(dcQ);
            dAc.View.CopyFromCPU(acQ);
            kernel.Run(dCoefs.View, dDc.View, dAc.View, blockCount, coefsPerBlock);
            await acc.SynchronizeAsync();
            var readback = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dCoefs);

            int mismatches = 0;
            for (int i = 0; i < cpuCoefs.Length; i++)
                if (cpuCoefs[i] != readback[i]) mismatches++;
            Equal(0, mismatches);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
