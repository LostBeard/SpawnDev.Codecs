// Tests for Vp8ForwardWalsh4x4Kernel. Bit-exact vs CPU reference
// Vp8ForwardTransform.ShortWalsh4x4 across every backend.

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
    public async Task Vp8ForwardWalsh4x4Kernel_ZeroInput_ProducesAllZero()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Vp8ForwardWalsh4x4Kernel(acc);
            var input = new short[16];
            using var dIn = acc.Allocate1D<short>(16);
            using var dOut = acc.Allocate1D<short>(16);
            dIn.View.CopyFromCPU(input);
            kernel.Run(dIn.View, dOut.View, blockCount: 1);
            await acc.SynchronizeAsync();
            var readback = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dOut);
            for (int i = 0; i < 16; i++) Equal((short)0, readback[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp8ForwardWalsh4x4Kernel_RandomInput_MatchesCpuReference()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Vp8ForwardWalsh4x4Kernel(acc);
            const int blockCount = 16;
            var rng = new Random(42);
            var input = new short[blockCount * 16];
            for (int i = 0; i < input.Length; i++) input[i] = (short)rng.Next(-512, 512);

            var cpuOut = new short[blockCount * 16];
            for (int b = 0; b < blockCount; b++)
                Vp8ForwardTransform.ShortWalsh4x4(
                    input.AsSpan(b * 16, 16), rowStrideShorts: 4,
                    cpuOut.AsSpan(b * 16, 16));

            using var dIn = acc.Allocate1D<short>(blockCount * 16);
            using var dOut = acc.Allocate1D<short>(blockCount * 16);
            dIn.View.CopyFromCPU(input);
            kernel.Run(dIn.View, dOut.View, blockCount);
            await acc.SynchronizeAsync();
            var readback = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dOut);
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
