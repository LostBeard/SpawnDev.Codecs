// Tests for Vp8InverseWalsh4x4Kernel - bit-exact vs Vp8InverseTransform.ShortInvWalsh4x4.

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
    public async Task Vp8InverseWalsh4x4Kernel_ZeroInput_ProducesZero()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var kernel = new Vp8InverseWalsh4x4Kernel(acc);
            const int blockCount = 4;
            var input = new short[blockCount * 16];

            using var dIn = acc.Allocate1D<short>(blockCount * 16);
            using var dOut = acc.Allocate1D<short>(blockCount * 16);
            dIn.View.CopyFromCPU(input);
            kernel.Run(dIn.View, dOut.View, blockCount);
            await acc.SynchronizeAsync();
            var readback = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dOut);
            var gpuOut = new short[blockCount * 16];
            readback.AsSpan(0, gpuOut.Length).CopyTo(gpuOut);

            int nonZero = 0;
            for (int i = 0; i < gpuOut.Length; i++)
                if (gpuOut[i] != 0) nonZero++;
            Equal(0, nonZero);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp8InverseWalsh4x4Kernel_RandomInput_MatchesCpuReference()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var kernel = new Vp8InverseWalsh4x4Kernel(acc);
            const int blockCount = 16;
            var rng = new Random(2027);
            var input = new short[blockCount * 16];
            for (int i = 0; i < input.Length; i++) input[i] = (short)rng.Next(-512, 512);

            // CPU reference output.
            var cpuOut = new short[blockCount * 16];
            for (int b = 0; b < blockCount; b++)
            {
                Vp8InverseTransform.ShortInvWalsh4x4(
                    input.AsSpan(b * 16, 16),
                    cpuOut.AsSpan(b * 16, 16));
            }

            // GPU kernel output.
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
