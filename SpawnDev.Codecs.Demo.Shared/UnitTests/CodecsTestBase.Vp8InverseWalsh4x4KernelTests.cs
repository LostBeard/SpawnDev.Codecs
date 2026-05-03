// Tests for Vp8InverseWalsh4x4Kernel - bit-exact vs Vp8InverseTransform.ShortInvWalsh4x4.
// Uses GPU-side verification (Rule 5a): never download the full output
// buffer; upload the CPU reference and count mismatches on the device.

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
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Vp8InverseWalsh4x4Kernel(acc);
            const int blockCount = 4;
            int total = blockCount * 16;

            using var dIn = acc.Allocate1D<short>(total);
            using var dOut = acc.Allocate1D<short>(total);
            dIn.View.MemSetToZero();
            kernel.Run(dIn.View, dOut.View, blockCount);
            await acc.SynchronizeAsync();

            // Compare on GPU vs an all-zero expected buffer.
            int mismatches = await GpuTestVerifyCodecs.CountShortMismatches(
                acc, dOut.View, new short[total], total);
            Equal(0, mismatches);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp8InverseWalsh4x4Kernel_RandomInput_MatchesCpuReference()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Vp8InverseWalsh4x4Kernel(acc);
            const int blockCount = 16;
            int total = blockCount * 16;
            var rng = new Random(2027);
            var input = new short[total];
            for (int i = 0; i < input.Length; i++) input[i] = (short)rng.Next(-512, 512);

            // CPU reference output.
            var cpuOut = new short[total];
            for (int b = 0; b < blockCount; b++)
            {
                Vp8InverseTransform.ShortInvWalsh4x4(
                    input.AsSpan(b * 16, 16),
                    cpuOut.AsSpan(b * 16, 16));
            }

            // GPU kernel output - stays on device.
            using var dIn = acc.Allocate1D<short>(total);
            using var dOut = acc.Allocate1D<short>(total);
            dIn.View.CopyFromCPU(input);
            kernel.Run(dIn.View, dOut.View, blockCount);
            await acc.SynchronizeAsync();

            int mismatches = await GpuTestVerifyCodecs.CountShortMismatches(
                acc, dOut.View, cpuOut, total);
            Equal(0, mismatches);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
